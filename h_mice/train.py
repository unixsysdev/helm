"""
H-MICE Training Loop

Features:
  - Bimodal loss: supervised scaffolding (0-2000) → geometric distortion (2001+)
  - Triple NaN guard: forward, backward, gradient
  - DeepSeekV3-style L2 load balancing
  - Zero-sync router telemetry: L1=[H:62% E:20% S:18%]
  - Streaming data pipeline with geometric metadata
"""

import os
import time
import geoopt
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import HMICETransformer
from data import HMICEDataset, generate_mock_data


def parse_args():
    p = argparse.ArgumentParser(description="H-MICE Training")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--steps", type=int, default=16000)
    p.add_argument("--warmup_steps", type=int, default=500)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--grad_clip", type=float, default=0.5)
    p.add_argument("--save_dir", type=str, default="/tmp/checkpoints/h_mice")
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mock", action="store_true", help="Use mock data for trial run")
    p.add_argument("--mock_steps", type=int, default=500, help="Steps for mock trial")
    # Bimodal loss params
    p.add_argument("--scaffold_steps", type=int, default=2000)
    p.add_argument("--scaffold_alpha", type=float, default=1.0, help="Scaffold loss weight")
    p.add_argument("--geom_lambda", type=float, default=0.01, help="Geometric distortion weight")
    return p.parse_args()


def find_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None
    pts = sorted(
        [f for f in os.listdir(save_dir) if f.startswith("hmice_step") and f.endswith(".pt")],
        key=lambda f: int(f.split("step")[1].split(".")[0])
    )
    return os.path.join(save_dir, pts[-1]) if pts else None


def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    geom_targets = torch.stack([b['geom_targets'] for b in batch])
    return input_ids, geom_targets


def lr_schedule(step, warmup, total, base_lr):
    if step < warmup:
        return base_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return base_lr * max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))


def main():
    import math
    args = parse_args()

    if args.mock:
        args.steps = args.mock_steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB")

    print(f"\nConfig:")
    print(f"  steps={args.steps}, bs={args.batch_size}, seq={args.seq_len}")
    print(f"  effective_bs={args.batch_size * args.grad_accum}")
    print(f"  tokens/step={args.batch_size * args.grad_accum * args.seq_len:,}")
    print(f"  lr={args.lr}, grad_clip={args.grad_clip}")
    print(f"  scaffold_steps={args.scaffold_steps}, alpha={args.scaffold_alpha}")
    print(f"  geom_lambda={args.geom_lambda}")
    print(f"  mock={'YES' if args.mock else 'NO'}")

    # --- Data (need texts first to train tokenizer) ---
    from tokenizer_utils import get_tokenizer

    if args.mock:
        print("\nGenerating mock data for trial run...")
        mock_data = generate_mock_data(n_samples=50000)
        train_texts = [s['text'] for s in mock_data]
    else:
        print("\nInitializing streaming 60/20/20 mix (CoT / Code / Text)...")
        mock_data = None
        train_texts = None  # Will use cached tokenizer

    # --- Tokenizer (8192 BPE) ---
    print("\nLoading/training 8192-BPE tokenizer...")
    tokenizer = get_tokenizer(train_texts=train_texts)
    vocab_size = tokenizer.vocab_size
    print(f"  Vocab: {vocab_size}")

    # --- Dataset ---
    if args.mock:
        target_tokens = args.mock_steps * args.batch_size * args.grad_accum * args.seq_len
        dataset = HMICEDataset(tokenizer, args.seq_len, target_tokens, mock_data=mock_data)
    else:
        target_tokens = args.steps * args.batch_size * args.grad_accum * args.seq_len
        dataset = HMICEDataset(tokenizer, args.seq_len, target_tokens)

    loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                        num_workers=0, pin_memory=True)
    loader_iter = iter(loader)
    print("  Data ready")

    # --- Model ---
    print(f"\nBuilding H-MICE v3 (vocab={vocab_size}, dim=1024, inter=2048, layers=16)...")
    model = HMICETransformer(
        vocab_size=vocab_size,
        dim=1024,
        n_layers=16,
        n_heads=16,
        inter_dim=2048,
        max_seq_len=args.seq_len,
    )
    total_params = model.count_params()
    print(f"  Total parameters: {total_params:,} ({total_params/1e6:.0f}M)")

    # --- Resume ---
    start_step = 0
    _resume_optimizer_state = None
    if args.resume:
        ckpt_path = find_latest_checkpoint(args.save_dir)
        if ckpt_path:
            print(f"\nResuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict = ckpt['model_state_dict']
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            loaded = model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded {len(state_dict) - len(loaded.unexpected_keys)}/{len(state_dict)} keys")
            start_step = ckpt.get('global_step', 0)
            _resume_optimizer_state = ckpt.get('optimizer_state_dict', None)
            print(f"  Restored step {start_step}")

    model = model.to(device)

    # --- Dual Optimizer ---
    hyp_params = model.get_hyperbolic_params()
    euc_params = model.get_euclidean_params()
    print(f"  Hyperbolic params (ManifoldParameter): {sum(p.numel() for p in hyp_params):,}")
    print(f"  Euclidean params: {sum(p.numel() for p in euc_params):,}")

    # RiemannianAdam for ManifoldParameter (embeddings on Lorentz)
    opt_hyp = geoopt.optim.RiemannianAdam(hyp_params, lr=args.lr, weight_decay=0.0, stabilize=10)
    # AdamW for everything else
    opt_euc = torch.optim.AdamW(euc_params, lr=args.lr, betas=(0.9, 0.95),
                                 weight_decay=0.1, fused=True)
    print(f"  RiemannianAdam (hyp, wd=0) + AdamW (euc, wd=0.1)")

    if _resume_optimizer_state:
        try:
            if 'opt_hyp' in _resume_optimizer_state:
                opt_hyp.load_state_dict(_resume_optimizer_state['opt_hyp'])
                opt_euc.load_state_dict(_resume_optimizer_state['opt_euc'])
                print("  Both optimizer states restored")
        except Exception as e:
            print(f"  Optimizer restore failed: {e}")

    for pg in opt_hyp.param_groups + opt_euc.param_groups:
        pg['lr'] = args.lr
    print(f"  Forced optimizer LR to {args.lr}")

    # --- Scheduler (on Euclidean optimizer) ---
    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt_euc, lr_lambda)
    sched_hyp = torch.optim.lr_scheduler.LambdaLR(opt_hyp, lr_lambda)
    for _ in range(start_step):
        scheduler.step()
        sched_hyp.step()

    # --- Compile ---
    print("\nCompiling model with torch.compile...")
    model = torch.compile(model)
    print("  Compiled (default mode)")

    # --- Training ---
    os.makedirs(args.save_dir, exist_ok=True)
    step = start_step
    accum_loss = 0.0
    accum_count = 0
    total_tokens = 0
    step_start = time.time()

    print(f"\n{'='*60}")
    print(f"Starting training from step {start_step}")
    print(f"{'='*60}\n")

    while step < args.steps:
        try:
            input_ids, geom_targets = next(loader_iter)
        except StopIteration:
            if args.mock:
                mock_data = generate_mock_data(n_samples=50000)
                dataset = HMICEDataset(tokenizer, args.seq_len,
                                       args.steps * args.batch_size * args.grad_accum * args.seq_len,
                                       mock_data=mock_data)
                loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                    num_workers=0, pin_memory=True)
                loader_iter = iter(loader)
                continue
            break

        input_ids = input_ids.to(device)
        geom_targets = geom_targets.to(device)

        inputs = input_ids[:, :-1].contiguous()
        targets = input_ids[:, 1:].contiguous()
        geom_tgt = geom_targets[:, :-1].contiguous()

        # --- Forward ---
        try:
            logits, l1_logits, geom_loss = model(inputs)
            logits = logits[:, :, :vocab_size]

            B, T, V = logits.shape
            text_loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
                ignore_index=tokenizer.pad_token_id,
            )

            # --- Bimodal Loss ---
            if step < args.scaffold_steps:
                # Phase 1: Supervised scaffolding
                scaffold_loss = F.cross_entropy(
                    l1_logits.reshape(B * T, 3),
                    geom_tgt.reshape(B * T),
                )
                loss = text_loss + args.scaffold_alpha * scaffold_loss
            else:
                # Phase 2: Geometric distortion penalty
                loss = text_loss + args.geom_lambda * geom_loss

            loss = loss / args.grad_accum

            # Guard 1: NaN in forward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Step {step}: NaN/Inf loss detected, skipping batch")
                del logits, loss, l1_logits, geom_loss
                opt_hyp.zero_grad(set_to_none=True)
                opt_euc.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                accum_loss = 0.0
                accum_count = 0
                continue

            loss.backward()
        except RuntimeError as e:
            if 'nan' in str(e).lower() or 'inf' in str(e).lower() or 'out of memory' in str(e).lower():
                print(f"  Step {step}: {str(e)[:80]}, skipping batch")
                opt_hyp.zero_grad(set_to_none=True)
                opt_euc.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                accum_loss = 0.0
                accum_count = 0
                continue
            raise

        accum_loss += loss.item() * args.grad_accum
        accum_count += 1
        total_tokens += input_ids.numel()

        # --- Gradient Accumulation Step ---
        if accum_count >= args.grad_accum:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Guard 2: NaN in gradients
            has_nan_grad = torch.isnan(grad_norm) or torch.isinf(grad_norm)
            if not has_nan_grad:
                for p in model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break

            if has_nan_grad:
                print(f"  Step {step+1}: NaN gradient detected, purging and skipping")
                opt_hyp.zero_grad(set_to_none=True)
                opt_euc.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                accum_loss = 0.0
                accum_count = 0
                continue

            opt_hyp.step()
            opt_euc.step()
            scheduler.step()
            sched_hyp.step()
            opt_hyp.zero_grad(set_to_none=True)
            opt_euc.zero_grad(set_to_none=True)
            step += 1
            avg_loss = accum_loss / accum_count

            # --- Log ---
            if step % args.log_every == 0:
                elapsed = time.time() - step_start
                ms_per_step = elapsed / args.log_every * 1000
                tok_per_s = (args.batch_size * args.grad_accum * args.seq_len * args.log_every) / elapsed
                current_lr = scheduler.get_last_lr()[0]

                # Router telemetry (zero-sync)
                l1_dist = model._orig_mod.get_l1_distribution() if hasattr(model, '_orig_mod') else model.get_l1_distribution()
                h_pct = int(l1_dist[0].item() * 100)  # Only .item() on accumulated EMA
                e_pct = int(l1_dist[1].item() * 100)
                s_pct = int(l1_dist[2].item() * 100)

                phase = "scaffold" if step < args.scaffold_steps else "geometric"
                eta_min = (args.steps - step) * ms_per_step / 60000
                print(
                    f"Step {step:5d}/{args.steps} | loss={avg_loss:.4f} | "
                    f"lr={current_lr:.2e} | {ms_per_step:.0f}ms/step | "
                    f"{tok_per_s:.0f} tok/s | grad={grad_norm:.2f} | "
                    f"L1=[H:{h_pct}% E:{e_pct}% S:{s_pct}%] | "
                    f"{phase} | ETA {eta_min:.0f}m"
                )
                step_start = time.time()

            # --- Save ---
            if step % args.save_every == 0:
                save_path = os.path.join(args.save_dir, f"hmice_step{step}.pt")
                torch.save({
                    'global_step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': {
                        'opt_hyp': opt_hyp.state_dict(),
                        'opt_euc': opt_euc.state_dict(),
                    },
                    'loss': avg_loss,
                    'config': {
                        'arch': 'H-MICE-v3-L16W1024-E9-interleaved',
                        'vocab_size': vocab_size,
                        'dim': 1024,
                        'inter_dim': 2048,
                        'n_layers': 16,
                        'n_heads': 16,
                        'experts': '4E+4H+1S',
                    }
                }, save_path)
                print(f"  Saved: {save_path}")

            # Reset accumulators
            accum_loss = 0.0
            accum_count = 0

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
