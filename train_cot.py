"""
HELM-D CoT Reasoning Engine — 133M Pretraining
================================================
Train a 133M hyperbolic transformer from scratch on a 60/20/20 mix:
  60% OpenThoughts CoT reasoning (math, code, science)
  20% Python code (Stack-Edu / python-edu)
  20% Educational text (Cosmopedia-v2)

Architecture:
  - 32K TinyLlama tokenizer (dense coverage, no dead tokens)
  - 4096 context (full CoT traces fit in one pass)
  - L6W384A6 (6 layers, width 384, 6 heads)
  - Lorentz manifold throughout

Data pipeline:
  - Streaming via interleave_datasets (no pre-download)
  - Tokenize on-the-fly, prefetch with DataLoader workers

Run locally (AMD Strix Halo, ROCm):
  toolbox run -c llama-rocm-7.2 bash -c 'source venv/bin/activate && python train_cot.py'

Run on H200:
  python3.12 -O train_cot.py --batch_size 32 --grad_accum 4
"""

import os
import sys
import time
import math
import argparse
import signal

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helm-src'))

import torch

# === TF32: unlock Tensor Cores for FP32 matmuls (7x faster than CUDA cores) ===
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset, interleave_datasets
from geoopt.optim import RiemannianAdam

from helm.hypercore.manifolds import Lorentz
from helm.modules.helm_d import LTransformerDecoder


# ==============================================================================
# Streaming Dataset
# ==============================================================================

class StreamingCoTDataset(IterableDataset):
    """60/20/20 streaming mix: CoT / Code / Text.
    
    Tokenizes on-the-fly, packs into fixed-length chunks.
    No pre-download — starts training in seconds.
    """

    def __init__(self, tokenizer, seq_len=4096, target_tokens=2_660_000_000, hf_token=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.target_tokens = target_tokens
        self.hf_token = hf_token
        self.eos_id = tokenizer.eos_token_id

    def _open_streams(self):
        """Open 3 parallel HF streams."""
        # 60% — OpenThoughts CoT reasoning
        cot = load_dataset(
            "open-thoughts/OpenThoughts-114k",
            split="train",
            streaming=True,
            token=self.hf_token,
        )

        # 20% — Python code (educational)
        code = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "python-edu",
            split="train",
            streaming=True,
            token=self.hf_token,
        )

        # 20% — Educational text (Cosmopedia)
        text = load_dataset(
            "HuggingFaceTB/smollm-corpus",
            "cosmopedia-v2",
            split="train",
            streaming=True,
            token=self.hf_token,
        )

        # Interleave at 60/20/20 ratio
        mixed = interleave_datasets(
            [cot, code, text],
            probabilities=[0.60, 0.20, 0.20],
            seed=42,
            stopping_strategy="all_exhausted",
        )
        return mixed

    def _extract_text(self, example):
        """Extract text from any of the 3 dataset formats."""
        # OpenThoughts: conversations list with role/content
        if 'conversations' in example and example['conversations']:
            parts = []
            for turn in example['conversations']:
                role = turn.get('role', turn.get('from', ''))
                content = turn.get('content', turn.get('value', ''))
                if role and content:
                    parts.append(f"<|{role}|>\n{content}")
            return '\n'.join(parts)

        # Code (python-edu): has 'text' directly or blob content
        if 'text' in example and example['text']:
            return example['text']

        # Cosmopedia: has 'text'
        if 'content' in example:
            return example['content']

        # Fallback
        for key in ['text', 'content', 'code', 'document']:
            if key in example and example[key]:
                return str(example[key])
        return ''

    def __iter__(self):
        """Yield seq_len chunks with shuffle buffer to prevent domain clustering."""
        import random as _rng
        SHUFFLE_BUFFER = 512  # accumulate chunks before shuffling

        mixed = self._open_streams()
        token_buffer = []
        chunk_buffer = []
        total_tokens = 0

        for example in mixed:
            if total_tokens >= self.target_tokens:
                break

            text = self._extract_text(example)
            if len(text) < 50:
                continue

            ids = self.tokenizer.encode(
                text[:16384],
                add_special_tokens=False,
            )
            token_buffer.extend(ids)
            token_buffer.append(self.eos_id)
            total_tokens += len(ids) + 1

            while len(token_buffer) >= self.seq_len:
                chunk = token_buffer[:self.seq_len]
                token_buffer = token_buffer[self.seq_len:]
                chunk_buffer.append(chunk)

            # Shuffle and yield when buffer full
            if len(chunk_buffer) >= SHUFFLE_BUFFER:
                _rng.shuffle(chunk_buffer)
                for c in chunk_buffer:
                    yield {'input_ids': torch.tensor(c, dtype=torch.long)}
                chunk_buffer = []

        # Flush remaining
        if chunk_buffer:
            _rng.shuffle(chunk_buffer)
            for c in chunk_buffer:
                yield {'input_ids': torch.tensor(c, dtype=torch.long)}


# ==============================================================================
# Model Initialization
# ==============================================================================

def init_riemannian_embeddings(model, vocab_size, width):
    """Initialize embeddings as Riemannian Normal on Lorentz manifold.
    
    Each token starts as random noise perfectly on the -1 surface:
      x0 = sqrt(1 + ||x_spatial||^2)
      x_spatial ~ N(0, 0.01)
    """
    with torch.no_grad():
        spatial = torch.randn(vocab_size, width - 1) * 0.01
        time_comp = torch.sqrt(1.0 + (spatial ** 2).sum(dim=-1, keepdim=True))
        emb = torch.cat([time_comp, spatial], dim=-1)

        # Verify constraint
        constraint = -(emb[:, 0] ** 2) + (emb[:, 1:] ** 2).sum(dim=-1)
        print(f"  Lorentz constraint: {constraint.mean():.6f} ± {constraint.std():.6f}")

        # Set into model
        for name, param in model.named_parameters():
            if 'embed' in name and 'embedding' in name and param.shape == emb.shape:
                param.data.copy_(emb)
                print(f"  Set {name} -> {list(param.shape)}")
                return True

        # Try alternative key patterns
        for name, param in model.named_parameters():
            if param.shape[0] == vocab_size and param.shape[1] == width:
                param.data.copy_(emb)
                print(f"  Set {name} -> {list(param.shape)}")
                return True

    print("  WARNING: Could not find embedding parameter!")
    return False


def init_output_mapping(model, vocab_size, width):
    """Initialize output mapping matrix with small random weights."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'mapping' in name and param.shape[0] == vocab_size:
                param.data.normal_(0, 0.01)
                print(f"  Set {name} -> {list(param.shape)}")
                return True
    print("  WARNING: Could not find mapping parameter!")
    return False


# ==============================================================================
# Training Loop
# ==============================================================================

def check_disk_space(path):
    st = os.statvfs(path if os.path.exists(path) else '.')
    free = st.f_bavail * st.f_frsize / 1e9
    total = st.f_blocks * st.f_frsize / 1e9
    return free, total


def find_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None
    ckpts = [f for f in os.listdir(save_dir) if f.startswith('cot_step') and f.endswith('.pt')]
    if not ckpts:
        return None
    ckpts.sort(key=lambda x: int(x.split('step')[1].split('.')[0]))
    return os.path.join(save_dir, ckpts[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=16000)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--target_tokens', type=int, default=2_660_000_000)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_dir', type=str, default='checkpoints/cot')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--sample_every', type=int, default=200)
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--reproject_every', type=int, default=100)
    parser.add_argument('--hf_token', type=str, default=None)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"VRAM: {vram_gb:.1f} GB")
    else:
        print(f"Device: {device}")

    print(f"\nConfig:")
    print(f"  steps={args.steps}, bs={args.batch_size}, seq={args.seq_len}")
    print(f"  effective_bs={args.batch_size * args.grad_accum}")
    print(f"  tokens/step={args.batch_size * args.grad_accum * args.seq_len:,}")
    print(f"  target_tokens={args.target_tokens:,}")
    print(f"  lr={args.lr}, grad_clip={args.grad_clip}")

    # --- Tokenizer ---
    print("\nLoading TinyLlama 32K tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)
    print(f"  Vocab: {vocab_size}")

    # --- Streaming Dataset ---
    print("\nInitializing streaming 60/20/20 mix (CoT / Code / Text)...")
    stream_ds = StreamingCoTDataset(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        target_tokens=args.target_tokens,
        hf_token=args.hf_token,
    )
    dataloader = DataLoader(
        stream_ds,
        batch_size=args.batch_size,
        num_workers=2,
        prefetch_factor=4,
        pin_memory=True,
    )
    print("  Streaming ready (no pre-download)")

    # --- Model ---
    WIDTH = 768
    print(f"\nBuilding HELM-D (vocab={vocab_size}, width={WIDTH}, ctx={args.seq_len})...")
    model = LTransformerDecoder(
        manifold_in=Lorentz(1.0),
        manifold_hidden=Lorentz(1.0),
        manifold_out=Lorentz(1.0),
        arch="L16W768A12",
        vocab_size=vocab_size,
        context_length=args.seq_len,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params:,} ({num_params / 1e6:.0f}M)")

    # --- Initialize fresh embeddings ---
    print("\nInitializing Riemannian Normal embeddings...")
    init_riemannian_embeddings(model, vocab_size, WIDTH)
    init_output_mapping(model, vocab_size, WIDTH)

    # --- Resume ---
    start_step = 0
    _resume_optimizer_state = None
    if args.resume:
        ckpt_path = find_latest_checkpoint(args.save_dir)
        if ckpt_path:
            print(f"\nResuming from: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # Strip torch.compile's '_orig_mod.' prefix from state dict keys
            state_dict = ckpt['model_state_dict']
            state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            loaded_keys = model.load_state_dict(state_dict, strict=False)
            print(f"  Loaded {len(state_dict) - len(loaded_keys.unexpected_keys)}/{len(state_dict)} keys")
            if loaded_keys.missing_keys:
                print(f"  Missing: {loaded_keys.missing_keys[:3]}...")
            start_step = ckpt.get('global_step', 0)
            _resume_optimizer_state = ckpt.get('optimizer_state_dict', None)
            print(f"  Restored step {start_step}")

    model = model.to(device)

    # --- Optimizer (dual-group: Euclidean gets decay, Manifold gets zero decay) ---
    import geoopt
    euclidean_params = []
    hyperbolic_params = []
    for name, param in model.named_parameters():
        if isinstance(param, geoopt.ManifoldParameter):
            hyperbolic_params.append(param)
        else:
            euclidean_params.append(param)
    print(f"  Euclidean params: {sum(p.numel() for p in euclidean_params):,}")
    print(f"  Hyperbolic params (zero decay): {sum(p.numel() for p in hyperbolic_params):,}")
    optimizer = RiemannianAdam([
        {"params": euclidean_params, "weight_decay": 0.01},
        {"params": hyperbolic_params, "weight_decay": 0.0},
    ], lr=args.lr)
    if _resume_optimizer_state:
        try:
            # Inject missing 'step' key (geoopt saves without it, newer PyTorch expects it)
            for k, v in _resume_optimizer_state.get("state", {}).items():
                if "step" not in v:
                    v["step"] = torch.tensor(float(start_step))
            optimizer.load_state_dict(_resume_optimizer_state)
            print("  Optimizer state restored (with momentum)")
        except Exception as e:
            print(f"  Optimizer state restore failed: {e}, using fresh")

    # Force LR from args (override checkpoint LR)
    for pg in optimizer.param_groups:
        pg["lr"] = args.lr
        pg["initial_lr"] = args.lr
    print(f"  Forced optimizer LR to {args.lr}")

    # --- LR Scheduler ---
    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.steps - args.warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    for _ in range(start_step):
        scheduler.step()

    # --- Compile ---
    if hasattr(torch, 'compile'):
        print("\nCompiling model with torch.compile...")
        try:
            model = torch.compile(model, mode='default')
            print("  Compiled (default mode)")
        except Exception as e:
            print(f"  Compile failed: {e}, running eager")

    # --- Training ---
    print(f"\n{'='*60}")
    print(f"Starting training from step {start_step}")
    print(f"{'='*60}\n")

    os.makedirs(args.save_dir, exist_ok=True)
    model.train()
    optimizer.zero_grad()

    step = start_step
    accum_loss = 0.0
    accum_count = 0
    step_start = time.time()
    total_tokens = 0

    data_iter = iter(dataloader)

    while step < args.steps:
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            print(f"\n  Dataset exhausted at step {step}, restarting stream...")
            data_iter = iter(dataloader)
            batch = next(data_iter)

        input_ids = batch['input_ids'].to(device)
        targets = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()

        # Forward
        try:
            logits = model(inputs)[:, :, :vocab_size]

            # Reshape for cross-entropy
            B, T, V = logits.shape
            loss = F.cross_entropy(
                logits.reshape(B * T, V),
                targets.reshape(B * T),
                ignore_index=tokenizer.pad_token_id,
            )
            loss = loss / args.grad_accum

            # Guard 1: NaN in forward pass — skip BEFORE backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  Step {step}: NaN/Inf loss detected, skipping batch")
                del logits, loss
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                accum_loss = 0.0
                accum_count = 0
                continue

            loss.backward()
        except RuntimeError as e:
            if 'nan' in str(e).lower() or 'inf' in str(e).lower() or 'out of memory' in str(e).lower():
                print(f"  Step {step}: {str(e)[:80]}, skipping batch")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                accum_loss = 0.0
                accum_count = 0
                continue
            raise

        accum_loss += loss.item() * args.grad_accum
        accum_count += 1
        total_tokens += input_ids.numel()

        # Gradient accumulation step
        if accum_count >= args.grad_accum:
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            # Guard: NaN in gradients — scan ALL parameters directly
            has_nan_grad = torch.isnan(grad_norm) or torch.isinf(grad_norm)
            if not has_nan_grad:
                for p in model.parameters():
                    if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                        has_nan_grad = True
                        break

            if has_nan_grad:
                print(f"  Step {step+1}: NaN gradient detected, purging and skipping")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                accum_loss = 0.0
                accum_count = 0
                continue

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            avg_loss = accum_loss / accum_count

            # --- Log ---
            if step % args.log_every == 0:
                elapsed = time.time() - step_start
                ms_per_step = elapsed / args.log_every * 1000
                tok_per_s = (args.batch_size * args.grad_accum * args.seq_len * args.log_every) / elapsed
                current_lr = scheduler.get_last_lr()[0]

                # Manifold check: Lorentz constraint should be -1.0, x₀ > 0
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if 'embed' in name and len(param.shape) == 2 and param.shape[0] == vocab_size:
                            sample = param[:100]
                            constraint = -(sample[:, 0] ** 2) + (sample[:, 1:] ** 2).sum(dim=-1)
                            m_mean, m_std = constraint.mean().item(), constraint.std().item()
                            x0_neg = (sample[:, 0] < 0).sum().item()
                            break
                    else:
                        m_mean, m_std, x0_neg = 0.0, 0.0, 0

                eta_min = (args.steps - step) * ms_per_step / 60000
                x0_str = f" x0_neg={x0_neg}" if x0_neg > 0 else ""
                print(
                    f"Step {step:5d}/{args.steps} | loss={avg_loss:.4f} | "
                    f"lr={current_lr:.2e} | {ms_per_step:.0f}ms/step | "
                    f"{tok_per_s:.0f} tok/s | grad={grad_norm:.2f} | "
                    f"manifold={m_mean:.4f}±{m_std:.4f}{x0_str} | "
                    f"ETA {eta_min:.0f}m"
                )
                step_start = time.time()

            # --- Sample ---
            if step % args.sample_every == 0:
                model.eval()
                with torch.no_grad():
                    prompt = "def fibonacci(n):"
                    input_ids_gen = tokenizer.encode(prompt, return_tensors='pt').to(device)
                    generated = input_ids_gen[0].tolist()
                    for _ in range(100):
                        inp = torch.tensor([generated[-args.seq_len:]], device=device)
                        out = model(inp)[:, -1, :vocab_size]
                        next_id = torch.argmax(out, dim=-1).item()
                        generated.append(next_id)
                        if next_id == tokenizer.eos_token_id:
                            break
                    text = tokenizer.decode(generated, skip_special_tokens=True)
                    print(f"\n  Sample @ step {step}:\n  {text[:300]}\n")
                model.train()

            # --- Re-project ---
            if step % args.reproject_every == 0:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if 'embed' in name and len(param.shape) == 2 and param.shape[0] == vocab_size:
                            spatial = param[:, 1:]
                            time_comp = torch.sqrt(1.0 + (spatial ** 2).sum(dim=-1, keepdim=True))
                            param[:, 0] = time_comp.squeeze(-1)

            # --- Save ---
            if step % args.save_every == 0:
                ckpt_path = os.path.join(args.save_dir, f"cot_step{step}.pt")
                torch.save({
                    'global_step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': avg_loss,
                    'total_tokens': total_tokens,
                    'config': {
                        'arch': 'L16W768A12',
                        'vocab_size': vocab_size,
                        'seq_len': args.seq_len,
                        'tokenizer': 'TinyLlama/TinyLlama-1.1B-Chat-v1.0',
                    },
                }, ckpt_path)
                print(f"  Saved: {ckpt_path}")

                # Rolling cleanup
                ckpts = sorted(
                    [f for f in os.listdir(args.save_dir) if f.startswith('cot_step')],
                    key=lambda x: int(x.split('step')[1].split('.')[0])
                )
                while len(ckpts) > 5:
                    old = os.path.join(args.save_dir, ckpts.pop(0))
                    os.remove(old)
                    print(f"  Removed old: {old}")

            accum_loss = 0.0
            accum_count = 0

    print(f"\n{'='*60}")
    print(f"Training complete at step {step}")
    print(f"Total tokens seen: {total_tokens:,}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
