"""
HYPER-HELM: Production H200 Training Script
Hardened for 2B+ token pretraining on NVIDIA H200 (141GB HBM3e).

Config: batch=256, seq=2048, 133M HELM-D (Qwen3 tokenizer, post-surgery)

NaN Failsafes:
  1. Periodic Lorentz manifold re-projection (every 100 steps)
  2. Aggressive gradient clipping (0.5 instead of 1.0)
  3. NaN detection with auto-rollback to last clean checkpoint
  4. Manifold constraint monitoring (t² - ||x||² should stay ≈ 1.0)

Usage:
  pip install geoopt transformers datasets huggingface_hub
  python3 -u train_h200.py --steps 4000 --batch_size 256 --seq_len 2048 2>&1 | tee train.log
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helm-src'))

import torch
import torch.nn.functional as F
import argparse
import time
import math
import copy
import shutil
import signal
import glob
import json
from pathlib import Path

from helm.hypercore.manifolds import Lorentz
from helm.modules.helm_d import LTransformerDecoder
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset, load_from_disk
from geoopt import ManifoldParameter
from geoopt.optim import RiemannianAdam

# Liger-Kernel: Triton-fused linear + cross-entropy (logits never in HBM)
try:
    from liger_kernel.transformers import LigerFusedLinearCrossEntropyLoss
    fused_ce_loss = LigerFusedLinearCrossEntropyLoss()
    print("✓ Liger-Kernel fused CE loaded — logits will NEVER materialize in VRAM")
except ImportError:
    fused_ce_loss = None
    print("⚠ Liger-Kernel not found — using BF16 standard CE (pip install liger-kernel)")


# ============================================================
# NaN FAILSAFE #1: Lorentz Manifold Re-projection
# ============================================================
def project_to_lorentz(model, c=1.0):
    """
    Re-project all embedding weights back onto the Lorentz manifold.
    Ensures: x₀ = sqrt(||x_space||² + 1/c)
    
    Called periodically to prevent drift from the manifold due to
    floating-point accumulation in gradient updates.
    """
    proj_count = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'embed' in name.lower() and 'embedding' in name.lower():
                # This is the token embedding — project each row
                space = param.data[:, 1:]
                space_sq = (space ** 2).sum(dim=-1, keepdim=True)
                param.data[:, 0:1] = (space_sq + c).sqrt()
                proj_count += 1
    return proj_count


# ============================================================
# NaN FAILSAFE #4: Manifold Constraint Monitor
# ============================================================
def check_manifold_constraint(model, c=1.0):
    """
    Verify the Lorentz constraint: t² - ||x||² = 1/c
    Returns mean and max deviation from the constraint.
    """
    for name, param in model.named_parameters():
        if 'embed' in name.lower() and 'embedding' in name.lower():
            with torch.no_grad():
                time_sq = param.data[:, 0] ** 2
                space_sq = (param.data[:, 1:] ** 2).sum(dim=-1)
                constraint = time_sq - space_sq  # Should be ≈ c
                deviation = (constraint - c).abs()
                return {
                    'mean_constraint': constraint.mean().item(),
                    'max_deviation': deviation.max().item(),
                    'mean_deviation': deviation.mean().item(),
                }
    return {'mean_constraint': 0, 'max_deviation': 0, 'mean_deviation': 0}


# ============================================================
# NaN FAILSAFE #3: NaN Detection + Auto-rollback
# ============================================================
# ============================================================
# SPOT INSTANCE: Graceful preemption handler
# ============================================================
_PREEMPTED = False
def _sigterm_handler(signum, frame):
    global _PREEMPTED
    _PREEMPTED = True
    print(f"\n{'!'*60}")
    print(f"  SIGTERM received — spot preemption detected!")
    print(f"  Saving emergency checkpoint on next step...")
    print(f"{'!'*60}\n")

signal.signal(signal.SIGTERM, _sigterm_handler)
signal.signal(signal.SIGINT, _sigterm_handler)


def check_disk_space(path='.'):
    """Check available disk space in GB at the given path."""
    import os
    stat = os.statvfs(path)
    free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
    total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
    return free_gb, total_gb


def find_latest_checkpoint(ckpt_dir):
    """Find the latest checkpoint in a directory by step number."""
    pattern = os.path.join(ckpt_dir, 'h200_step*.pt')
    files = glob.glob(pattern)
    if not files:
        return None
    # Extract step numbers and find max
    def get_step(f):
        try:
            return int(f.split('step')[1].split('.pt')[0])
        except:
            return -1
    latest = max(files, key=get_step)
    return latest


class RollingCheckpointer:
    """
    Saves checkpoints with rolling history. Keeps the last N checkpoints
    to allow recovery from NaN or spot preemption.
    """
    def __init__(self, save_dir, keep_last=5):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.saved_paths = []
    
    def save(self, step, model, optimizer, scheduler, extra=None):
        """Save checkpoint and prune old ones."""
        path = self.save_dir / f'h200_step{step}.pt'
        save_dict = {
            'global_step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        if extra:
            save_dict.update(extra)
        torch.save(save_dict, path)
        self.saved_paths.append(str(path))
        
        # Prune old checkpoints (keep last N)
        while len(self.saved_paths) > self.keep_last:
            old = self.saved_paths.pop(0)
            if os.path.exists(old) and 'final' not in old:
                os.remove(old)
                print(f"    Pruned old checkpoint: {os.path.basename(old)}")
        
        free_gb, _ = check_disk_space(str(self.save_dir))
        print(f"  💾 Saved: {path.name} (disk: {free_gb:.1f}GB free)")
        return str(path)


class NaNRollbackHandler:
    """
    Saves model state periodically. On NaN detection, automatically
    rolls back to last clean checkpoint and reduces learning rate.
    """
    def __init__(self, model, optimizer, scheduler, save_dir, max_rollbacks=5):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.max_rollbacks = max_rollbacks
        self.rollback_count = 0
        self.last_clean_state = None
        self.last_clean_step = 0
        self.lr_scale = 1.0

    def save_clean_state(self, step):
        """Save a snapshot of the model in its last known-good state."""
        self.last_clean_state = {
            'model': copy.deepcopy(self.model.state_dict()),
            'optimizer': copy.deepcopy(self.optimizer.state_dict()),
            'step': step,
        }
        self.last_clean_step = step

    def check_and_rollback(self, loss, step):
        """
        Check if loss is NaN. If so, rollback and reduce LR.
        Returns: (should_continue, new_step)
        """
        if not (torch.isnan(torch.tensor(loss)) or torch.isinf(torch.tensor(loss))):
            return True, step

        self.rollback_count += 1
        print(f"\n{'!'*60}")
        print(f"  NaN DETECTED at step {step}! Rollback #{self.rollback_count}")
        print(f"{'!'*60}")

        if self.rollback_count > self.max_rollbacks:
            print(f"  FATAL: Max rollbacks ({self.max_rollbacks}) exceeded. Stopping.")
            return False, step

        if self.last_clean_state is None:
            print(f"  FATAL: No clean state to rollback to.")
            return False, step

        # Rollback model and optimizer
        self.model.load_state_dict(self.last_clean_state['model'])
        self.optimizer.load_state_dict(self.last_clean_state['optimizer'])

        # Reduce learning rate by 50%
        self.lr_scale *= 0.5
        for pg in self.optimizer.param_groups:
            pg['lr'] *= 0.5

        # Re-project embeddings to manifold
        proj_count = project_to_lorentz(self.model)

        print(f"  Rolled back to step {self.last_clean_step}")
        print(f"  LR reduced by 50% (scale={self.lr_scale:.4f})")
        print(f"  Re-projected {proj_count} embedding(s) to Lorentz manifold")
        print(f"  Resuming from step {self.last_clean_step}...")
        print(f"{'!'*60}\n")

        return True, self.last_clean_step


def prepare_mixed_data(tokenizer, seq_len, num_wiki=100000, num_code=100000, hf_token=None):
    """Download and tokenize mixed Wikipedia + Python code."""
    cache_path = Path('data') / f'mixed_wiki_code_{seq_len}_h200'
    if cache_path.exists():
        print(f"Loading cached dataset from {cache_path}")
        return load_from_disk(str(cache_path))

    all_ids = []

    # --- Wikipedia ---
    print(f"Downloading {num_wiki} Wikipedia articles...")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
    count = 0
    for example in wiki:
        if count >= num_wiki:
            break
        text = example.get('text', '')
        if len(text) > 200:
            ids = tokenizer.encode(text, add_special_tokens=False)
            all_ids.extend(ids)
            all_ids.append(tokenizer.eos_token_id)
            count += 1
        if count % 20000 == 0 and count > 0:
            print(f"  Wiki: {count}/{num_wiki} ({len(all_ids):,} tokens)")
    wiki_tokens = len(all_ids)
    print(f"  Wiki total: {wiki_tokens:,} tokens")

    # --- Python Code (LOCAL FIRST — no download needed) ---
    print(f"\nCollecting Python code from local installed packages...")
    local_code_dirs = [
        "/opt/conda/lib/python3.12/",
        "/opt/conda/lib/python3.12/site-packages/torch/",
        "/opt/conda/lib/python3.12/site-packages/transformers/",
        "/opt/conda/lib/python3.12/site-packages/numpy/",
        "/opt/conda/lib/python3.12/site-packages/datasets/",
        "/opt/conda/lib/python3.12/site-packages/scipy/",
        "/opt/conda/lib/python3.12/site-packages/sklearn/",
        "/opt/conda/lib/python3.12/site-packages/pandas/",
        "/opt/conda/lib/python3.12/site-packages/geoopt/",
        "/opt/conda/lib/python3.12/site-packages/einops/",
        "/usr/lib/python3/",  # system python libs
    ]

    code_count = 0
    code_tokens_start = len(all_ids)
    for code_dir in local_code_dirs:
        if not os.path.exists(code_dir):
            continue
        for root, _, files in os.walk(code_dir):
            if code_count >= num_code:
                break
            for fname in files:
                if code_count >= num_code:
                    break
                if not fname.endswith('.py'):
                    continue
                fpath = os.path.join(root, fname)
                try:
                    with open(fpath, 'r', errors='ignore') as fh:
                        content = fh.read()
                    if len(content) > 100:
                        ids = tokenizer.encode(content, add_special_tokens=False)
                        all_ids.extend(ids)
                        all_ids.append(tokenizer.eos_token_id)
                        code_count += 1
                except:
                    pass
        if code_count % 5000 == 0 and code_count > 0:
            print(f"  Local code: {code_count}/{num_code} ({len(all_ids)-code_tokens_start:,} tokens)")

    local_code_tokens = len(all_ids) - code_tokens_start
    print(f"  Local code: {code_count:,} files → {local_code_tokens:,} tokens")

    # If local code is insufficient, try JSONL fallback
    if local_code_tokens == 0:
        jsonl_path = "/tmp/local_code/code.jsonl"
        if os.path.exists(jsonl_path):
            print(f"  Loading from {jsonl_path}...")
            import json as json_mod
            with open(jsonl_path) as f:
                for line in f:
                    if code_count >= num_code:
                        break
                    try:
                        text = json_mod.loads(line)['text']
                        if len(text) > 100:
                            ids = tokenizer.encode(text, add_special_tokens=False)
                            all_ids.extend(ids)
                            all_ids.append(tokenizer.eos_token_id)
                            code_count += 1
                    except:
                        pass
            print(f"  JSONL: {code_count:,} files loaded")

    code_loaded = code_count > 0
    if not code_loaded:
        print("  WARNING: No code data found!")

    code_tokens = len(all_ids) - wiki_tokens
    total = len(all_ids)
    print(f"\n  Code total: {code_tokens:,} tokens")
    print(f"  Combined: {total:,} tokens")
    print(f"  Wiki/Code ratio: {wiki_tokens/total*100:.0f}% / {code_tokens/total*100:.0f}%")

    # Chunk
    n_chunks = total // seq_len
    print(f"  → {n_chunks:,} chunks of {seq_len}")
    chunks = torch.tensor(all_ids[:n_chunks * seq_len], dtype=torch.long).reshape(n_chunks, seq_len)

    dataset = Dataset.from_dict({'input_ids': chunks.tolist()})
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(cache_path))
    return dataset


@torch.no_grad()
def generate_sample(model, tokenizer, prompt, max_tokens=80, temperature=0.7, device='cuda'):
    model.eval()
    ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    for _ in range(max_tokens):
        logits = model(ids)
        next_logits = logits[:, -1, :] / temperature
        values, _ = torch.topk(next_logits, 40)
        next_logits = torch.where(next_logits < values[:, -1:], -float('inf'), next_logits)
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=-1)
        if next_id.item() == tokenizer.eos_token_id or ids.shape[1] >= 512:
            break
    model.train()
    return tokenizer.decode(ids[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=16000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--seq_len', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--warmup_steps', type=int, default=500)
    parser.add_argument('--grad_clip', type=float, default=0.5)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--checkpoint', type=str, default='checkpoints/helm_d_qwen3_surgery.pt')
    parser.add_argument('--resume', action='store_true', help='Auto-resume from latest checkpoint in save_dir')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_every', type=int, default=100)  # Frequent for spot instances
    parser.add_argument('--keep_last', type=int, default=5)  # Rolling history
    parser.add_argument('--sample_every', type=int, default=500)
    parser.add_argument('--log_every', type=int, default=20)
    parser.add_argument('--reproject_every', type=int, default=100)  # Failsafe #1
    parser.add_argument('--rollback_save_every', type=int, default=200)  # Failsafe #3
    parser.add_argument('--num_wiki', type=int, default=100000)
    parser.add_argument('--num_code', type=int, default=100000)
    parser.add_argument('--hf_token', type=str, default=None)
    args = parser.parse_args()

    # Check disk space at startup
    free_gb, total_gb = check_disk_space(args.save_dir if os.path.exists(args.save_dir) else '.')
    print(f"\nDisk: {free_gb:.1f}GB free / {total_gb:.1f}GB total")
    if free_gb < 5:
        print(f"  WARNING: Low disk space! Need at least 5GB for checkpoints.")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print(f"Device: {device}")

    print(f"\nConfig:")
    print(f"  steps={args.steps}, bs={args.batch_size}, seq={args.seq_len}")
    print(f"  effective_bs={args.batch_size * args.grad_accum}")
    print(f"  tokens/step={args.batch_size * args.grad_accum * args.seq_len:,}")
    print(f"  target_tokens={args.steps * args.batch_size * args.grad_accum * args.seq_len:,}")
    print(f"  lr={args.lr}, grad_clip={args.grad_clip}")
    print(f"  NaN failsafes: reproject@{args.reproject_every}, rollback@{args.rollback_save_every}")

    # Tokenizer
    print("\nLoading Qwen3-30B-A3B tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = len(tokenizer)

    # Data
    dataset = prepare_mixed_data(tokenizer, args.seq_len, args.num_wiki, args.num_code, args.hf_token)
    print(f"Dataset: {len(dataset):,} chunks of {args.seq_len}")

    # Model
    print(f"\nBuilding HELM-D (vocab={vocab_size})...")
    model = LTransformerDecoder(
        manifold_in=Lorentz(1.0),
        manifold_hidden=Lorentz(1.0),
        manifold_out=Lorentz(1.0),
        arch="L6W384A6",
        vocab_size=vocab_size,
        context_length=args.seq_len,
    )

    # Load checkpoint — check for resume first
    resume_ckpt = None
    if args.resume:
        resume_ckpt = find_latest_checkpoint(args.save_dir)
        if resume_ckpt:
            print(f"Resuming from: {resume_ckpt}")
    
    ckpt_path = resume_ckpt or args.checkpoint
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt['model_state_dict'], strict=False)
        start_step = ckpt.get('global_step', 0)
        print(f"  Restored from step {start_step}: missing={len(missing)}, unexpected={len(unexpected)}")
        # Resume optimizer state if available
        if resume_ckpt and 'optimizer_state_dict' in ckpt:
            print(f"  Resuming optimizer state")
            # Will load after optimizer is created
            _resume_optimizer_state = ckpt['optimizer_state_dict']
            _resume_scheduler_state = ckpt.get('scheduler_state_dict', None)
        else:
            _resume_optimizer_state = None
            _resume_scheduler_state = None
        del ckpt
    else:
        start_step = 0
        _resume_optimizer_state = None
        _resume_scheduler_state = None
        print("No checkpoint found — training from scratch")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    model = model.to(device).train()

    # Check initial manifold constraint (Failsafe #4)
    constraint = check_manifold_constraint(model)
    print(f"\nInitial manifold constraint: {constraint}")

    # Optimizer
    hyp_params = [p for p in model.parameters() if isinstance(p, ManifoldParameter) and p.requires_grad]
    euc_params = [p for p in model.parameters() if not isinstance(p, ManifoldParameter) and p.requires_grad]
    print(f"Hyperbolic params: {len(hyp_params)}, Euclidean params: {len(euc_params)}")

    optimizer = RiemannianAdam([
        {'params': euc_params, 'lr': args.lr, 'weight_decay': 0.01},
        {'params': hyp_params, 'lr': args.lr * 0.1, 'weight_decay': 0.0},
    ])

    def lr_schedule(step):
        if step < args.warmup_steps:
            return step / args.warmup_steps
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)
    loss_fn = torch.nn.CrossEntropyLoss()

    # Resume optimizer/scheduler state if available
    if _resume_optimizer_state:
        try:
            optimizer.load_state_dict(_resume_optimizer_state)
            print("  Optimizer state restored")
        except Exception as e:
            print(f"  Could not restore optimizer: {e}")
    if _resume_scheduler_state:
        try:
            scheduler.load_state_dict(_resume_scheduler_state)
            print("  Scheduler state restored")
        except Exception as e:
            print(f"  Could not restore scheduler: {e}")

    # Initialize rolling checkpointer
    checkpointer = RollingCheckpointer(args.save_dir, keep_last=args.keep_last)

    # Initialize NaN rollback handler (Failsafe #3)
    nan_handler = NaNRollbackHandler(
        model, optimizer, scheduler,
        save_dir=os.path.join(args.save_dir, 'rollback'),
        max_rollbacks=5,
    )

    # Pre-training generation
    print(f"\n{'='*60}")
    print(f"BEFORE training:")
    print(f"{'='*60}")
    for p in ["Romania is a country", "def fibonacci(n):", "import os\n"]:
        print(f"  '{p[:30]}' → {generate_sample(model, tokenizer, p, device=device)[:120]}")

    # ============================================================
    # MAIN TRAINING LOOP
    # ============================================================
    print(f"\n{'='*60}")
    print(f"Training {args.steps} steps on H200")
    print(f"  {args.batch_size}×{args.seq_len} = {args.batch_size*args.seq_len:,} tok/step")
    print(f"{'='*60}\n")

    t0 = time.time()
    running_loss = 0.0
    data_idx = 0
    step = 0

    # Save initial clean state for rollback
    nan_handler.save_clean_state(0)

    # torch.compile: fuse Lorentz element-wise ops into Triton kernels
    print("Compiling model with torch.compile (first step will take 3-5 min)...")
    model = torch.compile(model, )

    while step < args.steps:
        optimizer.zero_grad()
        step_loss = 0.0

        for _ in range(args.grad_accum):
            batch = []
            for _ in range(args.batch_size):
                batch.append(dataset[data_idx % len(dataset)]['input_ids'])
                data_idx += 1
            ids = torch.tensor(batch, dtype=torch.long, device=device)

            input_ids = ids[:, :-1]
            targets = ids[:, 1:].reshape(-1)

            if fused_ce_loss is not None:
                # === LIGER FUSED LINEAR CE: logits tensor NEVER exists in HBM ===
                # Get hidden states (stop before mapping layer)
                max_len = input_ids.shape[-1]
                causal = model.attn_mask[:max_len, :max_len]
                token_embeddings = model.token_embed(input_ids)
                freqs_cis = model.freqs_complex[:max_len]
                decoder_features = token_embeddings
                for block in model.resblocks:
                    decoder_features = block(decoder_features, causal, freqs_cis)
                decoder_features = model.final_proj(decoder_features)
                hidden = model.ln_final(decoder_features)

                # Fused: linear projection + cross-entropy in one Triton kernel
                hidden_2d = hidden.reshape(-1, hidden.size(-1))
                loss = fused_ce_loss(hidden_2d, model.mapping.weight, targets)
                del hidden, hidden_2d, decoder_features
            else:
                # Fallback: standard forward with BF16 logits
                logits = model(input_ids)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets)
                del logits

            (loss / args.grad_accum).backward()
            step_loss += loss.item() / args.grad_accum

        # ============================================================
        # NaN FAILSAFE #3: Check for NaN and rollback if needed
        # ============================================================
        should_continue, new_step = nan_handler.check_and_rollback(step_loss, step)
        if not should_continue:
            print("FATAL: Training cannot continue. Saving emergency checkpoint.")
            break
        if new_step != step:
            step = new_step
            running_loss = 0.0
            continue

        # ============================================================
        # NaN FAILSAFE #2: Aggressive gradient clipping
        # ============================================================
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()
        scheduler.step()
        running_loss += step_loss

        # ============================================================
        # NaN FAILSAFE #1: Periodic manifold re-projection
        # ============================================================
        if (step + 1) % args.reproject_every == 0:
            project_to_lorentz(model)

        # ============================================================
        # NaN FAILSAFE #3: Save clean state for rollback
        # ============================================================
        if (step + 1) % args.rollback_save_every == 0:
            nan_handler.save_clean_state(step + 1)

        # Logging
        if (step + 1) % args.log_every == 0:
            avg_loss = running_loss / args.log_every
            elapsed = time.time() - t0
            ms_per_step = elapsed / (step + 1) * 1000
            tok_per_s = args.batch_size * args.grad_accum * (args.seq_len - 1) / (elapsed / (step + 1))
            eta = (args.steps - step - 1) * elapsed / (step + 1)

            # Failsafe #4: Monitor manifold constraint
            constraint = check_manifold_constraint(model)

            print(f"Step {step+1:5d}/{args.steps} | loss={avg_loss:.4f} | "
                  f"lr={scheduler.get_last_lr()[0]:.2e} | "
                  f"{ms_per_step:.0f}ms/step | {tok_per_s:.0f} tok/s | "
                  f"grad={grad_norm:.2f} | manifold={constraint['mean_constraint']:.4f}±{constraint['max_deviation']:.4f} | "
                  f"ETA {eta/60:.0f}m")
            running_loss = 0.0

        # Generation samples
        if (step + 1) % args.sample_every == 0:
            print(f"\n--- Samples at step {step+1} ---")
            prompts = [
                "The theory of",
                "Romania is a country in",
                "def fibonacci(n):",
                "import os\nos.path.",
                "for i in range(10):",
                "# Binary search algorithm\ndef binary_search(",
                "class NeuralNetwork(nn.Module):\n    def __init__(self",
            ]
            for p in prompts:
                out = generate_sample(model, tokenizer, p, device=device)
                print(f"  '{p[:40]}' →\n    {out[:200]}\n")

        # Rolling checkpointing (spot-instance resilient)
        if (step + 1) % args.save_every == 0:
            checkpointer.save(
                start_step + step + 1, model, optimizer, scheduler,
                extra={
                    'loss': step_loss,
                    'data_idx': data_idx,
                    'nan_rollbacks': nan_handler.rollback_count,
                }
            )

        # SPOT INSTANCE: Check for preemption signal
        if _PREEMPTED:
            print(f"\n🚨 Preemption! Emergency save at step {step+1}...")
            emergency_path = os.path.join(args.save_dir, f'h200_emergency_step{start_step+step+1}.pt')
            torch.save({
                'global_step': start_step + step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'data_idx': data_idx,
            }, emergency_path)
            print(f"  Emergency checkpoint saved: {emergency_path}")
            print(f"  Resume with: --resume --save_dir {args.save_dir}")
            break

        step += 1

    # Final
    elapsed = time.time() - t0
    tokens_processed = step * args.batch_size * args.grad_accum * args.seq_len
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Steps: {step}")
    print(f"  Time: {elapsed/60:.1f} minutes ({elapsed/step*1000:.0f}ms/step)")
    print(f"  Tokens: {tokens_processed:,}")
    print(f"  Throughput: {tokens_processed/elapsed:.0f} tok/s")
    print(f"  NaN rollbacks: {nan_handler.rollback_count}")
    print(f"  Final manifold constraint: {check_manifold_constraint(model)}")
    print(f"{'='*60}")

    # Final checkpoint (never pruned)
    save_path = os.path.join(args.save_dir, 'h200_final.pt')
    torch.save({
        'global_step': start_step + step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, save_path)
    print(f"Final checkpoint: {save_path}")
    print(f"To resume if needed: python3 train_h200.py --resume --save_dir {args.save_dir}")

    # Final generation
    print(f"\n--- Final generation ---")
    for p in ["The theory of general relativity",
              "Romania is a country in southeastern Europe",
              "def fibonacci(n):\n    if n <= 1:",
              "import torch\nimport torch.nn as nn\n\nclass",
              "for i in range(10):\n    print(",
              "# Binary search\ndef binary_search(arr, target):\n    low, high = 0, len(arr)",
              "#!/bin/bash\n# Deploy script\nset -e\n"]:
        out = generate_sample(model, tokenizer, p, max_tokens=120, device=device)
        print(f"\n  '{p[:50]}' →\n    {out[:300]}")


if __name__ == '__main__':
    main()
