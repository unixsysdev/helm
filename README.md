# HELM-D: H200 Optimized Pretraining

> Fork of [Graph-and-Geometric-Learning/helm](https://github.com/Graph-and-Geometric-Learning/helm) with hardware-optimized pretraining for NVIDIA H200 GPUs.

## What Changed

This fork introduces three key optimizations to make HELM-D's hyperbolic transformer architecture run efficiently on modern NVIDIA hardware (H200/A100):

### 1. Flash Attention 2 Integration (`lorentz_former_conv.py`)

The original `LorentzMultiheadAttention.full_attention()` materializes a full `[B, H, N, N]` attention matrix in HBM — O(N²) memory. We replaced it with Flash Attention 2, which computes attention block-by-block in SRAM.

**The Lorentz challenge**: FA2 computes Euclidean dot products, but hyperbolic attention requires the Minkowski inner product. We solve this by:
1. Running FA2 on the **spatial dimensions only** (stripping the Lorentz time coordinate)
2. Reconstructing the time coordinate after attention via `project()`: $x_0 = \sqrt{\|x_{1:d}\|^2 + c}$

This preserves the Lorentz manifold constraint while getting FA2's 5× memory reduction.

### 2. Liger Fused Linear Cross-Entropy (`train_h200.py`)

HELM-D uses the Qwen3 tokenizer (151,669 vocab) — the largest in the LLM ecosystem. The output logits tensor:
```
Batch × Seq × 151,669 × 4 bytes = 80 GB at batch=64 (FP32)
```

We use [Liger-Kernel](https://github.com/linkedin/Liger-Kernel)'s `LigerFusedLinearCrossEntropyLoss` which fuses the final linear projection and cross-entropy loss into a single Triton kernel. **The logits tensor never exists in HBM.**

### 3. Width 390 → 384 + RoPE Fix (`helm_d.py`)

The original `L6W390A6` architecture has `width=390`, which is not aligned with NVIDIA Tensor Core tile sizes (multiples of 64/128). We changed to `width=384`:

- Per-head dimension: 64 (power of 2 — perfect Tensor Core alignment)
- Enables Triton kernel compatibility (Liger, Flash Attention)
- MLP: 384×4 = 1536 (vs 1560)

**RoPE fix for Lorentz geometry**: In hyperbolic models, the per-head spatial dimension is `head_dim - 1` (removing the time coordinate), which can be odd. We patched `precompute_theta_pos_frequencies` and `apply_rotary_embeddings` to handle odd dimensions by applying rotary encoding to the first `d - (d % 2)` dimensions and passing the remainder through unchanged.

## Performance

Benchmarked on NVIDIA H200 (143 GB HBM3e), 130M parameter HELM-D, seq_len=2048:

| Configuration | ms/step | tok/s | VRAM | Batch |
|---|---|---|---|---|
| Original FP32 (chunked) | 5,966 | 43,917 | 131 GB | 64 |
| Selective BF16 logits | 3,601 | 72,770 | 74 GB | 16 |
| **FA2 + Liger + W384** | **3,255** | **80,495** | **106 GB** | **128** |

## Files

| File | Description |
|---|---|
| `train_h200.py` | H200 pretraining script with NaN failsafes, Lorentz re-projection, and Liger fused CE |
| `helm/modules/helm_d.py` | Modified: RoPE odd-dim fix for Lorentz geometry |
| `helm/hypercore/nn/attention/lorentz_former_conv.py` | Modified: Flash Attention 2 replacing O(N²) attention |

## Requirements

```bash
pip install flash-attn --no-build-isolation
pip install liger-kernel
pip install geoopt
```

## Usage

```bash
# Fresh pretraining on H200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python train_h200.py --save_dir /tmp/checkpoints

# Resume from checkpoint
python train_h200.py --resume --save_dir /tmp/checkpoints
```

## Safety Features

- **Lorentz manifold re-projection** every 100 steps — ensures $-x_0^2 + \|x\|^2 = -1$
- **NaN auto-rollback** with LR halving — reverts to last clean checkpoint on divergence
- **Gradient clipping** at 0.5 — prevents manifold constraint violation
- **Rolling checkpoints** — keeps last 5 checkpoints on disk

## Citation

Based on:
```bibtex
@article{helm2024,
  title={HELM: Hyperbolic Efficient Language Models},
  author={Graph and Geometric Learning Lab},
  year={2024}
}
```
