# HELM-D: H200 Optimized Hyperbolic Language Model

> Fork of [Graph-and-Geometric-Learning/helm](https://github.com/Graph-and-Geometric-Learning/helm) — a 130M hyperbolic transformer pretrained on NVIDIA H200 at 192K tokens/sec.

All computations live on the [Lorentz manifold](https://en.wikipedia.org/wiki/Hyperboloid_model): $-x_0^2 + x_1^2 + \dots + x_d^2 = -1$. The model uses hyperbolic embeddings, Lorentzian attention, and Riemannian optimization — making it natively suited for hierarchical data like code ASTs, dependency trees, and taxonomy structures.

---

## Pipeline Overview

```
Llama-3.1 HELM-D checkpoint (128K vocab, width=390)
        │
        ▼
┌──────────────────────────┐
│  1. Tokenizer Surgery    │  Llama→Qwen3 vocab swap via Lorentzian Fréchet Mean
│     tokenizer_surgery.py │  109K direct transfer + 42K novel tokens projected
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  2. Architecture Refit   │  Width 390→384 for Tensor Core alignment
│     helm_d.py            │  RoPE patched for Lorentz odd-dim
└──────────┬───────────────┘
           ▼
┌──────────────────────────┐
│  3. H200 Pretraining     │  Flash Attention 2, BF16 logits, torch.compile
│     train_h200.py        │  192K tok/s, 1.37s/step, ~6h to 4.2B tokens
└──────────────────────────┘
```

---

## 1. Tokenizer Surgery (`tokenizer_surgery.py`)

The original HELM-D uses the Llama-3.1 tokenizer (128,256 tokens). We replace it with the **Qwen3-30B-A3B tokenizer** (151,669 tokens) — the largest in the LLM ecosystem — for maximum downstream compatibility.

### The Problem
Swapping tokenizers requires transferring the embedding matrix, which lives on the Lorentz manifold — every row satisfies $-x_0^2 + \sum x_i^2 = -1$. New tokens need geometrically consistent initialization to preserve manifold constraints.

### The Solution: Three-Case Transfer

| Case | Count | Method |
|---|---|---|
| **Matching tokens** (same string in both vocabs) | 109,547 (72%) | Direct 1:1 coordinate copy |
| **Qwen3-only tokens** (not in Llama) | 42,097 (28%) | Decompose into Llama sub-tokens → **Lorentzian Fréchet Mean** |
| **Undecomposable** | 25 (<0.1%) | Riemannian Normal initialization |

The **Lorentzian Fréchet Mean** computes the geometric centroid on the hyperboloid. For a Qwen3 token like `"самостоятельно"`, we encode it with the Llama tokenizer to get sub-tokens, extract their hyperbolic embeddings, and compute the Einstein midpoint — the point that minimizes the sum of squared Lorentzian distances.

```bash
python tokenizer_surgery.py
# Requires: meta-llama/Llama-3.1-8B tokenizer access
# Input:  HELM-D checkpoint with Llama vocab (128,256 × 390)
# Output: HELM-D checkpoint with Qwen3 vocab (151,669 × 390)
```

---

## 2. Architecture Optimizations

### Width 390 → 384 (`helm_d.py`)

The original `L6W390A6` architecture has `width=390`, which doesn't align with NVIDIA Tensor Core tile sizes (multiples of 64/128). We changed to `width=384`:

- Per-head dimension: `384 / 6 = 64` — perfect Tensor Core alignment
- MLP: `384 × 4 = 1536` (vs 1560)
- Enables Triton kernel compatibility

### RoPE Fix for Lorentz Geometry (`helm_d.py`, `lorentz_former_conv.py`)

In hyperbolic models, the per-head spatial dimension is `head_dim - 1` (removing the Lorentz time coordinate). At width=384 with 6 heads: `64 - 1 = 63` (odd). RoPE requires even dimensions for its complex-number encoding.

**Fix**: `precompute_theta_pos_frequencies` rounds down to the nearest even number. `apply_rotary_embeddings` applies rotary encoding to the first 62 dimensions and passes the 63rd through unchanged.

### Flash Attention 2 (`lorentz_former_conv.py`)

The original `full_attention` materializes a full `[B, H, N, N]` attention matrix — O(N²) memory. We replaced it with Flash Attention 2.

**The Lorentz challenge**: FA2 computes Euclidean dot products, but hyperbolic attention requires the Minkowski inner product $\langle x, y \rangle_\mathcal{L} = -x_0 y_0 + \sum x_i y_i$.

**Solution**: Run FA2 on **spatial dimensions only** (strip the time coordinate `x_0`), then reconstruct the time coordinate after attention via the manifold projection: $x_0 = \sqrt{\|x_{1:d}\|^2 + 1}$.

### Selective BF16 Output Projection (`helm_d.py`)

The output projection `nn.Linear(384, 151669)` is **purely Euclidean** — no hyperbolic math. We cast this single layer to BF16:

```python
logits = F.linear(features.to(torch.bfloat16), self.mapping.weight.to(torch.bfloat16))
```

All upstream Lorentz operations (embeddings, attention, RMSNorm) remain in strict FP32 to preserve the manifold constraint.

### torch.compile (`train_h200.py`)

`torch.compile(model)` fuses the many small Lorentz element-wise operations (project, sqrt, concat, Minkowski hack) into optimized Triton kernels via TorchInductor.

> **Note**: `mode="max-autotune"` and `mode="reduce-overhead"` crash on CUDAGraphs due to dynamic `index_select` in LorentzEmbeddings. Default mode works.

### Python `-O` Flag

The original HELM codebase contains 30+ `assert not torch.isnan(U).any()` checks in the manifold code (`pseudohyperboloid.py`). Each triggers a GPU→CPU synchronization, stalling the pipeline. Running with `python -O` strips all assert statements.

Debug `print()` calls in the manifold hot path were also removed.

### geoopt Compatibility Patch

geoopt's `torch.norm(x, p=2, dim=dim)` in `lorentz/math.py` is incompatible with torch.compile's tracer. Patched to `torch.linalg.vector_norm(x, ord=2, dim=dim)`.

---

## 3. Performance

Benchmarked on NVIDIA H200 (143 GB HBM3e), 130M parameter HELM-D, seq_len=2048:

| Configuration | ms/step | tok/s | VRAM | Speedup |
|---|---|---|---|---|
| Original FP32 (chunked) | 5,966 | 43,917 | 131 GB | 1.0× |
| Selective BF16 logits | 3,601 | 72,770 | 74 GB | 1.7× |
| FA2 + BF16 (width=384) | 1,875 | 140,025 | 85 GB | 3.2× |
| **FA2 + BF16 + torch.compile** | **1,370** | **192,000** | **85 GB** | **4.4×** |

---

## Training

### Requirements

```bash
pip install flash-attn --no-build-isolation
pip install geoopt transformers datasets
```

### Run

```bash
# Fresh pretraining on H200
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -O train_h200.py --save_dir /tmp/checkpoints

# Resume from checkpoint
python -O train_h200.py --resume --save_dir /tmp/checkpoints
```

### Config

| Parameter | Value |
|---|---|
| Architecture | L6W384A6 (6 layers, 384 width, 6 heads) |
| Parameters | 130M |
| Tokenizer | Qwen3-30B-A3B (151,669 vocab) |
| Batch | 32 × 4 grad_accum = 128 effective |
| Sequence length | 2048 |
| Learning rate | 6e-4 (500-step warmup, cosine decay) |
| Optimizer | RiemannianAdam |
| Data | 100K Wikipedia (en) + 100K Python |

### Safety Features

- **Lorentz re-projection** every 100 steps — ensures $-x_0^2 + \|x\|^2 = -1$
- **NaN auto-rollback** — reverts to last clean checkpoint with 50% LR reduction
- **Gradient clipping** at 0.5
- **Rolling checkpoints** — keeps last 5 on disk

---

## Files

| File | Description |
|---|---|
| `tokenizer_surgery.py` | Llama→Qwen3 embedding transfer via Lorentzian Fréchet Mean |
| `train_h200.py` | H200 pretraining with FA2, BF16, torch.compile, NaN failsafes |
| `helm/modules/helm_d.py` | RoPE odd-dim fix, BF16 output projection |
| `helm/hypercore/nn/attention/lorentz_former_conv.py` | Flash Attention 2 with Minkowski-compatible spatial attention |

---

## Known Issues

- **torch.compile modes**: `max-autotune` and `reduce-overhead` crash with `CUDAGraphs index_select` error in LorentzEmbeddings. Only default mode works.
- **Width 390**: The original dimension is not Tensor Core aligned. Triton kernels may illegal-memory-access at this width.
- **geoopt + torch.compile**: Requires patching `torch.norm` → `torch.linalg.vector_norm` in geoopt's `lorentz/math.py`.

---

## Citation

Based on:
```bibtex
@article{helm2024,
  title={Hyperbolic Efficient Language Models},
  author={Graph and Geometric Learning Lab},
  year={2024},
  url={https://github.com/Graph-and-Geometric-Learning/helm}
}
```
