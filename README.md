# HELM-D: Hyperbolic Chain-of-Thought Reasoning Engine

> Fork of [Graph-and-Geometric-Learning/helm](https://github.com/Graph-and-Geometric-Learning/helm) — a **200M parameter** fully hyperbolic transformer trained on NVIDIA H200 for structured reasoning.
>
> **Checkpoints**: [datasysdev/helm-d-130m-hyperbolic](https://huggingface.co/datasysdev/helm-d-130m-hyperbolic) on HuggingFace

All computations live on the [Lorentz manifold](https://en.wikipedia.org/wiki/Hyperboloid_model): $-x_0^2 + x_1^2 + \dots + x_d^2 = -1$. The model uses hyperbolic embeddings, Lorentzian attention, and Riemannian optimization — making it natively suited for hierarchical data like code ASTs, dependency trees, and chain-of-thought reasoning traces.

---

## Current Training Run

Training a **200M parameter** HELM-D from scratch on a multi-domain reasoning corpus:

| Parameter | Value |
|---|---|
| Architecture | `L16W768A12` (16 layers, 768 width, 12 heads) |
| Parameters | **200M** (175.8M Euclidean + 24.6M Hyperbolic) |
| Tokenizer | TinyLlama 32K (dense coverage, no dead tokens) |
| Context | 4096 tokens (full CoT traces fit in one pass) |
| Throughput | **130K tok/s** on single H200 |
| Optimizer | Dual-group RiemannianAdam (see below) |
| Learning Rate | 3e-4, cosine decay with 500-step warmup |
| Gradient Clip | 0.5 |
| Manifold | Lorentz $-x_0^2 + \|x\|^2 = -1$, verified at 1.0000±0.0000 |

### Training Data (60/20/20 Mix)

| Domain | Weight | Source | Purpose |
|---|---|---|---|
| CoT Reasoning | 60% | [OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k) | Math, code, science reasoning with `<think>` traces |
| Python Code | 20% | [SmolLM-Corpus python-edu](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | Educational Python |
| Text | 20% | [SmolLM-Corpus cosmopedia-v2](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) | General knowledge |

Streamed via `interleave_datasets` with a **512-chunk shuffle buffer** to prevent domain clustering (see Architecture Decisions below).

---

## Key Changes from Upstream HELM

### 1. Tokenizer: Llama-3.1 → TinyLlama 32K

The original HELM uses the Llama-3.1 tokenizer (128K vocab). We switched to **TinyLlama's 32K tokenizer** for the CoT training run:

- **Dense coverage**: No dead tokens — every token gets trained
- **Smaller embedding matrix**: 32K × 768 vs 128K × 768 — significant VRAM savings
- **Better for small models**: 200M params can't support 128K vocab efficiently

### 2. Architecture: L6W384A6 → L16W768A12

Scaled up from the original 31M parameter toy model to a **200M parameter** engine:

| | Original | Ours |
|---|---|---|
| Layers | 6 | **16** |
| Width | 390 | **768** |
| Heads | 6 | **12** |
| Head dim | 65 | **64** (Tensor Core aligned) |
| Parameters | 31M | **200M** |

### 3. Dual-Group Optimizer (Matching Original Authors)

The original HELM repo uses **two separate optimizers**: AdamW for Euclidean params and RiemannianAdam for hyperbolic params, with `weight_decay=0.0` on manifold parameters.

We implement this as a single RiemannianAdam with dual parameter groups:

```python
optimizer = RiemannianAdam([
    {"params": euclidean_params, "weight_decay": 0.01},   # 175.8M params
    {"params": hyperbolic_params, "weight_decay": 0.0},   # 24.6M params
], lr=3e-4)
```

**Why**: Standard L2 weight decay pulls parameters toward the Euclidean origin `[0,0,...,0]`, which is **not on the Lorentz manifold**. Applying decay to manifold parameters causes the optimizer to constantly drag embeddings off the $-1$ surface, then the `expmap` projection violently snaps them back — destabilizing training.

### 4. Shuffle Buffer Dataloader

The streaming `interleave_datasets` interleaves at the **document** level. Since OpenThoughts reasoning traces can be 4,000-16,000 tokens (1-4 consecutive 4096-token chunks), the model receives bursts of pure math followed by bursts of pure code — causing catastrophic loss spikes.

**Fix**: A 512-chunk shuffle buffer accumulates tokenized chunks before yielding, ensuring every batch is a representative mix of all 3 domains:

```
Documents → Tokenize → Pack into 4096-token chunks → Buffer (512) → Shuffle → Yield to GPU
```

This eliminated gradient spikes of 46+ and stabilized the loss descent.

### 5. TF32 Tensor Core Acceleration

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

Throughput: **40K → 130K tok/s** (3.25× speedup). All upstream Lorentz operations remain in FP32 — only matmul operations use TF32's 10-bit mantissa through the Tensor Cores.

### 6. LR Override on Checkpoint Resume

PyTorch's `optimizer.load_state_dict()` restores the learning rate from the checkpoint, silently overriding CLI arguments. We force the LR after restore:

```python
for pg in optimizer.param_groups:
    pg["lr"] = args.lr
    pg["initial_lr"] = args.lr
```

---

## Quick Start

### Requirements

```bash
pip install torch flash-attn --no-build-isolation
pip install geoopt transformers datasets
```

### Training on H200

```bash
export PYTHONPATH=/path/to/helm-src:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Fresh training
python3 -O train_cot.py \
    --batch_size 16 --grad_accum 8 \
    --lr 3e-4 --seq_len 4096 \
    --save_dir /tmp/checkpoints/cot \
    --log_every 1

# Resume from checkpoint
python3 -O train_cot.py \
    --batch_size 16 --grad_accum 8 \
    --lr 3e-4 --save_dir /tmp/checkpoints/cot \
    --log_every 1 --resume
```

### Generation Test

```bash
python3 test_gen.py --checkpoint /tmp/checkpoints/cot/cot_step5000.pt
```

---

## Architecture Decisions

### Gradient Clipping: 1.0 → 0.5

The original authors use `grad_clip=1.0` on a 6-layer model. At 16 layers, gradient variance compounds across 10 additional layers. Clip of 0.5 on 16 layers is physically equivalent to 1.0 on 6 layers.

### LR Scaling: 4e-4 → 3e-4

The original authors use `lr=4e-4` on a 31M model. As parameter count and depth scale, optimal learning rates must decrease. 3e-4 is the correct scaling for 200M parameters.

### Flash Attention 2

FA2 computes Euclidean dot products, but hyperbolic attention requires the Minkowski inner product $\langle x, y \rangle_{\mathcal{L}} = -x_0 y_0 + \sum x_i y_i$. We run FA2 on **spatial dimensions only** (strip the time coordinate), then reconstruct via manifold projection: $x_0 = \sqrt{\|x_{1:d}\|^2 + 1}$.

### Periodic Re-projection

Embeddings are snapped back to $-x_0^2 + \|x\|^2 = -1$ every 100 steps to correct constraint drift from mixed-precision gradient updates.

---

## Files

| File | Description |
|---|---|
| `train_cot.py` | **Main training script** — 200M HELM-D with streaming 60/20/20 mix, shuffle buffer, dual optimizer |
| `test_gen.py` | Temperature sweep generation test with repetition penalty grid |
| `train_h200.py` | H200 pretraining with FA2, BF16, torch.compile (130M seed model) |
| `train_h200_130m.py` | 130M config (L6W384A6) for seed training |
| `tokenizer_surgery.py` | Llama→Qwen3 embedding transfer via Lorentzian Fréchet Mean |
| `upscale_130m_to_1b.py` | Network Morphism: 130M→1.37B (Lorentz zero-pad + layer cloning) |
| `setup_h200.sh` | H200 environment setup (CUDA, PyTorch, Flash Attention) |
| `helm/modules/helm_d.py` | HELM-D decoder with RoPE odd-dim fix, BF16 output projection |
| `helm/hypercore/` | Lorentz manifold operations, Riemannian optimizers |

---

## Known Issues

- **torch.compile modes**: `max-autotune` and `reduce-overhead` crash with CUDAGraphs in LorentzEmbeddings. Only default mode works.
- **geoopt + torch.compile**: Requires patching `torch.norm` → `torch.linalg.vector_norm` in geoopt's `lorentz/math.py`.
- **Tokenizer max length warnings**: TinyLlama tokenizer reports max_length=2048 but we use 4096 seq_len — this is harmless (we handle truncation ourselves).

---

## Citation

Based on:
```bibtex
@article{he2025helm,
  title={HELM: Hyperbolic Large Language Models via Mixture-of-Curvature Experts},
  author={He, Neil and Anand, Rishabh and Madhu, Hiren and Maatouk, Ali and Krishnaswamy, Smita and Tassiulas, Leandros and Yang, Menglin and Ying, Rex},
  journal={arXiv preprint arXiv:2505.24722},
  year={2025},
}
```

## License

MIT — see [LICENSE](LICENSE).
