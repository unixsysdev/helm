# H-MICE: Hierarchical Mixture of Curvature Experts

A **420M parameter** sparse transformer decoder with a **Product Manifold** (ℍ×𝔼×𝕊) and a 2-tier hierarchical router that dynamically routes tokens to geometry-specific experts.

## Architecture

```
Token → Embedding → [16 × Transformer Block] → LM Head → Logits
                         │
                    ┌────┴────┐
                    │  Attn   │  ← Flat Euclidean tangent space
                    ├─────────┤
                    │ H-MICE  │  ← Mixture of Curvature Experts
                    │  MoE    │
                    └────┬────┘
                         │
          ┌──────────────┼──────────────┐
          │              │              │
    L1 Router (Top-1)    │              │
          │              │              │
    ┌─────┴─────┐   ┌───┴───┐   ┌─────┴─────┐
    │ Hyperbolic │   │ Eucl. │   │ Spherical │
    │ (4 exp.)   │   │(4 exp)│   │ (4 exp.)  │
    │ Lorentz    │   │ Flat  │   │ 64D bottl.│
    └─────┬─────┘   └───┬───┘   └─────┬─────┘
          └──────────────┼──────────────┘
                         │
              expmap₀ → d² penalty → log₀
                         │
                   Flat residual stream
```

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Residual stream | Flat Euclidean | Preserves 130k tok/s; no Lorentzian attention overhead |
| Attention | Standard MHA + RoPE | Operates entirely in tangent space |
| Spherical expert | 64D bottleneck | 1024D unit sphere is hollow (curse of dimensionality) |
| Geometric loss | `p_selected × d²` | Backprop whip: router self-corrects via gradient suppression |
| Cross-layer | log₀ back to flat | All expert outputs are flat before residual add |
| L2 balancing | DeepSeekV3-style bias | Prevents expert collapse |

### Sizing

| Component | Params | Active/Token |
|---|---|---|
| Token Embedding (32K × 1024) | 32.8M | 32.8M |
| 16 Blocks × Attention | 67.1M | 67.1M |
| 16 Blocks × H-MICE MoE (12 experts) | ~320M | ~20M |
| **Total** | **~420M** | **~120M** |

## Training: Bimodal Loss

### Phase 1: Supervised Scaffolding (Steps 0–2000)
```
L_total = L_text + α × CrossEntropy(L1_logits, geom_targets)
```
Heuristic tags force the L1 router to learn geometry assignments:
- **Hyperbolic** (tag 2): `<think>` blocks, indented code (trees, recursion)
- **Spherical** (tag 1): datetime, trigonometry, modular arithmetic
- **Euclidean** (tag 0): standard prose (default)

### Phase 2: Geometric Distortion (Steps 2001+)
```
L_total = L_text + λ × Σ p_selected(x) · d_M(0, E(x))²
```
Router self-corrects: if a token is routed to the wrong geometry, the manifold distance explodes, `p_selected × d²` is massive, and the gradient crushes that routing probability.

## Quick Start

```bash
# Trial run with mock data (validates router)
python h_mice/train.py --mock --mock_steps 500

# Full training
python h_mice/train.py --batch_size 16 --grad_accum 8 --lr 3e-4 \
  --save_dir /tmp/checkpoints/h_mice --seq_len 4096
```

## Data Pipeline

Streaming 60/20/20 mix:
- **60%** OpenThoughts-114k (CoT reasoning)
- **20%** Code (Python)
- **20%** FineWeb-Edu (text)

512-chunk shuffle buffer prevents domain clustering. Per-token geometry tags generated on-the-fly via regex heuristics.

## Safety

- Triple NaN guard: forward loss → backward RuntimeError → gradient scan
- `torch.cuda.empty_cache()` on skip to prevent OOM
- geoopt arcosh/sqrt clamping patches
- `_orig_mod.` prefix stripping for torch.compile checkpoints

## Requirements

- PyTorch 2.x with CUDA
- geoopt
- transformers (for tokenizer)
- datasets (for streaming data)
