# H-MICE: Hierarchical Mixture of Curvature Experts

A **~600M parameter** sparse transformer decoder using a **Log-Euclidean Tangent Space** architecture with **interleaved Dense/MoE blocks** and **9 geometry-specialized experts**.

## Architecture

```
Token → ManifoldParameter Embedding (ON Lorentz, 8192×1025)
  │
  logmap₀ (single projection to flat)
  │
  ▼  Flat Euclidean Residual Stream (1024-dim)
  │
  ├── Block 1  [Dense]  Attn + SwiGLU FFN (1024→2048→1024)
  ├── Block 2  [MoE]    Attn + H-MICE Router → 9 Experts
  ├── Block 3  [Dense]
  ├── Block 4  [MoE]
  ├── ...
  ├── Block 15 [Dense]
  └── Block 16 [MoE]
  │
  RMSNorm → LM Head (1024→8192) → Logits
```

### Interleaved Block Topology

| Block Type | Layers | Contents |
|---|---|---|
| **Dense** (odd: 1,3,5...) | 8 blocks | Attn + SwiGLU FFN (1024→2048→1024) |
| **MoE** (even: 2,4,6...) | 8 blocks | Attn + H-MICE 2-tier Router → 9 Experts |

### The 9-Expert Array (Per MoE Block)

```
L1 Router → 3 logits [Euclidean, Hyperbolic, Spherical]
                │
    ┌───────────┼───────────┐
    │           │           │
  L2 Router   L2 Router   (single)
    │           │           │
 ┌──┼──┐    ┌──┼──┐        │
 E1 E2 E3 E4  H1 H2 H3 H4    S1
 (flat)     (k=.2 .5 1 2)  (64D bottleneck)
```

| Expert | Geometry | Curvature | Operation |
|---|---|---|---|
| E1–E4 | Euclidean | — | Pure flat SwiGLU FFN |
| H1 | Hyperbolic | k=0.2 (fixed) | SwiGLU + exp₀ → Lorentz dist² |
| H2 | Hyperbolic | k=0.5 (fixed) | SwiGLU + exp₀ → Lorentz dist² |
| H3 | Hyperbolic | k=1.0 (fixed) | SwiGLU + exp₀ → Lorentz dist² |
| H4 | Hyperbolic | k=2.0 (fixed) | SwiGLU + exp₀ → Lorentz dist² |
| S1 | Spherical | unit sphere | 1024→64→SwiGLU→projx→64→1024 |

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Embeddings | `ManifoldParameter` on Lorentz | Exponential volume for hierarchical data |
| Entry | Single `logmap₀` at start | No NaN compounding from chained manifold ops |
| Residual stream | Flat Euclidean | 130k tok/s throughput, numerical stability |
| Experts | Standard `nn.Linear` SwiGLU | Stable, fast, `torch.compile` friendly |
| Hyperbolic curvatures | **Fixed** k=[0.2, 0.5, 1.0, 2.0] | Prevents optimizer from collapsing manifold |
| Spherical expert | 64D bottleneck | Avoids hollow sphere (curse of dimensionality) |
| Geometric loss | `p_selected × d²` | Backprop whip: router self-corrects |
| Vocab | 8192 BPE | VRAM conserved for MoE experts |
| Optimizer | Dual: RiemannianAdam(emb) + AdamW(rest) | ManifoldParam needs Riemannian updates |

### Sizing

| Component | Params | Active/Token |
|---|---|---|
| Embedding (8192 × 1025, ManifoldParameter) | 8.4M | 8.4M |
| LM Head (1024 × 8192) | 8.4M | 8.4M |
| 8 Dense Blocks (Attn + FFN) | ~84M | ~84M |
| 8 MoE Blocks (Attn + 9 Experts) | ~490M | ~94M |
| **Total** | **~590M** | **~195M** |

## Training: Bimodal Loss

### Phase 1: Supervised Scaffolding (Steps 0–2000)
```
L_total = L_text + α × CrossEntropy(L1_logits, geom_targets)
```

Geometry tags assigned via **offset-mapped regex projection**:
1. Tokenize with `return_offsets_mapping=True`
2. Regex on raw string: `<think>` blocks → Hyperbolic, datetime/trig → Spherical
3. Project char spans to token spans via offset mapping

### Phase 2: Geometric Distortion (Steps 2001+)
```
L_total = L_text + λ × Σ p_selected(x) · d²_M(0, E(x))
```
**Backprop whip**: router self-corrects via gradient from manifold distance.

## Quick Start

```bash
# Trial run (validates router + manifold embedding)
cd h_mice && python test_routing.py

# Full training
python h_mice/train.py --batch_size 16 --grad_accum 8 --lr 3e-4 \
  --save_dir /tmp/checkpoints/h_mice --seq_len 4096
```

## Safety

- Triple NaN guard: forward → backward → gradient scan
- Interleaved dense blocks stabilize manifold-free residual flow
- Fixed curvatures prevent optimizer collapse
- `torch.cuda.empty_cache()` on OOM recovery
