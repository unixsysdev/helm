# H-MICE: Hierarchical Mixture of Curvature Experts

A **~600M parameter** sparse transformer decoder using a **Log-Euclidean Tangent Sandwich** architecture with **curved residual stream**, **interleaved Dense/MoE blocks**, and **9 geometry-specialized experts**.

## Architecture

```
Token → ManifoldParameter Embedding (ON Lorentz, 8192×1025)
  │
  ▼  CURVED Residual Stream (D+1 dim, Lorentz manifold)
  │
  ├── Block 1  [Dense]  log₀→Norm→SpatialAttn→Pad→NormPost→exp₀
  │                     log₀→Norm→SwiGLU→Pad→NormPost→exp₀
  ├── Block 2  [MoE]   log₀→Norm→SpatialAttn→Pad→NormPost→exp₀
  │                     log₀→Norm→MoE(9 experts)→Pad→NormPost→exp₀
  ├── Block 3  [Dense]
  ├── Block 4  [MoE]
  ├── ...
  ├── Block 15 [Dense]
  └── Block 16 [MoE]
  │
  log₀ → RMSNorm(spatial) → LM Head (1024→8192) → Logits
```

### The Tangent Sandwich (Per Sub-Layer)

```
x_manifold ──→ logmap₀ ──→ v_tangent (D+1)
                              │
                          RMSNorm(D+1)
                              │
                          v[..., 1:]  ← strip time (spatial-only, D-dim)
                              │
                    ┌─────────┴─────────┐
                    │  Flat Compute:     │
                    │  SpatialAttn(FA2)  │
                    │  or SwiGLU FFN     │
                    │  or MoE Experts    │
                    └─────────┬─────────┘
                              │
                          F.pad(0, D+1)  ← reconstruct time=0
                              │
                          v + expert_out  ← tangent residual add
                              │
                          RMSNorm(D+1)  ← POST-ADDITION norm
                              │
                    x_new = expmap₀ ──→ back on manifold
```

**Key insight**: The post-addition RMSNorm before `expmap₀` prevents tangent magnitude blowup through 16 layers while remaining fully learnable (no destructive clamping).

### Interleaved Block Topology

| Block Type | Layers | Contents |
|---|---|---|
| **Dense** (even idx: 0,2,4...) | 8 blocks | SpatialAttn + SwiGLU FFN |
| **MoE** (odd idx: 1,3,5...) | 8 blocks | SpatialAttn + H-MICE 2-tier Router → 9 Experts |

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
| H1 | Hyperbolic | k=0.2 (fixed) | SwiGLU + exp₀→Lorentz dist² |
| H2 | Hyperbolic | k=0.5 (fixed) | SwiGLU + exp₀→Lorentz dist² |
| H3 | Hyperbolic | k=1.0 (fixed) | SwiGLU + exp₀→Lorentz dist² |
| H4 | Hyperbolic | k=2.0 (fixed) | SwiGLU + exp₀→Lorentz dist² |
| S1 | Spherical | unit sphere | 1024→64→SwiGLU→projx→64→1024 |

### Numerical Stability: Pre-Norm + Zero-Init

| Technique | Purpose |
|---|---|
| **Zero-init `wo`** (attention output) | Step-0: `attn_out = 0` → identity pass-through |
| **Zero-init `w2`** (FFN/expert down-proj) | Step-0: `expert_out = 0` → identity pass-through |
| **Pre-addition RMSNorm** (D+1 tangent) | Normalize tangent before expert computation |
| **Post-addition RMSNorm** (D+1 tangent) | Bound tangent magnitude before `expmap₀` |
| **`stabilize=10`** (RiemannianAdam) | Re-project embeddings onto Lorentz every 10 steps |

At Step 0: `expmap₀(logmap₀(x) + 0) = x` → perfect identity through all 16 layers. No NaN, no exponential blowup.

### Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Residual stream | **Curved (Lorentz D+1)** | True exponential volume through all layers |
| Embeddings | `ManifoldParameter` on Lorentz | Maintained by RiemannianAdam |
| Attention | Spatial-only FA2 (`is_causal=True`) | Time=0 in tangent space → skip wasted compute |
| Experts | `nn.Linear` SwiGLU (zero-init'd) | Stable, fast, `torch.compile` friendly |
| Curvatures | **Fixed** k=[0.2, 0.5, 1.0, 2.0] | Prevents optimizer from collapsing manifold |
| Spherical expert | 64D bottleneck + ε-safe projx | Avoids hollow sphere + zero-vector NaN |
| Geometric loss | `p_selected × d²` | Backprop whip: router self-corrects |
| Vocab | 32K (Mistral-v0.1 baseline) | Strict parameter diet |
| Optimizer | Dual: RiemannianAdam + AdamW | ManifoldParam needs Riemannian updates |

### Sizing

| Component | Params | Active/Token |
|---|---|---|
| Embedding (32000 × 1025, ManifoldParameter) | 32.8M | 32.8M |
| LM Head (1024 × 32000) | 32.8M | 32.8M |
| 8 Dense Blocks (Attn + FFN + 4×RMSNorm) | ~84M | ~84M |
| 8 MoE Blocks (Attn + 9 Experts + 4×RMSNorm) | ~490M | ~94M |
| **Total** | **~640M** | **~244M** |

## Training: Bimodal Loss

### Dual Optimizer Protocol

```python
opt_manifold = geoopt.optim.RiemannianAdam(manifold_params, lr=1e-5, stabilize=10)
opt_euclidean = torch.optim.AdamW(euclidean_params, lr=1e-4, weight_decay=0.1)
```

### Phase 1: Supervised Scaffolding (Steps 0–2000)
```
L_total = L_text + α × CrossEntropy(L1_logits, geom_targets)
```

Geometry tags assigned via **offset-mapped regex projection**:
1. Tokenize with `return_offsets_mapping=True`
2. Regex on raw string: `<think>` / `def` / `class` → Hyperbolic, `cos`/dates → Spherical
3. Project char spans to token spans via offset mapping

### Phase 2: Geometric Distortion (Steps 2001+)
```
L_total = L_text + λ × Σ p_selected(x) · d²_M(0, E(x))
```
**Backprop whip**: router self-corrects via gradient from manifold distance.

## Telemetry

```
Step 100: loss=0.150 text=0.140 scaf=0.010 gn=2.0 acc=99% mc=-1.0000 L1=[E:32 H:32 S:34]
```

| Metric | Meaning |
|---|---|
| `mc` | Manifold constraint: -(x₀²) + Σxᵢ² (expect -1.0) |
| `L1=[E:32 H:32 S:34]` | Geometry class EMA distribution (zero-sync) |
| `acc` | Router scaffold accuracy (Phase 1) |
| `gn` | Gradient norm (post-clip) |

## Quick Start

```bash
# Trial run (validates router + manifold + FA2)
cd h_mice && python test_routing.py

# Full training (H200)
python h_mice/train.py --batch_size 16 --grad_accum 8 --lr 3e-4 \
  --save_dir /tmp/checkpoints/h_mice --seq_len 4096
```

## Safety

- **Zero-init identity**: wo + w2 = 0 → step-0 pass-through (no random blowup)
- **Post-addition RMSNorm**: bounded tangent magnitude before expmap₀
- **Triple NaN guard**: forward → backward → gradient scan
- **RiemannianAdam stabilize=10**: re-project embeddings onto hyperboloid
- **Fixed curvatures**: prevents optimizer-driven manifold collapse
- **`torch.cuda.empty_cache()`** on OOM recovery
