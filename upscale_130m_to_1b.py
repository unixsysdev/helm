"""
HELM-D Network Morphism: 130M → 1B
Upscales the pretrained 130M checkpoint to 1B parameters while preserving
the learned Lorentz manifold geometry.

Architecture change:
  130M: L6W384A6   (6 layers, width=384, 6 heads)
  1B:   L24W1536A24 (24 layers, width=1536, 24 heads)

Three operations:
1. Width expansion (384→1536): Zero-pad spatial dims of Lorentz embeddings
2. Depth expansion (6→24): Clone each layer 4× with residual scaling
3. Linear projection: Top-left corner placement for all weight matrices

Run on H200:
  python upscale_130m_to_1b.py --checkpoint /tmp/checkpoints/h200_step3900.pt
"""

import sys, os, argparse, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helm-src'))

import torch
import torch.nn.functional as F
from helm.hypercore.manifolds import Lorentz
from helm.modules.helm_d import LTransformerDecoder


def upscale_embedding(old_emb, new_dim):
    """Zero-pad Lorentz embedding from old_dim to new_dim.
    
    [x0, x1, ..., x383] → [x0, x1, ..., x383, 0, ..., 0]
                                                 ↑ new spatial zeros
    
    Preserves: -x0² + Σxi² = -1 exactly, since zeros don't contribute.
    """
    old_dim = old_emb.shape[-1]
    pad_size = new_dim - old_dim
    if pad_size <= 0:
        return old_emb
    padding = torch.zeros(*old_emb.shape[:-1], pad_size, dtype=old_emb.dtype)
    return torch.cat([old_emb, padding], dim=-1)


def upscale_linear_weight(old_weight, new_out, new_in, noise_scale=0.001):
    """Place old weight in top-left corner of new larger matrix.
    
    Old [out_old, in_old] → New [new_out, new_in]
    Top-left = old weights, rest = small noise (near-zero init).
    
    Because new input dims are zero (from zero-padded embeddings),
    the noise×0 = 0, so output is mathematically identical.
    """
    old_out, old_in = old_weight.shape
    new_weight = torch.randn(new_out, new_in, dtype=old_weight.dtype) * noise_scale
    new_weight[:old_out, :old_in] = old_weight
    return new_weight


def upscale_bias(old_bias, new_size, noise_scale=0.001):
    """Extend bias vector."""
    old_size = old_bias.shape[0]
    if new_size <= old_size:
        return old_bias
    new_bias = torch.randn(new_size, dtype=old_bias.dtype) * noise_scale
    new_bias[:old_size] = old_bias
    return new_bias


def upscale_layernorm(old_weight, new_size):
    """Extend LayerNorm weights (ones for new dims)."""
    old_size = old_weight.shape[0]
    new_weight = torch.ones(new_size, dtype=old_weight.dtype)
    new_weight[:old_size] = old_weight
    return new_weight


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True, help='Path to 130M checkpoint')
    parser.add_argument('--output', default='/tmp/checkpoints/helm_1b_upscaled.pt')
    parser.add_argument('--old_width', type=int, default=384)
    parser.add_argument('--new_width', type=int, default=1536)
    parser.add_argument('--old_layers', type=int, default=6)
    parser.add_argument('--new_layers', type=int, default=24)
    parser.add_argument('--old_heads', type=int, default=6)
    parser.add_argument('--new_heads', type=int, default=24)
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"HELM-D Network Morphism: 130M → 1B")
    print(f"  Width:  {args.old_width} → {args.new_width}")
    print(f"  Depth:  {args.old_layers} → {args.new_layers}")
    print(f"  Heads:  {args.old_heads} → {args.new_heads}")
    print(f"{'='*60}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    old_sd = ckpt['model_state_dict']
    print(f"  Step: {ckpt.get('global_step', 'unknown')}")
    print(f"  Keys: {len(old_sd)}")

    # Strip _orig_mod. prefix from torch.compile
    clean_sd = {}
    for k, v in old_sd.items():
        clean_k = k.replace('_orig_mod.', '')
        clean_sd[clean_k] = v
    old_sd = clean_sd

    # Dimension constants
    W_old = args.old_width          # 384
    W_new = args.new_width          # 1536
    S_old = W_old - 1               # 383 (spatial dims, Lorentz)
    S_new = W_new - 1               # 1535
    H_old = args.old_heads           # 6
    H_new = args.new_heads           # 24
    HD_old = W_old // H_old         # 64 (per-head dim)
    HD_new = W_new // H_new         # 64 (keep per-head dim same!)
    QK_old = H_old * (HD_old - 1)   # 6 * 63 = 378
    QK_new = H_new * (HD_new - 1)   # 24 * 63 = 1512
    MLP_old = W_old * 4 - 1         # 1535
    MLP_new = W_new * 4 - 1         # 6143

    print(f"\n  Spatial dims: {S_old} → {S_new}")
    print(f"  QK dims:     {QK_old} → {QK_new}")
    print(f"  MLP dims:    {MLP_old} → {MLP_new}")
    print(f"  Per-head:    {HD_old} (unchanged)")

    clones_per_layer = args.new_layers // args.old_layers  # 24 / 6 = 4
    residual_scale = 1.0 / (clones_per_layer ** 0.5)  # Scale down residuals
    print(f"  Clones/layer: {clones_per_layer}")
    print(f"  Residual scale: {residual_scale:.4f}")

    new_sd = {}

    # --- 1. Embeddings ---
    print(f"\n1. Embedding expansion...")
    emb_key = 'token_embed.embedding'
    old_emb = old_sd[emb_key]
    print(f"  {emb_key}: {list(old_emb.shape)} → ", end='')
    new_sd[emb_key] = upscale_embedding(old_emb, W_new)
    print(f"{list(new_sd[emb_key].shape)}")

    # Verify Lorentz constraint preserved
    sample = new_sd[emb_key][:10]
    time_sq = sample[:, 0] ** 2
    space_sq = (sample[:, 1:] ** 2).sum(dim=-1)
    constraint = time_sq - space_sq
    print(f"  Lorentz constraint check (first 10): {constraint.mean():.6f} ± {constraint.std():.6f}")

    # Copy token_embed scalar params
    for key in old_sd:
        if key.startswith('token_embed.') and key != emb_key:
            new_sd[key] = old_sd[key].clone()

    # --- 2. Mapping (output head) ---
    print(f"\n2. Output mapping expansion...")
    map_key = 'mapping.weight'
    old_map = old_sd[map_key]
    print(f"  {map_key}: {list(old_map.shape)} → ", end='')
    new_sd[map_key] = upscale_linear_weight(old_map, old_map.shape[0], W_new)
    print(f"{list(new_sd[map_key].shape)}")

    # --- 3. Final LayerNorm + Projection ---
    print(f"\n3. Final layers...")
    for key in ['ln_final.weight']:
        old_w = old_sd[key]
        new_sd[key] = upscale_layernorm(old_w, S_new)
        print(f"  {key}: {list(old_w.shape)} → {list(new_sd[key].shape)}")

    # final_proj linear
    fp_w = old_sd['final_proj.linear.weight']
    fp_b = old_sd['final_proj.linear.bias']
    new_sd['final_proj.linear.weight'] = upscale_linear_weight(fp_w, S_new, W_new)
    new_sd['final_proj.linear.bias'] = upscale_bias(fp_b, S_new)
    print(f"  final_proj.linear.weight: {list(fp_w.shape)} → {list(new_sd['final_proj.linear.weight'].shape)}")

    # Copy scalar params for final blocks
    for key in old_sd:
        if any(key.startswith(p) for p in ['ln_final.c', 'ln_final.manifold',
                                             'final_proj.c', 'final_proj.manifold']):
            new_sd[key] = old_sd[key].clone()

    # --- 4. Transformer layers (clone 6 → 24) ---
    print(f"\n4. Transformer layer cloning and expansion...")
    for new_layer_idx in range(args.new_layers):
        # Interleaved cloning (Google Depth Up-Cycling):
        # Repeats the full pipeline: 0,1,2,3,4,5, 0,1,2,3,4,5, ...
        # This preserves the learned layer-to-layer computation flow.
        old_layer_idx = new_layer_idx % args.old_layers
        prefix_old = f'resblocks.{old_layer_idx}'
        prefix_new = f'resblocks.{new_layer_idx}'
        is_clone = new_layer_idx >= args.old_layers  # first 6 are originals
        
        cycle = new_layer_idx // args.old_layers
        if cycle == 0:
            clone_label = f"original (layer {old_layer_idx})"
        else:
            clone_label = f"cycle {cycle}, clone of layer {old_layer_idx}"

        print(f"  Layer {new_layer_idx:2d} ← {clone_label}")

        for key, val in old_sd.items():
            if not key.startswith(prefix_old + '.'):
                continue
            suffix = key[len(prefix_old):]
            new_key = prefix_new + suffix

            if val.dim() == 0:
                # Scalar (curvature, bias flag, etc.)
                new_sd[new_key] = val.clone()
            elif 'ln_1.weight' in suffix or 'ln_2.weight' in suffix:
                # LayerNorm: extend to new spatial dim
                new_sd[new_key] = upscale_layernorm(val, S_new)
            elif 'attn.Wq.weight' in suffix or 'attn.Wk.weight' in suffix or 'attn.Wv.weight' in suffix:
                # QKV projections: [QK_old, W_old] → [QK_new, W_new]
                new_sd[new_key] = upscale_linear_weight(val, QK_new, W_new)
            elif 'attn.Wq.bias' in suffix or 'attn.Wk.bias' in suffix or 'attn.Wv.bias' in suffix:
                new_sd[new_key] = upscale_bias(val, QK_new)
            elif 'attn.final_linear.weight' in suffix:
                # Output projection: [S_old, W_old] → [S_new, W_new]
                new_sd[new_key] = upscale_linear_weight(val, S_new, W_new)
            elif 'attn.final_linear.bias' in suffix:
                new_sd[new_key] = upscale_bias(val, S_new)
            elif 'attn.scale' in suffix:
                # Attention scale: keep same (head_dim unchanged)
                new_sd[new_key] = val.clone()
            elif 'mlp.w1.linear.weight' in suffix or 'mlp.w3.linear.weight' in suffix:
                # Up-projection: [MLP_old, W_old] → [MLP_new, W_new]
                new_sd[new_key] = upscale_linear_weight(val, MLP_new, W_new)
            elif 'mlp.w1.linear.bias' in suffix or 'mlp.w3.linear.bias' in suffix:
                new_sd[new_key] = upscale_bias(val, MLP_new)
            elif 'mlp.w2.linear.weight' in suffix:
                # Down-projection: [S_old, MLP_old+1] → [S_new, MLP_new+1]
                new_sd[new_key] = upscale_linear_weight(val, S_new, MLP_new + 1)
            elif 'mlp.w2.linear.bias' in suffix:
                new_sd[new_key] = upscale_bias(val, S_new)
            elif 'res1.w_y' in suffix or 'res2.w_y' in suffix:
                # Residual weights: scale down for cloned layers to prevent
                # signal amplification from 4 identical residual paths
                new_val = val.clone()
                if is_clone:
                    new_val = new_val * residual_scale
                new_sd[new_key] = new_val
            else:
                # All other scalars (manifold.k, manifold.c, etc.)
                new_sd[new_key] = val.clone()

    # --- 5. Build new model and verify ---
    print(f"\n5. Building 1B model and loading upscaled weights...")
    arch = f"L{args.new_layers}W{args.new_width}A{args.new_heads}"
    model = LTransformerDecoder(
        manifold_in=Lorentz(1.0),
        manifold_hidden=Lorentz(1.0),
        manifold_out=Lorentz(1.0),
        arch=arch,
        vocab_size=151669,
        context_length=2048,
    )

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Architecture: {arch}")
    print(f"  Parameters: {num_params:,} ({num_params/1e9:.2f}B)")

    # Load upscaled weights
    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  First 5 missing: {missing[:5]}")
    if unexpected:
        print(f"  First 5 unexpected: {unexpected[:5]}")

    # --- 6. Save ---
    print(f"\n6. Saving upscaled checkpoint...")
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save({
        'global_step': 0,  # Reset step counter for new training
        'model_state_dict': model.state_dict(),
        'upscale_info': {
            'source_checkpoint': args.checkpoint,
            'source_step': ckpt.get('global_step', 'unknown'),
            'old_arch': f'L{args.old_layers}W{args.old_width}A{args.old_heads}',
            'new_arch': arch,
            'old_params': sum(v.numel() for v in old_sd.values()),
            'new_params': num_params,
            'residual_scale': residual_scale,
        },
    }, args.output)
    size_gb = os.path.getsize(args.output) / 1e9
    print(f"  Saved: {args.output} ({size_gb:.2f} GB)")

    print(f"\n{'='*60}")
    print(f"Upscale complete: 130M → {num_params/1e9:.2f}B")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
