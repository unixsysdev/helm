"""
H-MICE: Hierarchical Mixture of Curvature Experts (400M)

A sparse transformer decoder with a Product Manifold (H×E×S) and a 2-tier
hierarchical router that dynamically routes tokens to geometry-specific experts.

Architecture:
  - Residual stream: Flat Euclidean tangent space
  - Attention: Standard MHA in flat space (RoPE)
  - MoE: 3 geometries × 4 experts = 12 total, Top-1 × Top-1
  - Expert output: manifold fold → geometric loss → log₀ back to flat

Sizing:
  - Hidden: 1024, Layers: 16, Heads: 16
  - Total params: ~400M, Active per token: ~150M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt


# =============================================================================
# RoPE (Rotary Position Embeddings)
# =============================================================================

def precompute_rope(dim: int, max_seq: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex64


def apply_rope(x: torch.Tensor, freqs: torch.Tensor):
    B, H, T, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)
    out = torch.view_as_real(x_complex * freqs).reshape(B, H, T, D)
    return out.type_as(x)


# =============================================================================
# Multi-Head Attention (Flat Euclidean Space)
# =============================================================================

class FlatAttention(nn.Module):
    """Standard MHA operating in flat tangent space. No manifold ops."""

    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor, freqs: torch.Tensor, mask: torch.Tensor):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)

        # Scaled dot-product attention
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=(mask is None))
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)


# =============================================================================
# Expert FFN
# =============================================================================

class ExpertFFN(nn.Module):
    """Standard FFN expert: Linear → SiLU → Linear."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)  # gate

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =============================================================================
# Spherical Expert (with 64D bottleneck)
# =============================================================================

class SphericalExpert(nn.Module):
    """Expert that operates in a 64D spherical subspace."""

    def __init__(self, dim: int, hidden_dim: int, sphere_dim: int = 64):
        super().__init__()
        self.sphere_dim = sphere_dim
        self.manifold = geoopt.manifolds.Sphere()

        # Bottleneck projections
        self.down_proj = nn.Linear(dim, sphere_dim, bias=False)
        self.up_proj = nn.Linear(sphere_dim, dim, bias=False)

        # FFN in sphere space
        self.w1 = nn.Linear(sphere_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, sphere_dim, bias=False)
        self.w3 = nn.Linear(sphere_dim, hidden_dim, bias=False)

    def forward(self, x):
        """Returns (flat_output, manifold_point_for_loss)."""
        # Down-project to sphere subspace
        x_low = self.down_proj(x)

        # FFN in low-dim space
        h = self.w2(F.silu(self.w1(x_low)) * self.w3(x_low))

        # Fold onto sphere (for geometric loss)
        h_sphere = self.manifold.projx(h)  # Project onto unit sphere

        # Back to flat (log map is identity at origin for sphere → just use projected value)
        x_low_out = h  # Return pre-projection for residual (flat)

        # Up-project back to full dim
        return self.up_proj(x_low_out), h_sphere


# =============================================================================
# H-MICE MoE Layer
# =============================================================================

class HMICEMoE(nn.Module):
    """
    Hierarchical Mixture of Curvature Experts.

    L1: Route to geometry (Hyperbolic / Euclidean / Spherical) — Top-1
    L2: Route to sub-expert within geometry — Top-1

    All routing in flat tangent space. Manifold mapping only for loss penalty.
    """

    def __init__(self, dim: int, n_experts_per_geom: int = 4, ffn_hidden: int = 768,
                 sphere_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.n_geom = 3  # H, E, S
        self.n_experts_per_geom = n_experts_per_geom
        hidden_dim = ffn_hidden

        # Manifold instances
        self.manifold_hyp = geoopt.manifolds.Lorentz()
        self.manifold_sph = geoopt.manifolds.Sphere()
        # Euclidean needs no special manifold

        # L1 Geometry Router
        self.l1_router = nn.Linear(dim, self.n_geom, bias=False)

        # L2 Expert Routers (one per geometry)
        self.l2_routers = nn.ModuleList([
            nn.Linear(dim, n_experts_per_geom, bias=False)
            for _ in range(self.n_geom)
        ])

        # L2 Load balancing biases (DeepSeekV3-style)
        self.l2_biases = nn.ParameterList([
            nn.Parameter(torch.zeros(n_experts_per_geom))
            for _ in range(self.n_geom)
        ])

        # Experts: Hyperbolic (4) + Euclidean (4) + Spherical (4)
        self.hyp_experts = nn.ModuleList([ExpertFFN(dim, hidden_dim) for _ in range(n_experts_per_geom)])
        self.euc_experts = nn.ModuleList([ExpertFFN(dim, hidden_dim) for _ in range(n_experts_per_geom)])
        self.sph_experts = nn.ModuleList([SphericalExpert(dim, hidden_dim, sphere_dim) for _ in range(n_experts_per_geom)])

        # Router telemetry EMA (no grad, no sync)
        self.register_buffer('l1_ema', torch.ones(3) / 3, persistent=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, T, D] flat tangent space vectors

        Returns:
            output: [B, T, D] flat tangent space vectors
            l1_logits: [B, T, 3] for supervised loss (detached for telemetry)
            geom_loss: scalar geometric distortion penalty
        """
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)

        # --- L1 Geometry Routing ---
        l1_logits = self.l1_router(x_flat)  # [BT, 3]
        l1_probs = F.softmax(l1_logits, dim=-1)  # [BT, 3] — MUST stay in autograd graph
        l1_idx = l1_logits.argmax(dim=-1)   # [BT] — Top-1 hard routing

        # Gather p_selected for each token (stays in graph for backprop whip)
        p_selected = l1_probs.gather(1, l1_idx.unsqueeze(1)).squeeze(1)  # [BT]

        # Telemetry: update EMA (zero-sync, detached)
        with torch.no_grad():
            l1_onehot = F.one_hot(l1_idx, self.n_geom).float()
            l1_dist = l1_onehot.mean(dim=0)
            self.l1_ema = 0.99 * self.l1_ema + 0.01 * l1_dist

        # --- Dispatch to geometry groups ---
        output = torch.zeros_like(x_flat)
        geom_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for geom_id in range(self.n_geom):
            mask = (l1_idx == geom_id)  # [BT] boolean
            if not mask.any():
                continue

            tokens = x_flat[mask]  # [N, D]
            p_sel = p_selected[mask]  # [N] — router prob, in autograd graph

            # L2 routing within this geometry
            l2_logits = self.l2_routers[geom_id](tokens) + self.l2_biases[geom_id]
            l2_idx = l2_logits.argmax(dim=-1)  # [N]

            # Dispatch to individual experts
            expert_out = torch.zeros_like(tokens)

            for exp_id in range(self.n_experts_per_geom):
                exp_mask = (l2_idx == exp_id)
                if not exp_mask.any():
                    continue

                exp_tokens = tokens[exp_mask]
                p_exp = p_sel[exp_mask]  # Router probs for these tokens

                if geom_id == 0:  # Hyperbolic
                    h = self.hyp_experts[exp_id](exp_tokens)
                    # Geometric fold: expmap0 to Lorentz for distance calc
                    h_padded = F.pad(h, (1, 0), value=0.0)  # [N_exp, D+1]
                    h_manifold = self.manifold_hyp.expmap0(h_padded)
                    d_sq = self.manifold_hyp.dist0(h_manifold).pow(2)
                    # CRITICAL: p_selected * d² — backprop whip to router
                    geom_loss = geom_loss + (p_exp * d_sq).mean()
                    expert_out[exp_mask] = h

                elif geom_id == 1:  # Euclidean
                    h = self.euc_experts[exp_id](exp_tokens)
                    d_sq = h.pow(2).sum(dim=-1)  # L2 distance from origin
                    geom_loss = geom_loss + (p_exp * d_sq).mean() * 0.01
                    expert_out[exp_mask] = h

                elif geom_id == 2:  # Spherical
                    h, h_sphere = self.sph_experts[exp_id](exp_tokens)
                    # Angular distance from north pole
                    origin = torch.zeros_like(h_sphere)
                    origin[..., 0] = 1.0
                    cos_dist = (h_sphere * origin).sum(dim=-1)
                    d_sq = (1 - cos_dist).pow(2)
                    geom_loss = geom_loss + (p_exp * d_sq).mean()
                    expert_out[exp_mask] = h

            output[mask] = expert_out

        output = output.reshape(B, T, D)
        l1_logits_out = l1_logits.reshape(B, T, self.n_geom)

        return output, l1_logits_out, geom_loss


# =============================================================================
# Transformer Block
# =============================================================================

class HMICEBlock(nn.Module):
    """Transformer block with flat attention + H-MICE MoE."""

    def __init__(self, dim: int, n_heads: int, n_experts_per_geom: int = 4):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.attn = FlatAttention(dim, n_heads)
        self.norm2 = nn.RMSNorm(dim)
        self.moe = HMICEMoE(dim, n_experts_per_geom)

    def forward(self, x, freqs, mask):
        # Attention (flat space)
        x = x + self.attn(self.norm1(x), freqs, mask)
        # MoE (with geometric routing)
        moe_out, l1_logits, geom_loss = self.moe(self.norm2(x))
        x = x + moe_out
        return x, l1_logits, geom_loss


# =============================================================================
# H-MICE Transformer
# =============================================================================

class HMICETransformer(nn.Module):
    """
    400M Hierarchical Mixture of Curvature Experts Transformer.

    The residual stream operates entirely in flat Euclidean tangent space.
    Manifold mappings act strictly as boundary layers around expert blocks.
    """

    def __init__(self, vocab_size: int, dim: int = 1024, n_layers: int = 16,
                 n_heads: int = 16, n_experts_per_geom: int = 4,
                 max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # Token embedding (flat space)
        self.tok_emb = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            HMICEBlock(dim, n_heads, n_experts_per_geom)
            for _ in range(n_layers)
        ])

        # Output
        self.norm_out = nn.RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.tok_emb.weight

        # RoPE frequencies
        self.register_buffer('freqs', precompute_rope(dim // n_heads, max_seq_len))

        # Causal mask
        mask = torch.triu(torch.full((max_seq_len, max_seq_len), float('-inf')), diagonal=1)
        self.register_buffer('causal_mask', mask)

    def forward(self, input_ids: torch.Tensor):
        """
        Args:
            input_ids: [B, T] token indices

        Returns:
            logits: [B, T, V]
            all_l1_logits: [B, T, 3] from last layer (for supervised loss)
            total_geom_loss: scalar
        """
        B, T = input_ids.shape
        x = self.tok_emb(input_ids)

        mask = self.causal_mask[:T, :T]
        total_geom_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        last_l1_logits = None

        for block in self.blocks:
            x, l1_logits, geom_loss = block(x, self.freqs, mask)
            total_geom_loss = total_geom_loss + geom_loss
            last_l1_logits = l1_logits

        x = self.norm_out(x)
        logits = self.lm_head(x)

        return logits, last_l1_logits, total_geom_loss / self.n_layers

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        unique = total - self.tok_emb.weight.numel()  # tied weights
        return total, unique

    def get_l1_distribution(self):
        """Get averaged L1 routing EMA across all layers."""
        emas = torch.stack([b.moe.l1_ema for b in self.blocks])
        return emas.mean(dim=0)
