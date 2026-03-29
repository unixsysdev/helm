"""
H-MICE v3: Interleaved Dense/MoE Log-Euclidean Architecture (~600M)

ARCHITECTURE:
  - 16 layers, interleaved:
    - Odd blocks (1,3,5...): Dense SwiGLU FFN (1024→2048→1024)
    - Even blocks (2,4,6...): H-MICE MoE with 9 experts
  - 9 experts per MoE: 4 Euclidean + 4 Hyperbolic + 1 Spherical
  - Hyperbolic curvatures: FIXED k=[0.2, 0.5, 1.0, 2.0]
  - Spherical: 64D bottleneck (curse of dimensionality fix)
  - Flat Euclidean residual stream (no manifold boundary crossings = no NaN)
  - Top-1 routing across entire array

SIZING:
  hidden_dim = 1024
  intermediate_dim = 2048
  ~600M total, ~200M active per token
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import geoopt


# =============================================================================
# RoPE
# =============================================================================

def precompute_rope(dim: int, max_seq: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)


def apply_rope(x: torch.Tensor, freqs: torch.Tensor):
    B, H, T, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, T, D // 2, 2))
    freqs = freqs[:T].unsqueeze(0).unsqueeze(0)
    out = torch.view_as_real(x_complex * freqs).reshape(B, H, T, D)
    return out.type_as(x)


# =============================================================================
# Flat Attention
# =============================================================================

class SpatialAttention(nn.Module):
    """Spatial-only attention (helm-d pattern).
    
    Operates on D-dim flat spatial vectors (time component stripped by
    the tangent sandwich). Uses is_causal=True for FA2 fast kernel path.
    """
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wk = nn.Linear(dim, dim, bias=False)
        self.wv = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)
        # Zero-init output projection: step-0 identity pass-through
        nn.init.zeros_(self.wo.weight)

    def forward(self, x, freqs, mask=None):
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = apply_rope(q, freqs)
        k = apply_rope(k, freqs)
        # FA2: is_causal=True triggers the fused CUDA kernel (no mask materialization)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(out.transpose(1, 2).contiguous().view(B, T, D))


# =============================================================================
# SwiGLU FFN (Dense block and Euclidean/Hyperbolic experts)
# =============================================================================

class SwiGLUFFN(nn.Module):
    """SwiGLU: w2(SiLU(w1(x)) * w3(x)).  1024 → 2048 → 1024."""
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


# =============================================================================
# Spherical Expert (64D bottleneck)
# =============================================================================

class SphericalExpert(nn.Module):
    """1024 → 64 → sphere ops → 64 → 1024."""
    def __init__(self, dim: int, inter_dim: int, sphere_dim: int = 64):
        super().__init__()
        self.manifold = geoopt.manifolds.Sphere()
        self.down = nn.Linear(dim, sphere_dim, bias=False)
        self.up = nn.Linear(sphere_dim, dim, bias=False)
        self.w1 = nn.Linear(sphere_dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, sphere_dim, bias=False)
        self.w3 = nn.Linear(sphere_dim, inter_dim, bias=False)

    def forward(self, x):
        x_low = self.down(x)
        h = self.w2(F.silu(self.w1(x_low)) * self.w3(x_low))
        # Add epsilon before sphere projection to prevent projx(0) = NaN
        h_safe = h + 1e-8
        h_sphere = self.manifold.projx(h_safe)
        return self.up(h), h_sphere


# =============================================================================
# H-MICE MoE Layer (9 experts: 4E + 4H + 1S)
# =============================================================================

class HMICEMoE(nn.Module):
    """
    9-expert MoE with 2-tier routing.
    L1: [Euclidean, Hyperbolic, Spherical] → Top-1
    L2: Sub-expert within geometry → Top-1
    
    Fixed hyperbolic curvatures: k=[0.2, 0.5, 1.0, 2.0]
    Geometric loss: p_selected × d² (backprop whip)
    """
    HYP_CURVATURES = [0.2, 0.5, 1.0, 2.0]

    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.dim = dim

        # 4 Hyperbolic manifold instances (FIXED curvatures)
        self.hyp_manifolds = [geoopt.manifolds.Lorentz(k=k) for k in self.HYP_CURVATURES]

        # L1 Geometry Router → 3 logits
        self.l1_router = nn.Linear(dim, 3, bias=False)

        # L2 Routers: Euclidean(4), Hyperbolic(4), Spherical(1 — no router needed)
        self.l2_euc_router = nn.Linear(dim, 4, bias=False)
        self.l2_hyp_router = nn.Linear(dim, 4, bias=False)

        # L2 Load balance biases
        self.l2_euc_bias = nn.Parameter(torch.zeros(4))
        self.l2_hyp_bias = nn.Parameter(torch.zeros(4))

        # Experts (all standard nn.Linear)
        self.euc_experts = nn.ModuleList([SwiGLUFFN(dim, inter_dim) for _ in range(4)])
        self.hyp_experts = nn.ModuleList([SwiGLUFFN(dim, inter_dim) for _ in range(4)])
        self.sph_expert = SphericalExpert(dim, inter_dim)

        # Telemetry EMA
        self.register_buffer('l1_ema', torch.ones(3) / 3, persistent=False)

    def forward(self, x):
        """
        x: [B, T, D] flat
        Returns: output [B,T,D], l1_logits [B,T,3], geom_loss scalar
        """
        B, T, D = x.shape
        flat = x.reshape(B * T, D)

        # L1 routing
        l1_logits = self.l1_router(flat)  # [BT, 3]
        l1_probs = F.softmax(l1_logits, dim=-1)
        l1_idx = l1_logits.argmax(dim=-1)
        p_selected = l1_probs.gather(1, l1_idx.unsqueeze(1)).squeeze(1)

        # Telemetry (zero-sync)
        with torch.no_grad():
            self.l1_ema = 0.99 * self.l1_ema + 0.01 * F.one_hot(l1_idx, 3).float().mean(0)

        output = torch.zeros_like(flat)
        geom_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        # --- Euclidean (geom_id=0) ---
        euc_mask = (l1_idx == 0)
        if euc_mask.any():
            tokens = flat[euc_mask]
            p_sel = p_selected[euc_mask]
            l2 = (self.l2_euc_router(tokens) + self.l2_euc_bias).argmax(-1)
            euc_out = torch.zeros_like(tokens)
            for i in range(4):
                m = (l2 == i)
                if m.any():
                    h = self.euc_experts[i](tokens[m])
                    d_sq = h.pow(2).sum(-1)
                    geom_loss = geom_loss + (p_sel[m] * d_sq).mean() * 0.01
                    euc_out[m] = h
            output[euc_mask] = euc_out

        # --- Hyperbolic (geom_id=1) ---
        hyp_mask = (l1_idx == 1)
        if hyp_mask.any():
            tokens = flat[hyp_mask]
            p_sel = p_selected[hyp_mask]
            l2 = (self.l2_hyp_router(tokens) + self.l2_hyp_bias).argmax(-1)
            hyp_out = torch.zeros_like(tokens)
            for i in range(4):
                m = (l2 == i)
                if m.any():
                    h = self.hyp_experts[i](tokens[m])
                    # Geometric fold: exp₀ to Lorentz for distance
                    manifold = self.hyp_manifolds[i]
                    h_tangent = F.pad(h, (1, 0), value=0.0)
                    h_tangent = h_tangent.clamp(-10.0, 10.0)
                    h_on_M = manifold.expmap0(h_tangent)
                    d_sq = manifold.dist0(h_on_M).pow(2)
                    geom_loss = geom_loss + (p_sel[m] * d_sq).mean()
                    hyp_out[m] = h  # Output stays flat
            output[hyp_mask] = hyp_out

        # --- Spherical (geom_id=2, single expert) ---
        sph_mask = (l1_idx == 2)
        if sph_mask.any():
            tokens = flat[sph_mask]
            p_sel = p_selected[sph_mask]
            h, h_sphere = self.sph_expert(tokens)
            origin = torch.zeros_like(h_sphere)
            origin[..., 0] = 1.0
            d_sq = (1 - (h_sphere * origin).sum(-1)).pow(2)
            geom_loss = geom_loss + (p_sel * d_sq).mean()
            output[sph_mask] = h

        return output.reshape(B, T, D), l1_logits.reshape(B, T, 3), geom_loss


# =============================================================================
# Dense Block (Odd layers) — Pre-Norm + Zero-Init Tangent Sandwich
# =============================================================================

class DenseBlock(nn.Module):
    """Transformer block with true tangent sandwich.
    
    Residual stays on Lorentz manifold. Expert output zero-init'd
    so at step 0: exp₀(log₀(x) + 0) = x (perfect identity).
    """
    def __init__(self, dim: int, n_heads: int, inter_dim: int, manifold=None):
        super().__init__()
        self.manifold = manifold
        # Tangent-space norms (applied to D+1 tangent vectors)
        self.norm1 = nn.RMSNorm(dim + 1)
        self.attn = SpatialAttention(dim, n_heads)
        self.norm2 = nn.RMSNorm(dim + 1)
        self.ffn = SwiGLUFFN(dim, inter_dim)
        # Post-addition norms: control tangent magnitude before expmap0
        self.norm_post1 = nn.RMSNorm(dim + 1)
        self.norm_post2 = nn.RMSNorm(dim + 1)
        # Zero-init the down-projection so step-0 output = 0
        nn.init.zeros_(self.ffn.w2.weight)

    def forward(self, x, freqs):
        M = self.manifold
        # --- Attention ---
        v = M.logmap0(x)                              # [B,T,D+1] tangent
        v_normed = self.norm1(v)
        attn_out = self.attn(v_normed[..., 1:], freqs)  # Spatial-only FA2
        attn_tangent = F.pad(attn_out, (1, 0), value=0.0)     # → D+1
        x = M.expmap0(self.norm_post1(v + attn_tangent))       # Norm before exp₀

        # --- FFN ---
        v = M.logmap0(x)
        v_normed = self.norm2(v)
        ffn_out = self.ffn(v_normed[..., 1:])
        ffn_tangent = F.pad(ffn_out, (1, 0), value=0.0)
        x = M.expmap0(self.norm_post2(v + ffn_tangent))        # Norm before exp₀

        return x, None, torch.tensor(0.0, device=x.device, dtype=x.dtype)


# =============================================================================
# MoE Block (Even layers) — Pre-Norm + Zero-Init Tangent Sandwich
# =============================================================================

class MoEBlock(nn.Module):
    """H-MICE MoE block with true tangent sandwich.
    
    Same identity pass-through guarantee via zero-init'd experts.
    """
    def __init__(self, dim: int, n_heads: int, inter_dim: int, manifold=None):
        super().__init__()
        self.manifold = manifold
        self.norm1 = nn.RMSNorm(dim + 1)
        self.attn = SpatialAttention(dim, n_heads)
        self.norm2 = nn.RMSNorm(dim + 1)
        self.moe = HMICEMoE(dim, inter_dim)
        # Post-addition norms
        self.norm_post1 = nn.RMSNorm(dim + 1)
        self.norm_post2 = nn.RMSNorm(dim + 1)
        # Zero-init all expert down-projections
        for expert in self.moe.euc_experts:
            nn.init.zeros_(expert.w2.weight)
        for expert in self.moe.hyp_experts:
            nn.init.zeros_(expert.w2.weight)
        nn.init.zeros_(self.moe.sph_expert.w2.weight)

    def forward(self, x, freqs):
        M = self.manifold
        # --- Attention ---
        v = M.logmap0(x)
        v_normed = self.norm1(v)
        attn_out = self.attn(v_normed[..., 1:], freqs)  # Spatial-only FA2
        attn_tangent = F.pad(attn_out, (1, 0), value=0.0)
        x = M.expmap0(self.norm_post1(v + attn_tangent))

        # --- MoE ---
        v = M.logmap0(x)
        v_normed = self.norm2(v)
        moe_out, l1_logits, geom_loss = self.moe(v_normed[..., 1:])
        moe_tangent = F.pad(moe_out, (1, 0), value=0.0)
        x = M.expmap0(self.norm_post2(v + moe_tangent))

        return x, l1_logits, geom_loss


# =============================================================================
# H-MICE Transformer
# =============================================================================

class HMICETransformer(nn.Module):
    """
    ~600M Interleaved Dense/MoE Transformer.
    
    CURVED RESIDUAL STREAM:
    - Token embeddings: ManifoldParameter on Lorentz
    - Residual stream: STAYS on manifold through all 16 blocks
    - Each block: logmap₀ → flat compute → expmap₀ → curved residual add
    - Exit: single logmap₀ → flat → lm_head
    """
    def __init__(self, vocab_size: int, dim: int = 1024, n_layers: int = 16,
                 n_heads: int = 16, inter_dim: int = 2048, max_seq_len: int = 4096):
        super().__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.vocab_size = vocab_size

        # Shared manifold for the entire residual stream
        self.manifold = geoopt.manifolds.Lorentz(k=1.0)

        # Token embedding ON the Lorentz manifold (dim+1 for time component)
        emb_weights = self._init_lorentz_embeddings(vocab_size, dim)
        self.tok_emb = geoopt.ManifoldParameter(emb_weights, manifold=self.manifold)

        # Interleaved blocks — all receive the shared manifold
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            if i % 2 == 0:
                self.blocks.append(DenseBlock(dim, n_heads, inter_dim, manifold=self.manifold))
            else:
                self.blocks.append(MoEBlock(dim, n_heads, inter_dim, manifold=self.manifold))

        self.norm_out = nn.RMSNorm(dim)
        # LM head: flat dim → vocab (shapes differ from emb: D vs D+1)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        self.register_buffer('freqs', precompute_rope(dim // n_heads, max_seq_len))

    def _init_lorentz_embeddings(self, vocab_size, dim):
        """Initialize embeddings as points on the Lorentz manifold."""
        spatial = torch.randn(vocab_size, dim) * 0.01
        time = (1.0 + (spatial ** 2).sum(dim=-1, keepdim=True)).sqrt()
        return torch.cat([time, spatial], dim=-1)  # [V, D+1]

    def forward(self, input_ids):
        B, T = input_ids.shape

        # Embedding lookup → ON the manifold [B, T, D+1]
        x = self.tok_emb[input_ids]

        # x STAYS on the manifold through all blocks
        total_geom = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        last_l1 = None
        n_moe = 0

        for block in self.blocks:
            x, l1, gl = block(x, self.freqs)
            if l1 is not None:
                last_l1 = l1
                total_geom = total_geom + gl
                n_moe += 1

        # EXIT: single logmap₀ → flat for lm_head
        v_flat = self.manifold.logmap0(x)[..., 1:]  # [B, T, D]
        v_flat = self.norm_out(v_flat)
        logits = self.lm_head(v_flat)
        avg_geom = total_geom / max(n_moe, 1)
        return logits, last_l1, avg_geom

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def get_hyperbolic_params(self):
        """ManifoldParameter only — for RiemannianAdam (zero weight decay)."""
        return [p for p in self.parameters() if isinstance(p, geoopt.ManifoldParameter)]

    def get_euclidean_params(self):
        """All standard params — for AdamW (0.1 weight decay)."""
        return [p for p in self.parameters() if not isinstance(p, geoopt.ManifoldParameter)]

    def get_l1_distribution(self):
        emas = []
        for b in self.blocks:
            if hasattr(b, 'moe'):
                emas.append(b.moe.l1_ema)
        if emas:
            return torch.stack(emas).mean(0)
        return torch.ones(3) / 3


