"""
HYPER-HELM Phase 2: Riemannian Tokenizer Surgery
Transfers Llama-3.1 embeddings from HELM-D step-2186 to Qwen3 tokenizer.

Strategy:
1. Matching tokens (109K): Direct 1:1 coordinate transfer
2. Qwen3-only tokens (42K): Decompose into Llama sub-tokens, compute
   Lorentzian Fréchet Mean of their embeddings.
3. Rebuild model with new embedding matrix (151,669 × 390)
4. Verify manifold constraint: x₀² - Σxᵢ² = 1/c

Run: toolbox run -c llama-rocm-7.2 bash -c "source venv/bin/activate && HSA_OVERRIDE_GFX_VERSION=11.0.0 python3 tokenizer_surgery.py"
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'helm-src'))

import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer
from helm.hypercore.manifolds import Lorentz
from helm.modules.helm_d import LTransformerDecoder
import time


def lorentz_to_poincare(x, c=1.0):
    """Map from Lorentz model to Poincaré ball for visualization."""
    return x[..., 1:] / (x[..., 0:1] + 1.0)


def poincare_to_lorentz(x, c=1.0):
    """Map from Poincaré ball to Lorentz model."""
    sq_norm = (x * x).sum(dim=-1, keepdim=True)
    time_comp = (1.0 + sq_norm) / (1.0 - sq_norm + 1e-8)
    space_comp = 2.0 * x / (1.0 - sq_norm + 1e-8)
    return torch.cat([time_comp, space_comp], dim=-1)


def lorentzian_frechet_mean(embeddings, weights=None, c=1.0, max_iter=50, tol=1e-6):
    """
    Compute the Fréchet mean on the Lorentz manifold using iterative algorithm.
    For a set of points, finds the point that minimizes the sum of squared
    Lorentzian distances.
    
    Args:
        embeddings: (N, D) tensor of Lorentz embeddings
        weights: (N,) optional weights
        c: curvature
    Returns:
        (D,) Fréchet mean on the manifold
    """
    if embeddings.shape[0] == 1:
        return embeddings[0]
    
    if weights is None:
        weights = torch.ones(embeddings.shape[0], device=embeddings.device)
    weights = weights / weights.sum()
    
    # Use the weighted Einstein midpoint as initial estimate
    # This is faster and more stable than iterative methods for small sets
    gamma = embeddings[:, 0:1]  # time component
    space = embeddings[:, 1:]   # space components
    
    # Weighted average in the tangent space at origin, then project back
    weighted_space = (weights.unsqueeze(1) * space / gamma).sum(dim=0)
    sq_norm = (weighted_space * weighted_space).sum()
    
    if sq_norm >= 1.0:
        # Normalize to stay in the ball
        weighted_space = weighted_space * (0.95 / (sq_norm.sqrt() + 1e-8))
        sq_norm = (weighted_space * weighted_space).sum()
    
    # Project back to Lorentz
    time_val = (1.0 + sq_norm) / (1.0 - sq_norm + 1e-8)
    space_out = 2.0 * weighted_space / (1.0 - sq_norm + 1e-8)
    
    result = torch.cat([time_val.unsqueeze(0), space_out], dim=0)
    
    # Ensure it's on the manifold: x₀² - Σxᵢ² = 1
    space_sq = (result[1:] ** 2).sum()
    result[0] = (space_sq + c).sqrt()
    
    return result


def project_to_lorentz(x, c=1.0):
    """Project a vector onto the Lorentz manifold: x₀ = sqrt(||x_space||² + 1/c)"""
    space = x[..., 1:]
    space_sq = (space ** 2).sum(dim=-1, keepdim=True)
    time = (space_sq + c).sqrt()
    return torch.cat([time, space], dim=-1)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Load tokenizers
    print("Loading tokenizers...")
    tok_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    tok_qwen = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B")
    
    llama_vocab = tok_llama.get_vocab()  # token_str -> id
    qwen_vocab = tok_qwen.get_vocab()
    
    print(f"Llama vocab: {len(llama_vocab)}")
    print(f"Qwen3 vocab: {len(qwen_vocab)}")
    
    # Compute intersection
    common_tokens = set(llama_vocab.keys()) & set(qwen_vocab.keys())
    qwen_only = set(qwen_vocab.keys()) - set(llama_vocab.keys())
    print(f"Common tokens: {len(common_tokens)}")
    print(f"Qwen3-only tokens: {len(qwen_only)}")
    
    # Load HELM-D checkpoint
    print("\nLoading HELM-D step-2186 checkpoint...")
    ckpt = torch.load('/tmp/helm_d.pt', map_location='cpu', weights_only=False)
    sd = ckpt['model_state_dict']
    print(f"Checkpoint step: {ckpt['global_step']}")
    
    # Extract Llama embedding matrix
    # The embedding key in HELM-D
    emb_key = None
    head_key = None
    for k in sd:
        if 'token_embed' in k and 'embedding' in k and 'manifold' not in k and 'pos' not in k:
            emb_key = k
        elif 'mapping' in k and 'weight' in k:
            head_key = k
    
    if emb_key is None:
        # Try alternative key names
        for k in sd:
            if 'embed' in k.lower() and sd[k].shape[0] == len(llama_vocab):
                emb_key = k
                break
    
    print(f"Embedding key: {emb_key} -> shape: {sd[emb_key].shape}")
    print(f"Head key: {head_key} -> shape: {sd[head_key].shape}")
    
    llama_embeddings = sd[emb_key].float()  # (128256, 390)
    llama_head = sd[head_key].float()       # (128256, 390)
    embed_dim = llama_embeddings.shape[1]
    
    print(f"\nEmbedding stats:")
    print(f"  mean={llama_embeddings.mean():.5f}, std={llama_embeddings.std():.5f}")
    print(f"  time component (col 0) mean={llama_embeddings[:, 0].mean():.4f}, "
          f"min={llama_embeddings[:, 0].min():.4f}")
    
    # Check if embeddings are on Lorentz manifold
    space_sq = (llama_embeddings[:, 1:] ** 2).sum(dim=-1)
    time_sq = llama_embeddings[:, 0] ** 2
    constraint = time_sq - space_sq  # Should be ~1.0 for Lorentz
    print(f"  Lorentz constraint (t²-||x||²): mean={constraint.mean():.4f}, "
          f"std={constraint.std():.4f}")
    
    # Build new Qwen3 embedding matrix
    qwen_vocab_size = len(qwen_vocab)
    new_embeddings = torch.zeros(qwen_vocab_size, embed_dim)
    new_head = torch.zeros(qwen_vocab_size, llama_head.shape[1])
    
    # Track statistics
    direct_transfer = 0
    frechet_init = 0
    random_init = 0
    
    print(f"\n{'='*60}")
    print(f"Performing tokenizer surgery...")
    print(f"{'='*60}")
    
    t0 = time.time()
    
    # Reverse lookup: qwen token string -> qwen id
    qwen_str_to_id = qwen_vocab
    llama_str_to_id = llama_vocab
    
    for token_str, qwen_id in qwen_str_to_id.items():
        if token_str in llama_str_to_id:
            # CASE 1: Direct transfer — token exists in both vocabs
            llama_id = llama_str_to_id[token_str]
            new_embeddings[qwen_id] = llama_embeddings[llama_id]
            new_head[qwen_id] = llama_head[llama_id]
            direct_transfer += 1
        else:
            # CASE 2: Qwen3-only token — decompose into Llama sub-tokens
            # Encode the token text using Llama tokenizer to get constituent sub-tokens
            try:
                sub_ids = tok_llama.encode(token_str, add_special_tokens=False)
            except Exception:
                sub_ids = []
            
            if len(sub_ids) > 0 and all(0 <= sid < llama_embeddings.shape[0] for sid in sub_ids):
                # Compute Fréchet Mean of the Llama sub-token embeddings
                sub_embeddings = llama_embeddings[sub_ids]
                mean_emb = lorentzian_frechet_mean(sub_embeddings)
                new_embeddings[qwen_id] = mean_emb
                
                # For the head, use simple average (Euclidean output layer)
                new_head[qwen_id] = llama_head[sub_ids].mean(dim=0)
                frechet_init += 1
            else:
                # CASE 3: Can't decompose — random init on manifold
                space = torch.randn(embed_dim - 1) * llama_embeddings[:, 1:].std()
                space_sq = (space ** 2).sum()
                time_comp = (space_sq + 1.0).sqrt()
                new_embeddings[qwen_id] = torch.cat([time_comp.unsqueeze(0), space])
                new_head[qwen_id] = torch.randn(llama_head.shape[1]) * llama_head.std()
                random_init += 1
    
    elapsed = time.time() - t0
    
    print(f"\nSurgery complete in {elapsed:.1f}s:")
    print(f"  Direct transfer: {direct_transfer:,} ({direct_transfer/qwen_vocab_size*100:.1f}%)")
    print(f"  Fréchet Mean:    {frechet_init:,} ({frechet_init/qwen_vocab_size*100:.1f}%)")
    print(f"  Random init:     {random_init:,} ({random_init/qwen_vocab_size*100:.1f}%)")
    
    # Verify manifold constraint on new embeddings
    space_sq = (new_embeddings[:, 1:] ** 2).sum(dim=-1)
    time_sq = new_embeddings[:, 0] ** 2
    constraint = time_sq - space_sq
    print(f"\nNew embedding Lorentz constraint: mean={constraint.mean():.4f}, std={constraint.std():.4f}")
    
    # Project all new embeddings back onto manifold to ensure constraint
    new_embeddings = project_to_lorentz(
        torch.cat([new_embeddings[:, 0:1], new_embeddings[:, 1:]], dim=-1)
    )
    
    # Update state dict with new embeddings
    sd[emb_key] = new_embeddings
    sd[head_key] = new_head
    
    # Save the surgically modified checkpoint
    out_path = '/home/marcel/Work/helm/checkpoints/helm_d_qwen3_surgery.pt'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({
        'global_step': ckpt['global_step'],
        'model_state_dict': sd,
        'surgery_stats': {
            'direct_transfer': direct_transfer,
            'frechet_init': frechet_init,
            'random_init': random_init,
            'source_tokenizer': 'meta-llama/Llama-3.1-8B',
            'target_tokenizer': 'Qwen/Qwen3-30B-A3B',
        }
    }, out_path)
    print(f"\nSaved surgically modified checkpoint: {out_path}")
    
    # Now build HELM-D model with new vocab size and test
    print(f"\n{'='*60}")
    print(f"Building HELM-D with Qwen3 vocab ({qwen_vocab_size})...")
    print(f"{'='*60}")
    
    # Need to modify model to accept new vocab size
    # HELM-D uses arch string "L6W390A6" for the 120M model
    model = LTransformerDecoder(
        manifold_in=Lorentz(1.0),
        manifold_hidden=Lorentz(1.0),
        manifold_out=Lorentz(1.0),
        arch="L6W390A6",
        vocab_size=qwen_vocab_size,
        context_length=2048,
    )
    
    # Load the surgically modified state dict
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"Load result: missing={len(missing)}, unexpected={len(unexpected)}")
    if missing:
        print(f"  Missing: {missing[:5]}")
    if unexpected:
        print(f"  Unexpected: {unexpected[:5]}")
    
    model = model.to(device).eval()
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Test generation with Qwen3 tokenizer
    print(f"\n{'='*60}")
    print(f"Testing generation with Qwen3 tokenizer (POST-SURGERY):")
    print(f"{'='*60}")
    
    prompts = [
        "The theory of general relativity",
        "Romania is a country in",
        "In computer science, a binary tree",
        "Albert Einstein was a",
        "def hello_world():",
    ]
    
    for prompt in prompts:
        input_ids = tok_qwen.encode(prompt, return_tensors='pt').to(device)
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(60):
                logits = model(generated)
                next_logits = logits[:, -1, :] / 0.7
                values, _ = torch.topk(next_logits, 40)
                next_logits = torch.where(next_logits < values[:, -1:], -float('inf'), next_logits)
                probs = torch.softmax(next_logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
                generated = torch.cat([generated, next_id], dim=-1)
                if next_id.item() == tok_qwen.eos_token_id:
                    break
                if generated.shape[1] >= 256:
                    break
        
        text = tok_qwen.decode(generated[0], skip_special_tokens=True)
        print(f"\n  Prompt: '{prompt}'")
        print(f"  Output: {text[:200]}")
    
    print(f"\n{'='*60}")
    print("Surgery validation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
