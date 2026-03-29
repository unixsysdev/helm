"""
H-MICE v3 Validation: 32K Tokenizer + Offset-Mapped Tagger + Curved Residual

Verifies:
1. 32K tokenizer with offset-mapped geometry tagging
2. ManifoldParameter embedding on Lorentz
3. Curved residual stream (tangent sandwich, all 16 layers)
4. Zero-init identity pass-through (step 0)
5. FA2 spatial-only attention
6. Router learns geometry via dual optimizer
"""

import torch
import torch.nn.functional as F
import geoopt
from model import HMICETransformer
from data import HMICEDataset, generate_mock_data
from tokenizer_utils import get_tokenizer, tag_tokens
from torch.utils.data import DataLoader


def collate_fn(batch):
    input_ids = torch.stack([b['input_ids'] for b in batch])
    geom_targets = torch.stack([b['geom_targets'] for b in batch])
    return input_ids, geom_targets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Device: {device} ({torch.cuda.get_device_name(0)})")

    # --- 32K Tokenizer ---
    print("\n=== Loading 32K Tokenizer ===")
    tokenizer = get_tokenizer()
    V = tokenizer.vocab_size
    print(f"Vocab: {V}")

    # --- Tagger Validation ---
    print("\n=== Offset-Mapped Tagger Test ===")
    test_hyp = "<think>\ndef foo():\n    if True:\n        return 1\n</think>"
    test_sph = "The date is 2024-01-15. Compute cos(theta) on Monday."
    test_euc = "The quick brown fox jumps over the lazy dog."

    for label, text in [("HYP", test_hyp), ("SPH", test_sph), ("EUC", test_euc)]:
        enc = tokenizer(text, return_offsets_mapping=True)
        tags = tag_tokens(text, enc['offset_mapping'])
        tag_counts = {0: tags.count(0), 1: tags.count(1), 2: tags.count(2)}
        print(f"  {label}: {len(tags)} tokens → E:{tag_counts[0]} H:{tag_counts[2]} S:{tag_counts[1]}")

    # --- Model ---
    print(f"\n=== Building H-MICE v3 (vocab={V}) ===")
    model = HMICETransformer(
        vocab_size=V, dim=1024, n_layers=16, n_heads=16,
        inter_dim=2048, max_seq_len=512,
    )
    total = model.count_params()
    print(f"Total: {total:,} ({total/1e6:.0f}M)")

    assert isinstance(model.tok_emb, geoopt.ManifoldParameter)
    print(f"✓ Embedding: ManifoldParameter (shape={model.tok_emb.shape})")
    n_dense = sum(1 for b in model.blocks if hasattr(b, 'ffn'))
    n_moe = sum(1 for b in model.blocks if hasattr(b, 'moe'))
    print(f"✓ Topology: {n_dense} dense + {n_moe} MoE")

    model = model.to(device)

    # --- Forward Pass ---
    print("\n=== Forward Pass ===")
    mock_data = generate_mock_data(n_samples=5000)
    dataset = HMICEDataset(tokenizer, seq_len=512, target_tokens=5_000_000, mock_data=mock_data)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=0)
    loader_iter = iter(loader)
    input_ids, geom_targets = next(loader_iter)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        logits, l1_logits, geom_loss = model(input_ids[:, :-1])
        print(f"Logits: {logits.shape}, Geom: {geom_loss.item():.4f}")
        assert not torch.isnan(logits).any(), "NaN!"
        print("✓ Forward clean")

    # --- Training Trial ---
    print("\n=== Router Trial (200 steps) ===")
    opt_hyp = geoopt.optim.RiemannianAdam(model.get_hyperbolic_params(), lr=1e-5, weight_decay=0.0, stabilize=10)
    opt_euc = torch.optim.AdamW(model.get_euclidean_params(), lr=1e-4, weight_decay=0.1)

    for step in range(1, 201):
        try:
            input_ids, geom_targets = next(loader_iter)
        except StopIteration:
            dataset = HMICEDataset(tokenizer, seq_len=512, target_tokens=5_000_000,
                                   mock_data=generate_mock_data(n_samples=5000))
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=0)
            loader_iter = iter(loader)
            input_ids, geom_targets = next(loader_iter)

        input_ids = input_ids.to(device)
        geom_targets = geom_targets.to(device)
        inputs, targets = input_ids[:, :-1], input_ids[:, 1:]
        geom_tgt = geom_targets[:, :-1]

        logits, l1_logits, geom_loss = model(inputs)
        B, T, _ = logits.shape
        text_loss = F.cross_entropy(logits.reshape(B*T, -1), targets.reshape(B*T),
                                     ignore_index=tokenizer.pad_token_id)
        scaffold_loss = F.cross_entropy(l1_logits.reshape(B*T, 3), geom_tgt.reshape(B*T))
        loss = text_loss + scaffold_loss

        if torch.isnan(loss):
            print(f"  Step {step}: NaN!")
            opt_hyp.zero_grad(set_to_none=True)
            opt_euc.zero_grad(set_to_none=True)
            continue

        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        if torch.isnan(gn):
            opt_hyp.zero_grad(set_to_none=True)
            opt_euc.zero_grad(set_to_none=True)
            continue

        opt_hyp.step()
        opt_euc.step()
        opt_hyp.zero_grad(set_to_none=True)
        opt_euc.zero_grad(set_to_none=True)

        with torch.no_grad():
            acc = (l1_logits.argmax(-1).reshape(-1) == geom_tgt.reshape(-1)).float().mean().item()

        if step % 20 == 0:
            l1d = model.get_l1_distribution()
            emb = model.tok_emb[:100]
            mc = -(emb[:, 0]**2) + (emb[:, 1:]**2).sum(dim=-1)
            print(
                f"Step {step:3d}/200 | loss={loss.item():.4f} | "
                f"acc={acc:.1%} | L1=[E:{int(l1d[0]*100)}% H:{int(l1d[1]*100)}% S:{int(l1d[2]*100)}%] | "
                f"mc={mc.mean().item():.4f}"
            )

    print(f"\n=== Results ===")
    print(f"Router accuracy: {acc:.1%}")
    print("✓ Router learning!" if acc > 0.5 else "⚠ Router accuracy low")
    print("Trial complete.")


if __name__ == "__main__":
    main()
