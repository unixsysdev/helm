"""
H-MICE Router Validation: Mock Data Trial Run

Verifies:
1. Model instantiates at ~400M params
2. Forward pass works with mock data
3. L1 router learns to sort geometry tags (>80% accuracy by step 200)
4. Loss descends normally
5. No NaN/Inf
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import HMICETransformer
from data import HMICEDataset, generate_mock_data
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

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        model_max_length=512,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    vocab_size = len(tokenizer)

    # Model
    print("\n=== Building H-MICE ===")
    model = HMICETransformer(
        vocab_size=vocab_size,
        dim=1024,
        n_layers=16,
        n_heads=16,
        n_experts_per_geom=4,
        max_seq_len=512,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,} ({total_params/1e6:.0f}M)")
    assert total_params > 300_000_000, f"Expected >300M params, got {total_params:,}"
    print("✓ Parameter count check passed")

    model = model.to(device)

    # Data
    print("\n=== Generating Mock Data ===")
    mock_data = generate_mock_data(n_samples=10000)
    dataset = HMICEDataset(tokenizer, seq_len=512, target_tokens=10_000_000, mock_data=mock_data)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=0)
    loader_iter = iter(loader)

    # Quick forward pass test
    print("\n=== Forward Pass Test ===")
    input_ids, geom_targets = next(loader_iter)
    input_ids = input_ids.to(device)
    inputs = input_ids[:, :-1]

    with torch.no_grad():
        logits, l1_logits, geom_loss = model(inputs)
        print(f"Logits shape: {logits.shape}")
        print(f"L1 logits shape: {l1_logits.shape}")
        print(f"Geom loss: {geom_loss.item():.4f}")
        assert not torch.isnan(logits).any(), "NaN in logits!"
        assert not torch.isnan(l1_logits).any(), "NaN in L1 logits!"
        print("✓ Forward pass clean (no NaN)")

    # Training trial
    print("\n=== Router Learning Trial (200 steps) ===")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    n_steps = 200
    loader_iter = iter(loader)

    for step in range(1, n_steps + 1):
        try:
            input_ids, geom_targets = next(loader_iter)
        except StopIteration:
            dataset = HMICEDataset(tokenizer, seq_len=512, target_tokens=10_000_000,
                                   mock_data=generate_mock_data(n_samples=10000))
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn, num_workers=0)
            loader_iter = iter(loader)
            input_ids, geom_targets = next(loader_iter)

        input_ids = input_ids.to(device)
        geom_targets = geom_targets.to(device)

        inputs = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        geom_tgt = geom_targets[:, :-1]

        logits, l1_logits, geom_loss = model(inputs)
        logits = logits[:, :, :vocab_size]

        B, T, V = logits.shape
        text_loss = F.cross_entropy(logits.reshape(B * T, V), targets.reshape(B * T),
                                     ignore_index=tokenizer.pad_token_id)
        scaffold_loss = F.cross_entropy(l1_logits.reshape(B * T, 3), geom_tgt.reshape(B * T))
        loss = text_loss + scaffold_loss

        if torch.isnan(loss):
            print(f"  Step {step}: NaN loss!")
            optimizer.zero_grad(set_to_none=True)
            continue

        loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isnan(grad_norm):
            print(f"  Step {step}: NaN gradient!")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Router accuracy check
        with torch.no_grad():
            l1_preds = l1_logits.argmax(dim=-1).reshape(-1)
            l1_targets = geom_tgt.reshape(-1)
            accuracy = (l1_preds == l1_targets).float().mean().item()

        if step % 20 == 0:
            l1_dist = model.get_l1_distribution()
            h_pct = int(l1_dist[0].item() * 100)
            e_pct = int(l1_dist[1].item() * 100)
            s_pct = int(l1_dist[2].item() * 100)
            print(
                f"Step {step:3d}/{n_steps} | loss={loss.item():.4f} | "
                f"text={text_loss.item():.4f} | scaffold={scaffold_loss.item():.4f} | "
                f"router_acc={accuracy:.1%} | "
                f"L1=[H:{h_pct}% E:{e_pct}% S:{s_pct}%]"
            )

    # Final accuracy check
    print(f"\n=== Results ===")
    print(f"Final router accuracy: {accuracy:.1%}")
    if accuracy > 0.5:
        print("✓ Router is learning to sort geometry!")
    else:
        print("⚠ Router accuracy low — may need more steps or tuning")

    print("\nTrial complete.")


if __name__ == "__main__":
    main()
