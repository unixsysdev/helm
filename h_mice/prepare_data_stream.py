"""
H-MICE Streaming Data Producer

Streams from HuggingFace datasets, tokenizes with 32K Mistral tokenizer,
applies offset-mapped geometry tagging, and writes sequential binary chunks
to disk for the training consumer.

Usage:
    python prepare_data_stream.py --out_dir /data/h_mice_chunks --chunk_tokens 100000000

Each chunk produces:
    tokens_chunk_0.bin   (int32 memmap, shape=[chunk_tokens])
    geom_chunk_0.bin     (int8 memmap, shape=[chunk_tokens])
    chunk_0.ready        (empty sentinel file)

The .ready sentinel is written AFTER the data files are fully flushed,
so the consumer can safely memmap without partial reads.
"""

import argparse
import os
import time
import numpy as np
from tokenizer_utils import get_tokenizer, tag_tokens


def stream_dataset(split_ratios=None):
    """
    Stream the 60/20/20 mix: CoT reasoning / Code / General text.
    Yields raw text strings indefinitely.
    """
    from datasets import load_dataset, interleave_datasets

    if split_ratios is None:
        split_ratios = [0.6, 0.2, 0.2]

    print("  Initializing streaming datasets...")

    # CoT / Deep Reasoning (60%) — native multi-step traces
    cot = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)
    def _format_cot(x):
        parts = []
        for turn in (x.get("conversations") or []):
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))
            if role == "human":
                parts.append(content)
            elif role in ("gpt", "assistant") and content:
                parts.append(f"<think>\n{content}\n</think>")
        return {"text": "\n".join(parts)}
    cot = cot.map(_format_cot)

    # Python Code (20%) — educational Python from SmolLM-Corpus
    code = load_dataset("HuggingFaceTB/smollm-corpus", "python-edu",
                         split="train", streaming=True)

    # General Text (20%) — knowledge from SmolLM-Corpus
    text = load_dataset("HuggingFaceTB/smollm-corpus", "cosmopedia-v2",
                         split="train", streaming=True)

    combined = interleave_datasets(
        [cot, code, text],
        probabilities=split_ratios,
        stopping_strategy="all_exhausted",
    )

    for sample in combined:
        t = sample.get("text", "")
        if t and len(t) > 50:
            yield t


def stream_mock_data():
    """Fall back to mock data for testing."""
    from data import generate_mock_data
    mock = generate_mock_data(n_samples=100000)
    while True:
        for sample in mock:
            yield sample["text"]


def produce_chunks(out_dir: str, chunk_tokens: int, seq_len: int = 4096,
                   max_chunks: int = -1, mock: bool = False):
    """
    Main producer loop. Tokenizes + tags streaming text, packs into
    fixed-length sequences, and writes binary chunks to disk.
    """
    os.makedirs(out_dir, exist_ok=True)
    tokenizer = get_tokenizer()
    pad_id = tokenizer.pad_token_id

    print(f"\n=== Producer Config ===")
    print(f"  out_dir:      {out_dir}")
    print(f"  chunk_tokens: {chunk_tokens:,}")
    print(f"  seq_len:      {seq_len}")
    print(f"  max_chunks:   {'∞' if max_chunks < 0 else max_chunks}")

    # Allocate buffers
    tok_buf = np.zeros(chunk_tokens, dtype=np.int32)
    geo_buf = np.zeros(chunk_tokens, dtype=np.int8)
    buf_pos = 0
    chunk_idx = 0
    total_tokens = 0
    t_start = time.time()

    # Skip already-produced chunks
    while os.path.exists(os.path.join(out_dir, f"tokens_chunk_{chunk_idx}.bin")):
        chunk_idx += 1
        total_tokens += chunk_tokens
    if chunk_idx > 0:
        print(f"  Resuming from chunk {chunk_idx} ({total_tokens:,} tokens already on disk)")

    text_stream = stream_mock_data() if mock else stream_dataset()

    for text in text_stream:
        if 0 < max_chunks <= chunk_idx:
            break

        # Tokenize with offset mapping
        enc = tokenizer(
            text,
            truncation=True,
            max_length=seq_len,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        ids = enc["input_ids"]
        tags = tag_tokens(text, enc["offset_mapping"])

        n = len(ids)
        if n == 0:
            continue

        # Fill buffer
        space = chunk_tokens - buf_pos
        take = min(n, space)
        tok_buf[buf_pos:buf_pos + take] = ids[:take]
        geo_buf[buf_pos:buf_pos + take] = tags[:take]
        buf_pos += take

        # Chunk full → flush to disk
        if buf_pos >= chunk_tokens:
            _flush_chunk(out_dir, chunk_idx, tok_buf, geo_buf, chunk_tokens)
            elapsed = time.time() - t_start
            total_tokens += chunk_tokens
            rate = total_tokens / elapsed if elapsed > 0 else 0
            print(f"  Chunk {chunk_idx}: {chunk_tokens:,} tokens "
                  f"({total_tokens:,} total, {rate:,.0f} tok/s)")

            chunk_idx += 1
            buf_pos = 0
            tok_buf[:] = 0
            geo_buf[:] = 0

            # Leftover from this text
            leftover = n - take
            if leftover > 0:
                tok_buf[:leftover] = ids[take:]
                geo_buf[:leftover] = tags[take:]
                buf_pos = leftover

    # Flush partial final chunk
    if buf_pos > 0:
        # Pad remainder
        tok_buf[buf_pos:] = pad_id
        geo_buf[buf_pos:] = 0
        _flush_chunk(out_dir, chunk_idx, tok_buf, geo_buf, chunk_tokens)
        total_tokens += buf_pos
        print(f"  Chunk {chunk_idx} (final, partial): {buf_pos:,} tokens")
        chunk_idx += 1

    elapsed = time.time() - t_start
    # Signal consumer: no more chunks coming
    done_path = os.path.join(out_dir, "DONE")
    with open(done_path, 'w') as f:
        f.write(f"{chunk_idx}\n")
    print(f"\n=== Producer Done ===")
    print(f"  {chunk_idx} chunks, {total_tokens:,} tokens in {elapsed:.1f}s")


def _flush_chunk(out_dir, idx, tok_buf, geo_buf, chunk_tokens):
    """Write token + geom arrays to disk atomically via tmp→rename."""
    tok_path = os.path.join(out_dir, f"tokens_chunk_{idx}.bin")
    geo_path = os.path.join(out_dir, f"geom_chunk_{idx}.bin")
    tok_tmp = tok_path + ".tmp"
    geo_tmp = geo_path + ".tmp"

    # Write to .tmp files
    tok_buf.tofile(tok_tmp)
    geo_buf.tofile(geo_tmp)

    # Sync to disk
    with open(tok_tmp, 'rb') as f:
        os.fsync(f.fileno())
    with open(geo_tmp, 'rb') as f:
        os.fsync(f.fileno())

    # Atomic rename: consumer sees file appear all at once
    os.rename(tok_tmp, tok_path)
    os.rename(geo_tmp, geo_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H-MICE Streaming Data Producer")
    parser.add_argument("--out_dir", type=str, default="/data/h_mice_chunks")
    parser.add_argument("--chunk_tokens", type=int, default=100_000_000,
                        help="Tokens per chunk (default: 100M)")
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--max_chunks", type=int, default=-1,
                        help="Stop after N chunks (-1 = infinite)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock data instead of streaming")
    args = parser.parse_args()

    produce_chunks(
        out_dir=args.out_dir,
        chunk_tokens=args.chunk_tokens,
        seq_len=args.seq_len,
        max_chunks=args.max_chunks,
        mock=args.mock,
    )
