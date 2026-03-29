"""
H-MICE Data Pipeline

Two modes:
  1. ChunkStreamDataset: IterableDataset that polls for binary chunks from the
     producer (prepare_data_stream.py), loads via memmap, yields 4096-token sequences.
     When a chunk is exhausted, advances to the next. Waits if chunk not ready.

  2. HMICEDataset: In-memory mock dataset (for test_routing.py validation).
     Tokenizes + tags on the fly from mock_data.

Geometry tags (offset-mapped projection):
  0 = Euclidean (standard text)
  1 = Spherical (cyclical: dates, trig, modular arithmetic)
  2 = Hyperbolic (hierarchical: <think>, indented code, def/class)
"""

import os
import time
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from tokenizer_utils import tag_tokens


# =============================================================================
# Mock Data Generator (for validation)
# =============================================================================

def generate_mock_data(n_samples: int = 10000, seed: int = 42):
    """Generate synthetic data with clear geometric signals for router validation."""
    rng = random.Random(seed)
    samples = []

    hyp_templates = [
        "<think>\nLet me work through this step by step.\n"
        "First, we need to consider the base case.\n"
        "    If n == 0, return 1\n"
        "    If n == 1, return 1\n"
        "Then for the recursive case:\n"
        "    result = fibonacci(n-1) + fibonacci(n-2)\n"
        "    return result\n"
        "This gives us the answer.\n</think>",

        "def binary_search(arr, target):\n"
        "    left, right = 0, len(arr) - 1\n"
        "    while left <= right:\n"
        "        mid = (left + right) // 2\n"
        "        if arr[mid] == target:\n"
        "            return mid\n"
        "        elif arr[mid] < target:\n"
        "            left = mid + 1\n"
        "        else:\n"
        "            right = mid - 1\n"
        "    return -1\n",

        "<think>\nTo solve this polynomial equation:\n"
        "    ax^2 + bx + c = 0\n"
        "We apply the quadratic formula:\n"
        "    x = (-b ± sqrt(b^2 - 4ac)) / (2a)\n"
        "    discriminant = b^2 - 4ac\n"
        "    if discriminant > 0:\n"
        "        two real roots\n"
        "    elif discriminant == 0:\n"
        "        one repeated root\n"
        "    else:\n"
        "        complex roots\n</think>",
    ]

    sph_templates = [
        "The current datetime is 2024-01-15 14:30:00 UTC.\n"
        "Converting timezone offset. The day is Monday.\n"
        "Next rotation on Tuesday at cos(2*pi*t/T).\n",

        "import math\n"
        "angle = math.radians(45)\n"
        "x = math.cos(angle)\n"
        "y = math.sin(angle)\n"
        "rotation_matrix = [[cos(theta), -sin(theta)],\n"
        "                    [sin(theta), cos(theta)]]\n",

        "Schedule: Monday through Friday.\n"
        "The cycle repeats every 7 days.\n"
        "January, February, March — quarterly rotation.\n"
        "Frequency: 60Hz, period = 1/frequency seconds.\n",
    ]

    euc_templates = [
        "The weather today is partly cloudy with a high of 72 degrees.\n"
        "Rain is expected later in the evening.\n"
        "Remember to bring an umbrella if you go outside.\n",

        "The quick brown fox jumps over the lazy dog.\n"
        "This sentence contains every letter of the alphabet.\n"
        "It is commonly used for font display testing.\n",

        "Machine learning models require large amounts of data.\n"
        "The training process involves iterative optimization.\n"
        "Neural networks learn hierarchical representations.\n",
    ]

    for _ in range(n_samples):
        r = rng.random()
        if r < 0.4:
            text = rng.choice(hyp_templates)
        elif r < 0.6:
            text = rng.choice(sph_templates)
        else:
            text = rng.choice(euc_templates)
        samples.append({'text': text})

    return samples


# =============================================================================
# Chunk Stream Dataset (Producer-Consumer IterableDataset)
# =============================================================================

class ChunkStreamDataset(IterableDataset):
    """
    Polls for pre-tokenized binary chunks written by prepare_data_stream.py.

    Each chunk pair (atomic rename from .tmp → .bin):
      tokens_chunk_{i}.bin  (int32, flat 1-D array)
      geom_chunk_{i}.bin    (int8, flat 1-D array)

    Yields (input_ids, geom_targets) tensors of shape [seq_len].
    When current chunk is exhausted, advances. If next chunk is not ready, waits.
    """

    def __init__(self, data_dir: str, seq_len: int = 4096, start_chunk: int = 0):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.start_chunk = start_chunk

    def _chunk_exists(self, idx):
        return os.path.exists(
            os.path.join(self.data_dir, f"tokens_chunk_{idx}.bin"))

    def _producer_done(self):
        return os.path.exists(os.path.join(self.data_dir, "DONE"))

    def __iter__(self):
        chunk_idx = self.start_chunk

        while True:
            tok_path = os.path.join(self.data_dir, f"tokens_chunk_{chunk_idx}.bin")
            geo_path = os.path.join(self.data_dir, f"geom_chunk_{chunk_idx}.bin")

            # --- Poll until chunk appears or producer signals DONE ---
            while not os.path.exists(tok_path):
                if self._producer_done():
                    return  # No more chunks coming
                time.sleep(1)

            # --- Memmap read (zero-copy from NVMe) ---
            tokens = np.memmap(tok_path, dtype=np.int32, mode='r')
            geom = np.memmap(geo_path, dtype=np.int8, mode='r')

            num_seqs = len(tokens) // self.seq_len

            # --- Yield sequences from this chunk ---
            for i in range(num_seqs):
                start = i * self.seq_len
                end = start + self.seq_len
                yield {
                    'input_ids': torch.from_numpy(
                        np.array(tokens[start:end]).astype(np.int64)),
                    'geom_targets': torch.from_numpy(
                        np.array(geom[start:end]).astype(np.int64)),
                }

            # --- Free memmap and advance ---
            del tokens, geom
            chunk_idx += 1


# =============================================================================
# In-Memory Dataset (for test_routing.py / mock validation)
# =============================================================================

class HMICEDataset(IterableDataset):
    """
    In-memory streaming dataset for validation.
    Tokenizes + tags on the fly from mock_data.
    """

    def __init__(self, tokenizer, seq_len: int = 4096, target_tokens: int = 8_400_000_000,
                 mock_data=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.target_tokens = target_tokens
        self.eos_id = tokenizer.eos_token_id or 2
        self.pad_id = tokenizer.pad_token_id or 0
        self.mock_data = mock_data

    def _extract_text(self, example):
        if 'conversations' in example and example['conversations']:
            parts = []
            for turn in example['conversations']:
                role = turn.get('role', turn.get('from', ''))
                content = turn.get('content', turn.get('value', ''))
                if role and content:
                    parts.append(f"<|{role}|>\n{content}")
            return '\n'.join(parts)
        for key in ['text', 'content', 'code', 'document']:
            if key in example and example[key]:
                return str(example[key])
        return ''

    def _tokenize_and_tag(self, text: str):
        enc = self.tokenizer(text, add_special_tokens=False,
                              return_offsets_mapping=True, truncation=True, max_length=16384)
        ids = enc['input_ids']
        tags = tag_tokens(text, enc['offset_mapping'])
        return ids, tags

    def __iter__(self):
        import random as _rng
        SHUFFLE_BUFFER = 512
        mixed = iter(self.mock_data) if self.mock_data else iter([])
        token_buffer, tag_buffer, chunk_buffer = [], [], []
        total_tokens = 0

        for example in mixed:
            if total_tokens >= self.target_tokens:
                break
            text = self._extract_text(example)
            if len(text) < 50:
                continue
            ids, tags = self._tokenize_and_tag(text)
            token_buffer.extend(ids)
            tag_buffer.extend(tags)
            token_buffer.append(self.eos_id)
            tag_buffer.append(0)
            total_tokens += len(ids) + 1

            while len(token_buffer) >= self.seq_len:
                chunk_buffer.append((token_buffer[:self.seq_len], tag_buffer[:self.seq_len]))
                token_buffer = token_buffer[self.seq_len:]
                tag_buffer = tag_buffer[self.seq_len:]

            if len(chunk_buffer) >= SHUFFLE_BUFFER:
                _rng.shuffle(chunk_buffer)
                for ids_c, tags_c in chunk_buffer:
                    yield {
                        'input_ids': torch.tensor(ids_c, dtype=torch.long),
                        'geom_targets': torch.tensor(tags_c, dtype=torch.long),
                    }
                chunk_buffer = []

        if chunk_buffer:
            _rng.shuffle(chunk_buffer)
            for ids_c, tags_c in chunk_buffer:
                yield {
                    'input_ids': torch.tensor(ids_c, dtype=torch.long),
                    'geom_targets': torch.tensor(tags_c, dtype=torch.long),
                }
