"""
H-MICE Data Pipeline: Streaming dataset with offset-mapped geometry tagging.

Geometry tags assigned via char-span projection:
  1. Tokenize with return_offsets_mapping=True
  2. Regex on raw string → char spans
  3. Project spans to tokens via offset_mapping

Tags:
  0 = Euclidean (standard text)
  1 = Spherical (cyclical: dates, trig, modular arithmetic)
  2 = Hyperbolic (hierarchical: <think>, indented code, def/class)
"""

import re
import random
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets
from tokenizer_utils import tag_tokens


# =============================================================================
# Mock Data Generator
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
# Streaming Dataset with Offset-Mapped Geometry Tagging
# =============================================================================

class HMICEDataset(IterableDataset):
    """
    Streaming dataset yielding (input_ids, geom_targets) chunks.

    - 60/20/20 mix of CoT / Code / Text
    - Offset-mapped geometry tagging via regex char-span projection
    - 512-chunk shuffle buffer
    """

    def __init__(self, tokenizer, seq_len: int = 4096, target_tokens: int = 8_400_000_000,
                 mock_data=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.target_tokens = target_tokens
        self.eos_id = tokenizer.eos_token_id or 2
        self.pad_id = tokenizer.pad_token_id or 0
        self.mock_data = mock_data

    def _open_streams(self):
        if self.mock_data is not None:
            return iter(self.mock_data)

        cot = load_dataset("open-thoughts/OpenThoughts-114k", split="train", streaming=True)
        code = load_dataset("HuggingFaceTB/smol-smoltalk", split="train", streaming=True)
        text = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)

        mixed = interleave_datasets(
            [cot, code, text],
            probabilities=[0.60, 0.20, 0.20],
            seed=42,
            stopping_strategy="all_exhausted",
        )
        return mixed

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
        """
        Tokenize with offset mapping and apply geometry tagging.
        Returns (token_ids, geometry_tags).
        """
        enc = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=16384,
        )
        ids = enc['input_ids']
        offsets = enc['offset_mapping']
        tags = tag_tokens(text, offsets)
        return ids, tags

    def __iter__(self):
        import random as _rng
        SHUFFLE_BUFFER = 512

        mixed = self._open_streams()
        token_buffer = []
        tag_buffer = []
        chunk_buffer = []
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
            tag_buffer.append(0)  # EOS is Euclidean
            total_tokens += len(ids) + 1

            while len(token_buffer) >= self.seq_len:
                chunk_ids = token_buffer[:self.seq_len]
                chunk_tags = tag_buffer[:self.seq_len]
                token_buffer = token_buffer[self.seq_len:]
                tag_buffer = tag_buffer[self.seq_len:]
                chunk_buffer.append((chunk_ids, chunk_tags))

            if len(chunk_buffer) >= SHUFFLE_BUFFER:
                _rng.shuffle(chunk_buffer)
                for ids_chunk, tags_chunk in chunk_buffer:
                    yield {
                        'input_ids': torch.tensor(ids_chunk, dtype=torch.long),
                        'geom_targets': torch.tensor(tags_chunk, dtype=torch.long),
                    }
                chunk_buffer = []

        if chunk_buffer:
            _rng.shuffle(chunk_buffer)
            for ids_chunk, tags_chunk in chunk_buffer:
                yield {
                    'input_ids': torch.tensor(ids_chunk, dtype=torch.long),
                    'geom_targets': torch.tensor(tags_chunk, dtype=torch.long),
                }
