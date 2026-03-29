"""
H-MICE Data Pipeline: Streaming dataset with geometric metadata heuristics.

Yields (input_ids, labels, geom_targets) where geom_targets is a per-token
geometry tag:
  0 = Euclidean (standard text)
  1 = Spherical (cyclical patterns: %, datetime, sin, cos)
  2 = Hyperbolic (hierarchical: <think> tags, indented code)
"""

import re
import random
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, interleave_datasets


# =============================================================================
# Geometry Heuristic Tagger
# =============================================================================

# Regex patterns for spherical detection
SPHERICAL_PATTERNS = re.compile(
    r'\b(sin|cos|tan|atan|asin|acos|datetime|timedelta|strftime|'
    r'modulo|periodic|cycle|frequency|rotation|angle|degree|radian|'
    r'Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|'
    r'January|February|March|April|May|June|July|August|September|'
    r'October|November|December)\b|%[dfsYmHMS]',
    re.IGNORECASE
)

# Patterns for hyperbolic detection
HYPERBOLIC_INDENT = re.compile(r'^(\s{4,}|\t+)')  # 4+ spaces or tabs
THINK_OPEN = re.compile(r'<think>', re.IGNORECASE)
THINK_CLOSE = re.compile(r'</think>', re.IGNORECASE)


def tag_text_geometry(text: str) -> list:
    """
    Tag each line of text with a geometry heuristic.
    Returns list of (line, tag) tuples.

    Tag 0: Euclidean (default)
    Tag 1: Spherical (cyclical patterns)
    Tag 2: Hyperbolic (hierarchical/nested)
    """
    lines = text.split('\n')
    tags = []
    in_think = False

    for line in lines:
        if THINK_OPEN.search(line):
            in_think = True

        if in_think:
            tags.append((line, 2))  # Hyperbolic
        elif HYPERBOLIC_INDENT.match(line):
            # Indented code (def, class, nested logic)
            tags.append((line, 2))  # Hyperbolic
        elif SPHERICAL_PATTERNS.search(line):
            tags.append((line, 1))  # Spherical
        else:
            tags.append((line, 0))  # Euclidean

        if THINK_CLOSE.search(line):
            in_think = False

    return tags


def tag_tokens_geometry(text: str, tokenizer, token_ids: list) -> list:
    """
    Assign geometry tags to token IDs based on text heuristics.
    Maps line-level tags to token-level tags using character offsets.
    """
    tagged_lines = tag_text_geometry(text)

    # Build character-to-tag mapping
    char_tags = []
    for line, tag in tagged_lines:
        char_tags.extend([tag] * (len(line) + 1))  # +1 for newline

    # For each token, find its approximate character position
    # Simple heuristic: distribute tags proportionally across tokens
    n_tokens = len(token_ids)
    n_chars = len(char_tags)

    if n_chars == 0 or n_tokens == 0:
        return [0] * n_tokens

    token_tags = []
    for i in range(n_tokens):
        char_pos = int(i * n_chars / n_tokens)
        char_pos = min(char_pos, n_chars - 1)
        token_tags.append(char_tags[char_pos])

    return token_tags


# =============================================================================
# Mock Data Generator (for trial runs)
# =============================================================================

def generate_mock_data(n_samples: int = 10000, seed: int = 42):
    """Generate synthetic data with clear geometric signals for router validation."""
    rng = random.Random(seed)
    samples = []

    # Templates for each geometry
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
        "Converting to strftime format: %Y-%m-%d %H:%M:%S\n"
        "The day of the week is Monday.\n"
        "Next rotation occurs on Tuesday at cos(2*pi*t/T).\n",

        "import math\n"
        "angle = math.radians(45)\n"
        "x = math.cos(angle)\n"
        "y = math.sin(angle)\n"
        "rotation_matrix = [[cos(theta), -sin(theta)],\n"
        "                    [sin(theta), cos(theta)]]\n",

        "Schedule: Monday through Friday, 9am to 5pm.\n"
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
# Streaming Dataset with Geometric Metadata
# =============================================================================

class HMICEDataset(IterableDataset):
    """
    Streaming dataset that yields (input_ids, geom_targets) chunks.

    - 60/20/20 mix of CoT / Code / Text
    - Heuristic geometry tagging per token
    - 512-chunk shuffle buffer
    """

    def __init__(self, tokenizer, seq_len: int = 4096, target_tokens: int = 8_400_000_000,
                 mock_data=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.target_tokens = target_tokens
        self.eos_id = tokenizer.eos_token_id or 2
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

            text_trunc = text[:16384]
            ids = self.tokenizer.encode(text_trunc, add_special_tokens=False)

            # Get per-token geometry tags
            tags = tag_tokens_geometry(text_trunc, self.tokenizer, ids)

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
