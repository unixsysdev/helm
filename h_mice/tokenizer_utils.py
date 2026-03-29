"""
H-MICE Tokenizer: Custom 8192-vocab BPE + Offset-Mapped Geometry Tagger

1. Train/load a 8192-token BPE tokenizer via HuggingFace `tokenizers`
2. Tag geometry via regex on raw string → project to token spans via offset_mapping
3. No heuristics on token_ids — everything is char-span based

Usage:
    from tokenizer_utils import get_tokenizer, tag_tokens

    tokenizer = get_tokenizer(train_texts=corpus)  # trains if not cached
    enc = tokenizer(text, return_offsets_mapping=True)
    tags = tag_tokens(text, enc.offset_mapping)
"""

import os
import re
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

TOKENIZER_DIR = os.path.join(os.path.dirname(__file__), "tokenizer_8k")
VOCAB_SIZE = 8192


# =============================================================================
# Regex patterns for geometry assignment
# =============================================================================

# Hyperbolic: hierarchical / tree structures (tag=2)
HYP_PATTERNS = [
    r'<think>.*?</think>',          # CoT reasoning blocks
    r'^\s{4,}\S.*$',                # Indented code (depth ≥ 1)
    r'\bdef\s+\w+',                 # Function definitions
    r'\bclass\s+\w+',              # Class definitions
    r'\bif\s+.+?:',                # Conditionals
    r'\bfor\s+.+?:',              # For loops
    r'\bwhile\s+.+?:',            # While loops
    r'\btry\s*:',                  # Try blocks
    r'\bexcept\s+',               # Except blocks
    r'\breturn\b',                 # Returns (implicit tree)
    r'\bimport\s+',               # Imports (dependency tree)
]

# Spherical: cyclical / periodic patterns (tag=1)
SPH_PATTERNS = [
    r'\d{4}[-/]\d{2}[-/]\d{2}',   # Dates (YYYY-MM-DD)
    r'\d{2}:\d{2}(:\d{2})?',      # Times (HH:MM:SS)
    r'\b(sin|cos|tan|arcsin|arccos|arctan)\b',  # Trig functions
    r'\b(pi|π)\b',                 # Pi
    r'\bmod\b|\b%\s',             # Modular arithmetic
    r'\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
    r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\b',
    r'\b\d{1,2}(st|nd|rd|th)\b',  # Ordinals (often cyclical context)
]


def _compile_patterns():
    """Pre-compile regex patterns for speed."""
    hyp = [re.compile(p, re.MULTILINE | re.DOTALL) for p in HYP_PATTERNS]
    sph = [re.compile(p, re.IGNORECASE) for p in SPH_PATTERNS]
    return hyp, sph


_HYP_RE, _SPH_RE = _compile_patterns()


# =============================================================================
# Geometry Tagger (offset-mapped projection)
# =============================================================================

def tag_tokens(text: str, offset_mapping: list) -> list:
    """
    Assign geometry tags to tokens via char-span projection.

    Step A: offset_mapping from tokenizer (list of (start_char, end_char))
    Step B: Regex on raw string → char spans
    Step C: Project char spans to token tags

    Returns: list of ints, same length as offset_mapping
        0 = Euclidean (default)
        1 = Spherical
        2 = Hyperbolic
    """
    n_tokens = len(offset_mapping)
    tags = [0] * n_tokens

    # Collect hyperbolic char spans
    hyp_spans = []
    for pat in _HYP_RE:
        for m in pat.finditer(text):
            hyp_spans.append((m.start(), m.end()))

    # Collect spherical char spans
    sph_spans = []
    for pat in _SPH_RE:
        for m in pat.finditer(text):
            sph_spans.append((m.start(), m.end()))

    # Project: if token span overlaps a regex span, assign tag
    for i, (tok_start, tok_end) in enumerate(offset_mapping):
        if tok_start == tok_end:  # Special/padding tokens
            continue

        # Check hyperbolic first (higher priority)
        for s, e in hyp_spans:
            if tok_start < e and tok_end > s:  # Overlap
                tags[i] = 2
                break
        else:
            # Check spherical
            for s, e in sph_spans:
                if tok_start < e and tok_end > s:  # Overlap
                    tags[i] = 1
                    break

    return tags


# =============================================================================
# BPE Tokenizer Training / Loading
# =============================================================================

def train_tokenizer(texts: list, save_dir: str = TOKENIZER_DIR):
    """
    Train a BPE tokenizer with vocab_size=8192 from a corpus of strings.
    Saves to disk for reuse.
    """
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        min_frequency=2,
        show_progress=True,
    )

    # Train from iterator
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Save raw tokenizer
    tokenizer.save(os.path.join(save_dir, "tokenizer.json"))

    # Wrap as HuggingFace PreTrainedTokenizerFast
    fast_tok = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
    )
    fast_tok.save_pretrained(save_dir)
    print(f"  Tokenizer saved to {save_dir} (vocab={fast_tok.vocab_size})")
    return fast_tok


def load_tokenizer(save_dir: str = TOKENIZER_DIR):
    """Load a previously trained tokenizer."""
    tok = PreTrainedTokenizerFast.from_pretrained(save_dir)
    return tok


def get_tokenizer(train_texts: list = None, force_retrain: bool = False):
    """
    Get the 8192-BPE tokenizer. Trains from train_texts if not cached.
    """
    tok_path = os.path.join(TOKENIZER_DIR, "tokenizer.json")

    if os.path.exists(tok_path) and not force_retrain:
        print(f"  Loading cached tokenizer from {TOKENIZER_DIR}")
        return load_tokenizer()

    if train_texts is None:
        raise ValueError("No cached tokenizer found and no train_texts provided")

    print(f"  Training 8192-BPE tokenizer on {len(train_texts)} texts...")
    return train_tokenizer(train_texts)
