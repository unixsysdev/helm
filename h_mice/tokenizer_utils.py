"""
H-MICE Tokenizer: 32K Baseline + Offset-Mapped Geometry Tagger

1. Use a standard 32K tokenizer (Qwen2.5-0.5B) via HuggingFace
2. Tag geometry via regex on raw string → project to token spans via offset_mapping
3. No heuristics on token_ids — everything is char-span based

Usage:
    from tokenizer_utils import get_tokenizer, tag_tokens

    tokenizer = get_tokenizer()
    enc = tokenizer(text, return_offsets_mapping=True)
    tags = tag_tokens(text, enc['offset_mapping'])
"""

import re
from transformers import AutoTokenizer

TOKENIZER_NAME = "mistralai/Mistral-7B-v0.1"
VOCAB_SIZE = 32000  # Strict 32K diet


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
# Tokenizer Loading
# =============================================================================

_CACHED_TOKENIZER = None


def get_tokenizer(**kwargs):
    """
    Get the 32K baseline tokenizer (Qwen2.5).
    Supports return_offsets_mapping for geometry tagging.
    """
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is not None:
        return _CACHED_TOKENIZER

    print(f"  Loading tokenizer: {TOKENIZER_NAME}")
    tok = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    _CACHED_TOKENIZER = tok
    print(f"  Vocab: {tok.vocab_size}")
    return tok
