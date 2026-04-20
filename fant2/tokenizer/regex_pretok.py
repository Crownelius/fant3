"""
Regex pre-tokenizer for FANT 2 BPE.

This is the same Llama 3 / GPT-4 style regex split that handles:
  - Contractions ('s, 't, 're, ...)
  - Letters (Unicode-aware via \\p{L})
  - Digits (max 3 in a group, prevents the model from learning specific large numbers)
  - Punctuation
  - Whitespace (preserved)

The pattern is run once before BPE merges. It guarantees that BPE never crosses
a token-class boundary, which prevents the well-known "puzzling token" pathology
where BPE learns merges like "Apple is" or " 12345".

Reference:
    - https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
    - Llama 3 tokenizer.json (the same regex)
"""

# The Llama 3 / GPT-4 / cl100k_base regex (Unicode-aware, requires 'regex' package).
# Broken across lines for readability — joined verbatim by the helper below.
GPT4_PATTERN_PARTS = [
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",                   # English contractions
    r"[^\r\n\p{L}\p{N}]?+\p{L}+",                       # Letter run optionally preceded by 1 non-alphanum/non-digit
    r"\p{N}{1,3}",                                      # 1-3 digits
    r" ?[^\s\p{L}\p{N}]++[\r\n]*",                      # Punctuation, optionally preceded by 1 space, possibly trailing newlines
    r"\s*[\r\n]",                                       # Whitespace then a newline
    r"\s+(?!\S)",                                       # Trailing whitespace
    r"\s+",                                             # Whitespace
]

GPT4_REGEX_PATTERN = "|".join(GPT4_PATTERN_PARTS)


def get_pretok_pattern() -> str:
    """Return the compiled-ready regex pattern string."""
    return GPT4_REGEX_PATTERN


def split_for_bpe(text: str):
    """
    Run the pretokenizer over a text and yield each piece.

    Uses the `regex` package (NOT the stdlib `re`), because we need
    Unicode property classes \\p{L}, \\p{N}.
    """
    try:
        import regex
    except ImportError as e:
        raise ImportError(
            "FANT 2 tokenizer requires the `regex` package (not stdlib `re`) "
            "for Unicode property support. Install with: pip install regex"
        ) from e
    pattern = regex.compile(GPT4_REGEX_PATTERN)
    for m in pattern.finditer(text):
        yield m.group(0)
