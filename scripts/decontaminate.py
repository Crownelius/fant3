#!/usr/bin/env python3
"""
Benchmark decontamination for FANT 3 training.

Implements the standard 13-gram rule (DeepMind Gopher, Meta Llama):
  - Download benchmark test sets (GSM8K, MATH-500, MMLU).
  - Compute the SHA-1 hash of every sliding 13-word n-gram of every
    test question.
  - A training document is flagged as contaminated if ANY of its 13-grams
    hash-matches ANY benchmark n-gram.

The filter is used in TWO ways:
  1. As a report — run this script directly to scan the 6-source
     distillation mix and print a per-source contamination rate.
  2. As a filter — import `is_contaminated()` into the data pipeline and
     drop contaminated documents before tokenization.

Usage:
    python scripts/decontaminate.py                   # build cache + report
    python scripts/decontaminate.py --n-docs 5000     # report on 5K docs per source
    python scripts/decontaminate.py --rebuild-cache   # force re-download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable, Set

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

NGRAM_N = 13
CACHE_DIR = _HERE / "output" / "decontamination"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_PATH = CACHE_DIR / "ngram_hashes.json"


# -----------------------------------------------------------------------------
# N-gram hashing
# -----------------------------------------------------------------------------

_WORD_RE = re.compile(r"[A-Za-z0-9]+")

def _tokens(text: str) -> list[str]:
    """Lowercased word-like tokens. Deliberately aggressive normalisation so
    we catch surface variation (extra punctuation, quoting, etc.)."""
    return [m.group(0).lower() for m in _WORD_RE.finditer(text)]


def _ngrams(tokens: list[str], n: int = NGRAM_N) -> Iterable[str]:
    if len(tokens) < n:
        return
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def ngram_hashes_of(text: str, n: int = NGRAM_N) -> Set[str]:
    return {_hash(g) for g in _ngrams(_tokens(text), n)}


# -----------------------------------------------------------------------------
# Benchmark sources we decontaminate against
# -----------------------------------------------------------------------------

BENCHMARKS = [
    # (short_name, hf_id, config, split, question_field)
    ("gsm8k",    "gsm8k",           "main",   "test",       "question"),
    ("math500",  "HuggingFaceH4/MATH-500", None, "test",    "problem"),
    ("mmlu",     "cais/mmlu",       "all",    "test",       "question"),
]


def build_hash_cache(rebuild: bool = False) -> dict[str, Set[str]]:
    """Return {benchmark_name: set_of_ngram_hashes}."""
    if CACHE_PATH.exists() and not rebuild:
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {k: set(v) for k, v in raw.items()}

    from datasets import load_dataset

    cache: dict[str, Set[str]] = {}
    for name, hf_id, cfg, split, field in BENCHMARKS:
        print(f"  building hashes for {name} ({hf_id}) ...", flush=True)
        try:
            ds = load_dataset(hf_id, cfg, split=split) if cfg else load_dataset(hf_id, split=split)
        except Exception as e:
            print(f"    [skip] {hf_id}: {e}")
            continue
        hashes: Set[str] = set()
        n_q = 0
        for ex in ds:
            q = str(ex.get(field, "")).strip()
            if not q:
                continue
            hashes |= ngram_hashes_of(q)
            n_q += 1
        cache[name] = hashes
        print(f"    {n_q} questions -> {len(hashes)} unique {NGRAM_N}-grams")

    # Persist
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump({k: sorted(v) for k, v in cache.items()}, f)
    print(f"  cache saved to {CACHE_PATH}")
    return cache


# -----------------------------------------------------------------------------
# Public API: is_contaminated(text) -> bool
# -----------------------------------------------------------------------------

_GLOBAL_HASHES: Set[str] | None = None


def _load_global():
    global _GLOBAL_HASHES
    if _GLOBAL_HASHES is None:
        cache = build_hash_cache(rebuild=False)
        _GLOBAL_HASHES = set()
        for v in cache.values():
            _GLOBAL_HASHES |= v
    return _GLOBAL_HASHES


def is_contaminated(text: str) -> bool:
    """True if `text` contains any 13-gram that appears in any benchmark test
    question. Use as a filter in the training data pipeline."""
    hashes = _load_global()
    for g in _ngrams(_tokens(text)):
        if _hash(g) in hashes:
            return True
    return False


def contamination_details(text: str) -> list[tuple[str, str]]:
    """For reporting — return list of (matching_ngram, example) tuples. Slower
    than is_contaminated() but diagnostic."""
    cache = build_hash_cache(rebuild=False)
    out = []
    toks = _tokens(text)
    for g in _ngrams(toks):
        h = _hash(g)
        for name, hashes in cache.items():
            if h in hashes:
                out.append((name, g))
                break
    return out


# -----------------------------------------------------------------------------
# Report mode: scan the 6-source distillation mix
# -----------------------------------------------------------------------------

def _report(n_docs: int):
    """Stream a sample of each training source and count contaminated docs."""
    from datasets import load_dataset
    from fant2.data.formats import extract_text, DatasetFormat

    MIX = [
        ("HuggingFaceFW/fineweb-edu",                                "default",               "train", "text",          DatasetFormat.FLAT_TEXT),
        ("crownelius/Opus-4.6-Reasoning-3300x",                      None,                    "train", "problem",       DatasetFormat.PROBLEM_THINK_SOLUTION),
        ("ianncity/KIMI-K2.5-1000000x",                              "General-Distillation",  "train", "messages",      DatasetFormat.MESSAGES),
        ("AI-MO/NuminaMath-CoT",                                     None,                    "train", "problem",       DatasetFormat.PROBLEM_SOLUTION),
        ("mlabonne/FineTome-100k",                                   None,                    "train", "conversations", DatasetFormat.CONVERSATIONS),
        ("Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",       "stage1",                "train", "output",        DatasetFormat.INPUT_OUTPUT),
    ]

    # Warm the cache once
    _load_global()

    print()
    print(f"Scanning {n_docs} docs per source for {NGRAM_N}-gram matches against")
    print(f"GSM8K + MATH-500 + MMLU test questions.")
    print()
    print(f'{"source":<55} {"seen":>6} {"contam":>7} {"rate":>6}')
    print("-" * 82)

    for hf_id, cfg, split, text_key, fmt in MIX:
        try:
            ds = load_dataset(hf_id, cfg, split=split, streaming=True)
        except Exception as e:
            print(f"{hf_id:<55} [skip] {e}")
            continue
        seen = 0
        contam = 0
        kwargs = {"fmt": fmt, "text_key": text_key}
        if fmt == DatasetFormat.INPUT_OUTPUT:
            kwargs["output_key"] = text_key
            kwargs["input_key"] = "input"
        for ex in ds:
            if seen >= n_docs:
                break
            try:
                text = extract_text(ex, **kwargs)
            except Exception:
                continue
            if not text or len(text) < 16:
                continue
            seen += 1
            if is_contaminated(text):
                contam += 1
        rate = contam / max(seen, 1) * 100
        print(f"{hf_id:<55} {seen:>6} {contam:>7} {rate:>5.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=2000,
                    help="docs per source to scan (default 2000)")
    ap.add_argument("--rebuild-cache", action="store_true",
                    help="force re-download of benchmark test sets")
    ap.add_argument("--test", action="store_true",
                    help="quick self-test on a known-contaminated string")
    args = ap.parse_args()

    # Build / load cache
    print(f"Decontamination cache: {CACHE_PATH}")
    cache = build_hash_cache(rebuild=args.rebuild_cache)
    total = sum(len(v) for v in cache.values())
    print(f"  total unique {NGRAM_N}-gram hashes across benchmarks: {total}")

    if args.test:
        # GSM8K-style test
        probe = ("Natalia sold clips to 48 of her friends in April, and then she sold "
                 "half as many clips in May. How many clips did Natalia sell altogether "
                 "in April and May?")
        print(f"\nSelf-test probe: {probe[:60]}...")
        print(f"  contaminated: {is_contaminated(probe)}")
        details = contamination_details(probe)[:5]
        for name, g in details:
            print(f"    matched by {name}: '{g}'")
        return

    _report(args.n_docs)


if __name__ == "__main__":
    main()
