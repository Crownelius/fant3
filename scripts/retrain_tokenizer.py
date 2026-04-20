#!/usr/bin/env python3
"""
Retrain the FANT 2 BPE tokenizer on the current distillation mix.

Fix 1b of the 5-fix campaign (2026-04-19). The old tokenizer was fit only on
FineWeb-Edu (5000 docs) — it badly oversubscribes on JSON, math LaTeX, and the
special ChatML tokens. Retrain on a representative sample of the real training
mix so:

  * Numerals and math punctuation get shorter codes
  * Opus 4.6 reasoning vocabulary gets first-class merges
  * The `<|answer|>` / `<|/answer|>` tokens get their IDs reserved

Usage:
    python scripts/retrain_tokenizer.py \
        --n-docs 100000 \
        --out output/tokenizer/tokenizer_v2.json

Datasets sampled (weights are approximate, trimmed to --n-docs total):
    40%  FineWeb-Edu           (base prose distribution)
    20%  Crownelius Opus 4.6   (our distillation target distribution)
    15%  Kimi K2.5             (long reasoning traces)
    10%  NuminaMath CoT        (math LaTeX)
    10%  FineTome 100K         (high-quality SFT)
     5%  Superior Reasoning    (input/output reasoning pairs)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Run from the project root
_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from fant2.data.formats import extract_text, DatasetFormat  # noqa: E402
from fant2.tokenizer.bpe import FANT2Tokenizer            # noqa: E402


MIX = [
    # (hf_id, config, split, text_key, format, weight, input_key_or_None)
    ("HuggingFaceFW/fineweb-edu",                                "default",               "train", "text",          DatasetFormat.FLAT_TEXT,              0.40, None),
    ("crownelius/Opus-4.6-Reasoning-3300x",                      None,                    "train", "problem",       DatasetFormat.PROBLEM_THINK_SOLUTION, 0.20, None),
    ("ianncity/KIMI-K2.5-1000000x",                              "General-Distillation",  "train", "messages",      DatasetFormat.MESSAGES,               0.15, None),
    ("AI-MO/NuminaMath-CoT",                                     None,                    "train", "problem",       DatasetFormat.PROBLEM_SOLUTION,       0.10, None),
    ("mlabonne/FineTome-100k",                                   None,                    "train", "conversations", DatasetFormat.CONVERSATIONS,          0.10, None),
    ("Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",       "stage1",                "train", "output",        DatasetFormat.INPUT_OUTPUT,           0.05, "input"),
]


def stream_docs(n_total: int, verbose: bool = True):
    """Yield `n_total` plain-text docs according to the MIX weights."""
    from datasets import load_dataset  # import lazily

    total_budget = n_total
    for hf_id, cfg, split, text_key, fmt, w, input_key in MIX:
        budget = int(n_total * w)
        if budget <= 0:
            continue
        try:
            ds = load_dataset(hf_id, cfg, split=split, streaming=True)
        except Exception as e:
            if verbose:
                print(f"  [skip] {hf_id}: {e}")
            continue

        kwargs = {"fmt": fmt, "text_key": text_key}
        if fmt == DatasetFormat.INPUT_OUTPUT:
            kwargs["output_key"] = text_key      # text_key IS the output column here
            if input_key is not None:
                kwargs["input_key"] = input_key

        yielded = 0
        t0 = time.time()
        for ex in ds:
            if yielded >= budget:
                break
            try:
                s = extract_text(ex, **kwargs)
            except Exception:
                continue
            if s and len(s) >= 32:
                yield s
                yielded += 1
        if verbose:
            dt = time.time() - t0
            print(f"  {hf_id}: {yielded}/{budget} docs in {dt:.1f}s")
        total_budget -= yielded

    if verbose:
        print(f"  total shortfall: {total_budget}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-docs", type=int, default=100_000,
                    help="total docs for BPE training (split across MIX by weight)")
    ap.add_argument("--vocab-size", type=int, default=32_768)
    ap.add_argument("--out", type=str, default="output/tokenizer/tokenizer_v2.json")
    ap.add_argument("--smoke", action="store_true",
                    help="use n_docs=1000 for a quick test")
    args = ap.parse_args()

    if args.smoke:
        args.n_docs = 1000
        args.out = args.out.replace("tokenizer_v2.json", "tokenizer_v2_smoke.json")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Retraining FANT 2 tokenizer")
    print(f"  n_docs     : {args.n_docs}")
    print(f"  vocab_size : {args.vocab_size}")
    print(f"  out        : {out_path}")
    print()

    t0 = time.time()
    tok = FANT2Tokenizer.train_from_iterator(
        stream_docs(args.n_docs),
        vocab_size=args.vocab_size,
        min_frequency=2,
        show_progress=True,
    )
    dt = time.time() - t0
    print(f"\nTrained in {dt:.1f}s")

    tok.save(str(out_path))
    print(f"Saved to {out_path}")

    # Sanity-check special tokens survived
    special = ["<|bos|>", "<|eos|>", "<|im_start|>", "<|im_end|>",
               "<|think|>", "<|/think|>", "<|answer|>", "<|/answer|>"]
    print("\nSpecial-token IDs:")
    for t in special:
        try:
            ids = tok.encode(t)
            print(f"  {t:15s} -> {ids}")
        except Exception as e:
            print(f"  {t:15s} -> ERROR {e}")


if __name__ == "__main__":
    main()
