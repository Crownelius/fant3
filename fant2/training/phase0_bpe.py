"""
Phase 0: Train the FANT 2 BPE tokenizer.

This phase does NOT train any neural-network weights. It trains a byte-level
BPE tokenizer (vocab=32768) on a streaming text corpus and saves it as a
HuggingFace-compatible JSON file.

The trained tokenizer is used by ALL later phases (1-6) to convert raw text
to token ids; without it, they cannot run.

Usage
-----

    # Synthetic (offline) test — trains on the SEED_CORPUS, fast
    python -m fant2.training.phase0_bpe --n-docs 1000 --out-path data/tokenizer.json

    # HuggingFace FineWeb-Edu (first 100k docs ≈ a few GB, ~10-20 min on CPU)
    python -m fant2.training.phase0_bpe --use-hf --n-docs 100000 \\
           --out-path data/tokenizer.json
"""

import argparse
import itertools
import os
import sys
import time
from typing import Iterable, Iterator

from ..constants import VOCAB_SIZE
from ..tokenizer import FANT2Tokenizer
from ..data import SyntheticStream, HuggingFaceStream, SEED_CORPUS


# -----------------------------------------------------------------------------
# Small stream helpers
# -----------------------------------------------------------------------------

def _capped_stream(stream: Iterable[str], n_docs: int) -> Iterator[str]:
    """Yield at most n_docs strings, printing a progress dot every 10k."""
    t0 = time.time()
    for i, s in enumerate(stream):
        if i >= n_docs:
            return
        if (i + 1) % 10_000 == 0:
            dt = time.time() - t0
            print(f"    fed {i+1:,} docs ({(i+1)/max(dt,1e-6):.0f} docs/s)")
        yield s


def _seed_stream_repeat(n_docs: int) -> Iterator[str]:
    """Yield n_docs strings by cycling through SEED_CORPUS (for offline test)."""
    for i in range(n_docs):
        yield SEED_CORPUS[i % len(SEED_CORPUS)]


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="FANT 2 Phase 0: Train the BPE tokenizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out-path", type=str, default="data/tokenizer.json",
                   help="where to write the trained tokenizer JSON")
    p.add_argument("--vocab-size", type=int, default=VOCAB_SIZE,
                   help="target BPE vocabulary size (includes reserved specials)")
    p.add_argument("--min-frequency", type=int, default=2,
                   help="minimum merge frequency during BPE training")
    p.add_argument("--n-docs", type=int, default=10_000,
                   help="number of training documents to consume from the stream")
    p.add_argument("--use-hf", action="store_true",
                   help="stream from HuggingFace (default: cycle through SEED_CORPUS)")
    p.add_argument("--hf-dataset", type=str, default=None,
                   help="HF dataset name override (only with --use-hf)")
    p.add_argument("--seed-repeat", action="store_true",
                   help="cycle through the built-in SEED_CORPUS (useful for unit tests)")
    args = p.parse_args()

    # ----- Select stream -----
    if args.use_hf:
        text_stream = HuggingFaceStream(dataset_name=args.hf_dataset)
        print(f"  Streaming from HuggingFace ({args.hf_dataset or 'default cascade'})")
    elif args.seed_repeat:
        text_stream = _seed_stream_repeat(args.n_docs)
        print(f"  Using seed-corpus repeat ({len(SEED_CORPUS)} unique sentences)")
    else:
        text_stream = SyntheticStream()
        print(f"  Using SyntheticStream (random seed-corpus concatenations)")

    # ----- Cap to n_docs -----
    # _seed_stream_repeat is already capped; everything else needs to be wrapped.
    if not args.seed_repeat:
        text_stream = _capped_stream(text_stream, args.n_docs)

    # ----- Train -----
    print(f"=== FANT 2 Phase 0: Training BPE tokenizer ===")
    print(f"  target vocab_size: {args.vocab_size:,}")
    print(f"  min_frequency:     {args.min_frequency}")
    print(f"  n_docs:            {args.n_docs:,}")
    print(f"  out_path:          {args.out_path}")

    t0 = time.time()
    tok = FANT2Tokenizer.train_from_iterator(
        iterator=text_stream,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        show_progress=True,
    )
    dt = time.time() - t0
    print(f"  training time: {dt:.1f}s")

    # ----- Save -----
    out_dir = os.path.dirname(args.out_path) or "."
    os.makedirs(out_dir, exist_ok=True)
    tok.save(args.out_path)
    print(f"  saved to {args.out_path} (vocab_size={tok.vocab_size:,})")

    # ----- Round-trip sanity check -----
    sample = "FANT 2 tokenizer test: 2 + 3 = 5. The Apollonian gasket is fractal."
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    round_trip = tok.decode(ids, skip_special_tokens=True)
    print(f"  round-trip check:")
    print(f"    input : {sample!r}")
    print(f"    ids   : {ids[:16]}... ({len(ids)} total)")
    print(f"    output: {round_trip!r}")

    print("=== Phase 0 complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
