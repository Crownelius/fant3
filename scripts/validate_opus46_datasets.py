"""
Pre-launch validator for the Opus 4.6 recipe.

Pulls 2 samples from each new dataset, checks the format extractor outputs
a non-empty ChatML string, and builds the full InterleavedMultiDatasetStream
for 10 iterations to catch any registry / format wiring bugs before we
commit to a 2500-step training run.

Usage:
    PYTHONPATH=. python scripts/validate_opus46_datasets.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fant2.data.registry import TRAINING_DATASETS, get_dataset
from fant2.data.formats import DatasetFormat, extract_text
from fant2.data import InterleavedMultiDatasetStream


TO_VALIDATE = [
    "opus46-crownelius-3300x",
    "opus46-teichai-887x",
]

FULL_MIX = [
    ("kimi-k25-distill",        0.25),
    ("kimi-k25-math",           0.10),
    ("superior-reasoning-s1",   0.10),
    ("opus46-crownelius-3300x", 0.15),
    ("opus46-teichai-887x",     0.10),
    ("numina-math-cot",         0.10),
    ("finetome-100k",           0.10),
    ("fineweb-edu",             0.10),
]


def main() -> None:
    print("=" * 60)
    print("  Opus 4.6 dataset validator")
    print("=" * 60)

    # ---- Step 1: check each new dataset loads + extracts cleanly ----
    for name in TO_VALIDATE:
        print(f"\n[{name}]")
        entry = TRAINING_DATASETS[name]
        print(f"  hf_id:   {entry.hf_id}")
        print(f"  config:  {entry.config}")
        print(f"  format:  {entry.format.value}")
        print(f"  text_key: {entry.text_key}")

        try:
            ds = get_dataset(name, streaming=True)
            it = iter(ds)
            for i in range(2):
                ex = next(it)
                print(f"\n  --- sample {i+1} ---")
                # show schema
                cols = list(ex.keys())
                print(f"  columns: {cols}")
                # extract + show first 250 chars
                text = extract_text(
                    ex,
                    fmt=entry.format,
                    text_key=entry.text_key,
                    input_key=entry.input_key or "input",
                    output_key=entry.output_key or "output",
                )
                print(f"  extracted len: {len(text)}")
                print(f"  preview: {text[:250]!r}")
                assert text, f"{name} sample {i+1} extracted to empty string"
        except Exception as e:
            print(f"  ERROR: {type(e).__name__}: {e}")
            sys.exit(1)

    # ---- Step 2: build the full mixed stream, pull 20 items ----
    print("\n" + "=" * 60)
    print("  Full HF_MIX stream test (20 iterations)")
    print("=" * 60)

    hf_names   = [n for n, _ in FULL_MIX]
    hf_weights = [w for _, w in FULL_MIX]

    stream = InterleavedMultiDatasetStream(
        dataset_names=hf_names, weights=hf_weights,
    )

    counts = {n: 0 for n in hf_names}
    it = iter(stream)
    for i in range(20):
        try:
            text = next(it)
        except Exception as e:
            print(f"  ERROR at step {i}: {type(e).__name__}: {e}")
            sys.exit(1)
        assert text, f"stream yielded empty at step {i}"
        # stream doesn't tell us which source — just verify it's producing text
        if i < 3:
            print(f"  [{i}] len={len(text):5d}  preview: {text[:100]!r}")

    print("\n  ✓ Stream produced 20 non-empty items with no errors")
    print("\n" + "=" * 60)
    print("  VALIDATION PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
