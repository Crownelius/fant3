#!/usr/bin/env python
"""
Rebuild fant_code.zip for Colab uploads.

Writes a fresh archive to `fant2/fant_code.zip`, picking up:
  - fant2/**             (legacy runtime)
  - fant3/**             (current architecture)
  - scripts/**           (utilities: eval, decontaminate, FSS, etc.)
  - notebooks/*.ipynb + _patch_*.py
  - tests/**
  - bendvm/** + its demos
  - output/decontamination/ngram_hashes.json   (457k SHA-1 cache)
  - output/tokenizer/tokenizer_v2.json         (vocab)

Excluded on purpose: __pycache__, *.pyc, *.pt (checkpoints), .openrouter_key,
old output directories aside from the two listed above, venv/, .git/.
"""

from __future__ import annotations
import argparse
import os
import zipfile
from pathlib import Path

# ---------------------------------------------------------------------------
# Inclusion / exclusion rules
# ---------------------------------------------------------------------------

INCLUDE_DIRS = ["fant2", "fant3", "scripts", "tests", "bendvm", "notebooks"]
INCLUDE_SPECIFIC_FILES = [
    Path("output/decontamination/ngram_hashes.json"),
    Path("output/tokenizer/tokenizer_v2.json"),
]
EXCLUDE_SUFFIXES = (".pyc", ".pyo", ".pt", ".pth", ".bin", ".safetensors",
                    ".zip", ".log")
EXCLUDE_DIRNAMES = ("__pycache__", ".ipynb_checkpoints", ".git", ".venv",
                    "venv", "node_modules")
EXCLUDE_FILENAMES = (".openrouter_key", ".env", ".DS_Store")


def _skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root).as_posix()
    name = path.name
    if name in EXCLUDE_FILENAMES:
        return True
    if path.suffix.lower() in EXCLUDE_SUFFIXES:
        return True
    parts = path.relative_to(root).parts
    for part in parts:
        if part in EXCLUDE_DIRNAMES:
            return True
    return False


def _iter_files(root: Path):
    for d in INCLUDE_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not _skip(p, root):
                yield p
    for f in INCLUDE_SPECIFIC_FILES:
        p = root / f
        if p.is_file():
            yield p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--out",  type=Path, default=None,
                    help="output path (default: <root>/fant_code.zip)")
    args = ap.parse_args()

    root = args.root.resolve()
    out  = (args.out or (root / "fant_code.zip")).resolve()

    print(f"building {out} from {root}")
    files = sorted(_iter_files(root))
    print(f"  collected {len(files)} files")

    total_bytes = 0
    with zipfile.ZipFile(out, mode="w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as z:
        for p in files:
            arcname = p.relative_to(root).as_posix()
            z.write(p, arcname=arcname)
            total_bytes += p.stat().st_size

    size_mb = out.stat().st_size / 1e6
    print(f"  wrote {len(files)} files ({total_bytes/1e6:.2f} MB raw -> {size_mb:.2f} MB compressed)")


if __name__ == "__main__":
    main()
