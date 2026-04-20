"""
Shared helpers for the 7 phase training scripts.

Each `phaseN_*.py` script is deliberately thin: it parses CLI args, calls
`build_everything(...)` from here, then customizes a `TrainConfig` with the
phase-specific defaults and launches `FANT2Trainer.train()`.

Centralizing the boilerplate here means:

  * model / tokenizer / data construction is identical across phases
  * resume logic lives in one place
  * CLI arg parsing is consistent

Nothing phase-specific goes in this file.
"""

import argparse
import os
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import torch

from ..config import FANT2Config, fant2_default, fant2_tiny, fant2_750m, fant2_2b
from ..model import FANT2Model
from ..tokenizer import FANT2Tokenizer
from ..data import (
    HuggingFaceStream,
    InterleavedMultiDatasetStream,
    SyntheticStream,
    TokenizedBatchStream,
    make_default_stream,
)


# -----------------------------------------------------------------------------
# Preset table
# -----------------------------------------------------------------------------

PRESETS = {
    "tiny":    fant2_tiny,     # 5M-stored, CPU fast test
    "default": fant2_default,  # 60M-stored / 200M-active, the locked config
    "750m":    fant2_750m,     # 742M-stored, fits RTX 3060 12GB
    "2b":      fant2_2b,       # 2B-stored / ~3.6B-active, needs A100
}


# -----------------------------------------------------------------------------
# Argument parser factory
# -----------------------------------------------------------------------------

def make_phase_parser(phase: int, description: str) -> argparse.ArgumentParser:
    """
    Build an argparse.ArgumentParser with the standard set of FANT 2 training flags.

    Every phase script uses this parser (plus any phase-specific args appended).
    """
    p = argparse.ArgumentParser(
        description=f"FANT 2 Phase {phase}: {description}",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ----- Model -----
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="default",
                   help="model size preset")
    p.add_argument("--resume", type=str, default=None,
                   help="checkpoint path to resume from")

    # ----- Data -----
    p.add_argument("--tokenizer", type=str, default="data/tokenizer.json",
                   help="path to a trained FANT2Tokenizer JSON")
    p.add_argument("--use-hf", action="store_true",
                   help="stream from HuggingFace datasets (default: synthetic)")
    p.add_argument("--hf-dataset", type=str, default=None,
                   help="HF dataset registry key (if --use-hf)")
    p.add_argument("--hf-datasets", type=str, default=None,
                   help="comma-separated registry keys for multi-dataset interleave")
    p.add_argument("--hf-weights", type=str, default=None,
                   help="comma-separated sampling weights (must match --hf-datasets)")

    # ----- Training -----
    p.add_argument("--n-steps", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--muon-lr", type=float, default=1e-3)
    p.add_argument("--adam-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)

    # ----- Hardware -----
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action="store_true",
                   help="use bfloat16 (recommended on RTX 3060+)")
    p.add_argument("--no-grad-checkpoint", action="store_true")
    p.add_argument("--no-8bit-adam", action="store_true",
                   help="disable 8-bit AdamW (required on CPU)")

    # ----- Output -----
    p.add_argument("--out-dir", type=str, default=f"output/fant2_phase{phase}")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=50)

    return p


# -----------------------------------------------------------------------------
# Construction helpers
# -----------------------------------------------------------------------------

def build_tokenizer(args) -> FANT2Tokenizer:
    """
    Load the tokenizer from --tokenizer. If the file does not exist, print a
    helpful error telling the user to run phase0 first.
    """
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(
            f"Tokenizer not found at {args.tokenizer}.\n"
            "Run phase 0 first to train one:\n"
            "    python -m fant2.training.phase0_bpe --out-path data/tokenizer.json\n"
            f"Or pass --tokenizer <path> to use an existing one."
        )
    print(f"  Loading tokenizer from {args.tokenizer}")
    return FANT2Tokenizer.load(args.tokenizer)


def build_model(args) -> Tuple[FANT2Model, FANT2Config]:
    """Build a fresh FANT2Model from the named preset."""
    cfg = PRESETS[args.preset]()
    print(f"  Building model (preset={args.preset})")
    print(cfg.summary())
    model = FANT2Model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total stored params: {n_params/1e6:.2f}M")
    return model, cfg


def build_stream(
    args,
    tokenizer: FANT2Tokenizer,
) -> Iterable:
    """Build a TokenizedBatchStream from the CLI args.

    Supports three modes:
      1. ``--use-hf`` with a single ``--hf-dataset`` key — loads one dataset
         with format-aware extraction via the registry.
      2. ``--use-hf`` with ``--hf-datasets`` (comma-separated) — interleaves
         multiple datasets with optional weights.
      3. No ``--use-hf`` — falls back to SyntheticStream.
    """
    if args.use_hf:
        multi = getattr(args, "hf_datasets", None)
        if multi:
            # Multi-dataset interleave mode
            names = [n.strip() for n in multi.split(",") if n.strip()]
            weights_raw = getattr(args, "hf_weights", None)
            weights = None
            if weights_raw:
                weights = [float(w) for w in weights_raw.split(",")]
            text_stream = InterleavedMultiDatasetStream(
                dataset_names=names,
                weights=weights,
            )
            print(f"  Using interleaved HF stream ({len(names)} datasets)")
        elif args.hf_dataset:
            # Single dataset with registry format
            from ..data.registry import ALL_DATASETS
            from ..data.formats import DatasetFormat
            entry = ALL_DATASETS.get(args.hf_dataset)
            if entry:
                text_stream = HuggingFaceStream(
                    dataset_name=entry.hf_id,
                    dataset_config=entry.config,
                    split=entry.split,
                    text_key=entry.text_key,
                    format_type=entry.format,
                    input_key=entry.input_key,
                    output_key=entry.output_key,
                )
            else:
                text_stream = HuggingFaceStream(dataset_name=args.hf_dataset)
            print(f"  Using HuggingFace stream ({args.hf_dataset})")
        else:
            text_stream = HuggingFaceStream()
            print(f"  Using HuggingFace stream (default cascade)")
    else:
        text_stream = SyntheticStream()
        print(f"  Using SyntheticStream (offline fallback)")

    return TokenizedBatchStream(
        text_stream=text_stream,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=getattr(args, "seq_len", 1024),
        device="cpu",  # the trainer moves to device
    )


def build_everything(args) -> Tuple[FANT2Model, FANT2Config, Iterable, FANT2Tokenizer]:
    """Bundle: tokenizer + model + stream, all constructed from CLI args."""
    tokenizer = build_tokenizer(args)
    model, cfg = build_model(args)
    if tokenizer.vocab_size != cfg.vocab_size:
        print(
            f"  WARNING: tokenizer.vocab_size={tokenizer.vocab_size} != "
            f"cfg.vocab_size={cfg.vocab_size}. Continuing; make sure this is intended."
        )
    stream = build_stream(args, tokenizer)
    return model, cfg, stream, tokenizer
