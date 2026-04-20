"""
FANT 2 — scaled training with N3 SleepGate + all datasets.

Supports two scale presets:
  --scale 750m   742M stored params — fits RTX 3060 12GB (default)
  --scale 2b     2.025B stored params — needs A100-40GB+

N3 SleepGate memory consolidation is integrated from the start.

Dataset mix (Phase 2 pretraining):
  60%  fineweb-edu          — bulk clean web text
  15%  cosmopedia-v2        — synthetic educational text
   5%  finetome-100k        — curated fine-tuning corpus
   5%  kimi-k25-distill     — reasoning with <think> CoT
   5%  kimi-k25-stem        — multilingual STEM
   5%  superior-reasoning-s1 — stage 1 low-temp reasoning
   5%  numina-math-cot      — math chain-of-thought

Usage:
    # Smoke test (verify pipeline works)
    PYTHONPATH=. python scripts/train_2b.py --phase 2 --smoke

    # Phase 2 pretraining on RTX 3060 (750M default)
    PYTHONPATH=. python scripts/train_2b.py --phase 2 --n-steps 50000

    # Phase 2 with 2B scale (needs A100)
    PYTHONPATH=. python scripts/train_2b.py --phase 2 --scale 2b --n-steps 50000

    # Custom dataset mix
    PYTHONPATH=. python scripts/train_2b.py --phase 2 \\
        --datasets fineweb-edu,kimi-k25-distill \\
        --weights 0.8,0.2
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import json
import math

import torch
import torch.nn as nn

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fant2.config import fant2_750m, fant2_2b, FANT2Config
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.data import (
    InterleavedMultiDatasetStream,
    TokenizedBatchStream,
    SyntheticStream,
)
from fant2.training import TrainConfig, FANT2Trainer
from fant2.training.campaign_n import run_sleep_consolidation


# ── Dataset mixes per phase ──────────────────────────────────────────────

PHASE2_MIX = {
    "datasets": [
        "fineweb-edu",
        "cosmopedia-v2",
        "finetome-100k",
        "kimi-k25-distill",
        "kimi-k25-stem",
        "superior-reasoning-s1",
        "numina-math-cot",
    ],
    "weights": [0.60, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05],
}

PHASE34_MIX = {
    "datasets": [
        "kimi-k25-distill",
        "kimi-k25-math",
        "kimi-k25-science",
        "superior-reasoning-s1",
        "superior-reasoning-s2",
        "numina-math-cot",
        "finetome-100k",
        "infinity-instruct",
        "logic-puzzles",
    ],
    "weights": [0.25, 0.15, 0.10, 0.15, 0.10, 0.10, 0.05, 0.05, 0.05],
}

PHASE6_MIX = {
    "datasets": [
        "tulu3-sft",
        "magpie-pro",
        "finetome-100k",
    ],
    "weights": [0.40, 0.40, 0.20],
}


# ── Phase-specific training configs ──────────────────────────────────────

def get_train_config(phase: int, cfg: FANT2Config, args) -> TrainConfig:
    """Build a TrainConfig appropriate for the given phase and scale."""
    use_cuda = (args.device != "cpu")
    base = dict(
        phase=phase,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        grad_accum=args.grad_accum,
        grad_clip=1.0,
        bf16=use_cuda,
        use_8bit_adam=use_cuda,
        grad_checkpoint=use_cuda,
        log_every=args.log_every,
        save_every=args.save_every,
        out_dir=os.path.join(args.out_dir, f"phase{phase}"),
        device=args.device,
        # N3 SleepGate — proven winner from Campaign N
        sleep_consolidate_every=args.sleep_every,
        sleep_merge_threshold=args.sleep_threshold,
        sleep_staleness_horizon=args.sleep_staleness,
    )

    if phase in (1, 2):
        # Pretraining: higher LR
        base.update(
            muon_lr=args.muon_lr or 5e-4,
            adam_lr=args.adam_lr or 1.5e-4,
            weight_decay=0.01,
        )
    elif phase in (3, 4):
        # Reasoning fine-tune: lower LR
        base.update(
            muon_lr=args.muon_lr or 2e-4,
            adam_lr=args.adam_lr or 5e-5,
            weight_decay=0.005,
        )
    elif phase == 5:
        # GRPO: very low LR (spec §8: lr=5e-7)
        base.update(
            muon_lr=args.muon_lr or 5e-7,
            adam_lr=args.adam_lr or 5e-7,
            weight_decay=0.0,
        )
    elif phase == 6:
        # SimPO+KTO: moderate LR
        base.update(
            muon_lr=args.muon_lr or 1e-5,
            adam_lr=args.adam_lr or 1e-5,
            weight_decay=0.001,
        )

    return TrainConfig(**base)


# ── VRAM estimation ──────────────────────────────────────────────────────

SCALE_PRESETS = {
    "750m": fant2_750m,
    "2b":   fant2_2b,
}


def estimate_vram(cfg: FANT2Config, n_params: int, batch_size: int, seq_len: int) -> dict:
    """Rough VRAM estimate for planning."""
    param_bytes = n_params * 2       # bf16
    grad_bytes = n_params * 2        # bf16 gradients
    optim_bytes = n_params * 2       # 8-bit AdamW (2 moments × 1 byte each)
    # With grad checkpointing, activation memory is roughly:
    # sqrt(n_layers) × batch × seq × dim × 2
    activation_bytes = (
        math.sqrt(cfg.n_layers) * batch_size * seq_len * cfg.dim * 2
    )
    total = param_bytes + grad_bytes + optim_bytes + activation_bytes
    return {
        "params_gb": param_bytes / 1e9,
        "grads_gb": grad_bytes / 1e9,
        "optim_gb": optim_bytes / 1e9,
        "activation_gb": activation_bytes / 1e9,
        "total_gb": total / 1e9,
    }


# ── Main training loop ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FANT 2 — 2B-scale training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Scale & phase
    parser.add_argument("--scale", type=str, default="750m", choices=list(SCALE_PRESETS.keys()),
                        help="model scale preset")
    parser.add_argument("--phase", type=int, default=2, choices=[1, 2, 3, 4, 5, 6],
                        help="training phase")
    parser.add_argument("--datasets", type=str, default=None,
                        help="comma-separated dataset registry keys (overrides phase default)")
    parser.add_argument("--weights", type=str, default=None,
                        help="comma-separated sampling weights (must match --datasets)")
    parser.add_argument("--smoke", action="store_true",
                        help="smoke test: 20 steps, synthetic data, verify no NaN")

    # Training schedule
    parser.add_argument("--n-steps", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="sequence length (default: from config preset)")
    parser.add_argument("--grad-accum", type=int, default=8,
                        help="gradient accumulation steps (effective batch = batch_size × grad_accum)")
    parser.add_argument("--muon-lr", type=float, default=None,
                        help="Muon LR override (default: phase-specific)")
    parser.add_argument("--adam-lr", type=float, default=None,
                        help="AdamW LR override (default: phase-specific)")

    # N3 SleepGate
    parser.add_argument("--sleep-every", type=int, default=200,
                        help="N3 SleepGate consolidation frequency (0=off)")
    parser.add_argument("--sleep-threshold", type=float, default=0.92,
                        help="N3 merge cosine similarity threshold")
    parser.add_argument("--sleep-staleness", type=int, default=500,
                        help="N3 staleness eviction horizon")

    # Hardware
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")

    # I/O
    parser.add_argument("--resume", type=str, default=None, help="checkpoint to resume from")
    parser.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    parser.add_argument("--out-dir", type=str, default=None,
                        help="output directory (default: output/fant2_{scale})")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--log-every", type=int, default=50)

    args = parser.parse_args()

    # ── Build config ──
    cfg = SCALE_PRESETS[args.scale]()

    # ── Fill defaults from config ──
    if args.seq_len is None:
        args.seq_len = cfg.max_seq_len
    if args.out_dir is None:
        args.out_dir = f"output/fant2_{args.scale}"

    # ── Smoke-test overrides ──
    if args.smoke:
        print("=== SMOKE TEST MODE ===")
        args.n_steps = 20
        args.batch_size = 2
        args.seq_len = 128
        args.grad_accum = 1
        args.save_every = 999999
        args.log_every = 1

    print(f"\n{'='*60}")
    print(f"FANT 2 — {args.scale.upper()} Training (Phase {args.phase})")
    print(f"{'='*60}")
    print(cfg.summary())

    # ── Build tokenizer ──
    tok_candidates = [
        args.tokenizer,
        "output/option_i/tokenizer.json",
        "output/option_b/tokenizer.json",
    ]
    tokenizer = None
    for tok_path in tok_candidates:
        if os.path.exists(tok_path):
            tokenizer = FANT2Tokenizer.load(tok_path)
            print(f"\n  Tokenizer loaded from {tok_path}: vocab_size={tokenizer.vocab_size}")
            break
    if tokenizer is None:
        print(f"\n  ERROR: no tokenizer found. Checked: {tok_candidates}")
        print(f"  Run Phase 0 first, or pass --tokenizer <path>")
        sys.exit(1)

    # ── Build model ──
    print(f"\n  Building FANT2Model ({args.scale} preset)...")
    t0 = time.time()
    model = FANT2Model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model built in {time.time()-t0:.1f}s — {n_params/1e6:.1f}M stored params")

    # ── VRAM estimate ──
    vram = estimate_vram(cfg, n_params, args.batch_size, args.seq_len)
    print(f"\n--- VRAM Estimate ---")
    print(f"  Parameters:   {vram['params_gb']:.1f} GB")
    print(f"  Gradients:    {vram['grads_gb']:.1f} GB")
    print(f"  Optimizer:    {vram['optim_gb']:.1f} GB")
    print(f"  Activations:  {vram['activation_gb']:.1f} GB (with grad checkpoint)")
    print(f"  TOTAL:        {vram['total_gb']:.1f} GB")

    if args.device == "cuda":
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"  GPU memory:   {gpu_mem:.1f} GB")
        if vram["total_gb"] > gpu_mem * 0.9:
            print(f"  WARNING: estimated VRAM ({vram['total_gb']:.1f} GB) may exceed GPU")
            print(f"           Consider reducing --batch-size / --seq-len")

    # Resume from checkpoint
    if args.resume:
        print(f"  Loading checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        else:
            model.load_state_dict(ckpt)
        print(f"  Checkpoint loaded OK")

    # ── Build data stream ──
    if args.smoke:
        print(f"\n  Using SyntheticStream (smoke test)")
        text_stream = SyntheticStream()
    elif args.datasets:
        # Custom dataset list
        names = [n.strip() for n in args.datasets.split(",")]
        weights = None
        if args.weights:
            weights = [float(w) for w in args.weights.split(",")]
        text_stream = InterleavedMultiDatasetStream(
            dataset_names=names, weights=weights,
        )
        print(f"\n  Using custom interleave: {names}")
    else:
        # Phase-specific default mix
        if args.phase in (1, 2):
            mix = PHASE2_MIX
        elif args.phase in (3, 4):
            mix = PHASE34_MIX
        elif args.phase == 6:
            mix = PHASE6_MIX
        else:
            # Phase 5 (GRPO) uses procedural data, not this mix
            mix = PHASE34_MIX

        text_stream = InterleavedMultiDatasetStream(
            dataset_names=mix["datasets"],
            weights=mix["weights"],
        )
        print(f"\n  Using Phase {args.phase} default mix ({len(mix['datasets'])} datasets)")
        for name, w in zip(mix["datasets"], mix["weights"]):
            print(f"    {name:30s}  {w*100:5.1f}%")

    stream = TokenizedBatchStream(
        text_stream=text_stream,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device="cpu",
    )

    # ── Build trainer ──
    train_cfg = get_train_config(args.phase, cfg, args)
    print(f"\n  TrainConfig:")
    print(f"    phase={train_cfg.phase}, steps={train_cfg.n_steps}")
    print(f"    batch={train_cfg.batch_size}x{train_cfg.seq_len}, accum={train_cfg.grad_accum}")
    eff = train_cfg.batch_size * train_cfg.seq_len * train_cfg.grad_accum
    print(f"    effective_batch_tokens = {eff:,}")
    print(f"    muon_lr={train_cfg.muon_lr}, adam_lr={train_cfg.adam_lr}")
    print(f"    N3 SleepGate: every {train_cfg.sleep_consolidate_every} steps "
          f"(threshold={train_cfg.sleep_merge_threshold}, staleness={train_cfg.sleep_staleness_horizon})")

    if args.resume:
        train_cfg.resume_from = args.resume

    trainer = FANT2Trainer(model, train_cfg, stream)

    # ── Train ──
    print(f"\n{'='*60}")
    print(f"Starting Phase {args.phase} training — {train_cfg.n_steps} steps")
    print(f"{'='*60}\n")

    trainer.train()

    # ── Results summary ──
    if args.device == "cuda":
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    if args.smoke:
        print(f"\n  SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
