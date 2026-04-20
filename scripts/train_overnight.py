"""
FANT 2 — overnight training with N3 SleepGate + distillation mix.

Best architecture we've tested: **N3 SleepGate** (+5.3pp at 5M scale,
59.9% vs 54.6% baseline on the 1K eval).  This script keeps N3 enabled
and adds sequence-level distillation from existing teacher-generated
corpora, plus the Elephant-Alpha local cache if present.

Defaults are tuned for the RTX 3060 12GB + an overnight (≈10h) wall clock:

    preset:    default (60M stored)    <- known to train end-to-end on 3060
    batch:     4 × seq 1024, grad_accum 4 -> 16k effective tokens/step
    bf16:      on (CUDA)
    8-bit AdamW + grad-checkpoint
    N3:        every 200 steps, merge=0.92, staleness=500
    steps:     30,000  (≈ 8-10 h on RTX 3060; ~500M training tokens)

Data mix (Opus 4.6 seasoning on a Kimi K2.5 backbone, all sequence-level
distillation + a slice of clean web text):
    25%  kimi-k25-distill              Kimi K2.5 general reasoning w/ <think>
    10%  kimi-k25-math                 Kimi K2.5 math CoT
    10%  superior-reasoning-s1         gpt-oss-120b reasoning (stage 1, low-temp)
    15%  opus46-crownelius-3300x       Claude Opus 4.6 reasoning (problem/think/solution)
    10%  opus46-teichai-887x           Claude Opus 4.6 reasoning (messages)
    10%  numina-math-cot               math CoT
    10%  finetome-100k                 curated instruction pairs
    10%  fineweb-edu                   clean web text anchor

Rationale (see memory/project_opus46_distillation_plan.md):
- User pick: Crownelius + TeichAI Opus 4.6 datasets ("the very best datasets")
- Kimi stays the largest single voice (25%) to avoid catastrophic forgetting
  from the prior Kimi-heavy step_500 warm-start
- Combined Opus 4.6 = 25% — seasoning-heavy but not dominant; with only
  3,046 rows, higher weighting would cycle samples ~30+ times
- Elephant Alpha cache dropped entirely (pivoted to Claude Opus 4.6 as the
  frontier teacher; cache preserved at data/distill_cache/elephant_alpha.jsonl
  for future use)

Usage:
    # smoke-test the pipeline (20 steps, synthetic data)
    PYTHONPATH=. python scripts/train_overnight.py --smoke

    # full overnight run
    PYTHONPATH=. python scripts/train_overnight.py

    # override scale
    PYTHONPATH=. python scripts/train_overnight.py --scale 750m --n-steps 5000
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import Iterable, List, Optional

import torch

# project root -> path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fant2.config import fant2_default, fant2_tiny, fant2_750m, fant2_2b, FANT2Config
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.data import (
    InterleavedMultiDatasetStream,
    MixedStream,
    SyntheticStream,
    TokenizedBatchStream,
)
from fant2.data.openrouter_teacher import LocalJSONLStream, DEFAULT_CACHE
from fant2.training import TrainConfig, FANT2Trainer


# --------------------------------------------------------------------------- #
# Scale table                                                                  #
# --------------------------------------------------------------------------- #

SCALES = {
    "tiny":    fant2_tiny,     # 5M
    "default": fant2_default,  # 60M
    "750m":    fant2_750m,     # 742M
    "2b":      fant2_2b,       # 2B (needs A100)
}


# --------------------------------------------------------------------------- #
# Data mix                                                                     #
# --------------------------------------------------------------------------- #

#   (registry-key, weight)  — Opus 4.6 recipe (post-step-500 warm restart)
HF_MIX: List[tuple] = [
    # Kimi K2.5 backbone (rebalanced down from 50% → 35% to make room
    # for Opus 4.6, but still the single largest voice for continuity
    # with the step_500 warm-start that was Kimi-heavy).
    ("kimi-k25-distill",        0.25),
    ("kimi-k25-math",           0.10),
    # gpt-oss-120b reasoning anchor
    ("superior-reasoning-s1",   0.10),
    # Claude Opus 4.6 reasoning (NEW — Crownelius + TeichAI)
    ("opus46-crownelius-3300x", 0.15),
    ("opus46-teichai-887x",     0.10),
    # Supplementary curated / web
    ("numina-math-cot",         0.10),
    ("finetome-100k",           0.10),
    ("fineweb-edu",             0.10),
]
# Elephant Alpha local cache — legacy, no longer mixed into the overnight run.
# Kept as an import so the --elephant-cache flag still works for future recipes.
ELEPHANT_WEIGHT = 0.00


def build_text_stream(args) -> Iterable:
    """Assemble the final text stream (HF mix + optional Elephant cache)."""
    if args.smoke:
        print("  [smoke] SyntheticStream only")
        return SyntheticStream()

    hf_names   = [n for n, _ in HF_MIX]
    hf_weights = [w for _, w in HF_MIX]

    hf_stream = InterleavedMultiDatasetStream(
        dataset_names=hf_names, weights=hf_weights,
    )

    # Optional Elephant Alpha local cache. Default weight is 0 in the current
    # recipe — kept as a flag for backward compatibility / future recipes.
    if ELEPHANT_WEIGHT <= 0.0:
        print(f"  [data] HF-only mix ({len(hf_names)} datasets), weights:")
        for name, w in HF_MIX:
            print(f"         {w*100:>4.1f}%  {name}")
        return hf_stream

    has_cache = (
        os.path.exists(args.elephant_cache) and
        os.path.getsize(args.elephant_cache) > 0
    )
    if not has_cache:
        print(f"  [data] Elephant cache not found at {args.elephant_cache}; "
              f"HF-only mix ({len(hf_names)} datasets)")
        return hf_stream

    # Local + HF mix (legacy path, not used by the default Opus 4.6 recipe)
    local = LocalJSONLStream(args.elephant_cache, loop=True, shuffle=True)
    try:
        with open(args.elephant_cache, "r", encoding="utf-8") as fh:
            n_cache = sum(1 for _ in fh)
    except OSError:
        n_cache = 0

    hf_total_w = 1.0 - ELEPHANT_WEIGHT
    print(f"  [data] Elephant cache found ({n_cache} samples): "
          f"{hf_total_w*100:.0f}% HF + {ELEPHANT_WEIGHT*100:.0f}% local")
    return MixedStream(
        streams=[hf_stream, local],
        weights=[hf_total_w, ELEPHANT_WEIGHT],
        names=["hf_mix", "elephant_cache"],
    )


# --------------------------------------------------------------------------- #
# VRAM check                                                                   #
# --------------------------------------------------------------------------- #

def vram_estimate(cfg: FANT2Config, n_params: int, batch: int, seq: int) -> dict:
    pb = n_params * 2        # bf16 params
    gb = n_params * 2        # bf16 grads
    ob = n_params * 2        # 8-bit AdamW (2 × 1 byte)
    ab = math.sqrt(max(cfg.n_layers, 1)) * batch * seq * cfg.dim * 2
    return {
        "params_gb": pb / 1e9,
        "grads_gb":  gb / 1e9,
        "optim_gb":  ob / 1e9,
        "acts_gb":   ab / 1e9,
        "total_gb":  (pb + gb + ob + ab) / 1e9,
    }


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(
        description="FANT 2 overnight trainer — N3 SleepGate + distillation mix",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ----- model / scale -----
    p.add_argument("--scale", choices=list(SCALES.keys()), default="default")
    p.add_argument("--resume", type=str, default=None)

    # ----- training schedule -----
    p.add_argument("--n-steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=None,
                   help="default: from the config preset")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--muon-lr", type=float, default=5e-4)
    p.add_argument("--adam-lr", type=float, default=1.5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)

    # ----- N3 SleepGate (the proven winner) -----
    p.add_argument("--sleep-every", type=int, default=200,
                   help="N3 consolidation frequency (0 disables)")
    p.add_argument("--sleep-threshold", type=float, default=0.92)
    p.add_argument("--sleep-staleness", type=int, default=500)

    # ----- I/O -----
    p.add_argument("--elephant-cache", type=str, default=DEFAULT_CACHE,
                   help="local Elephant-Alpha JSONL distillation cache")
    p.add_argument("--tokenizer", type=str, default="data/tokenizer.json")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--save-every", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=50)

    # ----- hardware / mode -----
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--no-grad-ckpt", action="store_true",
                   help="disable gradient checkpointing (trades VRAM for speed)")
    p.add_argument("--no-8bit-adam", action="store_true",
                   help="disable 8-bit AdamW")
    p.add_argument("--smoke", action="store_true",
                   help="20 steps, synthetic data, verify no NaN")

    args = p.parse_args()

    # ----- resolve defaults -----
    cfg: FANT2Config = SCALES[args.scale]()
    if args.seq_len is None:
        args.seq_len = min(cfg.max_seq_len, 1024)
    if args.out_dir is None:
        args.out_dir = f"output/overnight_{args.scale}"
    use_cuda = (args.device == "cuda")

    if args.smoke:
        print("=== SMOKE TEST ===")
        args.n_steps      = 20
        args.batch_size   = 2
        args.seq_len      = 128
        args.grad_accum   = 1
        args.save_every   = 9_999_999
        args.log_every    = 1
        args.sleep_every  = 5

    print("=" * 60)
    print(f"  FANT 2 overnight — scale={args.scale}  device={args.device}")
    print("=" * 60)
    print(cfg.summary())

    # ----- tokenizer -----
    tok_candidates = [
        args.tokenizer,
        "output/option_i/tokenizer.json",
        "output/option_b/tokenizer.json",
        "output/tokenizer/tokenizer.json",
    ]
    tokenizer: Optional[FANT2Tokenizer] = None
    for cand in tok_candidates:
        if os.path.exists(cand):
            tokenizer = FANT2Tokenizer.load(cand)
            print(f"\n  Tokenizer: {cand}  vocab={tokenizer.vocab_size}")
            break
    if tokenizer is None:
        print("\n  ERROR: no tokenizer found.  Checked:", tok_candidates)
        sys.exit(1)

    # ----- model -----
    print(f"\n  Building FANT2Model ({args.scale})...")
    t0 = time.time()
    model = FANT2Model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Built in {time.time()-t0:.1f}s  stored={n_params/1e6:.1f}M")

    vram = vram_estimate(cfg, n_params, args.batch_size, args.seq_len)
    print(f"\n  VRAM (est): params {vram['params_gb']:.1f} + "
          f"grads {vram['grads_gb']:.1f} + optim {vram['optim_gb']:.1f} "
          f"+ acts {vram['acts_gb']:.1f} = {vram['total_gb']:.1f} GB")
    if use_cuda:
        gpu_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU memory: {gpu_gb:.1f} GB")
        if vram["total_gb"] > gpu_gb * 0.9:
            print("  WARNING: estimate exceeds 90% of GPU memory.  "
                  "Consider --batch-size or --seq-len.")

    # ----- resume -----
    if args.resume:
        print(f"\n  Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict") or ckpt.get("model") or ckpt
        model.load_state_dict(state)

    # ----- data -----
    text_stream = build_text_stream(args)
    stream = TokenizedBatchStream(
        text_stream=text_stream,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        device="cpu",
    )

    # ----- trainer cfg -----
    train_cfg = TrainConfig(
        phase=2,                    # FEP MoE — Phase 2 is the bulk pretraining
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        grad_accum=args.grad_accum,
        muon_lr=args.muon_lr,
        adam_lr=args.adam_lr,
        weight_decay=args.weight_decay,
        grad_clip=1.0,
        bf16=use_cuda,
        use_8bit_adam=use_cuda and not args.no_8bit_adam,
        grad_checkpoint=use_cuda and not args.no_grad_ckpt,
        log_every=args.log_every,
        save_every=args.save_every,
        out_dir=args.out_dir,
        device=args.device,
        # N3 SleepGate — proven winner (+5.3pp)
        sleep_consolidate_every=args.sleep_every,
        sleep_merge_threshold=args.sleep_threshold,
        sleep_staleness_horizon=args.sleep_staleness,
    )
    if args.resume:
        train_cfg.resume_from = args.resume

    print(f"\n  TrainConfig:")
    eff_tok = train_cfg.batch_size * train_cfg.seq_len * train_cfg.grad_accum
    print(f"    phase=2  n_steps={train_cfg.n_steps}")
    print(f"    batch={train_cfg.batch_size} x seq={train_cfg.seq_len} x accum={train_cfg.grad_accum} "
          f"-> {eff_tok:,} effective tokens/step")
    total_tok = eff_tok * train_cfg.n_steps
    print(f"    projected total tokens: {total_tok/1e9:.2f}B")
    print(f"    muon_lr={train_cfg.muon_lr}  adam_lr={train_cfg.adam_lr}  "
          f"wd={train_cfg.weight_decay}")
    print(f"    N3 SleepGate: every {train_cfg.sleep_consolidate_every} steps  "
          f"thr={train_cfg.sleep_merge_threshold}  stale={train_cfg.sleep_staleness_horizon}")
    print(f"    out: {train_cfg.out_dir}")

    trainer = FANT2Trainer(model, train_cfg, stream)

    print("\n" + "=" * 60)
    print(f"  Training for {train_cfg.n_steps} steps")
    print("=" * 60 + "\n")

    t_start = time.time()
    trainer.train()
    wall = time.time() - t_start

    print(f"\n  Done in {wall/3600:.2f}h")
    if use_cuda:
        print(f"  Peak VRAM: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
    if args.smoke:
        print("  SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
