"""
Option G — Benchmark + ramped-up training.

Goal: measure FANT 2's perplexity on a held-out synthetic stream, then
do a substantially longer training run and demonstrate at least a 50%
reduction in perplexity vs. the baseline.

Why perplexity (not GSM8K/HellaSwag/ARC):
  * The user-imposed constraint forbids training on public benchmarks.
    Eval on them is allowed but the tiny preset has zero hope of any
    above-random accuracy on benchmarks designed for ~7B+ models, so
    the signal would be noise.
  * Perplexity is continuous, deterministic, fast, and self-contained
    (uses the same `SyntheticStream` source distribution at a held-out
    seed). It cleanly captures whether the model has learned the
    training distribution.
  * "50-100% improvement" is unambiguous on perplexity: a 50%
    improvement = perplexity halved. We aim for ≥ 2x reduction.

Procedure:
  1. Build a tiny model + reuse the existing tokenizer
  2. Create a held-out batch stream from `SyntheticStream(seed=999)`
  3. Load the existing 60-step baseline ckpt (option_e/phase4_boot/final.pt)
  4. Bench → record baseline perplexity
  5. Resume + train for ~1800 more steps on `SyntheticStream(seed=33)`
     (the same training distribution as the bootstrap)
  6. Bench again → record final perplexity
  7. Print the relative improvement and pass/fail vs. the 50% target

This script uses the existing `evaluate_perplexity` from `fant2.bench`.

Run:
    PYTHONPATH=. python scripts/option_g_benchmark_rampup.py
"""

from __future__ import annotations

import math
import os
import time

import torch

from fant2.bench import evaluate_perplexity
from fant2.config import fant2_tiny
from fant2.data import SEED_CORPUS, SyntheticStream, TokenizedBatchStream
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training import TrainConfig, FANT2Trainer


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

BASELINE_CKPT  = "output/option_e/phase4_boot/final.pt"
BASELINE_STEPS = 60        # the existing bootstrap is 60 Phase 2 steps
RAMP_STEPS     = 1800      # 30x more training during the ramp-up
SEQ_LEN        = 64        # within tiny preset's max_seq_len=128
BATCH_SIZE     = 4
TRAIN_SEED     = 33        # matches the bootstrap so the resume is "more of the same"
HELDOUT_SEED   = 999       # different seed = held-out
N_EVAL_BATCHES = 80        # eval batches per benchmark call
TARGET_REL_IMPROVEMENT = 0.50  # require ≥ 50% lower perplexity

OUT_BASE = "output/option_g"
OUT_RAMP = os.path.join(OUT_BASE, "ramp")
TOK_PATH = "output/option_e/tokenizer.json"  # reuse the option_e tokenizer


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def get_tokenizer() -> FANT2Tokenizer:
    if os.path.exists(TOK_PATH):
        print(f"  reusing tokenizer at {TOK_PATH}")
        return FANT2Tokenizer.load(TOK_PATH)
    print(f"  training a fresh BPE tokenizer at {TOK_PATH}")
    def gen():
        for i in range(5000):
            yield SEED_CORPUS[i % len(SEED_CORPUS)]
    tok = FANT2Tokenizer.train_from_iterator(
        iterator=gen(), vocab_size=4096, min_frequency=2, show_progress=False,
    )
    os.makedirs(os.path.dirname(TOK_PATH) or ".", exist_ok=True)
    tok.save(TOK_PATH)
    return tok


def make_eval_stream(tokenizer):
    """Held-out batch stream — different seed from training."""
    text = SyntheticStream(seed=HELDOUT_SEED)
    return TokenizedBatchStream(
        text_stream=text, tokenizer=tokenizer,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device="cpu",
    )


def make_train_stream(tokenizer):
    """Training batch stream — same seed as the bootstrap."""
    text = SyntheticStream(seed=TRAIN_SEED)
    return TokenizedBatchStream(
        text_stream=text, tokenizer=tokenizer,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device="cpu",
    )


def benchmark(model, tokenizer, label: str) -> dict:
    print()
    print(f"  -- benchmarking: {label} --")
    eval_stream = make_eval_stream(tokenizer)
    t0 = time.time()
    res = evaluate_perplexity(
        model, eval_stream, max_batches=N_EVAL_BATCHES, verbose=False,
    )
    dt = time.time() - t0
    print(f"    avg NLL  = {res['loss']:.4f}")
    print(f"    perplexity = {res['perplexity']:.3f}")
    print(f"    n_tokens = {res['n_tokens']}")
    print(f"    eval time = {dt:.1f}s ({N_EVAL_BATCHES} batches × "
          f"{BATCH_SIZE}×{SEQ_LEN})")
    return res


def build_trainer(tokenizer, n_steps: int, resume_from: str | None) -> FANT2Trainer:
    cfg = fant2_tiny()
    model = FANT2Model(cfg)
    train_stream = make_train_stream(tokenizer)
    train_cfg = TrainConfig(
        phase=2, n_steps=n_steps,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        muon_lr=1e-3, adam_lr=3e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.1,
        fep_kl_beta_max=0.5,
        # Anneal across the full ramp so KL bites slowly as training progresses.
        fep_kl_anneal_steps=max(n_steps, 1),
        telemetry_every=2000, tikkun_every=2000, fana_every=10000,
        log_every=max(1, n_steps // 12),
        save_every=10000,
        out_dir=OUT_RAMP,
        resume_from=resume_from,
        device="cpu",
        bf16=False, grad_checkpoint=False, use_8bit_adam=False,
    )
    return FANT2Trainer(model, train_cfg, train_stream)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print(" FANT 2 — Option G: benchmark + ramped-up training")
    print(" (perplexity on held-out SyntheticStream — no benchmarks touched)")
    print("=" * 64)

    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(OUT_RAMP, exist_ok=True)
    tokenizer = get_tokenizer()

    if not os.path.exists(BASELINE_CKPT):
        print(f"  ✗ baseline ckpt not found at {BASELINE_CKPT}")
        print(f"    run `python scripts/option_e_phase5_grpo.py` first to bootstrap it")
        return 1

    # ---------- Phase A: benchmark the baseline ----------
    print()
    print(f"  ===== Phase A: bench baseline ({BASELINE_STEPS} steps) =====")
    baseline_trainer = build_trainer(
        tokenizer, n_steps=1, resume_from=BASELINE_CKPT,
    )
    print(f"  loaded baseline at step {baseline_trainer.step}")
    baseline_res = benchmark(
        baseline_trainer.model, tokenizer,
        label=f"baseline ({baseline_trainer.step} steps)",
    )

    # ---------- Phase B: ramp up training ----------
    print()
    print(f"  ===== Phase B: ramp up training (+{RAMP_STEPS} steps) =====")
    ramp_trainer = build_trainer(
        tokenizer, n_steps=RAMP_STEPS, resume_from=BASELINE_CKPT,
    )
    t0 = time.time()
    ramp_trainer.train()
    dt = time.time() - t0
    print(f"  ramp-up done in {dt:.1f}s "
          f"({dt / RAMP_STEPS * 1000:.1f} ms/step)")
    print(f"  trainer at step {ramp_trainer.step}")

    # ---------- Phase C: benchmark the ramped model ----------
    ramp_res = benchmark(
        ramp_trainer.model, tokenizer,
        label=f"ramped ({ramp_trainer.step} steps)",
    )

    # ---------- Report ----------
    print()
    print("=" * 64)
    print(" RESULTS")
    print("=" * 64)
    bp = baseline_res["perplexity"]
    rp = ramp_res["perplexity"]
    bn = baseline_res["loss"]
    rn = ramp_res["loss"]
    rel_improvement = (bp - rp) / bp if bp > 0 else float("nan")
    nll_drop = bn - rn

    print(f"  baseline ({BASELINE_STEPS:>5d} steps):  ppl = {bp:9.3f}   nll = {bn:.4f}")
    print(f"  ramped   ({BASELINE_STEPS + RAMP_STEPS:>5d} steps):  ppl = {rp:9.3f}   nll = {rn:.4f}")
    print()
    print(f"  perplexity reduction: {bp - rp:9.3f}  ({rel_improvement * 100:+.1f}%)")
    print(f"  NLL    reduction: {nll_drop:9.4f}  "
          f"({nll_drop / bn * 100 if bn > 0 else float('nan'):+.1f}%)")
    print(f"  perplexity ratio  ramped/baseline = {rp / bp:.4f}")
    print(f"  perplexity ratio  baseline/ramped = {bp / rp:.4f} x")
    print()

    target_pct = TARGET_REL_IMPROVEMENT * 100
    if not (math.isfinite(bp) and math.isfinite(rp)):
        print(f"  ✗ FAIL: non-finite perplexity (baseline={bp}, ramped={rp})")
        return 1
    if rel_improvement < TARGET_REL_IMPROVEMENT:
        print(f"  ✗ FAIL: improvement {rel_improvement * 100:+.1f}% "
              f"is below the {target_pct:.0f}% target")
        return 1
    print(f"  ✓ PASS: improvement {rel_improvement * 100:+.1f}% "
          f"meets or exceeds the {target_pct:.0f}% target")
    if rel_improvement >= 1.00:
        print(f"  ✓✓ DOUBLE-PASS: improvement {rel_improvement * 100:+.1f}% "
              f"also meets the 100% stretch target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
