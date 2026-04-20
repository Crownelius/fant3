"""
Campaign N — unified experiment runner for N3, N6, N7 and all combinations.

Three structural levers (no auxiliary losses):
  N3 — SleepGate: periodic memory consolidation
  N6 — G2RPO-A:  gold reasoning traces in training data
  N7 — SEC:      self-evolving curriculum via Multi-Armed Bandit

All 7 variants:
  n3      — SleepGate only
  n6      — gold reasoning only
  n7      — curriculum only
  n3_n6   — SleepGate + gold reasoning
  n3_n7   — SleepGate + curriculum
  n6_n7   — gold reasoning + curriculum
  n3_n6_n7 — all three

Usage:
    PYTHONPATH=. python scripts/run_campaign_n.py --variant n3
    PYTHONPATH=. python scripts/run_campaign_n.py --variant n6_n7
    PYTHONPATH=. python scripts/run_campaign_n.py --variant n3_n6_n7
"""

from __future__ import annotations

import argparse
import math
import os
import re
import sys
import time
import json
from typing import Iterator, List

import torch

from fant2.config import fant2_tiny
from fant2.data import TokenizedBatchStream
from fant2.inference import FANT2Generator
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training import TrainConfig, FANT2Trainer
from fant2.training.phase5_rollout import ProceduralMathStream, format_prompt
from fant2.training.campaign_n import (
    GuidedMathTextStream,
    PlainMathTextStream,
    CurriculumMathTextStream,
    run_sleep_consolidation,
)


# =====================================================================
# Configuration
# =====================================================================

OPTION_I_CKPT = "output/option_i/pretrain/final.pt"
OPTION_I_TOK  = "output/option_i/tokenizer.json"

N_STEPS    = 2500
SEQ_LEN    = 128
BATCH_SIZE = 8
TRAIN_SEED = 11
EVAL_SEED  = 9999
EVAL_MAX_VALUE = 12
EVAL_N_PROBLEMS = 1000
EVAL_MAX_NEW_TOKENS = 64

OUTPUT_GATE_INIT     = 0.1
CURVATURE_THRESHOLD  = 1.0

# N3 settings
SLEEP_CONSOLIDATE_EVERY = 100   # consolidate every 100 steps
SLEEP_MERGE_THRESHOLD   = 0.92
SLEEP_STALENESS_HORIZON = 200

# N7 settings
CURRICULUM_UCB_C = 1.5  # UCB1 exploration constant

VALID_VARIANTS = [
    "n3", "n6", "n7",
    "n3_n6", "n3_n7", "n6_n7",
    "n3_n6_n7",
]


# =====================================================================
# Helpers (reused from option_n1)
# =====================================================================

_ANSWER_TAG = re.compile(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>")
_ANY_NUM    = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_answer(text: str) -> str | None:
    m = _ANSWER_TAG.search(text)
    if m:
        return m.group(1)
    nums = _ANY_NUM.findall(text)
    if nums:
        return nums[-1]
    return None


def wilson_ci(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def evaluate_procedural_math(
    model, tokenizer, *, seed, max_value, n_problems,
    max_new_tokens, device="cpu", verbose=True,
) -> dict:
    gen = FANT2Generator(model, tokenizer, device=device)
    stream = ProceduralMathStream(seed=seed, max_value=max_value)
    it = iter(stream)
    correct = total = extracted_count = 0
    examples_log: List[dict] = []
    t0 = time.time()
    for i in range(n_problems):
        try:
            ex = next(it)
        except StopIteration:
            break
        prompt = format_prompt(ex.question)
        completion = gen.generate(
            prompt, max_new_tokens=max_new_tokens,
            greedy=True, return_full_text=False,
        )
        pred = _extract_answer(completion)
        if pred is not None:
            extracted_count += 1
        is_correct = (pred is not None) and (
            pred.lstrip("0") == ex.gold_answer.lstrip("0")
            or pred == ex.gold_answer
            or (pred == "0" and ex.gold_answer == "0")
        )
        if pred is not None and not is_correct:
            try:
                is_correct = float(pred) == float(ex.gold_answer)
            except ValueError:
                pass
        total += 1
        if is_correct:
            correct += 1
        if i < 10:
            examples_log.append({
                "q": ex.question, "gold": ex.gold_answer,
                "completion": completion[:300], "pred": pred,
                "correct": is_correct,
            })
        if verbose and (i + 1) % 100 == 0:
            lo, hi = wilson_ci(correct, total)
            print(f"    [{i+1}/{n_problems}] acc={correct}/{total}="
                  f"{correct/max(total,1):.3f} CI=[{lo:.3f}, {hi:.3f}]",
                  flush=True)
    dt = time.time() - t0
    lo, hi = wilson_ci(correct, total)
    return {
        "correct": correct, "total": total,
        "accuracy": correct / max(total, 1),
        "wilson_ci_95": [lo, hi],
        "extraction_rate": extracted_count / max(total, 1),
        "wall_seconds": dt,
        "first_examples": examples_log,
    }


# =====================================================================
# Model setup helpers
# =====================================================================

def bump_output_gates(model, value):
    n = 0
    for m in model.modules():
        if hasattr(m, "output_gate") and isinstance(m.output_gate, torch.nn.Parameter):
            with torch.no_grad():
                m.output_gate.fill_(value)
            n += 1
    return n


def bump_curvature_threshold(model, value):
    prev = float(model.memory.curvature_threshold)
    model.memory.curvature_threshold = float(value)
    return prev


def memory_diagnostics(model) -> dict:
    mem = model.memory
    fills = mem.fill_rates()
    curvs = mem.curvature_statistics()
    alpha_pl = mem.estimate_power_law_exponent("alpha")
    beta_pl = mem.estimate_power_law_exponent("beta")
    gates = []
    for m in model.modules():
        if hasattr(m, "output_gate") and isinstance(m.output_gate, torch.nn.Parameter):
            gates.append(float(m.output_gate.item()))
    return {
        "fills": fills, "curvature": curvs,
        "alpha_power_law_exp": alpha_pl,
        "beta_power_law_exp": beta_pl,
        "output_gates": gates,
        "curvature_threshold": float(mem.curvature_threshold),
    }


# =====================================================================
# Parse variant into lever flags
# =====================================================================

def parse_variant(variant: str) -> dict:
    """Parse variant string into boolean lever flags."""
    parts = set(variant.lower().split("_"))
    return {
        "use_n3": "n3" in parts,
        "use_n6": "n6" in parts,
        "use_n7": "n7" in parts,
    }


# =====================================================================
# Build text stream based on lever flags
# =====================================================================

def make_text_stream(flags: dict, seed: int, max_value: int):
    """Build the appropriate text stream based on active levers."""
    use_n6 = flags["use_n6"]
    use_n7 = flags["use_n7"]

    if use_n7:
        # N7 curriculum (optionally with N6 gold reasoning)
        return CurriculumMathTextStream(
            seed=seed,
            use_gold_reasoning=use_n6,
            c=CURRICULUM_UCB_C,
        )
    elif use_n6:
        # N6 gold reasoning only
        return GuidedMathTextStream(seed=seed, max_value=max_value)
    else:
        # N3-only or baseline — plain text stream
        return PlainMathTextStream(seed=seed, max_value=max_value)


# =====================================================================
# Build trainer
# =====================================================================

def build_trainer(tokenizer, text_stream, flags: dict, n_steps: int,
                  resume_from: str | None) -> FANT2Trainer:
    cfg = fant2_tiny()
    assert tokenizer.vocab_size <= cfg.vocab_size
    model = FANT2Model(cfg)

    train_stream = TokenizedBatchStream(
        text_stream=text_stream, tokenizer=tokenizer,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device="cpu",
    )

    train_cfg = TrainConfig(
        phase=4, n_steps=n_steps,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        muon_lr=8e-4, adam_lr=2e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.05,
        fep_kl_beta_max=0.2,
        fep_kl_anneal_steps=max(n_steps, 1),
        # NO auxiliary losses (lesson from N1)
        ortho_alpha=0.0,
        var_alpha=0.0,
        # N3 SleepGate
        sleep_consolidate_every=(
            SLEEP_CONSOLIDATE_EVERY if flags["use_n3"] else 0
        ),
        sleep_merge_threshold=SLEEP_MERGE_THRESHOLD,
        sleep_staleness_horizon=SLEEP_STALENESS_HORIZON,
        telemetry_every=2000, tikkun_every=2000, fana_every=10000,
        log_every=max(1, n_steps // 25),
        save_every=500,
        out_dir=os.path.join("output", f"campaign_{flags['variant_name']}"),
        resume_from=resume_from,
        device="cpu",
        bf16=False, grad_checkpoint=False, use_8bit_adam=False,
    )
    return FANT2Trainer(model, train_cfg, train_stream)


# =====================================================================
# Custom training loop (needed for N7 curriculum feedback)
# =====================================================================

def custom_train_loop(
    trainer: FANT2Trainer,
    text_stream,
    flags: dict,
    n_steps: int,
):
    """
    Run training with N7 curriculum feedback.

    If N7 is active, feeds training CE loss back to the curriculum
    scheduler after each step. Otherwise behaves exactly like
    trainer.train().
    """
    use_n7 = flags["use_n7"]

    if not use_n7:
        # No curriculum feedback needed — use standard training loop
        trainer.train()
        return

    # Custom loop for N7: intercept losses and feed to MAB
    c = trainer.cfg
    print(f"=== FANT 2 Phase {c.phase} training, {c.n_steps} steps ===")
    print(f"  out_dir: {c.out_dir}")
    print(f"  N7 curriculum active: UCB_c={CURRICULUM_UCB_C}")

    data_iter = iter(trainer.data_stream)
    t0 = time.time()
    running = {}
    n_running = 0

    for step in range(trainer.step + 1, trainer.step + n_steps + 1):
        trainer.step = step

        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(trainer.data_stream)
            batch = next(data_iter)

        losses = trainer.train_step(batch)

        # N7: Feed CE loss back to curriculum scheduler
        if hasattr(text_stream, 'scheduler'):
            ce_loss = losses.get("ce", losses.get("total", 0.0))
            text_stream.scheduler.update_reward(ce_loss)

        # Running averages
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + v
        n_running += 1

        # Periodic Tikkun + fana
        if step % c.tikkun_every == 0:
            n_repaired = trainer.model.tikkun_repair_all()
            if n_repaired > 0:
                print(f"  [step {step}] Tikkun repaired {n_repaired} layers")
        if step % c.fana_every == 0:
            trainer.model.fana_dropout_all(p=0.5)

        # N3: SleepGate consolidation (handled by trainer if configured)
        # (already integrated in trainer.train_step flow via trainer.train() path)
        if (c.sleep_consolidate_every > 0
                and step % c.sleep_consolidate_every == 0
                and hasattr(trainer.model, "memory")):
            run_sleep_consolidation(
                trainer.model,
                merge_threshold=c.sleep_merge_threshold,
                staleness_horizon=c.sleep_staleness_horizon,
            )

        # Periodic logging
        if step % c.log_every == 0:
            avg = {k: v / max(n_running, 1) for k, v in running.items()}
            dt = time.time() - t0
            tps = (step * c.batch_size * c.seq_len) / max(dt, 1e-6)
            parts = [f"[step {step:6d}]"]
            for k in ["ce", "fep_kl", "z_loss", "succ"]:
                if k in avg:
                    parts.append(f"{k}={avg[k]:.4f}")
            if "total" in avg:
                parts.append(f"total={avg['total']:.4f}")
            parts.append(f"({tps:.0f} tok/s)")
            # Add curriculum stats if N7
            if hasattr(text_stream, 'scheduler'):
                parts.append(f"[MAB: {text_stream.scheduler.summary()}]")
            print("  " + "  ".join(parts))
            running = {}
            n_running = 0

        # Periodic checkpoint
        if step % c.save_every == 0:
            trainer.save_checkpoint(
                os.path.join(c.out_dir, f"step_{step}.pt")
            )

    # Final save
    trainer.save_checkpoint(os.path.join(c.out_dir, "final.pt"))
    print(f"=== Training done. Total time: {(time.time() - t0):.1f}s ===")


# =====================================================================
# Main
# =====================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Campaign N experiment runner"
    )
    parser.add_argument(
        "--variant", required=True,
        choices=VALID_VARIANTS,
        help="Which lever combination to run",
    )
    args = parser.parse_args()
    variant = args.variant
    flags = parse_variant(variant)
    flags["variant_name"] = variant

    # Banner
    levers = []
    if flags["use_n3"]:
        levers.append("N3-SleepGate")
    if flags["use_n6"]:
        levers.append("N6-GoldReasoning")
    if flags["use_n7"]:
        levers.append("N7-Curriculum")
    lever_str = " + ".join(levers) if levers else "NONE"

    print("=" * 64)
    print(f" FANT 2 — Campaign N: {lever_str}")
    print(f" Base: L1.5 | Variant: {variant}")
    print("=" * 64)
    print()
    if flags["use_n3"]:
        print(f"  N3 SleepGate: consolidate every {SLEEP_CONSOLIDATE_EVERY} steps")
        print(f"    merge_threshold={SLEEP_MERGE_THRESHOLD}, "
              f"staleness_horizon={SLEEP_STALENESS_HORIZON}")
    if flags["use_n6"]:
        print(f"  N6 G2RPO-A: gold step-by-step reasoning traces")
    if flags["use_n7"]:
        print(f"  N7 SEC: UCB1 curriculum, c={CURRICULUM_UCB_C}")
        print(f"    arms: easy(5), medium(8), hard(12)")
    print(f"  eval_n = {EVAL_N_PROBLEMS} (1K for statistical power)")
    print()

    # Check prerequisite
    if not os.path.exists(OPTION_I_CKPT):
        print(f"  x Option I checkpoint not found at {OPTION_I_CKPT}")
        return 1

    out_dir = os.path.join("output", f"campaign_{variant}")
    os.makedirs(out_dir, exist_ok=True)

    tokenizer = FANT2Tokenizer.load(OPTION_I_TOK)

    # ===== Phase A: pre-ramp eval (200 samples) =====
    print("  ===== Phase A: pre-ramp eval (200 samples) =====")
    text_stream_pre = PlainMathTextStream(seed=TRAIN_SEED, max_value=EVAL_MAX_VALUE)
    pre_trainer = build_trainer(
        tokenizer, text_stream_pre, {**flags, "variant_name": variant},
        n_steps=1, resume_from=OPTION_I_CKPT,
    )
    bump_output_gates(pre_trainer.model, OUTPUT_GATE_INIT)
    bump_curvature_threshold(pre_trainer.model, CURVATURE_THRESHOLD)

    pre_diag = memory_diagnostics(pre_trainer.model)
    pre_res = evaluate_procedural_math(
        pre_trainer.model, tokenizer,
        seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=200, max_new_tokens=EVAL_MAX_NEW_TOKENS,
    )
    print(f"  pre-ramp: {pre_res['correct']}/{pre_res['total']} = "
          f"{pre_res['accuracy']:.1%}")

    # ===== Phase B: Phase 4 ramp with active levers =====
    print()
    print(f"  ===== Phase B: Phase 4 ramp ({N_STEPS} steps, {lever_str}) =====")
    text_stream = make_text_stream(flags, seed=TRAIN_SEED, max_value=EVAL_MAX_VALUE)
    trainer = build_trainer(
        tokenizer, text_stream, {**flags, "variant_name": variant},
        n_steps=N_STEPS, resume_from=OPTION_I_CKPT,
    )
    bump_output_gates(trainer.model, OUTPUT_GATE_INIT)
    bump_curvature_threshold(trainer.model, CURVATURE_THRESHOLD)

    t0 = time.time()
    train_exc = None
    try:
        custom_train_loop(trainer, text_stream, flags, N_STEPS)
    except (KeyboardInterrupt, Exception) as exc:
        train_exc = exc
        print(f"  ! training interrupted: {type(exc).__name__}: {exc}")
    dt = time.time() - t0
    steps_done = max(trainer.step, 1)
    print(f"  ramp done in {dt / 60:.1f} min ({dt / steps_done * 1000:.0f} ms/step)")

    final_ckpt = os.path.join(out_dir, "final.pt")
    if train_exc is not None:
        torch.save({
            "model": trainer.model.state_dict(),
            "step": trainer.step,
            "halted_early": True,
        }, final_ckpt)

    # ===== Phase C: post-ramp 1K eval =====
    print()
    print(f"  ===== Phase C: post-ramp 1K eval =====")
    post_diag = memory_diagnostics(trainer.model)

    # Print curriculum summary if N7
    if flags["use_n7"] and hasattr(text_stream, 'scheduler'):
        print(f"  N7 final MAB: {text_stream.scheduler.summary()}")

    post_res = evaluate_procedural_math(
        trainer.model, tokenizer,
        seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=EVAL_N_PROBLEMS, max_new_tokens=EVAL_MAX_NEW_TOKENS,
    )
    lo, hi = post_res["wilson_ci_95"]
    print(f"  post-ramp: {post_res['correct']}/{post_res['total']} = "
          f"{post_res['accuracy']:.1%} CI=[{lo:.3f}, {hi:.3f}]")

    # ===== Report =====
    print()
    print("=" * 64)
    print(f" RESULTS — Campaign {variant} ({lever_str})")
    print("=" * 64)
    print(f"  pre-ramp:  {pre_res['correct']}/{pre_res['total']} = "
          f"{pre_res['accuracy']*100:.1f}%")
    print(f"  post-ramp: {post_res['correct']}/{post_res['total']} = "
          f"{post_res['accuracy']*100:.1f}%  CI=[{lo:.3f}, {hi:.3f}]")
    print()
    print("  Comparison baselines (1K evals):")
    print(f"    L1.5 baseline:  546/1000 = 54.6%  CI=[0.515, 0.577]")
    print(f"    M4 classifier:  512/1000 = 51.2%  CI=[0.481, 0.543]")
    print(f"    N1 ortho+var:    76/1000 =  7.6%  (catastrophic)")
    print(f"    {variant:12s}: {post_res['correct']:>4d}/{post_res['total']} = "
          f"{post_res['accuracy']*100:.1f}%  CI=[{lo:.3f}, {hi:.3f}]")
    print()

    # Build config for JSON
    config = {
        "phase": 4,
        "variant": variant,
        "levers": levers,
        "output_gate_init": OUTPUT_GATE_INIT,
        "curvature_threshold": CURVATURE_THRESHOLD,
        "n_steps": N_STEPS, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
        "train_seed": TRAIN_SEED, "eval_seed": EVAL_SEED,
        "eval_max_value": EVAL_MAX_VALUE, "eval_n_problems": EVAL_N_PROBLEMS,
    }
    if flags["use_n3"]:
        config.update({
            "sleep_consolidate_every": SLEEP_CONSOLIDATE_EVERY,
            "sleep_merge_threshold": SLEEP_MERGE_THRESHOLD,
            "sleep_staleness_horizon": SLEEP_STALENESS_HORIZON,
        })
    if flags["use_n7"]:
        config["curriculum_ucb_c"] = CURRICULUM_UCB_C
        if hasattr(text_stream, 'scheduler'):
            config["curriculum_final_summary"] = text_stream.scheduler.summary()

    results = {
        "config": config,
        "pre_ramp": pre_res,
        "post_ramp": post_res,
        "pre_memory": pre_diag,
        "post_memory": post_diag,
        "ckpt": final_ckpt,
        "tokenizer": OPTION_I_TOK,
    }
    results_json = os.path.join(out_dir, "results.json")
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  results JSON: {results_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
