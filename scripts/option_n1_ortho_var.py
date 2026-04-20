"""
Option N1 — Expert orthogonality + router variance losses.

Campaign N lever #1: arXiv:2505.22323 (Advancing Expert Specialization)

Base: L1.5 (Phase 4, no gate/filter/Coconut/SpiralThinker)
Change: +expert orthogonality loss +router variance loss
Expected: Better expert specialization → improved routing → higher accuracy

Two new loss terms added to fep_unified_loss:
  - L_ortho: ||A_i^T A_j||_F for all expert pairs within each megapool
    Pushes Kronecker A-factors toward mutual orthogonality, forcing each
    expert to process genuinely different token types.
  - L_var: -Var(router_logits) — negative variance = maximize decisiveness
    Pushes the router toward more confident selections, reducing the
    "soft overlap" where multiple experts get similar weights.

Hyperparameters (from paper + scale adaptation):
  ortho_alpha = 0.01  (paper default)
  var_alpha   = 0.01  (paper default)

Run:
    PYTHONPATH=. python scripts/option_n1_ortho_var.py
"""

from __future__ import annotations

import os
import re
import sys
import time
import json
import math
from typing import Iterator, List

import torch

from fant2.config import fant2_tiny
from fant2.data import TokenizedBatchStream
from fant2.inference import FANT2Generator
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training import TrainConfig, FANT2Trainer
from fant2.training.phase5_rollout import ProceduralMathStream, format_prompt


# Configuration — L1.5 base, NO classifier changes, just N1 ortho+var

OPTION_I_CKPT = "output/option_i/pretrain/final.pt"
OPTION_I_TOK  = "output/option_i/tokenizer.json"

OUT_BASE = "output/option_n1"
OUT_RAMP = os.path.join(OUT_BASE, "math_ramp")
RESULTS_JSON = os.path.join(OUT_BASE, "results.json")

N_STEPS    = 2500
SEQ_LEN    = 128
BATCH_SIZE = 8
TRAIN_SEED = 11
EVAL_SEED  = 9999
EVAL_MAX_VALUE = 12
EVAL_N_PROBLEMS = 1000   # 1K for statistical power
EVAL_MAX_NEW_TOKENS = 64

OUTPUT_GATE_INIT     = 0.1
CURVATURE_THRESHOLD  = 1.0

# N1 lever settings (paper defaults, no classifier fixes)
ORTHO_ALPHA = 0.01   # expert orthogonality loss weight
VAR_ALPHA   = 0.01   # router variance loss weight


# Procedural math text stream — same as L1.5 (uses fixed generator)

class ProceduralMathTextStream:
    def __init__(self, seed: int, max_value: int = 12):
        self.stream = ProceduralMathStream(seed=seed, max_value=max_value)

    def __iter__(self) -> Iterator[str]:
        for ex in self.stream:
            prompt = format_prompt(ex.question)
            answer_block = (
                f" Let me work it out. The answer is {ex.gold_answer}.\n"
                f"</think>\n"
                f"<answer>{ex.gold_answer}</answer>"
            )
            yield prompt + answer_block


def make_train_stream(tokenizer: FANT2Tokenizer) -> TokenizedBatchStream:
    text = ProceduralMathTextStream(seed=TRAIN_SEED, max_value=EVAL_MAX_VALUE)
    return TokenizedBatchStream(
        text_stream=text, tokenizer=tokenizer,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device="cpu",
    )


# Procedural math eval (1K with Wilson CI)

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
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def evaluate_procedural_math(
    model, tokenizer, *, seed: int, max_value: int, n_problems: int,
    max_new_tokens: int, device: str = "cpu", verbose: bool = True,
) -> dict:
    gen = FANT2Generator(model, tokenizer, device=device)
    stream = ProceduralMathStream(seed=seed, max_value=max_value)
    it = iter(stream)
    correct = 0
    total = 0
    extracted_count = 0
    examples_log: List[dict] = []
    t0 = time.time()
    for i in range(n_problems):
        try:
            ex = next(it)
        except StopIteration:
            break
        prompt = format_prompt(ex.question)
        completion = gen.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            greedy=True,
            return_full_text=False,
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
                "q": ex.question,
                "gold": ex.gold_answer,
                "completion": completion[:300],
                "pred": pred,
                "correct": is_correct,
            })
        if verbose and (i + 1) % 100 == 0:
            lo, hi = wilson_ci(correct, total)
            print(f"    [{i+1}/{n_problems}] acc={correct}/{total}={correct/max(total,1):.3f} "
                  f"CI=[{lo:.3f}, {hi:.3f}]", flush=True)
    dt = time.time() - t0
    lo, hi = wilson_ci(correct, total)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
        "wilson_ci_95": [lo, hi],
        "extraction_rate": extracted_count / max(total, 1),
        "wall_seconds": dt,
        "first_examples": examples_log,
    }


# Memory subsystem instrumentation

def bump_output_gates(model, value: float) -> int:
    n_touched = 0
    for module in model.modules():
        if hasattr(module, "output_gate") and isinstance(
            module.output_gate, torch.nn.Parameter
        ):
            with torch.no_grad():
                module.output_gate.fill_(value)
            n_touched += 1
    return n_touched


def bump_curvature_threshold(model, value: float) -> float:
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
    for module in model.modules():
        if hasattr(module, "output_gate") and isinstance(
            module.output_gate, torch.nn.Parameter
        ):
            gates.append(float(module.output_gate.item()))

    return {
        "fills": fills,
        "curvature": curvs,
        "alpha_power_law_exp": alpha_pl,
        "beta_power_law_exp": beta_pl,
        "output_gates": gates,
        "curvature_threshold": float(mem.curvature_threshold),
    }


# Trainer construction — L1.5 base + N1 ortho+var losses

def build_trainer(tokenizer, n_steps: int, resume_from: str | None) -> FANT2Trainer:
    cfg = fant2_tiny()
    # NO classifier changes — pure L1.5 base
    assert tokenizer.vocab_size <= cfg.vocab_size
    model = FANT2Model(cfg)
    train_stream = make_train_stream(tokenizer)
    train_cfg = TrainConfig(
        phase=4, n_steps=n_steps,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        muon_lr=8e-4, adam_lr=2e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.05,
        fep_kl_beta_max=0.2,
        fep_kl_anneal_steps=max(n_steps, 1),
        # Campaign N1
        ortho_alpha=ORTHO_ALPHA,
        var_alpha=VAR_ALPHA,
        telemetry_every=2000, tikkun_every=2000, fana_every=10000,
        log_every=max(1, n_steps // 25),
        save_every=500,
        out_dir=OUT_RAMP,
        resume_from=resume_from,
        device="cpu",
        bf16=False, grad_checkpoint=False, use_8bit_adam=False,
    )
    return FANT2Trainer(model, train_cfg, train_stream)


# Main

def main() -> int:
    print("=" * 64)
    print(" FANT 2 — Campaign N1: Expert Orthogonality + Router Variance")
    print(" Base: L1.5 | Lever: arXiv:2505.22323")
    print("=" * 64)
    print()
    print(f"  ortho_alpha = {ORTHO_ALPHA}  (expert orthogonality loss weight)")
    print(f"  var_alpha   = {VAR_ALPHA}  (router variance loss weight)")
    print(f"  base        = L1.5 (no classifier changes)")
    print(f"  eval_n      = {EVAL_N_PROBLEMS} (1K for statistical power)")
    print()

    if not os.path.exists(OPTION_I_CKPT):
        print(f"  x Option I checkpoint not found at {OPTION_I_CKPT}")
        return 1

    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(OUT_RAMP, exist_ok=True)

    tokenizer = FANT2Tokenizer.load(OPTION_I_TOK)

    # ---------- Phase A: pre-ramp eval (200 for speed) ----------
    print("  ===== Phase A: pre-ramp eval (200 samples) =====")
    pre_trainer = build_trainer(tokenizer, n_steps=1, resume_from=OPTION_I_CKPT)
    bump_output_gates(pre_trainer.model, OUTPUT_GATE_INIT)
    bump_curvature_threshold(pre_trainer.model, CURVATURE_THRESHOLD)

    pre_diag = memory_diagnostics(pre_trainer.model)
    pre_res = evaluate_procedural_math(
        pre_trainer.model, tokenizer,
        seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=200, max_new_tokens=EVAL_MAX_NEW_TOKENS,
    )
    print(f"  pre-ramp: {pre_res['correct']}/{pre_res['total']} = {pre_res['accuracy']:.1%}")

    # ---------- Phase B: Phase 4 ramp with N1 losses ----------
    print()
    print(f"  ===== Phase B: Phase 4 ramp ({N_STEPS} steps, N1 active) =====")
    trainer = build_trainer(tokenizer, n_steps=N_STEPS, resume_from=OPTION_I_CKPT)
    bump_output_gates(trainer.model, OUTPUT_GATE_INIT)
    bump_curvature_threshold(trainer.model, CURVATURE_THRESHOLD)

    t0 = time.time()
    train_exc = None
    try:
        trainer.train()
    except (KeyboardInterrupt, Exception) as exc:
        train_exc = exc
        print(f"  ! training interrupted: {type(exc).__name__}: {exc}")
    dt = time.time() - t0
    steps_done = max(trainer.step, 1)
    print(f"  ramp done in {dt / 60:.1f} min ({dt / steps_done * 1000:.0f} ms/step)")

    final_ckpt = os.path.join(OUT_RAMP, "final.pt")
    if train_exc is not None:
        torch.save({
            "model": trainer.model.state_dict(),
            "step":  trainer.step,
            "halted_early": True,
        }, final_ckpt)

    # ---------- Phase C: post-ramp 1K eval ----------
    print()
    print(f"  ===== Phase C: post-ramp 1K eval =====")
    post_diag = memory_diagnostics(trainer.model)
    print(f"  post-ramp curvature: alpha_pl={post_diag['alpha_power_law_exp']:.3f}, "
          f"beta_pl={post_diag['beta_power_law_exp']:.3f}")

    post_res = evaluate_procedural_math(
        trainer.model, tokenizer,
        seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=EVAL_N_PROBLEMS, max_new_tokens=EVAL_MAX_NEW_TOKENS,
    )
    lo, hi = post_res["wilson_ci_95"]
    print(f"  post-ramp: {post_res['correct']}/{post_res['total']} = {post_res['accuracy']:.1%} "
          f"CI=[{lo:.3f}, {hi:.3f}]")

    # ---------- Report ----------
    print()
    print("=" * 64)
    print(" RESULTS — Campaign N1 (L1.5 + ortho + variance)")
    print("=" * 64)
    print(f"  pre-ramp:  {pre_res['correct']}/{pre_res['total']} = {pre_res['accuracy']*100:.1f}%")
    print(f"  post-ramp: {post_res['correct']}/{post_res['total']} = {post_res['accuracy']*100:.1f}%  "
          f"CI=[{lo:.3f}, {hi:.3f}]")
    print()
    print("  Comparison baselines (1K evals):")
    print(f"    L1.5 baseline:  546/1000 = 54.6%  CI=[0.515, 0.577]")
    print(f"    M4 classifier:  512/1000 = 51.2%  CI=[0.481, 0.543]")
    print(f"    M4-EBM energy:   64/1000 =  6.4%  (catastrophic)")
    print(f"    N1 ortho+var:   {post_res['correct']}/{post_res['total']} = "
          f"{post_res['accuracy']*100:.1f}%  CI=[{lo:.3f}, {hi:.3f}]")
    print()

    results = {
        "config": {
            "phase": 4,
            "variant": "n1_ortho_var",
            "ortho_alpha": ORTHO_ALPHA,
            "var_alpha": VAR_ALPHA,
            "output_gate_init": OUTPUT_GATE_INIT,
            "curvature_threshold": CURVATURE_THRESHOLD,
            "n_steps": N_STEPS, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_seed": TRAIN_SEED, "eval_seed": EVAL_SEED,
            "eval_max_value": EVAL_MAX_VALUE, "eval_n_problems": EVAL_N_PROBLEMS,
        },
        "pre_ramp": pre_res,
        "post_ramp": post_res,
        "pre_memory": pre_diag,
        "post_memory": post_diag,
        "ckpt": final_ckpt,
        "tokenizer": OPTION_I_TOK,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  results JSON: {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
