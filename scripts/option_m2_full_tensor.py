"""
Option M2 — M1 + Coconut full-tensor feedback + SpiralThinker alignment.

Builds on M1 (Think-at-Hard gate + Phi-4 difficulty filter) by turning on two
additional levers from the literature scan:

  #2 Coconut full-tensor feedback (arxiv:2412.06769)
     Instead of collapsing pass-1's final_hidden to a single pooled [B, D]
     mean (the legacy L1.5 path), M2 prepends the last K positions as a
     [B, K, D] sequence. This preserves per-position structure so pass 2 can
     attend over pass-1's *thinking* rather than one averaged vector. K=8
     gives 8 virtual reasoning tokens at the start of every pass-2 sample.

  #6 SpiralThinker progressive alignment (arxiv:2511.08983)
     In addition to the legacy MSE consistency penalty (h1 ↔ h2), add a
     cosine-alignment penalty `(1 - cos(h1_detached, h2)).mean()`. This is
     SpiralThinker's "progressive alignment" anchor that keeps pass 2 from
     drifting off the pass-1 manifold while still allowing refinement on
     the hard tokens. Weight 0.1 matches SpiralThinker's reported best for
     a single-refinement loop on small models.

Everything else matches M1:
  - Think-at-Hard gate on (threshold 0.7)
  - Phi-4 difficulty filter [5, 80]
  - L1.5 curvature_threshold = 1.0
  - 2500 steps, fant2_tiny, seed 11
  - Start from Option I checkpoint

Baselines to beat:
    K     (phase 2 only):             80.0% procedural
    L1    (phase 4, β=0 bug):         54.5% procedural
    L1.5  (phase 4, β fixed):         61.5% procedural
    M1    (L1.5 + #1 gate + #5 filt): <M1-accuracy>   ← M2 target

Run:
    PYTHONPATH=. python scripts/option_m2_full_tensor.py
"""

from __future__ import annotations

import os
import re
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


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OPTION_I_CKPT = "output/option_i/pretrain/final.pt"
OPTION_I_TOK  = "output/option_i/tokenizer.json"

OUT_BASE = "output/option_m2"
OUT_RAMP = os.path.join(OUT_BASE, "math_ramp")
RESULTS_JSON = os.path.join(OUT_BASE, "results.json")

N_STEPS    = 2500
SEQ_LEN    = 128
BATCH_SIZE = 8
TRAIN_SEED = 11
EVAL_SEED  = 9999
EVAL_MAX_VALUE = 12
EVAL_N_PROBLEMS = 200
EVAL_MAX_NEW_TOKENS = 64

OUTPUT_GATE_INIT     = 0.1
CURVATURE_THRESHOLD  = 1.0

# Option M feature flags
PHASE4_GATE_ENABLED   = True
PHASE4_GATE_THRESHOLD = 0.7
PHASE4_PREPEND_K      = 8     # M2: Coconut full-tensor feedback, 8 virtual reasoning tokens
PHASE4_ALIGNMENT_WT   = 0.1   # M2: SpiralThinker progressive alignment weight
PHASE4_CLASS_UPSTREAM = False # still legacy post-RMSNorm; M3 turns this on
PHASE4_CLASS_MODE     = "curvature"  # still legacy L2-norm; M3 turns this to ce_surprise

# Phi-4 difficulty filter bounds (#5 — kept from M1)
DIFFICULTY_MIN_ANS = 5
DIFFICULTY_MAX_ANS = 80


# -----------------------------------------------------------------------------
# Difficulty-filtered procedural math stream (#5 Phi-4 curriculum)
# -----------------------------------------------------------------------------

class DifficultyFilteredMathStream:
    """Wrap ProceduralMathStream, yield only problems in [min_ans, max_ans]."""

    def __init__(
        self,
        base: ProceduralMathStream,
        min_ans: int = DIFFICULTY_MIN_ANS,
        max_ans: int = DIFFICULTY_MAX_ANS,
    ):
        self.base = base
        self.min_ans = min_ans
        self.max_ans = max_ans

    def __iter__(self) -> Iterator:
        for ex in self.base:
            try:
                v = int(ex.gold_answer)
            except (ValueError, TypeError):
                continue
            if self.min_ans <= v <= self.max_ans:
                yield ex


class ProceduralMathTextStream:
    """Tokenized text stream for the Phase 4 trainer, with difficulty filter."""

    def __init__(self, seed: int, max_value: int = 12):
        base = ProceduralMathStream(seed=seed, max_value=max_value)
        self.stream = DifficultyFilteredMathStream(base)

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


# -----------------------------------------------------------------------------
# Procedural math eval — same as L1.5 / M1
# -----------------------------------------------------------------------------

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
        if i < 5:
            examples_log.append({
                "q": ex.question,
                "gold": ex.gold_answer,
                "completion": completion[:300],
                "pred": pred,
                "correct": is_correct,
            })
        if verbose and (i + 1) % 25 == 0:
            print(f"    [proc-math {i+1}/{n_problems}] acc={correct/max(total,1):.3f} "
                  f"({correct}/{total}) extracted={extracted_count}")
    dt = time.time() - t0
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
        "extraction_rate": extracted_count / max(total, 1),
        "wall_seconds": dt,
        "first_examples": examples_log,
    }


# -----------------------------------------------------------------------------
# Memory subsystem instrumentation (same as M1)
# -----------------------------------------------------------------------------

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


# -----------------------------------------------------------------------------
# Trainer construction — with Option M config flags applied to the model cfg
# -----------------------------------------------------------------------------

def apply_m_flags(cfg) -> None:
    """Apply the Option M2 phase-4 feature flags to a FANT2Config."""
    cfg.phase4_gate_enabled        = PHASE4_GATE_ENABLED
    cfg.phase4_gate_threshold      = PHASE4_GATE_THRESHOLD
    cfg.phase4_prepend_k           = PHASE4_PREPEND_K
    cfg.phase4_alignment_weight    = PHASE4_ALIGNMENT_WT
    cfg.phase4_classifier_upstream = PHASE4_CLASS_UPSTREAM
    cfg.phase4_classifier_mode     = PHASE4_CLASS_MODE


def build_trainer(tokenizer, n_steps: int, resume_from: str | None) -> FANT2Trainer:
    cfg = fant2_tiny()
    apply_m_flags(cfg)
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
        telemetry_every=2000, tikkun_every=2000, fana_every=10000,
        log_every=max(1, n_steps // 25),
        save_every=500,
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
    print(" FANT 2 — Option M2: Coconut feedback + SpiralThinker alignment")
    print(" Phase 4 ramp on M1 recipe + #2 #6 from literature scan")
    print("=" * 64)
    print()
    print(f"  phase4_gate_enabled       = {PHASE4_GATE_ENABLED}")
    print(f"  phase4_gate_threshold     = {PHASE4_GATE_THRESHOLD}")
    print(f"  phase4_prepend_k          = {PHASE4_PREPEND_K}  (Coconut full-tensor, M2)")
    print(f"  phase4_alignment_weight   = {PHASE4_ALIGNMENT_WT}  (SpiralThinker, M2)")
    print(f"  phase4_classifier_upstream= {PHASE4_CLASS_UPSTREAM} (M3 turns this on)")
    print(f"  phase4_classifier_mode    = {PHASE4_CLASS_MODE}")
    print(f"  curvature_threshold       = {CURVATURE_THRESHOLD}")
    print(f"  difficulty band           = [{DIFFICULTY_MIN_ANS}, {DIFFICULTY_MAX_ANS}]")
    print()

    if not os.path.exists(OPTION_I_CKPT):
        print(f"  ✗ Option I checkpoint not found at {OPTION_I_CKPT}")
        return 1
    if not os.path.exists(OPTION_I_TOK):
        print(f"  ✗ Option I tokenizer not found at {OPTION_I_TOK}")
        return 1

    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(OUT_RAMP, exist_ok=True)

    print("  loading Option I tokenizer + checkpoint")
    tokenizer = FANT2Tokenizer.load(OPTION_I_TOK)

    # ---------- Phase A: pre-ramp eval ----------
    print()
    print("  ===== Phase A: pre-ramp procedural-math eval =====")
    pre_trainer = build_trainer(tokenizer, n_steps=1, resume_from=OPTION_I_CKPT)
    print(f"  loaded at step {pre_trainer.step}")

    n_gates = bump_output_gates(pre_trainer.model, OUTPUT_GATE_INIT)
    prev_thresh = bump_curvature_threshold(pre_trainer.model, CURVATURE_THRESHOLD)
    print(f"  bumped {n_gates} output_gate(s) to {OUTPUT_GATE_INIT}")
    print(f"  bumped curvature_threshold {prev_thresh} -> {CURVATURE_THRESHOLD}")

    pre_diag = memory_diagnostics(pre_trainer.model)
    print(f"  pre-ramp fills:   {pre_diag['fills']}")

    pre_res = evaluate_procedural_math(
        pre_trainer.model, tokenizer,
        seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=EVAL_N_PROBLEMS, max_new_tokens=EVAL_MAX_NEW_TOKENS,
    )
    print(f"  pre-ramp: {pre_res['correct']}/{pre_res['total']} = {pre_res['accuracy']:.1%}")

    # ---------- Phase B: Phase 4 ramp ----------
    print()
    print(f"  ===== Phase B: Phase 4 ramp ({N_STEPS} steps) =====")
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
    print(f"  ramp done/halted in {dt / 60:.1f} min ({dt / steps_done * 1000:.0f} ms/step)")

    final_ckpt = os.path.join(OUT_RAMP, "final.pt")
    if train_exc is not None:
        torch.save({
            "model": trainer.model.state_dict(),
            "opt":   trainer.opt.state_dict(),
            "cfg":   trainer.cfg,
            "step":  trainer.step,
            "halted_early": True,
        }, final_ckpt)

    # ---------- Phase C: post-ramp diagnostics + eval ----------
    print()
    print("  ===== Phase C: post-ramp diagnostics + eval =====")
    post_diag = memory_diagnostics(trainer.model)
    print(f"  post-ramp fills:     {post_diag['fills']}")
    print(f"  post-ramp curvature: {post_diag['curvature']}")
    print(f"  alpha power-law exp: {post_diag['alpha_power_law_exp']:.3f} (target ~1.305)")
    print(f"  beta  power-law exp: {post_diag['beta_power_law_exp']:.3f}")

    post_res = evaluate_procedural_math(
        trainer.model, tokenizer,
        seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=EVAL_N_PROBLEMS, max_new_tokens=EVAL_MAX_NEW_TOKENS,
    )
    print(f"  post-ramp: {post_res['correct']}/{post_res['total']} = {post_res['accuracy']:.1%}")

    # ---------- Report ----------
    print()
    print("=" * 64)
    print(" RESULTS — Option M2 (Coconut full-tensor + SpiralThinker alignment)")
    print("=" * 64)
    print(f"  pre-ramp:  {pre_res['correct']}/{pre_res['total']} = {pre_res['accuracy']*100:.1f}%")
    print(f"  post-ramp: {post_res['correct']}/{post_res['total']} = {post_res['accuracy']*100:.1f}%")
    print(f"  delta: {(post_res['accuracy']-pre_res['accuracy'])*100:+.1f}pp")
    print()
    print("  Baseline comparison:")
    print(f"    K   (phase 2):      160/200 = 80.0%  (α=0.0, β=0.0)")
    print(f"    L1  (phase 4):      109/200 = 54.5%  (α=1.0, β=0.0)")
    print(f"    L1.5(phase 4):      123/200 = 61.5%  (α=1.0, β=1.0)")
    print(f"    M2  (phase 4+#1#2#5#6): {post_res['correct']}/{post_res['total']} = "
          f"{post_res['accuracy']*100:.1f}%  (α={post_diag['fills']['alpha_fill']:.1f}, "
          f"β={post_diag['fills']['beta_fill']:.1f})")
    print()

    results = {
        "config": {
            "phase": 4,
            "output_gate_init": OUTPUT_GATE_INIT,
            "curvature_threshold": CURVATURE_THRESHOLD,
            "phase4_gate_enabled": PHASE4_GATE_ENABLED,
            "phase4_gate_threshold": PHASE4_GATE_THRESHOLD,
            "phase4_prepend_k": PHASE4_PREPEND_K,
            "phase4_alignment_weight": PHASE4_ALIGNMENT_WT,
            "phase4_classifier_upstream": PHASE4_CLASS_UPSTREAM,
            "phase4_classifier_mode": PHASE4_CLASS_MODE,
            "difficulty_min_ans": DIFFICULTY_MIN_ANS,
            "difficulty_max_ans": DIFFICULTY_MAX_ANS,
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
