"""
Option M — Phase 4 Ablation Study.

M1 (Think-at-Hard gate + Phi-4 filter) dropped procedural math -9.5pp vs L1.5
despite improving the internal curvature statistics. This script isolates each
of M1's two features to identify which one caused the regression.

Usage:
    PYTHONPATH=. python scripts/option_m_ablation.py --variant gate_only
    PYTHONPATH=. python scripts/option_m_ablation.py --variant filter_only
    PYTHONPATH=. python scripts/option_m_ablation.py --variant filter_wide    # [1, 144] filter
    PYTHONPATH=. python scripts/option_m_ablation.py --variant gate_loose     # threshold 0.9
    PYTHONPATH=. python scripts/option_m_ablation.py --variant all            # runs all variants sequentially

Each variant runs 2500 Phase 4 steps from Option I, same recipe as L1.5 / M1,
and reports procedural-math accuracy vs the L1.5 61.5% baseline.

Variants:

  gate_only    — Think-at-Hard gate ON, Phi-4 filter OFF. Isolates #1's effect.
                 If this matches L1.5 or beats it, the filter was the culprit.

  filter_only  — Think-at-Hard gate OFF, Phi-4 filter [5, 80] ON.
                 Isolates #5's effect. If this matches L1.5 or beats it, the
                 gate was the culprit.

  filter_wide  — Filter expanded to [1, 144] — excludes only zero-answer
                 problems. Tests whether the filter's *narrowness* was the
                 issue, not the filtering itself.

  gate_loose   — Gate threshold raised 0.7 → 0.9. Tests whether Think-at-Hard
                 just needs a less aggressive cutoff at this model size.
"""

from __future__ import annotations

import argparse
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
# Constants
# -----------------------------------------------------------------------------

OPTION_I_CKPT = "output/option_i/pretrain/final.pt"
OPTION_I_TOK  = "output/option_i/tokenizer.json"

OUT_BASE_FMT = "output/option_m_ablation/{variant}"

N_STEPS    = 2500
SEQ_LEN    = 128
BATCH_SIZE = 8
TRAIN_SEED = 11
EVAL_SEED  = 9999
EVAL_MAX_VALUE = 12
EVAL_N_PROBLEMS = 200
EVAL_MAX_NEW_TOKENS = 64

OUTPUT_GATE_INIT    = 0.1
CURVATURE_THRESHOLD = 1.0


# -----------------------------------------------------------------------------
# Variant config
# -----------------------------------------------------------------------------

VARIANTS = {
    "gate_only": {
        "desc": "Think-at-Hard gate on, Phi-4 filter off",
        "gate_enabled": True,
        "gate_threshold": 0.7,
        "filter_enabled": False,
        "filter_min": 0,
        "filter_max": 10_000,
    },
    "filter_only": {
        "desc": "Phi-4 filter [5,80] on, gate off",
        "gate_enabled": False,
        "gate_threshold": 0.7,
        "filter_enabled": True,
        "filter_min": 5,
        "filter_max": 80,
    },
    "filter_wide": {
        "desc": "Phi-4 filter widened to [1,144], gate off",
        "gate_enabled": False,
        "gate_threshold": 0.7,
        "filter_enabled": True,
        "filter_min": 1,
        "filter_max": 144,
    },
    "gate_loose": {
        "desc": "Think-at-Hard gate at 0.9 threshold, Phi-4 filter off",
        "gate_enabled": True,
        "gate_threshold": 0.9,
        "filter_enabled": False,
        "filter_min": 0,
        "filter_max": 10_000,
    },
}


# -----------------------------------------------------------------------------
# Difficulty-filtered procedural math stream
# -----------------------------------------------------------------------------

class DifficultyFilteredMathStream:
    """Yield only problems whose integer gold answer is in [min_ans, max_ans]."""

    def __init__(self, base: ProceduralMathStream, min_ans: int, max_ans: int):
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
    """Tokenized text stream, optionally difficulty-filtered."""

    def __init__(
        self,
        seed: int,
        max_value: int,
        filter_enabled: bool,
        filter_min: int,
        filter_max: int,
    ):
        base = ProceduralMathStream(seed=seed, max_value=max_value)
        if filter_enabled:
            self.stream = DifficultyFilteredMathStream(base, filter_min, filter_max)
        else:
            self.stream = base

    def __iter__(self) -> Iterator[str]:
        for ex in self.stream:
            prompt = format_prompt(ex.question)
            answer_block = (
                f" Let me work it out. The answer is {ex.gold_answer}.\n"
                f"</think>\n"
                f"<answer>{ex.gold_answer}</answer>"
            )
            yield prompt + answer_block


def make_train_stream(
    tokenizer: FANT2Tokenizer,
    filter_enabled: bool,
    filter_min: int,
    filter_max: int,
) -> TokenizedBatchStream:
    text = ProceduralMathTextStream(
        seed=TRAIN_SEED,
        max_value=EVAL_MAX_VALUE,
        filter_enabled=filter_enabled,
        filter_min=filter_min,
        filter_max=filter_max,
    )
    return TokenizedBatchStream(
        text_stream=text, tokenizer=tokenizer,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device="cpu",
    )


# -----------------------------------------------------------------------------
# Procedural math eval (identical to M1/L1.5 — never filtered)
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
        if i < 3:
            examples_log.append({
                "q": ex.question,
                "gold": ex.gold_answer,
                "completion": completion[:200],
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
# Memory subsystem instrumentation
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
# Per-variant trainer builder
# -----------------------------------------------------------------------------

def apply_variant_flags(cfg, variant_cfg: dict) -> None:
    cfg.phase4_gate_enabled        = variant_cfg["gate_enabled"]
    cfg.phase4_gate_threshold      = variant_cfg["gate_threshold"]
    cfg.phase4_prepend_k           = 0       # legacy pooled (no Coconut in ablation)
    cfg.phase4_alignment_weight    = 0.0     # legacy MSE consistency only
    cfg.phase4_classifier_upstream = False
    cfg.phase4_classifier_mode     = "curvature"


def build_trainer(
    tokenizer,
    n_steps: int,
    resume_from: str | None,
    out_dir: str,
    variant_cfg: dict,
) -> FANT2Trainer:
    cfg = fant2_tiny()
    apply_variant_flags(cfg, variant_cfg)
    assert tokenizer.vocab_size <= cfg.vocab_size
    model = FANT2Model(cfg)
    train_stream = make_train_stream(
        tokenizer,
        filter_enabled=variant_cfg["filter_enabled"],
        filter_min=variant_cfg["filter_min"],
        filter_max=variant_cfg["filter_max"],
    )
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
        out_dir=out_dir,
        resume_from=resume_from,
        device="cpu",
        bf16=False, grad_checkpoint=False, use_8bit_adam=False,
    )
    return FANT2Trainer(model, train_cfg, train_stream)


# -----------------------------------------------------------------------------
# Run a single variant
# -----------------------------------------------------------------------------

def run_variant(variant: str, variant_cfg: dict) -> dict:
    out_base = OUT_BASE_FMT.format(variant=variant)
    out_ramp = os.path.join(out_base, "math_ramp")
    results_json = os.path.join(out_base, "results.json")
    os.makedirs(out_base, exist_ok=True)
    os.makedirs(out_ramp, exist_ok=True)

    print()
    print("=" * 64)
    print(f" Variant: {variant}")
    print(f" {variant_cfg['desc']}")
    print("=" * 64)
    print(f"  gate_enabled   = {variant_cfg['gate_enabled']}  (threshold={variant_cfg['gate_threshold']})")
    print(f"  filter_enabled = {variant_cfg['filter_enabled']}  "
          f"(range=[{variant_cfg['filter_min']}, {variant_cfg['filter_max']}])")
    print()

    tokenizer = FANT2Tokenizer.load(OPTION_I_TOK)

    # Pre-eval
    print("  ----- pre-eval -----")
    pre_trainer = build_trainer(
        tokenizer, n_steps=1, resume_from=OPTION_I_CKPT,
        out_dir=out_ramp, variant_cfg=variant_cfg,
    )
    bump_output_gates(pre_trainer.model, OUTPUT_GATE_INIT)
    bump_curvature_threshold(pre_trainer.model, CURVATURE_THRESHOLD)
    pre_res = evaluate_procedural_math(
        pre_trainer.model, tokenizer,
        seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=EVAL_N_PROBLEMS, max_new_tokens=EVAL_MAX_NEW_TOKENS,
    )
    print(f"  pre-ramp: {pre_res['correct']}/{pre_res['total']} = {pre_res['accuracy']:.1%}")

    # Phase 4 ramp
    print()
    print(f"  ----- {N_STEPS}-step Phase 4 ramp -----")
    trainer = build_trainer(
        tokenizer, n_steps=N_STEPS, resume_from=OPTION_I_CKPT,
        out_dir=out_ramp, variant_cfg=variant_cfg,
    )
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
    print(f"  ramp done/halted in {dt / 60:.1f} min ({dt / max(trainer.step, 1) * 1000:.0f} ms/step)")
    if train_exc is not None:
        torch.save({
            "model": trainer.model.state_dict(),
            "opt":   trainer.opt.state_dict(),
            "cfg":   trainer.cfg,
            "step":  trainer.step,
            "halted_early": True,
        }, os.path.join(out_ramp, "final.pt"))

    # Post-eval
    print()
    print("  ----- post-eval -----")
    post_diag = memory_diagnostics(trainer.model)
    print(f"  post-ramp fills:     {post_diag['fills']}")
    post_res = evaluate_procedural_math(
        trainer.model, tokenizer,
        seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=EVAL_N_PROBLEMS, max_new_tokens=EVAL_MAX_NEW_TOKENS,
    )
    print(f"  post-ramp: {post_res['correct']}/{post_res['total']} = {post_res['accuracy']:.1%}")

    results = {
        "variant": variant,
        "variant_cfg": variant_cfg,
        "pre_ramp": pre_res,
        "post_ramp": post_res,
        "post_memory": post_diag,
        "ramp_wall_seconds": dt,
    }
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print()
    print(f"  VARIANT {variant} SUMMARY")
    print(f"    accuracy: {post_res['accuracy']*100:.1f}%  "
          f"(L1.5 baseline: 61.5%, M1: 52.0%)")
    print(f"    delta vs L1.5: {(post_res['accuracy']-0.615)*100:+.1f}pp")
    print(f"    delta vs M1  : {(post_res['accuracy']-0.520)*100:+.1f}pp")
    print(f"    results JSON: {results_json}")
    return results


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()) + ["all"],
        required=True,
        help="Which ablation variant to run",
    )
    args = parser.parse_args()

    if not os.path.exists(OPTION_I_CKPT):
        print(f"  ✗ Option I checkpoint not found at {OPTION_I_CKPT}")
        return 1
    if not os.path.exists(OPTION_I_TOK):
        print(f"  ✗ Option I tokenizer not found at {OPTION_I_TOK}")
        return 1

    if args.variant == "all":
        all_results = {}
        for v, vcfg in VARIANTS.items():
            all_results[v] = run_variant(v, vcfg)
        print()
        print("=" * 64)
        print(" ABLATION SUMMARY")
        print("=" * 64)
        print(f"  L1.5 baseline:          123/200 = 61.5%")
        print(f"  M1 (gate+filter):       104/200 = 52.0%")
        for v, r in all_results.items():
            acc = r["post_ramp"]["accuracy"] * 100
            correct = r["post_ramp"]["correct"]
            total = r["post_ramp"]["total"]
            print(f"  {v:<15}: {correct}/{total} = {acc:.1f}%")
    else:
        run_variant(args.variant, VARIANTS[args.variant])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
