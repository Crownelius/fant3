"""
Evaluate a FANT 2 checkpoint on 1000 procedural-math problems.

200-sample evals have ~6.5% of the statistical power needed to detect
3-5pp effects. This script runs 1000 samples for reliable measurement.
Wilson 95% CI width drops from ~14pp to ~6pp, enabling real comparisons.

Run:
    PYTHONPATH=. python scripts/eval_1k.py --ckpt output/option_m4/math_ramp/final.pt
    PYTHONPATH=. python scripts/eval_1k.py --ckpt output/option_l1_5/math_ramp/final.pt
"""

from __future__ import annotations

import argparse
import math
import os
import re
import time
import json
from typing import List

import torch

from fant2.config import fant2_tiny
from fant2.inference import FANT2Generator
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training.phase5_rollout import ProceduralMathStream, format_prompt


TOKENIZER_PATH = "output/option_i/tokenizer.json"

# Eval config
EVAL_SEED = 9999
EVAL_MAX_VALUE = 12
EVAL_N_PROBLEMS = 1000
EVAL_MAX_NEW_TOKENS = 64

# Memory init
OUTPUT_GATE_INIT = 0.1
CURVATURE_THRESHOLD = 1.0

_ANSWER_TAG = re.compile(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>")
_ANY_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_answer(text: str) -> str | None:
    m = _ANSWER_TAG.search(text)
    if m:
        return m.group(1)
    nums = _ANY_NUM.findall(text)
    if nums:
        return nums[-1]
    return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0, centre - spread), min(1, centre + spread))


def bump_output_gates(model, value: float) -> int:
    n = 0
    for module in model.modules():
        if hasattr(module, "output_gate") and isinstance(
            module.output_gate, torch.nn.Parameter
        ):
            with torch.no_grad():
                module.output_gate.fill_(value)
            n += 1
    return n


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


def load_model(ckpt_path: str, cfg_overrides: dict | None = None):
    """Load a FANT2 model from checkpoint."""
    tokenizer = FANT2Tokenizer.load(TOKENIZER_PATH)
    cfg = fant2_tiny()
    # Apply any config overrides (e.g., M4's classifier flags)
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
    model = FANT2Model(cfg)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    bump_output_gates(model, OUTPUT_GATE_INIT)
    bump_curvature_threshold(model, CURVATURE_THRESHOLD)
    return model, tokenizer


def evaluate(model, tokenizer, n_problems: int = EVAL_N_PROBLEMS) -> dict:
    gen = FANT2Generator(model, tokenizer, device="cpu")
    stream = ProceduralMathStream(seed=EVAL_SEED, max_value=EVAL_MAX_VALUE)
    it = iter(stream)
    correct = 0
    total = 0
    extracted = 0
    examples: List[dict] = []
    t0 = time.time()

    for i in range(n_problems):
        try:
            ex = next(it)
        except StopIteration:
            break
        prompt = format_prompt(ex.question)
        completion = gen.generate(
            prompt, max_new_tokens=EVAL_MAX_NEW_TOKENS,
            greedy=True, return_full_text=False,
        )
        pred = _extract_answer(completion)
        if pred is not None:
            extracted += 1
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
            examples.append({
                "q": ex.question, "gold": ex.gold_answer,
                "completion": completion[:300], "pred": pred,
                "correct": is_correct,
            })
        if (i + 1) % 100 == 0:
            ci = wilson_ci(correct, total)
            print(f"  [{i+1:4d}/{n_problems}] acc={correct/total:.3f} "
                  f"({correct}/{total}) CI=[{ci[0]:.3f}, {ci[1]:.3f}] "
                  f"extracted={extracted}")

    dt = time.time() - t0
    ci = wilson_ci(correct, total)
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
        "wilson_ci_95": list(ci),
        "extraction_rate": extracted / max(total, 1),
        "wall_seconds": dt,
        "first_examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint")
    parser.add_argument("--n", type=int, default=EVAL_N_PROBLEMS)
    parser.add_argument("--label", default=None, help="Run label for output")
    parser.add_argument("--upstream", action="store_true",
                        help="Set phase4_classifier_upstream=True")
    parser.add_argument("--ce-surprise", action="store_true",
                        help="Set phase4_classifier_mode='ce_surprise'")
    args = parser.parse_args()

    label = args.label or os.path.basename(os.path.dirname(os.path.dirname(args.ckpt)))
    print(f"{'='*64}")
    print(f"  FANT 2 — 1K Eval: {label}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Problems: {args.n}")
    print(f"{'='*64}")

    overrides = {}
    if args.upstream:
        overrides["phase4_classifier_upstream"] = True
    if args.ce_surprise:
        overrides["phase4_classifier_mode"] = "ce_surprise"

    model, tokenizer = load_model(args.ckpt, overrides or None)
    diag = memory_diagnostics(model)
    print(f"  fills: {diag['fills']}")
    print(f"  alpha PLExp: {diag['alpha_power_law_exp']:.3f}")
    print(f"  beta  PLExp: {diag['beta_power_law_exp']:.3f}")
    print(f"  output gates: {diag['output_gates']}")
    print()

    results = evaluate(model, tokenizer, n_problems=args.n)
    ci = results["wilson_ci_95"]
    print()
    print(f"{'='*64}")
    print(f"  RESULT: {results['correct']}/{results['total']} = "
          f"{results['accuracy']*100:.1f}%")
    print(f"  Wilson 95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
    print(f"  CI width: {(ci[1]-ci[0])*100:.1f}pp")
    print(f"  Wall time: {results['wall_seconds']:.0f}s")
    print(f"{'='*64}")

    out_path = f"output/eval_1k_{label}.json"
    results["label"] = label
    results["checkpoint"] = args.ckpt
    results["memory"] = diag
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
