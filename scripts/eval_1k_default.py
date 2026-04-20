"""
Adapter of scripts/eval_1k.py that uses the `default` preset (84.8M stored)
instead of the `tiny` preset (5M). Everything else is identical — procedural
math stream, 1K problems, Wilson CI, greedy decoding with <answer>...</answer>
extraction.

Use this to evaluate checkpoints trained with `--scale default`.

Run:
    PYTHONPATH=. python scripts/eval_1k_default.py \
        --ckpt output/overnight_opus46/step_2000.pt --n 500
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

from fant2.config import fant2_default
from fant2.inference import FANT2Generator
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training.phase5_rollout import ProceduralMathStream, format_prompt


TOKENIZER_PATH = "output/option_i/tokenizer.json"

EVAL_SEED = 9999
EVAL_MAX_VALUE = 12
EVAL_N_PROBLEMS = 1000
EVAL_MAX_NEW_TOKENS = 64

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
    try:
        curvs = mem.curvature_statistics()
        alpha_pl = mem.estimate_power_law_exponent("alpha")
        beta_pl = mem.estimate_power_law_exponent("beta")
    except Exception:
        curvs = {}
        alpha_pl = 0.0
        beta_pl = 0.0
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
        "output_gate_count": len(gates),
        "output_gate_mean": sum(gates) / max(len(gates), 1),
        "curvature_threshold": float(mem.curvature_threshold),
    }


def load_model(ckpt_path: str, cfg_overrides: dict | None = None):
    """Load a FANT2 model from checkpoint, at `default` scale (84.8M stored)."""
    tokenizer = FANT2Tokenizer.load(TOKENIZER_PATH)
    cfg = fant2_default()
    if cfg_overrides:
        for k, v in cfg_overrides.items():
            setattr(cfg, k, v)
    model = FANT2Model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model built: stored={n_params/1e6:.1f}M")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (first: {missing[:2]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (first: {unexpected[:2]})")
    step = ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"
    print(f"  Loaded step: {step}")
    model.eval()
    bump_output_gates(model, OUTPUT_GATE_INIT)
    bump_curvature_threshold(model, CURVATURE_THRESHOLD)
    return model, tokenizer


def evaluate(model, tokenizer, n_problems: int, device: str = "cpu", log_every: int = 50) -> dict:
    gen = FANT2Generator(model, tokenizer, device=device)
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

        if i < 8:
            examples.append({
                "q": ex.question, "gold": ex.gold_answer,
                "completion": completion[:300], "pred": pred,
                "correct": is_correct,
            })
        if (i + 1) % log_every == 0 or (i + 1) == n_problems or i < 5:
            ci = wilson_ci(correct, total)
            elapsed = time.time() - t0
            rate = total / max(elapsed, 0.001)
            eta = (n_problems - total) / max(rate, 0.001)
            print(f"  [{i+1:4d}/{n_problems}] acc={correct/total:.3f} "
                  f"({correct}/{total}) CI=[{ci[0]:.3f}, {ci[1]:.3f}] "
                  f"extracted={extracted} ({rate:.2f}/s, ETA {eta/60:.1f}m)",
                  flush=True)

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
    parser.add_argument("--label", default=None)
    parser.add_argument("--upstream", action="store_true")
    parser.add_argument("--save-json", default=None)
    parser.add_argument("--device", default="cpu",
                        help="cpu or cuda (cuda needs ~300MB free VRAM)")
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    label = args.label or os.path.basename(os.path.dirname(args.ckpt))
    print(f"{'='*64}")
    print(f"  FANT 2 — 1K Eval (default 84.8M): {label}")
    print(f"  Checkpoint: {args.ckpt}")
    print(f"  Problems: {args.n}")
    print(f"{'='*64}")

    overrides = {}
    if args.upstream:
        overrides["phase4_classifier_upstream"] = True

    model, tokenizer = load_model(args.ckpt, overrides or None)
    diag = memory_diagnostics(model)
    print(f"  fills: {diag['fills']}")
    print(f"  output_gate: count={diag['output_gate_count']}  mean={diag['output_gate_mean']:.3f}")
    print(f"  curvature_threshold: {diag['curvature_threshold']}")

    if args.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda", dtype=torch.bfloat16)
        print(f"  device: cuda (bf16)")
    else:
        print(f"  device: cpu (fp32)")
    print(flush=True)

    result = evaluate(model, tokenizer, args.n, device=args.device, log_every=args.log_every)

    print()
    print(f"{'='*64}")
    print(f"  RESULT: {label}")
    print(f"{'='*64}")
    print(f"  Accuracy:     {result['accuracy']:.3f}  ({result['correct']}/{result['total']})")
    print(f"  Wilson 95% CI: [{result['wilson_ci_95'][0]:.3f}, {result['wilson_ci_95'][1]:.3f}]")
    print(f"  Extraction:   {result['extraction_rate']:.3f}")
    print(f"  Wall:         {result['wall_seconds']:.1f}s")
    print()
    print("  First 3 examples:")
    for ex in result["first_examples"][:3]:
        print(f"    Q: {ex['q']!r}")
        print(f"    gold={ex['gold']!r}  pred={ex['pred']!r}  correct={ex['correct']}")
        print(f"    completion: {ex['completion'][:160]!r}")
        print()

    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as fh:
            json.dump({"ckpt": args.ckpt, "n": args.n, **result,
                       "diag": diag}, fh, indent=2, default=str)
        print(f"  saved to {args.save_json}")


if __name__ == "__main__":
    main()
