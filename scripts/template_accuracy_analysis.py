"""
Per-template accuracy breakdown for the FANT 2 M4 checkpoint.

Generates the same 1000 procedural math problems as eval_1k.py (seed=9999,
max_value=12) and evaluates the M4 checkpoint on each one, tracking accuracy
per template type.  Also flags semantically paradoxical problems where the
place_group does not match the item (e.g. "garden has rows of books").

Expected wall-clock: ~33 minutes on CPU.

Run:
    PYTHONPATH=. python scripts/template_accuracy_analysis.py
"""

from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import torch

from fant2.config import fant2_tiny
from fant2.inference import FANT2Generator
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training.phase5_rollout import ProceduralMathStream, format_prompt


# ---------------------------------------------------------------------------
# Constants (same as eval_1k.py)
# ---------------------------------------------------------------------------
TOKENIZER_PATH = "output/option_i/tokenizer.json"
CKPT_PATH = "output/option_m4/math_ramp/final.pt"

EVAL_SEED = 9999
EVAL_MAX_VALUE = 12
EVAL_N_PROBLEMS = 1000
EVAL_MAX_NEW_TOKENS = 64

OUTPUT_GATE_INIT = 0.1
CURVATURE_THRESHOLD = 1.0

_ANSWER_TAG = re.compile(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>")
_ANY_NUM = re.compile(r"-?\d+(?:\.\d+)?")


# ---------------------------------------------------------------------------
# Semantic-mismatch definitions
# ---------------------------------------------------------------------------
# Which items make sense in which place_group.  If a (place, item) pair does
# not appear in this mapping, it is flagged as a mismatch.
_SENSIBLE_PLACE_ITEMS: Dict[str, set] = {
    "garden":  {"apples", "shells"},         # natural outdoor items
    "orchard": {"apples"},                   # orchards have fruit
    "field":   {"apples", "shells"},         # outdoor natural items
    "shelf":   {"books", "marbles", "stickers", "coins", "shells",
                "cards", "pencils"},         # things you put on a shelf
}

# Templates that use both {place_group} and {item}:
_PLACE_ITEM_TEMPLATES = {"multiplication_grid", "remainder_complement"}

# Regex to pull the place_group and item from the generated question text.
# multiplication_grid: "A <place> has <a> rows of <b> <item>."
_MG_RE = re.compile(r"^A (\w+) has \d+ rows of \d+ (\w+)\.")
# remainder_complement: "There are <a> <item> in a <place>."
_RC_RE = re.compile(r"^There are \d+ (\w+) in a (\w+)\.")


def _is_mismatch(question: str, template: str) -> bool:
    """Return True if the question has a semantically paradoxical
    place/item combination."""
    if template not in _PLACE_ITEM_TEMPLATES:
        return False

    if template == "multiplication_grid":
        m = _MG_RE.search(question)
        if m:
            place, item = m.group(1), m.group(2)
        else:
            return False
    elif template == "remainder_complement":
        m = _RC_RE.search(question)
        if m:
            item, place = m.group(1), m.group(2)
        else:
            return False
    else:
        return False

    sensible = _SENSIBLE_PLACE_ITEMS.get(place, set())
    return item not in sensible


# ---------------------------------------------------------------------------
# Helpers (same as eval_1k.py)
# ---------------------------------------------------------------------------

def _extract_answer(text: str) -> str | None:
    m = _ANSWER_TAG.search(text)
    if m:
        return m.group(1)
    nums = _ANY_NUM.findall(text)
    if nums:
        return nums[-1]
    return None


def _is_correct(pred: str | None, gold: str) -> bool:
    if pred is None:
        return False
    if pred.lstrip("0") == gold.lstrip("0"):
        return True
    if pred == gold:
        return True
    if pred == "0" and gold == "0":
        return True
    try:
        if float(pred) == float(gold):
            return True
    except ValueError:
        pass
    return False


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
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


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    tokenizer = FANT2Tokenizer.load(TOKENIZER_PATH)
    cfg = fant2_tiny()
    cfg.phase4_classifier_upstream = True
    cfg.phase4_classifier_mode = "ce_surprise"
    model = FANT2Model(cfg)
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    bump_output_gates(model, OUTPUT_GATE_INIT)
    bump_curvature_threshold(model, CURVATURE_THRESHOLD)
    return model, tokenizer


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 72)
    print("  FANT 2 -- Template Accuracy Analysis (1000 problems)")
    print(f"  Checkpoint: {CKPT_PATH}")
    print("=" * 72)

    model, tokenizer = load_model()
    gen = FANT2Generator(model, tokenizer, device="cpu")
    stream = ProceduralMathStream(seed=EVAL_SEED, max_value=EVAL_MAX_VALUE)
    it = iter(stream)

    # Per-template accumulators
    template_total: Dict[str, int] = defaultdict(int)
    template_correct: Dict[str, int] = defaultdict(int)

    # Mismatch tracking
    mismatch_count = 0
    mismatch_examples: List[str] = []

    t0 = time.time()

    for i in range(EVAL_N_PROBLEMS):
        ex = next(it)

        # --- Distribution counting ---
        template_total[ex.template] += 1

        # --- Semantic mismatch check ---
        if _is_mismatch(ex.question, ex.template):
            mismatch_count += 1
            if len(mismatch_examples) < 10:
                mismatch_examples.append(
                    f"  [{ex.template}] {ex.question}"
                )

        # --- Model evaluation ---
        prompt = format_prompt(ex.question)
        completion = gen.generate(
            prompt, max_new_tokens=EVAL_MAX_NEW_TOKENS,
            greedy=True, return_full_text=False,
        )
        pred = _extract_answer(completion)
        if _is_correct(pred, ex.gold_answer):
            template_correct[ex.template] += 1

        # Progress update every 100 problems
        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            total_correct = sum(template_correct.values())
            print(f"  [{i+1:4d}/{EVAL_N_PROBLEMS}] "
                  f"acc={total_correct/(i+1):.3f} "
                  f"elapsed={elapsed:.0f}s")

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # Results table
    # ------------------------------------------------------------------
    print()
    print("=" * 72)
    print("  PER-TEMPLATE ACCURACY")
    print("=" * 72)
    print(f"  {'template':<25s} {'count':>6s} {'correct':>8s} {'accuracy':>9s}   {'wilson 95% CI':>18s}")
    print("  " + "-" * 70)

    # Sort by template name for stable output
    all_templates = sorted(template_total.keys())
    grand_total = 0
    grand_correct = 0
    for tmpl in all_templates:
        n = template_total[tmpl]
        c = template_correct[tmpl]
        acc = c / n if n > 0 else 0.0
        lo, hi = wilson_ci(c, n)
        grand_total += n
        grand_correct += c
        print(f"  {tmpl:<25s} {n:>6d} {c:>8d} {acc:>8.1%}   [{lo:.3f}, {hi:.3f}]")

    print("  " + "-" * 70)
    overall_acc = grand_correct / grand_total if grand_total > 0 else 0.0
    lo, hi = wilson_ci(grand_correct, grand_total)
    print(f"  {'OVERALL':<25s} {grand_total:>6d} {grand_correct:>8d} "
          f"{overall_acc:>8.1%}   [{lo:.3f}, {hi:.3f}]")
    print()

    # ------------------------------------------------------------------
    # Distribution check
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  TEMPLATE DISTRIBUTION (expected ~125 each for 8 templates)")
    print("=" * 72)
    for tmpl in all_templates:
        n = template_total[tmpl]
        bar = "#" * (n // 5)
        print(f"  {tmpl:<25s} {n:>4d}  {bar}")
    print()

    # ------------------------------------------------------------------
    # Semantic mismatch report
    # ------------------------------------------------------------------
    print("=" * 72)
    print("  SEMANTIC MISMATCHES (place/item paradoxes)")
    print("=" * 72)
    print(f"  Total mismatches: {mismatch_count} / {EVAL_N_PROBLEMS}")
    if mismatch_examples:
        print()
        print("  First examples:")
        for ex_str in mismatch_examples:
            print(ex_str)
    print()
    print(f"  Wall time: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print("=" * 72)


if __name__ == "__main__":
    main()
