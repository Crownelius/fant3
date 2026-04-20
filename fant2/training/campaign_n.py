"""
Campaign N levers — structural interventions (no auxiliary losses).

Three non-loss levers for improving FANT 2 intelligence:

  N3 — SleepGate: periodic memory consolidation (merge similar, evict stale)
  N6 — G2RPO-A:  gold reasoning traces in training data (richer supervision)
  N7 — SEC:      self-evolving curriculum via Multi-Armed Bandit scheduling

All three are STRUCTURAL/DATA interventions, not loss modifications.
The N1 experiment proved that any auxiliary loss with alpha >= 0.01
destroys arithmetic at 5M scale (7.6% post-ramp vs 54.6% baseline).

These levers work through:
  N3: Better memory utilization (consolidate/evict between training steps)
  N6: Richer training signal (explicit step-by-step reasoning in gold text)
  N7: Adaptive data distribution (focus on what the model finds hardest)
"""

from __future__ import annotations

import math
import random
from typing import Iterator, List, Optional

from .phase5_rollout import (
    MathExample,
    ProceduralMathStream,
    format_prompt,
)


# =============================================================================
# N6 — Gold reasoning trace generation (G2RPO-A style)
# =============================================================================

# Per-template reasoning templates. These provide explicit step-by-step
# arithmetic traces that teach the model HOW to arrive at the answer,
# not just WHAT the answer is. At 5M scale, this supervised signal is
# much more tractable than RL exploration.

_REASONING_TEMPLATES = {
    "addition": (
        "Let me work it out step by step. "
        "{a} + {b} = {answer}."
    ),
    "subtraction": (
        "Let me work it out step by step. "
        "Starting with {a}, subtract {b}: {a} - {b} = {answer}."
    ),
    "multiplication_grid": (
        "Let me work it out step by step. "
        "{a} rows with {b} each: {a} times {b} = {answer}."
    ),
    "multiplication_pack": (
        "Let me work it out step by step. "
        "{a} packs with {b} each: {a} times {b} = {answer}."
    ),
    "division_even": (
        "Let me work it out step by step. "
        "Split {a} into {b} equal groups: {a} divided by {b} = {answer}."
    ),
    "rate": (
        "Let me work it out step by step. "
        "{a} per hour for {b} hours: {a} times {b} = {answer}."
    ),
    "remainder_complement": (
        "Let me work it out step by step. "
        "Total is {a}, subtract the {b} that are colored: {a} - {b} = {answer}."
    ),
    "weekly_repeat": (
        "Let me work it out step by step. "
        "{a} per day for {b} days: {a} times {b} = {answer}."
    ),
}

# Fallback for unknown templates
_FALLBACK_REASONING = "Let me work it out. The answer is {answer}."


def generate_gold_reasoning(ex: MathExample) -> str:
    """
    Generate a template-specific step-by-step reasoning trace.

    Uses the operands (a, b) stored on the MathExample to produce an
    explicit arithmetic derivation. Falls back to a generic trace if the
    template is unknown or operands are missing (a=b=0).
    """
    tmpl = _REASONING_TEMPLATES.get(ex.template)
    if tmpl is None or (ex.a == 0 and ex.b == 0):
        return _FALLBACK_REASONING.format(answer=ex.gold_answer)
    return tmpl.format(a=ex.a, b=ex.b, answer=ex.gold_answer)


class GuidedMathTextStream:
    """
    N6: ProceduralMathTextStream with gold reasoning traces.

    Instead of the generic "Let me work it out. The answer is X.",
    this stream produces template-specific step-by-step reasoning:
      "Let me work it out step by step. 5 + 3 = 8."

    This is the Phase 4 analog of G2RPO-A: instead of injecting gold
    traces into RL rollouts (Phase 5), inject them into the supervised
    training text. At 5M scale, explicit reasoning supervision is more
    effective than exploration-based RL.
    """

    def __init__(self, seed: int, max_value: int = 12):
        self.stream = ProceduralMathStream(seed=seed, max_value=max_value)

    def __iter__(self) -> Iterator[str]:
        for ex in self.stream:
            prompt = format_prompt(ex.question)
            reasoning = generate_gold_reasoning(ex)
            answer_block = (
                f" {reasoning}\n"
                f"</think>\n"
                f"<answer>{ex.gold_answer}</answer>"
            )
            yield prompt + answer_block


class PlainMathTextStream:
    """
    Baseline text stream (same as L1.5) for non-N6 variants.

    Produces the generic "Let me work it out. The answer is X." trace
    that the L1.5 baseline uses.
    """

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


# =============================================================================
# N7 — Self-Evolving Curriculum via Multi-Armed Bandit (SEC)
# =============================================================================

class CurriculumScheduler:
    """
    UCB1 Multi-Armed Bandit for adaptive difficulty scheduling.

    Arms correspond to difficulty bands (max_value for ProceduralMathStream):
      - easy:   max_value = 5  (single-digit operands, small answers)
      - medium: max_value = 8  (moderate operands)
      - hard:   max_value = 12 (full range, including large multiplications)

    The reward signal comes from the training loss: higher loss means more
    learning potential at that difficulty, so the MAB focuses there.

    The exploration constant c balances exploitation (focus on hardest)
    vs exploration (sample all bands to get accurate loss estimates).
    """

    ARMS: List[tuple] = [
        ("easy", 5),
        ("medium", 8),
        ("hard", 12),
    ]

    def __init__(self, c: float = 1.5):
        self.c = c
        self.n_arms = len(self.ARMS)
        self.counts = [1] * self.n_arms    # start at 1 to avoid div-by-zero
        self.sum_rewards = [0.0] * self.n_arms
        self.total_pulls = self.n_arms
        self.last_arm_idx = 0

    def _ucb1_score(self, idx: int) -> float:
        exploit = self.sum_rewards[idx] / self.counts[idx]
        explore = self.c * math.sqrt(
            math.log(self.total_pulls + 1) / self.counts[idx]
        )
        return exploit + explore

    def select_arm(self) -> int:
        """Select the next arm using UCB1. Returns arm index."""
        scores = [self._ucb1_score(i) for i in range(self.n_arms)]
        best = max(range(self.n_arms), key=lambda i: scores[i])
        self.last_arm_idx = best
        self.counts[best] += 1
        self.total_pulls += 1
        return best

    def update_reward(self, reward: float, arm_idx: Optional[int] = None):
        """Update reward for the given arm (default: last selected arm)."""
        idx = arm_idx if arm_idx is not None else self.last_arm_idx
        self.sum_rewards[idx] += reward

    def arm_name(self, idx: int) -> str:
        return self.ARMS[idx][0]

    def arm_max_value(self, idx: int) -> int:
        return self.ARMS[idx][1]

    def summary(self) -> str:
        """One-line summary of arm statistics."""
        parts = []
        for i, (name, _) in enumerate(self.ARMS):
            avg = self.sum_rewards[i] / max(self.counts[i], 1)
            parts.append(f"{name}={self.counts[i]}x(avg={avg:.3f})")
        return " ".join(parts)


class CurriculumMathTextStream:
    """
    N7: Text stream with MAB-scheduled difficulty.

    Each text comes from a difficulty band selected by UCB1. The training
    loss is fed back to the scheduler after each step to guide future
    selections.

    This stream tracks `self.scheduler` so the experiment runner can call
    `stream.scheduler.update_reward(loss)` after each training step.

    If use_gold_reasoning is True, also applies N6 gold reasoning traces.
    """

    def __init__(
        self,
        seed: int,
        use_gold_reasoning: bool = False,
        c: float = 1.5,
    ):
        self.scheduler = CurriculumScheduler(c=c)
        self.use_gold = use_gold_reasoning
        # One stream per difficulty arm
        self.streams = [
            ProceduralMathStream(seed=seed + i, max_value=mv)
            for i, (_, mv) in enumerate(CurriculumScheduler.ARMS)
        ]
        self._iters = [iter(s) for s in self.streams]

    def __iter__(self) -> Iterator[str]:
        return self

    def __next__(self) -> str:
        arm_idx = self.scheduler.select_arm()
        ex = next(self._iters[arm_idx])

        prompt = format_prompt(ex.question)
        if self.use_gold:
            reasoning = generate_gold_reasoning(ex)
            answer_block = (
                f" {reasoning}\n"
                f"</think>\n"
                f"<answer>{ex.gold_answer}</answer>"
            )
        else:
            answer_block = (
                f" Let me work it out. The answer is {ex.gold_answer}.\n"
                f"</think>\n"
                f"<answer>{ex.gold_answer}</answer>"
            )
        return prompt + answer_block


# =============================================================================
# N3 — SleepGate integration helper
# =============================================================================

def run_sleep_consolidation(
    model,
    merge_threshold: float = 0.92,
    staleness_horizon: int = 200,
    verbose: bool = True,
) -> dict:
    """
    Call sleep_consolidate() on the model's memory and log results.

    This is the N3 SleepGate lever: periodic memory consolidation that
    merges similar entries and evicts stale ones. The core method is
    implemented in ApollonianMemory.sleep_consolidate().

    Returns the consolidation stats dict.
    """
    stats = model.memory.sleep_consolidate(
        merge_threshold=merge_threshold,
        staleness_horizon=staleness_horizon,
    )
    if verbose and (stats["n_merged"] > 0 or stats["n_evicted"] > 0):
        print(
            f"    SleepGate: merged={stats['n_merged']}, "
            f"evicted={stats['n_evicted']}, "
            f"{stats['n_before']}→{stats['n_after']} entries"
        )
    return stats
