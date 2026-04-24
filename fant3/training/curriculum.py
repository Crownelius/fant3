"""Data-mix curriculum scheduler for FANT 3 training.

Source: arxiv:2604.16278 "Learning to Reason with Insight for Informal
Theorem Proving" (Li et al., 2026). The paper reports that a 3-stage
progressive SFT curriculum — Apprentice (Q, Proof) -> Journeyman
(Q, Sketch, Proof) -> Expert (Q, Techniques, Sketch, Proof) — closes
most of the RL gap at 1B-3B scale, the exact band FANT 3 targets.
1B-3B showed a "disproportionately large boost" vs flat SFT.

This module provides a backward-compatible extension of the 2-phase
pattern in scripts/runpod_train.py (sampler_A / sampler_B at a step
threshold) to an N-phase pattern with named presets.

Usage:
    from fant3.training.curriculum import PRESETS, get_active_phase

    curriculum = PRESETS["deepinsight_3phase"]
    phase = get_active_phase(step, total_steps, curriculum)
    # -> PhaseSpec with .name, .datasets, .weights, .seq_len

Design invariants (enforced by tests):
  1. Phase end_fracs are strictly increasing and last phase is 1.0.
  2. Weights are normalized to sum ~1.0 within each phase.
  3. Dataset keys reference fant2.data.registry.TRAINING_DATASETS.
  4. get_active_phase(step=0) returns the first phase.
  5. get_active_phase(step=total_steps) returns the last phase.
  6. Transitions are hard (no linear blend) — simpler + matches paper.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class PhaseSpec:
    """One phase of a training curriculum.

    Attributes:
        name:     Human-readable tag (e.g. "apprentice"). Logged per step.
        end_frac: Fraction of total_steps where this phase ends, in (0, 1].
                  The last phase must have end_frac = 1.0.
        datasets: Registry keys (strings) for fant2.data.registry.
        weights:  Sampling probabilities; must sum to ~1.0 and len == len(datasets).
        seq_len:  Sequence length used by the sampler during this phase.
    """
    name: str
    end_frac: float
    datasets: Tuple[str, ...]
    weights: Tuple[float, ...]
    seq_len: int = 1024

    def __post_init__(self):
        if len(self.datasets) != len(self.weights):
            raise ValueError(
                f"PhaseSpec {self.name}: {len(self.datasets)} datasets "
                f"vs {len(self.weights)} weights"
            )
        if not 0.0 < self.end_frac <= 1.0:
            raise ValueError(
                f"PhaseSpec {self.name}: end_frac must be in (0, 1], got {self.end_frac}"
            )
        total = sum(self.weights)
        if abs(total - 1.0) > 1e-3:
            raise ValueError(
                f"PhaseSpec {self.name}: weights sum to {total}, expected ~1.0"
            )
        for w in self.weights:
            if w < 0:
                raise ValueError(f"PhaseSpec {self.name}: negative weight {w}")


@dataclass(frozen=True)
class Curriculum:
    """A named sequence of PhaseSpecs."""
    name: str
    phases: Tuple[PhaseSpec, ...]

    def __post_init__(self):
        if not self.phases:
            raise ValueError(f"Curriculum {self.name}: empty phases")
        # end_fracs strictly increasing, last == 1.0
        prev = 0.0
        for p in self.phases:
            if p.end_frac <= prev:
                raise ValueError(
                    f"Curriculum {self.name}: phase {p.name} end_frac {p.end_frac} "
                    f"<= previous {prev}"
                )
            prev = p.end_frac
        if abs(self.phases[-1].end_frac - 1.0) > 1e-6:
            raise ValueError(
                f"Curriculum {self.name}: last phase end_frac must be 1.0, "
                f"got {self.phases[-1].end_frac}"
            )

    def all_datasets(self) -> Tuple[str, ...]:
        """All unique dataset keys across all phases."""
        seen = []
        for p in self.phases:
            for d in p.datasets:
                if d not in seen:
                    seen.append(d)
        return tuple(seen)


def get_active_phase(step: int, total_steps: int, curriculum: Curriculum) -> PhaseSpec:
    """Return the PhaseSpec active at the given step.

    step=0 -> first phase. step=total_steps -> last phase. Steps beyond
    total_steps also return the last phase (graceful for unlimited runs).
    """
    if total_steps <= 0:
        return curriculum.phases[0]
    frac = step / total_steps
    for phase in curriculum.phases:
        if frac <= phase.end_frac:
            return phase
    return curriculum.phases[-1]


def get_phase_index(step: int, total_steps: int, curriculum: Curriculum) -> int:
    """Return the index of the active phase (0-indexed)."""
    phase = get_active_phase(step, total_steps, curriculum)
    return curriculum.phases.index(phase)


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------
#
# Dataset key conventions (must match fant2/data/registry.py):
#   fineweb-edu, numina-math-cot, finetome-100k,
#   kimi-k25-distill, kimi-k25-math,
#   opus46-crownelius-3300x, sonnet46-120k,
#   nvidia-openmath-2, nvidia-openmath-reasoning,
#   nvidia-opencode-reasoning-2, nvidia-cascade2-sft-{math,chat,if,science},
#   nvidia-daring-anteater.
#
# DO NOT include decontamination-sensitive datasets here; is_contaminated()
# is applied downstream per-text in make_batch_sampler.


# --- Legacy 2-phase (exact match of scripts/runpod_train.py pre-curriculum) ---
#
# Preserved verbatim so --curriculum legacy_2phase reproduces the existing
# 50m-unlimited run bit-identically.

LEGACY_2PHASE = Curriculum(
    name="legacy_2phase",
    phases=(
        PhaseSpec(
            name="A",
            end_frac=8000.0 / 12000.0,  # matches default phase_a_steps / total_steps
            datasets=(
                "fineweb-edu",
                "nvidia-openmath-reasoning",
                "nvidia-opencode-reasoning-2",
                "nvidia-openmath-2",
                "opus46-crownelius-3300x",
                "kimi-k25-distill",
            ),
            weights=(0.35, 0.20, 0.10, 0.10, 0.15, 0.10),
            seq_len=1024,
        ),
        PhaseSpec(
            name="B",
            end_frac=1.0,
            datasets=(
                "nvidia-cascade2-sft-if",
                "sonnet46-120k",
                "nvidia-openmath-2",
                "nvidia-cascade2-sft-science",
                "nvidia-daring-anteater",
                "nvidia-cascade2-sft-chat",
            ),
            weights=(0.25, 0.30, 0.15, 0.10, 0.10, 0.10),
            seq_len=1024,
        ),
    ),
)


# --- DeepInsight 3-phase (arxiv:2604.16278) ---
#
# Paper maps:
#   Stage 1 Apprentice: (Q, Proof) - foundational math language skills
#   Stage 2 Journeyman: (Q, Sketch, Proof) - logical planning
#   Stage 3 Expert:     (Q, Technique, Sketch, Proof) - insight / core-technique recognition
#
# FANT 3 mapping (we don't have explicit Technique annotations yet, so we
# approximate the Expert phase with our highest-structure reasoning data:
# PROBLEM_THINK_SOLUTION-native Opus 4.6 Crownelius + curated Sonnet 4.6):

DEEPINSIGHT_3PHASE = Curriculum(
    name="deepinsight_3phase",
    phases=(
        # Apprentice: foundational LM, heavy raw text + simple structured problems.
        # Goal: acquire language + basic problem format before being asked to reason.
        PhaseSpec(
            name="apprentice",
            end_frac=0.25,
            datasets=(
                "fineweb-edu",
                "nvidia-openmath-2",
                "nvidia-opencode-reasoning-2",
                "numina-math-cot",
                "finetome-100k",
            ),
            weights=(0.55, 0.15, 0.10, 0.10, 0.10),
            seq_len=1024,
        ),
        # Journeyman: introduce proof-sketch / <think>-style traces.
        # Mix continues to include raw text (FineWeb) so language doesn't drift.
        PhaseSpec(
            name="journeyman",
            end_frac=0.65,
            datasets=(
                "fineweb-edu",
                "kimi-k25-distill",
                "sonnet46-120k",
                "nvidia-openmath-reasoning",
                "opus46-crownelius-3300x",
                "nvidia-cascade2-sft-if",
                "nvidia-opencode-reasoning-2",
            ),
            weights=(0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05),
            seq_len=1024,
        ),
        # Expert: heaviest weighting on PROBLEM_THINK_SOLUTION and graded-
        # quality reasoning data (Crownelius, Sonnet 4.6, Kimi math).
        # FineWeb retained at 5% anchor to prevent format collapse.
        PhaseSpec(
            name="expert",
            end_frac=1.0,
            datasets=(
                "opus46-crownelius-3300x",
                "sonnet46-120k",
                "kimi-k25-math",
                "nvidia-openmath-reasoning",
                "nvidia-cascade2-sft-science",
                "nvidia-cascade2-sft-math",
                "fineweb-edu",
            ),
            weights=(0.30, 0.20, 0.15, 0.10, 0.10, 0.10, 0.05),
            seq_len=1024,
        ),
    ),
)


# --- Flat 1-phase control (same mix throughout; baseline for A/B testing curriculum) ---
#
# Uses the Journeyman mix — the middle of the road. If curriculum beats flat,
# the curriculum worked. If not, the format / data was the issue.

FLAT_1PHASE = Curriculum(
    name="flat_1phase",
    phases=(
        PhaseSpec(
            name="flat",
            end_frac=1.0,
            datasets=(
                "fineweb-edu",
                "kimi-k25-distill",
                "sonnet46-120k",
                "nvidia-openmath-reasoning",
                "opus46-crownelius-3300x",
                "nvidia-cascade2-sft-if",
                "nvidia-opencode-reasoning-2",
            ),
            weights=(0.25, 0.20, 0.15, 0.15, 0.10, 0.10, 0.05),
            seq_len=1024,
        ),
    ),
)


PRESETS: dict[str, Curriculum] = {
    "legacy_2phase": LEGACY_2PHASE,
    "deepinsight_3phase": DEEPINSIGHT_3PHASE,
    "flat_1phase": FLAT_1PHASE,
}


def load_preset(name: str) -> Curriculum:
    """Load a curriculum by name, with a useful error on miss."""
    if name not in PRESETS:
        raise KeyError(
            f"Unknown curriculum preset {name!r}. "
            f"Available: {sorted(PRESETS)}"
        )
    return PRESETS[name]


def build_curriculum(
    name: str,
    phase_a_steps: int | None = None,
    total_steps: int | None = None,
) -> Curriculum:
    """Load a preset, optionally overriding the first phase boundary.

    Used by scripts/runpod_train.py for backward compatibility: the
    existing --phase-a-steps / --total-steps CLI flags override the
    hard-coded end_frac of the legacy_2phase preset. For other presets
    phase boundaries are driven by total_steps alone (fractions are
    baked into the preset).

    Args:
        name:           Preset key (e.g. "legacy_2phase", "deepinsight_3phase").
        phase_a_steps:  If set AND name == "legacy_2phase", overrides the
                        first phase's end_frac to phase_a_steps/total_steps.
                        Ignored for non-legacy presets.
        total_steps:    Required if phase_a_steps is set.

    Returns:
        Curriculum (possibly a fresh one with overridden end_frac).
    """
    base = load_preset(name)
    if name != "legacy_2phase" or phase_a_steps is None:
        return base
    if total_steps is None or total_steps <= 0:
        raise ValueError(
            "build_curriculum: total_steps must be positive when phase_a_steps is set"
        )
    new_end = float(phase_a_steps) / float(total_steps)
    if not 0.0 < new_end < 1.0:
        raise ValueError(
            f"build_curriculum: phase_a_steps {phase_a_steps} / total_steps "
            f"{total_steps} = {new_end}; must be in (0, 1)"
        )
    first, second = base.phases
    return Curriculum(
        name=base.name,
        phases=(
            PhaseSpec(
                name=first.name,
                end_frac=new_end,
                datasets=first.datasets,
                weights=first.weights,
                seq_len=first.seq_len,
            ),
            second,
        ),
    )
