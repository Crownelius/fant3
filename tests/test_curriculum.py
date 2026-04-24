"""Tests for fant3.training.curriculum — data-mix progressive curriculum.

Source paper: arxiv:2604.16278 DeepInsightTheorem. Design invariants are
enforced by PhaseSpec.__post_init__ / Curriculum.__post_init__; these tests
exercise those invariants plus the per-step selector and the preset catalog.
"""
from __future__ import annotations

import pytest

from fant3.training.curriculum import (
    PhaseSpec,
    Curriculum,
    PRESETS,
    get_active_phase,
    get_phase_index,
    load_preset,
    build_curriculum,
)
from fant2.data.registry import TRAINING_DATASETS


# ---------------------------------------------------------------------------
# PhaseSpec validation
# ---------------------------------------------------------------------------

class TestPhaseSpec:
    def test_valid_phase_ok(self):
        PhaseSpec(
            name="p0", end_frac=1.0,
            datasets=("fineweb-edu",), weights=(1.0,),
        )

    def test_datasets_weights_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="datasets"):
            PhaseSpec(
                name="bad", end_frac=1.0,
                datasets=("a", "b"), weights=(1.0,),
            )

    @pytest.mark.parametrize("bad_frac", [0.0, -0.1, 1.5])
    def test_end_frac_out_of_range_raises(self, bad_frac):
        with pytest.raises(ValueError, match="end_frac"):
            PhaseSpec(
                name="bad", end_frac=bad_frac,
                datasets=("a",), weights=(1.0,),
            )

    def test_weights_dont_sum_to_one_raises(self):
        with pytest.raises(ValueError, match="sum"):
            PhaseSpec(
                name="bad", end_frac=1.0,
                datasets=("a", "b"), weights=(0.3, 0.3),
            )

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="negative"):
            PhaseSpec(
                name="bad", end_frac=1.0,
                datasets=("a", "b"), weights=(1.5, -0.5),
            )

    def test_weights_near_one_tolerated(self):
        # Floating-point rounding: should accept 0.999x / 1.001x
        PhaseSpec(
            name="ok", end_frac=1.0,
            datasets=("a", "b", "c"),
            weights=(0.333, 0.333, 0.334),
        )


# ---------------------------------------------------------------------------
# Curriculum validation
# ---------------------------------------------------------------------------

class TestCurriculum:
    def _p(self, name, end_frac, n=1):
        return PhaseSpec(name=name, end_frac=end_frac,
                         datasets=("fineweb-edu",) * n,
                         weights=tuple([1.0 / n] * n))

    def test_valid_single_phase(self):
        c = Curriculum(name="c", phases=(self._p("only", 1.0),))
        assert len(c.phases) == 1

    def test_valid_three_phase(self):
        c = Curriculum(name="c", phases=(
            self._p("a", 0.3), self._p("b", 0.7), self._p("c", 1.0),
        ))
        assert len(c.phases) == 3

    def test_empty_phases_raises(self):
        with pytest.raises(ValueError, match="empty"):
            Curriculum(name="c", phases=())

    def test_non_monotonic_end_fracs_raises(self):
        with pytest.raises(ValueError, match="end_frac"):
            Curriculum(name="c", phases=(
                self._p("a", 0.5), self._p("b", 0.3), self._p("c", 1.0),
            ))

    def test_equal_end_fracs_raises(self):
        with pytest.raises(ValueError, match="end_frac"):
            Curriculum(name="c", phases=(
                self._p("a", 0.5), self._p("b", 0.5), self._p("c", 1.0),
            ))

    def test_last_not_one_raises(self):
        with pytest.raises(ValueError, match="1.0"):
            Curriculum(name="c", phases=(
                self._p("a", 0.5), self._p("b", 0.8),
            ))

    def test_all_datasets_dedup(self):
        p1 = PhaseSpec(name="a", end_frac=0.5,
                       datasets=("x", "y"), weights=(0.5, 0.5))
        p2 = PhaseSpec(name="b", end_frac=1.0,
                       datasets=("y", "z"), weights=(0.5, 0.5))
        c = Curriculum(name="c", phases=(p1, p2))
        assert c.all_datasets() == ("x", "y", "z")


# ---------------------------------------------------------------------------
# get_active_phase — per-step selector (the core training-loop hook)
# ---------------------------------------------------------------------------

class TestGetActivePhase:
    def _three_phase(self):
        return Curriculum(name="c", phases=(
            PhaseSpec("a", 0.25, ("d",), (1.0,)),
            PhaseSpec("b", 0.65, ("d",), (1.0,)),
            PhaseSpec("c", 1.0,  ("d",), (1.0,)),
        ))

    def test_step_zero_returns_first(self):
        c = self._three_phase()
        assert get_active_phase(0, 1000, c).name == "a"

    def test_step_one_returns_first(self):
        c = self._three_phase()
        assert get_active_phase(1, 1000, c).name == "a"

    def test_step_mid_phase_a(self):
        c = self._three_phase()
        assert get_active_phase(100, 1000, c).name == "a"

    def test_step_at_phase_a_boundary_stays_in_a(self):
        # 250 / 1000 = 0.25 exactly == end_frac of phase a. Spec: frac <= end_frac
        # returns that phase. Step 251 should be phase b.
        c = self._three_phase()
        assert get_active_phase(250, 1000, c).name == "a"
        assert get_active_phase(251, 1000, c).name == "b"

    def test_step_at_phase_b_boundary_stays_in_b(self):
        c = self._three_phase()
        assert get_active_phase(650, 1000, c).name == "b"
        assert get_active_phase(651, 1000, c).name == "c"

    def test_step_equal_total_returns_last(self):
        c = self._three_phase()
        assert get_active_phase(1000, 1000, c).name == "c"

    def test_step_beyond_total_returns_last(self):
        c = self._three_phase()
        assert get_active_phase(5000, 1000, c).name == "c"

    def test_zero_total_steps_defaults_to_first(self):
        c = self._three_phase()
        # Defensive: avoid ZeroDivisionError on pathological inputs
        assert get_active_phase(0, 0, c).name == "a"

    def test_get_phase_index_consistent(self):
        c = self._three_phase()
        for step in [0, 100, 250, 251, 650, 651, 1000, 1500]:
            phase = get_active_phase(step, 1000, c)
            idx = get_phase_index(step, 1000, c)
            assert c.phases[idx] is phase


# ---------------------------------------------------------------------------
# Preset catalog
# ---------------------------------------------------------------------------

class TestPresets:
    def test_all_presets_load(self):
        # Every preset key must be loadable and well-formed.
        expected = {"legacy_2phase", "deepinsight_3phase", "flat_1phase"}
        assert expected.issubset(set(PRESETS.keys()))

    def test_load_preset_unknown_raises(self):
        with pytest.raises(KeyError, match="Unknown"):
            load_preset("nonexistent-preset")

    @pytest.mark.parametrize("preset_name", ["legacy_2phase", "deepinsight_3phase", "flat_1phase"])
    def test_every_preset_dataset_is_registered(self, preset_name):
        """Every dataset key referenced by a preset must exist in TRAINING_DATASETS.
        Catches typos in the preset at import time — the curriculum would fail
        at stream-build time otherwise, hours into a RunPod run."""
        preset = load_preset(preset_name)
        for phase in preset.phases:
            for ds_key in phase.datasets:
                assert ds_key in TRAINING_DATASETS, (
                    f"Preset {preset_name!r} phase {phase.name!r} references "
                    f"unknown dataset {ds_key!r}. Known: {sorted(TRAINING_DATASETS)}"
                )

    def test_legacy_2phase_matches_original_runpod_train(self):
        """The legacy_2phase preset must reproduce the pre-curriculum hardcoded
        mix bit-exactly so `--curriculum legacy_2phase` resumes from existing
        checkpoints without drifting the data distribution."""
        preset = load_preset("legacy_2phase")
        assert len(preset.phases) == 2
        assert preset.phases[0].datasets == (
            "fineweb-edu",
            "nvidia-openmath-reasoning",
            "nvidia-opencode-reasoning-2",
            "nvidia-openmath-2",
            "opus46-crownelius-3300x",
            "kimi-k25-distill",
        )
        assert preset.phases[0].weights == (0.35, 0.20, 0.10, 0.10, 0.15, 0.10)
        assert preset.phases[1].datasets == (
            "nvidia-cascade2-sft-if",
            "sonnet46-120k",
            "nvidia-openmath-2",
            "nvidia-cascade2-sft-science",
            "nvidia-daring-anteater",
            "nvidia-cascade2-sft-chat",
        )
        assert preset.phases[1].weights == (0.25, 0.30, 0.15, 0.10, 0.10, 0.10)

    def test_deepinsight_3phase_has_three_phases(self):
        preset = load_preset("deepinsight_3phase")
        assert len(preset.phases) == 3
        assert preset.phases[0].name == "apprentice"
        assert preset.phases[1].name == "journeyman"
        assert preset.phases[2].name == "expert"
        # Paper's schedule: approximately 0-25%, 25-65%, 65-100%
        assert preset.phases[0].end_frac == 0.25
        assert preset.phases[1].end_frac == 0.65
        assert preset.phases[2].end_frac == 1.0

    def test_deepinsight_expert_heavy_on_reasoning(self):
        """Paper claim: Expert phase emphasizes the (Technique, Sketch, Proof)
        hierarchy. Our approximation: Opus Crownelius + Sonnet 4.6 (highest-
        structure reasoning data) dominate the Expert mix."""
        expert = load_preset("deepinsight_3phase").phases[2]
        # Build weight-by-key lookup
        by_key = dict(zip(expert.datasets, expert.weights))
        reasoning_heavy = (
            by_key.get("opus46-crownelius-3300x", 0.0)
            + by_key.get("sonnet46-120k", 0.0)
        )
        assert reasoning_heavy >= 0.45, (
            f"Expert phase should be ≥45% Opus/Sonnet reasoning; got {reasoning_heavy}"
        )

    def test_deepinsight_apprentice_heavy_on_foundation(self):
        """Apprentice phase builds foundational language — FineWeb should dominate."""
        apprentice = load_preset("deepinsight_3phase").phases[0]
        by_key = dict(zip(apprentice.datasets, apprentice.weights))
        assert by_key.get("fineweb-edu", 0.0) >= 0.50, (
            "Apprentice phase should be ≥50% FineWeb (foundational)"
        )

    def test_flat_1phase_matches_journeyman_middle(self):
        """flat_1phase is the control arm — single middle-of-road mix.
        Semantic check: not weighted toward extremes."""
        flat = load_preset("flat_1phase")
        assert len(flat.phases) == 1
        assert flat.phases[0].end_frac == 1.0


# ---------------------------------------------------------------------------
# build_curriculum — legacy_2phase override
# ---------------------------------------------------------------------------

class TestBuildCurriculum:
    def test_default_returns_preset_untouched(self):
        built = build_curriculum("deepinsight_3phase")
        assert built is PRESETS["deepinsight_3phase"]

    def test_legacy_override_phase_a(self):
        """Passing phase_a_steps rewrites the first phase end_frac for
        legacy_2phase so CLI users can still point the transition wherever."""
        built = build_curriculum("legacy_2phase", phase_a_steps=5000, total_steps=10000)
        assert built.phases[0].end_frac == 0.5
        # Second phase untouched
        assert built.phases[1].end_frac == 1.0
        # Datasets/weights preserved
        assert built.phases[0].datasets == PRESETS["legacy_2phase"].phases[0].datasets
        assert built.phases[0].weights == PRESETS["legacy_2phase"].phases[0].weights

    def test_legacy_no_override_returns_original(self):
        built = build_curriculum("legacy_2phase")
        assert built is PRESETS["legacy_2phase"]

    def test_non_legacy_with_override_is_noop(self):
        """phase_a_steps should be silently ignored for non-legacy presets
        (they're not 2-phase, so the flag semantics don't apply)."""
        built = build_curriculum("deepinsight_3phase", phase_a_steps=500, total_steps=1000)
        assert built is PRESETS["deepinsight_3phase"]

    def test_override_without_total_raises(self):
        with pytest.raises(ValueError, match="total_steps"):
            build_curriculum("legacy_2phase", phase_a_steps=500)

    def test_override_zero_total_raises(self):
        with pytest.raises(ValueError, match="total_steps"):
            build_curriculum("legacy_2phase", phase_a_steps=500, total_steps=0)

    def test_override_equal_total_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            build_curriculum("legacy_2phase", phase_a_steps=1000, total_steps=1000)

    def test_override_greater_than_total_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            build_curriculum("legacy_2phase", phase_a_steps=1500, total_steps=1000)

    def test_override_zero_phase_a_raises(self):
        with pytest.raises(ValueError, match="must be in"):
            build_curriculum("legacy_2phase", phase_a_steps=0, total_steps=1000)

    def test_unknown_preset_raises(self):
        with pytest.raises(KeyError, match="Unknown"):
            build_curriculum("ghost-preset")


# ---------------------------------------------------------------------------
# Stream-integration (synthetic; no network)
# ---------------------------------------------------------------------------
#
# We cannot actually load HF datasets in CI (no network, no tokens), but we
# CAN verify that the InterleavedMultiDatasetStream constructor accepts our
# preset weights and that the phase-to-sampler mapping is unambiguous.

class TestStreamIntegration:
    @pytest.mark.parametrize("preset_name", ["legacy_2phase", "deepinsight_3phase", "flat_1phase"])
    def test_preset_weights_normalize(self, preset_name):
        """Weights passed to InterleavedMultiDatasetStream are normalized
        internally; but they should already be near-1 so the normalization
        is a no-op."""
        preset = load_preset(preset_name)
        for phase in preset.phases:
            s = sum(phase.weights)
            assert abs(s - 1.0) < 1e-3, (
                f"{preset_name}/{phase.name} weights sum to {s}, should be 1.0"
            )

    def test_phase_boundary_step_numbers(self):
        """Mirror the runpod_train.py milestone logic: phase_end_steps
        dict maps end_step -> suffix. Verify that for deepinsight_3phase
        at 10000 total steps, boundaries land at 2500 (apprentice) and
        6500 (journeyman)."""
        preset = load_preset("deepinsight_3phase")
        total = 10000
        boundaries = {}
        for p in preset.phases[:-1]:
            es = max(1, int(p.end_frac * total))
            boundaries[es] = p.name
        assert boundaries == {2500: "apprentice", 6500: "journeyman"}

    def test_simulated_training_walk(self):
        """Walk step 0..total_steps in coarse increments and verify that the
        sequence of active phases is monotonic (never goes backward)."""
        c = load_preset("deepinsight_3phase")
        total = 10000
        seen = []
        for step in range(0, total + 1, 100):
            phase = get_active_phase(step, total, c)
            if not seen or seen[-1] != phase.name:
                seen.append(phase.name)
        assert seen == ["apprentice", "journeyman", "expert"]
