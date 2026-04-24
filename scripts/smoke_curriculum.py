"""Dry-run sanity check for the data-mix curriculum system.

Exercises the same code paths runpod_train.py uses for curriculum+stream
setup, without requiring CUDA, bitsandbytes, or live HF access. Runs locally
in <1 sec and should pass on every CI box before we spin up a RunPod GPU.

Run:
    python scripts/smoke_curriculum.py                 # all 3 presets at defaults
    python scripts/smoke_curriculum.py --preset deepinsight_3phase --total-steps 100000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fant3.training import (
    PRESETS,
    build_curriculum,
    get_active_phase,
)
from fant2.data.streaming import InterleavedMultiDatasetStream


def check_preset(name: str, total_steps: int, phase_a_steps: int | None = None,
                 construct_streams: bool = True) -> bool:
    """Validate one preset end-to-end. Returns True on success, False on any failure."""
    print(f"\n=== {name} @ total_steps={total_steps}"
          f"{f' phase_a_steps={phase_a_steps}' if phase_a_steps else ''} ===")

    # 1. Build curriculum
    try:
        curriculum = build_curriculum(
            name,
            phase_a_steps=phase_a_steps if name == "legacy_2phase" else None,
            total_steps=total_steps,
        )
    except (KeyError, ValueError) as e:
        print(f"  FAIL build_curriculum: {type(e).__name__}: {e}")
        return False

    print(f"  curriculum.name = {curriculum.name}")
    print(f"  n_phases = {len(curriculum.phases)}")
    for i, phase in enumerate(curriculum.phases):
        end_step = int(phase.end_frac * total_steps)
        print(f"    phase[{i}] {phase.name!r:>12s}  end_frac={phase.end_frac:.3f}  "
              f"end_step={end_step:>7d}  n_datasets={len(phase.datasets)}  "
              f"seq_len={phase.seq_len}")
        # Weight-by-dataset table
        for ds, w in zip(phase.datasets, phase.weights):
            print(f"        {ds:35s}  w={w:.3f}")
        wsum = sum(phase.weights)
        if abs(wsum - 1.0) > 1e-3:
            print(f"  FAIL: phase {phase.name!r} weights sum to {wsum}, expected 1.0")
            return False

    # 2. Phase walk — verify selector returns expected sequence
    phase_trace = []
    for step in range(0, total_steps + 1, max(1, total_steps // 50)):
        phase = get_active_phase(step, total_steps, curriculum)
        if not phase_trace or phase_trace[-1][1] != phase.name:
            phase_trace.append((step, phase.name))
    print(f"  phase walk: {' -> '.join(f'{n}@{s}' for s, n in phase_trace)}")
    expected_names = [p.name for p in curriculum.phases]
    observed_names = [n for _, n in phase_trace]
    if observed_names != expected_names:
        print(f"  FAIL: phase walk {observed_names} != expected {expected_names}")
        return False

    # 3. Milestone step/suffix logic (mirrors runpod_train.py)
    phase_end_steps = {}
    for p in curriculum.phases[:-1]:
        es = max(1, int(p.end_frac * total_steps))
        suffix = f"_phase{p.name}" if name == "legacy_2phase" else f"_phase_{p.name}"
        phase_end_steps[es] = suffix
    print(f"  milestone boundaries: {phase_end_steps}")

    # 4. Optionally construct streams (tests registry key validity + imports)
    if construct_streams:
        for i, phase in enumerate(curriculum.phases):
            try:
                s = InterleavedMultiDatasetStream(
                    list(phase.datasets),
                    weights=list(phase.weights),
                    seed=1000 * i,
                )
                # Don't iterate — that would hit HF network
                assert len(s.entries) == len(phase.datasets), (
                    f"stream dropped datasets: got {len(s.entries)} from "
                    f"{len(phase.datasets)} requested"
                )
            except Exception as e:
                print(f"  FAIL stream construction phase[{i}]: {type(e).__name__}: {e}")
                return False

    print("  OK")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=None,
                    help="Only check this preset (default: all)")
    ap.add_argument("--total-steps", type=int, default=12000)
    ap.add_argument("--phase-a-steps", type=int, default=None,
                    help="override legacy_2phase first boundary")
    ap.add_argument("--no-streams", action="store_true",
                    help="skip InterleavedMultiDatasetStream construction "
                         "(useful in environments without 'datasets' pkg)")
    args = ap.parse_args()

    presets_to_check = [args.preset] if args.preset else list(PRESETS.keys())
    print(f"Checking {len(presets_to_check)} preset(s) @ total_steps={args.total_steps}")

    all_ok = True
    for name in presets_to_check:
        ok = check_preset(
            name,
            total_steps=args.total_steps,
            phase_a_steps=args.phase_a_steps,
            construct_streams=not args.no_streams,
        )
        all_ok = all_ok and ok

    print()
    if all_ok:
        print(f"ALL {len(presets_to_check)} preset(s) OK")
        sys.exit(0)
    else:
        print("SOME PRESETS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
