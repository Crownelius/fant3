"""
Learning-rate schedulers for FANT 3 training.

Two post-warmup shapes:

* ``cosine`` — classical 1/2·(1+cos(pi·t/T)), the current notebook default.
* ``litim`` — compact-support quadratic 1 - (t/T)^2 clamped to [0, 1].
  Litim's "optimised regulator" (Phys. Rev. D 64 105007, CERN CDS record
  492334; hep-th/0103195) is the functional RG analogue of a finite
  momentum cut-off. Compared with cosine, it:

    * finishes exactly at zero (not asymptotically),
    * is smooth at the boundary (first derivative is zero at both ends),
    * minimises the step-to-step change in the effective action during
      perturbative flows — i.e. small phase-transition jitter.

Both functions take ``(step, warmup_steps, total_steps)`` and return a
scalar multiplier in ``[0, 1]`` that the caller multiplies by ``peak_lr``.
Warmup is always a linear ramp so existing notebooks drop in unchanged.
"""

from __future__ import annotations
import math


def cosine_schedule(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(0.0, step / max(1, warmup_steps))
    t = step - warmup_steps
    T = max(1, total_steps - warmup_steps)
    frac = min(1.0, t / T)
    return 0.5 * (1.0 + math.cos(math.pi * frac))


def litim_schedule(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return max(0.0, step / max(1, warmup_steps))
    t = step - warmup_steps
    T = max(1, total_steps - warmup_steps)
    frac = min(1.0, t / T)
    return max(0.0, 1.0 - frac * frac)


def schedule_multiplier(
    step: int,
    warmup_steps: int,
    total_steps: int,
    shape: str = "cosine",
) -> float:
    if shape == "cosine":
        return cosine_schedule(step, warmup_steps, total_steps)
    if shape == "litim":
        return litim_schedule(step, warmup_steps, total_steps)
    raise ValueError(f"Unknown schedule shape: {shape!r} (expected 'cosine' or 'litim')")
