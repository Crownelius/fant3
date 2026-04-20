"""
Run-time monitors that watch for collapse / pathology and trigger interventions.

Each monitor is a callable that takes (step, model, telemetry_snap) and returns
a dict of warnings + a list of intervention callbacks to run before the next
optimizer step. The trainer calls them periodically.

The monitors implement the FANT 2 spec §11 "early warning + auto-repair" rules:

  1. RouterCollapseMonitor:
     If any single expert receives > 50% of routing across a probe batch, OR
     if mean_jsd between domains < 0.05 after warmup, trigger:
       - Tikkun repair on all routers
       - Optional: bump fep_kl_beta by 1.5x
       - Optional: log + alert

  2. CerebellumChaosMonitor:
     If the reservoir's spectral radius drifts outside [0.85, 0.99], log warning.

  3. ApollonianStarvationMonitor:
     If the alpha or beta pack hasn't received any new entries in N steps,
     log warning that the curvature threshold may need tuning.

  4. ParisiSymmetricMonitor:
     If the Parisi P(q) entropy is too low (< 1.0), the system has collapsed
     to a single replica — that's the symmetric phase, NOT the desired RSB.
     Trigger fanā dropout to break the symmetry.

  5. FractalDimensionMonitor:
     If the box-counting dimension of routing decisions falls below 0.5,
     the routing has become too predictable (no fractal structure).
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .telemetry import TelemetrySnapshot


@dataclass
class MonitorReport:
    name: str
    severity: str  # "info" | "warn" | "critical"
    message: str
    intervention: Optional[Callable] = None


class RouterCollapseMonitor:
    """Watch for the FANT 350M failure mode (router collapse onto a single expert)."""

    def __init__(self, jsd_threshold: float = 0.05, after_step: int = 500):
        self.jsd_threshold = jsd_threshold
        self.after_step = after_step

    def __call__(self, step: int, model, snap: TelemetrySnapshot) -> List[MonitorReport]:
        out = []
        if step < self.after_step:
            return out
        if snap.router_jsd_mean is None:
            return out
        if snap.router_jsd_mean < self.jsd_threshold:
            def fix():
                n = model.tikkun_repair_all()
                model.fana_dropout_all(p=1.0)
                return f"tikkun repaired {n} layers + fanā shuffled all"
            out.append(MonitorReport(
                name="RouterCollapse",
                severity="critical",
                message=(
                    f"router_jsd_mean = {snap.router_jsd_mean:.4f} below threshold "
                    f"{self.jsd_threshold}. The router has collapsed (FANT 350M failure mode). "
                    "Triggering Tikkun + fanā repair."
                ),
                intervention=fix,
            ))
        return out


class CerebellumChaosMonitor:
    """Watch for the cerebellum reservoir drifting off the edge of chaos."""

    def __init__(self, low: float = 0.85, high: float = 0.99, every_n_steps: int = 1000):
        self.low = low
        self.high = high
        self.every_n_steps = every_n_steps

    def __call__(self, step: int, model, snap: TelemetrySnapshot) -> List[MonitorReport]:
        if step == 0 or step % self.every_n_steps != 0:
            return []
        if not hasattr(model, "cerebellum"):
            return []
        sr = model.cerebellum.estimate_spectral_radius()
        if sr < self.low or sr > self.high:
            return [MonitorReport(
                name="CerebellumChaos",
                severity="warn",
                message=f"cerebellum spectral radius = {sr:.4f}, outside target [{self.low}, {self.high}]",
                intervention=None,
            )]
        return []


class ApollonianStarvationMonitor:
    """Watch for either pack failing to fill — but only when stores have been
    attempted. Phase 2 bulk pretraining hardcodes `store_to_memory=False`, so a
    zero-fill pack there is expected, not a bug. We use `model.memory.global_step`
    (incremented inside `ApollonianMemory.store()`) as the "stores attempted"
    signal: if it's still 0, we've never written, so starvation warnings are
    suppressed. Only when writes ARE happening but nothing lands in the pack
    do we treat it as a threshold-tuning problem.
    """

    def __init__(self, after_step: int = 1000, fill_threshold: float = 0.05):
        self.after_step = after_step
        self.fill_threshold = fill_threshold

    def __call__(self, step: int, model, snap: TelemetrySnapshot) -> List[MonitorReport]:
        if step < self.after_step:
            return []

        # Guard against false alarms from phases that don't populate memory.
        # Apollonian.global_step is only incremented inside .store(); if it's
        # zero, this phase has never called store_to_memory=True.
        mem = getattr(model, "memory", None)
        if mem is None:
            return []
        gs = getattr(mem, "global_step", None)
        writes = int(gs.item()) if gs is not None else 0
        if writes == 0:
            return []  # phase doesn't populate memory — zero-fill is by design

        out = []
        if snap.apollonian_alpha_fill is not None and snap.apollonian_alpha_fill < self.fill_threshold:
            out.append(MonitorReport(
                name="ApollonianAlphaStarvation",
                severity="warn",
                message=(
                    f"alpha pack fill = {snap.apollonian_alpha_fill:.3f} after {step} steps "
                    f"({writes} writes attempted). Curvature threshold may be too high — "
                    "almost nothing classified as instance memory."
                ),
            ))
        if snap.apollonian_beta_fill is not None and snap.apollonian_beta_fill < self.fill_threshold:
            out.append(MonitorReport(
                name="ApollonianBetaStarvation",
                severity="warn",
                message=(
                    f"beta pack fill = {snap.apollonian_beta_fill:.3f} after {step} steps "
                    f"({writes} writes attempted). Curvature threshold may be too low — "
                    "almost nothing classified as schema memory."
                ),
            ))
        return out


class ParisiSymmetricMonitor:
    """Trigger fana when the Parisi distribution is degenerate."""

    def __init__(self, min_entropy: float = 1.0, after_step: int = 500):
        self.min_entropy = min_entropy
        self.after_step = after_step

    def __call__(self, step: int, model, snap: TelemetrySnapshot) -> List[MonitorReport]:
        if step < self.after_step:
            return []
        if snap.parisi_p_q_entropy is None:
            return []
        if snap.parisi_p_q_entropy < self.min_entropy:
            def fix():
                model.fana_dropout_all(p=1.0)
                return "fanā shuffled all (Parisi entropy too low → symmetric phase)"
            return [MonitorReport(
                name="ParisiSymmetric",
                severity="warn",
                message=(
                    f"Parisi P(q) entropy = {snap.parisi_p_q_entropy:.3f} below {self.min_entropy}. "
                    "System is in the symmetric phase (replicas have collapsed). Triggering fanā."
                ),
                intervention=fix,
            )]
        return []


class FractalDimensionMonitor:
    """Watch for the box-counting dim of router decisions collapsing."""

    def __init__(self, min_dim: float = 0.5, after_step: int = 1000):
        self.min_dim = min_dim
        self.after_step = after_step

    def __call__(self, step: int, model, snap: TelemetrySnapshot) -> List[MonitorReport]:
        if step < self.after_step:
            return []
        if snap.box_counting_dim is None:
            return []
        if snap.box_counting_dim < self.min_dim:
            return [MonitorReport(
                name="FractalDimensionLow",
                severity="warn",
                message=(
                    f"box-counting dim of routing = {snap.box_counting_dim:.3f} below {self.min_dim}. "
                    "Routing decisions have become too predictable."
                ),
            )]
        return []


# -----------------------------------------------------------------------------
# Default monitor bundle
# -----------------------------------------------------------------------------

def default_monitors() -> List[Callable]:
    """The standard FANT 2 monitor stack."""
    return [
        RouterCollapseMonitor(),
        CerebellumChaosMonitor(),
        ApollonianStarvationMonitor(),
        ParisiSymmetricMonitor(),
        FractalDimensionMonitor(),
    ]


def run_monitors(monitors: List[Callable], step: int, model, snap: TelemetrySnapshot) -> List[MonitorReport]:
    """Run every monitor and return all reports + run any interventions."""
    reports: List[MonitorReport] = []
    for m in monitors:
        reports.extend(m(step, model, snap))
    for r in reports:
        if r.intervention is not None:
            try:
                msg = r.intervention()
                r.message = r.message + f"  [intervention: {msg}]"
            except Exception as e:
                r.message = r.message + f"  [intervention failed: {e}]"
    return reports
