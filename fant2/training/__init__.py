"""FANT 2 training subpackage."""

from .optimizer import (
    Muon,
    HybridOptimizer,
    newton_schulz5,
    partition_params_for_muon,
)
from .losses import (
    cross_entropy_loss,
    router_z_loss,
    fep_kl_prior,
    llm_jepa_loss,
    success_estimator_loss,
    fep_unified_loss,
    effective_rank,
    calibration_loss,
    kto_loss,
    simpo_loss,
    dr_grpo_loss,
)
from .telemetry import (
    TelemetrySnapshot,
    intrinsic_dimension_twonn,
    martin_mahoney_alpha,
    box_counting_dimension,
    mfdfa_singularity_width,
    avalanche_exponent_tau,
    router_jsd_pairwise,
    parisi_p_q_entropy,
    collect_telemetry,
)
from .monitors import (
    MonitorReport,
    RouterCollapseMonitor,
    CerebellumChaosMonitor,
    ApollonianStarvationMonitor,
    ParisiSymmetricMonitor,
    FractalDimensionMonitor,
    default_monitors,
    run_monitors,
)
from .trainer import (
    TrainConfig,
    FANT2Trainer,
)

__all__ = [
    # Optimizer
    "Muon", "HybridOptimizer", "newton_schulz5", "partition_params_for_muon",
    # Losses
    "cross_entropy_loss", "router_z_loss", "fep_kl_prior", "llm_jepa_loss",
    "success_estimator_loss", "fep_unified_loss", "effective_rank", "calibration_loss",
    "kto_loss", "simpo_loss", "dr_grpo_loss",
    # Telemetry
    "TelemetrySnapshot", "intrinsic_dimension_twonn", "martin_mahoney_alpha",
    "box_counting_dimension", "mfdfa_singularity_width", "avalanche_exponent_tau",
    "router_jsd_pairwise", "parisi_p_q_entropy", "collect_telemetry",
    # Monitors
    "MonitorReport", "RouterCollapseMonitor", "CerebellumChaosMonitor",
    "ApollonianStarvationMonitor", "ParisiSymmetricMonitor", "FractalDimensionMonitor",
    "default_monitors", "run_monitors",
    # Trainer
    "TrainConfig", "FANT2Trainer",
]
