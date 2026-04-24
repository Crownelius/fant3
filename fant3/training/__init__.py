"""fant3.training — trainer utilities (schedulers, aux losses, curriculum)."""
from .schedulers import litim_schedule, cosine_schedule, schedule_multiplier
from .optim import precondition_router_grads_
from .curriculum import (
    PhaseSpec,
    Curriculum,
    PRESETS,
    get_active_phase,
    get_phase_index,
    load_preset,
    build_curriculum,
)

__all__ = [
    "litim_schedule",
    "cosine_schedule",
    "schedule_multiplier",
    "precondition_router_grads_",
    "PhaseSpec",
    "Curriculum",
    "PRESETS",
    "get_active_phase",
    "get_phase_index",
    "load_preset",
    "build_curriculum",
]
