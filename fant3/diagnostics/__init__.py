"""
fant3.diagnostics — post-hoc and training-time introspection tools for FANT 3.

``sae`` — offline Sparse Autoencoder (SAE) analysis of Apollonian memory
          contents. Not imported during training; for post-run analysis.
``spectral`` — cheap O(dim)-per-layer spectral-radius check for the
          Mythos-style LTI recurrence in MoR. SAFE to call every N training
          steps; runs under ``torch.no_grad()`` and reads only parameter
          tensors (no activations materialized).
"""

from .sae import ApollonianSAE, train_on_hidden_states, analyze_apollonian_memory
from .spectral import (
    MoRSpectralEntry,
    spectral_radius_report,
    format_report,
    assert_lti_stable,
)

__all__ = [
    "ApollonianSAE",
    "train_on_hidden_states",
    "analyze_apollonian_memory",
    "MoRSpectralEntry",
    "spectral_radius_report",
    "format_report",
    "assert_lti_stable",
]
