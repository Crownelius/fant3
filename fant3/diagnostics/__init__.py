"""
fant3.diagnostics — post-hoc introspection tools for FANT models.

NOT imported during training. All tools here are for offline analysis only.
"""

from .sae import ApollonianSAE, train_on_hidden_states, analyze_apollonian_memory

__all__ = [
    "ApollonianSAE",
    "train_on_hidden_states",
    "analyze_apollonian_memory",
]
