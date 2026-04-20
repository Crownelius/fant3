"""FANT 3 model components."""

from .attention       import MASAAtomBank, MASAAttention
from .matryoshka_moe  import MatryoshkaRouter, MatryoshkaMoEFFN
from .recursion       import MoRDepthRouter, MoRShared
from .etf             import simplex_etf, freeze_linear_to_etf, etf_quality
from .fant3_model     import FANT3Model, DenseBlock, MoEBlock, DenseSwiGLU

__all__ = [
    "MASAAtomBank", "MASAAttention",
    "MatryoshkaRouter", "MatryoshkaMoEFFN",
    "MoRDepthRouter", "MoRShared",
    "simplex_etf", "freeze_linear_to_etf", "etf_quality",
    "FANT3Model", "DenseBlock", "MoEBlock", "DenseSwiGLU",
]
