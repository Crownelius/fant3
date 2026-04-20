"""
FANT 2 — Fractal Atomic Neural Topology v2

A 60M-stored / 200M-active fractal-Apollonian language model with:
- 72 unique fractal-seed experts in 8 mega-pools of 9
- 3-level Kronecker weight hierarchy A ⊗ B ⊗ C
- Hierarchical Apollonian router with 6-tradition convergence fixes
- Hub attention (32 VEN-analog tokens) + 4 sinks + sliding window=128
- Parallel cerebellum module (echo-state reservoir + Mallat scattering)
- Apollonian α/β memory with cross-attention retrieval
- Custom byte-level BPE tokenizer (vocab=32768)
- Streaming HuggingFace datasets (≤10 GB disk)
- 7-phase training pipeline (BPE → JEPA → MoE → Calib → Refine → GRPO → SimPO)

Target hardware: single NVIDIA RTX 3060 12GB.
See ../fant2_architecture_spec.md for the full specification.
"""

__version__ = "2.0.0-dev"
__authors__ = ["FANT Team"]
