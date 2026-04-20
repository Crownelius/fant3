"""
FANT 3 — rebuild of FANT 2 with four new architectural primitives:

  1. Matryoshka MoE routing (arxiv:2509.26520)
     Nested coarse-to-fine expert selection. Expert-0 learns coarse behavior,
     +1 adds detail, etc. Enables elastic inference (trade quality for speed).

  2. MASA shared-atom attention (arxiv:2508.04581)
     All Q/K/V/O matrices across all layers decompose into a shared dictionary
     of "atom" matrices + per-layer coefficients. 66.7% attention param savings
     at BERT/ViT parity.

  3. Mixture-of-Recursions — MoR (arxiv:2507.10524)
     Per-token dynamic recursion depth over a shared layer stack. α-tagged
     tokens route to shallow passes, β-tagged to deep. Integrates with
     Apollonian curvature classification.

  4. Intermediate ETF freezing (arxiv:2412.00884)
     Router weights frozen to simplex ETFs after early calibration. Free
     parameter compression with no accuracy loss.

Preserved from FANT 2:
  - N3 SleepGate memory consolidation (the +5.3pp validated lever)
  - Apollonian dual α/β memory with 2026-04-16 Phase 4 fixes
  - Kronecker 3-level expert factorization
  - Cerebellum reservoir (fixed at ~25M params; does NOT scale with model)
  - Hub attention + attention sinks
  - bf16 + 8-bit AdamW + gradient checkpointing recipe

Target: 1B stored / ~100M active on RTX 3060 12GB.

See: C:/Users/rsfit/.claude/projects/C--FANT/memory/project_fant3_rebuild_proposal_2026_04_16.md
"""

from .config import FANT3Config, fant3_1b, fant3_742m, fant3_smoke

__all__ = ["FANT3Config", "fant3_1b", "fant3_742m", "fant3_smoke"]
