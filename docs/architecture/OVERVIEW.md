# FANT 3 Architecture Overview

**FANT (Fractal Atomic Neural Topology) version 3** is a research language model that combines several recent architectural innovations into a unified system targeting approximately 770 million to 1 billion stored parameters, designed to train on a single NVIDIA RTX 3060 12 GB GPU using bf16 (bfloat16) precision, 8-bit AdamW, and gradient checkpointing.

---

## Design Philosophy

FANT 3 is built around three ideas:

1. **Parameter efficiency through sharing.** Rather than giving every layer its own independent weight matrices, FANT 3 shares attention projections across layers (via MASA (Multi-head Attention with Shared Atoms)) and shares entire transformer blocks across recursion depths (via MoR (Mixture of Recursions)).

2. **Compute elasticity.** The Matryoshka MoE (Mixture of Experts) routing allows the model to activate fewer experts at inference time without retraining, enabling a smooth compute-accuracy trade-off.

3. **Biologically motivated memory.** Two complementary memory systems — SpinorApollonianMemory (long-term associative store split by Clifford-algebra chirality) and AHN (Artificial Hippocampus Networks, a short-term sliding-window + compressed long-term store) — complement the feedforward layers without adding parameters to the main optimization loop.

---

## High-Level Layer Plan

For the production `fant3_742m` preset (16 layers, 3 dense prefix, 2 dense suffix, 11 shared middle):

```
Layer 0..1   DenseBlock    MASA attention + dense SwiGLU FFN (n_dense_layers=2)
                            ↑ These layers use a standard SwiGLU FFN; no MoE.

Layer 2..12  MoRShared     ONE shared MoEBlock applied 1..2 times per token
                            ↑ 11 logical layers with the parameter cost of 1.

Layer 13..15 MoEBlock (×3) Distinct MoE suffix blocks (n_suffix=3)
                            ↑ Memory retrieval hooks in during these layers.

             CerebellumModule (fixed 768→7680→768, parallel residual, gated)
             ArtificialHippocampusNetwork (gated residual before final norm)
             RMSNorm (Root Mean Square Normalization) + tied LM head
```

For the default `fant3_1b` preset (20 layers, 3 dense prefix, 3 dense suffix, 14 shared middle), the same plan applies scaled up.

---

## Core Components (summary)

| Component | Purpose | Source |
|---|---|---|
| MASA attention | Share Q/K/V/O atom matrices across all layers; only per-layer scalar coefficients vary | arxiv:2508.04581 |
| GQA (Grouped-Query Attention) | `n_kv_heads < n_heads` reduces KV cache size | Ainslie et al. 2023 |
| Partial RoPE (Rotary Position Embedding) | Rotate only 25% of head dimensions (Phi-4-Mini style) | Su et al. 2021 |
| MoR shared block | Apply one shared MoEBlock 1–2 times per token, depth chosen by router | arxiv:2507.10524 |
| Matryoshka MoE FFN | Two-stage router: megapool selection + nested band expert activation | arxiv:2509.26520 |
| SpinorApollonianMemory | Dual α/β memory pack classified via Clifford Cl(2,1) spinor chirality | arxiv:2001.05866 |
| AHN | Sliding short-term KV window + compressed long-term memory, gated residual | ByteDance 2025 |
| Reservoir computing module | Fixed-size echo-state reservoir (spectral radius 0.95), parallel residual | Maass 2002, Jaeger 2001 |
| ETF (Equiangular Tight Frame) freezing | Freeze router weights to simplex ETF after calibration | arxiv:2412.00884 |
| RMSNorm | Pre-norm on all attention and FFN sub-layers | Zhang & Sennrich 2019 |
| Tied LM head | LM head shares weights with token embedding (saves `vocab_size × dim` params) | standard practice |

---

## Data Flow Summary

```
input_ids (B, T)
    → tok_emb           (B, T, dim)     token embedding
    → DenseBlock × n_dense_layers       pre-norm MASA + SwiGLU FFN
    → MoRShared                         shared MoEBlock × [1..n_recursion_depths] per token
    → MoEBlock × n_suffix               suffix MoE blocks; memory retrieval augments attention
    → CerebellumModule (if enabled)     parallel echo-state residual, gated
    → AHN (if enabled)                  hippocampal memory residual, gated
    → RMSNorm → lm_head                 tied projection to vocab_size
    → logits (B, T, vocab_size)
```

Cross-entropy loss is computed only when `targets` is provided. Memory store (to SpinorApollonianMemory) occurs only when `store_to_memory=True` (Phase 4+ training).

---

## Source Files

| File | Role |
|---|---|
| `fant3/config.py` | `FANT3Config` dataclass + all preset functions |
| `fant3/model/fant3_model.py` | Top-level `FANT3Model`, `DenseBlock`, `MoEBlock` |
| `fant3/model/attention.py` | `MASAAtomBank`, `MASAAttention` |
| `fant3/model/matryoshka_moe.py` | `MatryoshkaRouter`, `MatryoshkaMoEFFN` |
| `fant3/model/recursion.py` | `MoRDepthRouter`, `MoRShared` |
| `fant3/model/spinor_apollonian.py` | `SpinorApollonianMemory`, `clifford_bilinear` |
| `fant3/model/ahn.py` | `ArtificialHippocampusNetwork` |
| `fant3/model/etf.py` | `simplex_etf`, `freeze_linear_to_etf` |
| `fant2/model/cerebellum.py` | `CerebellumModule` (reservoir computing module, reused from FANT 2) |
| `fant2/model/apollonian.py` | `ApollonianMemory` (legacy scalar-curvature memory, kept for A/B comparison) |

---

## Hardware Target

The primary training target is a single NVIDIA RTX 3060 12 GB GPU using:

- bf16 (bfloat16) weights and activations
- 8-bit AdamW optimizer (bitsandbytes)
- Gradient checkpointing on all DenseBlock, MoEBlock, and MoR recursion passes
- Batch size 1–2, sequence length 512–1024, gradient accumulation 4–8

The 742m preset (`fant3_742m`) has been validated on an NVIDIA A100 96 GB (Colab) and on a local RTX 3060 for smoke tests. The 1b preset (`FANT3Config()` defaults) targets the A100 for the flagship pretrain run.
