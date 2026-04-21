# Architectural Decision Records (ADRs)

This directory contains ADRs (Architectural Decision Records) for FANT 3 in MADR (Markdown Any Decision Record) format.

Each ADR records a significant architectural decision: the context that motivated it, the choice made, the consequences, and the alternatives that were considered and rejected.

---

## Index

| ADR | Title | Status | Date |
|---|---|---|---|
| [0001](0001-matryoshka-moe-over-standard-moe.md) | Matryoshka MoE over Standard Top-k MoE | Accepted | 2026-04-16 |
| [0002](0002-spinor-apollonian-over-scalar-curvature.md) | SpinorApollonianMemory over Scalar Curvature Classifier | Accepted | 2026-04-19 |
| [0003](0003-nvidia-datasets-over-community.md) | NVIDIA Reasoning Datasets as Primary Training Signal | Accepted | 2026-04-19 |
| [0004](0004-gradient-checkpointing-for-742m-plus.md) | Gradient Checkpointing Mandatory at 742m and Above | Accepted | 2026-04-19 |

---

## How to read an ADR

Each ADR answers four questions:

1. **Context** — What problem were we facing? What were the constraints?
2. **Decision** — What did we choose, and at what level of detail?
3. **Consequences** — What are the tradeoffs? What did we gain and what did we give up?
4. **Alternatives Considered** — What did we explicitly reject, and why?

ADRs are write-once by convention: once accepted, the document is not edited. If a decision is reversed, a new ADR is written with status "Supersedes ADR NNNN".

---

## Decisions not yet written

The following decisions were made but not yet formalized as ADRs:

| Topic | Summary |
|---|---|
| MASA over standard GQA | All layers share `n_attention_atoms` basis matrices; per-layer rank-`masa_coef_rank` coefficients. Reduces parameter count while sharing structure across depth. |
| MoR over uniform depth | Per-token recursion depth chosen by a lightweight router; α tokens (shallow/instance) vs β tokens (deep/schema). |
| AHN as gated residual | AHN gate zero-initialized so it starts as a no-op, preserving training stability at step 0. |
| ETF router initialization | Simplex ETF (Equiangular Tight Frame) initialization prevents router collapse without any aux loss, then frozen after `etf_freeze_after_step`. |
| BPE tokenizer retrain (Fix 1b) | Tokenizer retrained on 6-source distillation mix; 10–18% compression gain over the base tokenizer; special token IDs preserved. |
| GSPO RL deferred to post-pretrain | Campaign N showed aux losses hurt at 5M scale; deferred Fix 3 (GSPO RL) until after the pretrain checkpoint is established. |
