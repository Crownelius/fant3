# ADR 0001: Matryoshka MoE (Mixture of Experts) over Standard Top-k MoE

## Status

Accepted (implemented 2026-04-16, production-validated 2026-04-19)

---

## Context

FANT 2 used a `HierarchicalApollonianRouter` with fixed top-k expert selection per token. This worked but had two problems:

1. **Fixed compute budget at inference**: top-k MoE activates exactly k experts regardless of token difficulty. Easy tokens waste compute on k experts; hard tokens cannot exceed k.
2. **Training instability at small scale**: with 72 experts in 8 megapools, the router needed careful aux-loss tuning (z-loss, KL-to-prior, Tikkun, Fanā) to prevent collapse. FANT 350M lost 94.5% of routing onto a single expert in one epoch without these defenses.

For FANT 3, we needed a routing scheme that:
- Supports elastic inference (vary compute at test time without retraining)
- Has a natural inductive bias toward coarse-to-fine reasoning
- Scales more gracefully to 1B parameters

Standard top-k MoE (as used in Mixtral, OLMoE, DeepSeek-MoE) satisfies none of these.

---

## Decision

Implement **Matryoshka MoE** (Wang et al., arXiv:2509.26520) with FANT-specific integration:

- Experts within each megapool are arranged in nested bands: band 0 = 1 expert (coarse), band 1 = 2 experts (adds first-order detail), band 2 = 4 experts, band L = 2^L experts
- A two-stage router selects (1) which megapool and (2) which Matryoshka level per token
- The level selection is trained with random level dropout during training so every prefix of experts is competent independently
- Lower-level experts see more training examples (monotone nesting) → better generalization
- Higher-level experts see only the residuals that lower levels cannot explain → natural specialization

FANT 3 default: `n_megapools=4`, `n_per_megapool=8`, `n_matryoshka_levels=2`. At 742m scale: 32 total experts.

**Implementation note (critical):** `MatryoshkaMoEFFN` allocates full-rank expert weights `torch.randn(n_experts, dim, 2*moe_hidden)`. The `kron_*` config fields inherited from FANT 2 are currently dead code. The correct parameter count formula is:

```
MoE params ≈ 4 × n_total_experts × dim × moe_hidden
```

Presets were calibrated to this formula on 2026-04-19 after discovering the original presets produced 6.6B (742m) and ~7B (1b) parameters.

---

## Consequences

**Benefits:**
- Elastic inference: at test time, can cap the Matryoshka level to reduce compute by 2x–4x with graceful degradation
- Coarse-to-fine inductive bias matches how humans approach problems (broad context first, details second)
- Simpler router collapse defenses: nested structure means even collapsed routing still activates expert 0, which is the most-trained expert
- Monotone improvement guarantee: more experts never hurts (level L+1 extends level L, never contradicts it)

**Drawbacks:**
- Band-based gating is more complex than flat top-k
- The Kronecker factorization from FANT 2 is currently disabled (experts are full-rank), losing the parameter efficiency of the `kron_*` design
- Expert count must be a power-of-2 per megapool for clean band boundaries

**Known issues:**
- `W_up` gather inside gradient checkpointing recompute creates large transient tensors `(M, band_size, D, 2*hidden)` — the dominant VRAM consumer at 742m. At B=2, T=1024, this caused a 93 GB peak on A100 96 GB. Solution: B=1 with GRAD_ACCUM=8.

---

## Alternatives Considered

**Standard top-k MoE (Mixtral / OLMoE style)**  
Rejected: no elastic inference, no coarse-to-fine inductive bias, requires more careful aux-loss tuning to prevent collapse.

**Mixture-of-Depths (token dropping)**  
Rejected: token dropping loses positional information and requires masked attention masks, incompatible with MASA (Multi-head Attention with Shared Atoms) shared-atom attention.

**Switch Transformer (top-1 routing)**  
Rejected: top-1 per token is too coarse for math/code reasoning; expert collapse risk is higher.

**Expert Choice routing (each expert picks tokens)**  
Noted as a future direction (consistent with FANT 3 theory per HF archive survey 2026-04-18), but deferred — requires fixed-size batch processing incompatible with variable-length streaming data.
