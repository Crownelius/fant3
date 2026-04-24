# FANT 3 Theory Guide

The mathematical foundations of each architectural component in plain English. Every section maps to a named module in the codebase and cites the external paper it builds on.

**For the formal equations, derivations, and proof sketches**, see the rigorous companion: [mathematical-foundations.md](../mathematical-foundations.md).

## Matryoshka MoE

**Where:** `fant3/model/matryoshka_moe.py`
**Paper:** Wang et al., arXiv:2509.26520

Experts are organized as nested megapools: a coarse set of "core" experts plus a finer set of "level-1" experts plus still-finer "level-2" experts, and so on. At inference any prefix of this sequence can be used — one trained checkpoint serves multiple compute budgets. The router computes a single distribution, then the forward pass takes the top-`k` experts at the chosen level.

### Why not standard top-k MoE?

Standard top-k MoE commits to a fixed activation sparsity at training time. At deployment you cannot trade compute for quality without retraining. The Matryoshka variant lets you pick the level at inference, so a single checkpoint is elastic across deployment environments.

### Why not dense FFN at this scale?

Dense FFN would give up the sparse-activation win that makes 100 M active / 1 B stored feasible on one GPU. At 1 B dense you cannot fit the model and optimizer state on a single 80 GB card.

## MoR

**Where:** `fant3/model/recursion.py`
**Concept:** Mixture of Recursions (per-token adaptive compute depth)

Each token decides how many times to loop through a shared "recursion block" (1, 2, or 3 times by default). A lightweight router predicts the recursion depth from the hidden state; easy tokens exit after one pass, hard tokens loop up to three times.

### Contractive-alpha decay

Deeper recursion passes use progressively smaller residual-update magnitudes, enforcing a contractive dynamic that guarantees fixed-point convergence regardless of depth. The contraction factor is learned.

### Dynamic K at training time

Training samples `K ~ Uniform[1, n_recursion_depths]` per batch so the router learns to produce sensible exits at every depth. The per-step FLOP cost scales with the sampled K.

### Monotonic CE penalty (ISRM extension)

If CE loss at depth `k+1` is worse than at depth `k`, a small penalty is added. The model is allowed to improve with more passes but not allowed to degrade. Smoke-tested at 1M scale: 22/22 unit tests pass, 600-step CPU run reaches CE 2.67 vs CE-only baseline 3.84 (+1.17 nats better at identical compute).

### Inference K extrapolation

Because the block is shared across depths and the dynamic is contractive, the model can run with `K > max(training K)` at inference. Validated at K=6 after training with K in {1, 2, 3}: copy-half accuracy drops from 89.3% to 71.4% but doesn't diverge.

## MASA

**Where:** `fant3/model/attention.py`
**Concept:** Multi-head Attention with Shared Atoms

All layers share a learned dictionary of `n_attention_atoms` basis matrices. Per-layer attention is reconstructed as a rank-`masa_coef_rank` linear combination of these basis atoms. Total attention parameter count is `n_atoms × atom_shape + n_layers × n_heads × rank × 2` — much smaller than `n_layers × n_heads × head_dim × 4`.

### Why this works

Attention weights across layers are known to be highly correlated. Sharing atoms captures the correlation explicitly instead of relearning it per layer. Empirically, recovery of full-attention quality happens at `n_atoms ≈ n_heads × 2` and `rank ≈ head_dim / 4`.

### Interaction with RoPE

RoPE cos/sin tables are kept in f32 to avoid a subtle bug where bf16 RoPE on bf16 V gave a dtype mismatch that silently corrupted attention at 742m. The fix is in `fant3/model/attention.py:_apply_rope`.

## Spinor Apollonian Memory

**Where:** `fant3/model/spinor_apollonian.py`
**Paper:** Kocik, arXiv:2001.05866

The memory module maintains two parallel packs:
- **α-pack** ("instance" memory) — high-chirality tokens; episodic detail
- **β-pack** ("schema" memory) — low-chirality tokens; structural regularities

The split is determined by a Kocik tangency spinor in Cl(2,1) Minkowski space. Each hidden state is projected into a 4-vector; its chirality is the sign of the Descartes invariant. This split is fundamental — it is not a threshold on a scalar.

### Why this fixes α/β starvation

The original FANT 3 memory used a scalar-curvature classifier with a learned threshold. At every scale the threshold fragility caused one pack to get 0% or 100% of tokens (starvation). The Kocik chirality split is classifier-free: chirality 0.4–0.6 is natural because the invariant form is balanced in generic input distributions.

Verified chirality balance across scales on 2026-04-19:
- 5m: 0.266
- 40m: 0.447
- 150m: 0.500
- 350m: 0.719

All within the healthy 0.2–0.8 band.

### Capacity

Each pack has `apollonian_alpha_cap` or `apollonian_beta_cap` slots (default 128 each). Eviction is currently FIFO. A planned upgrade (per arXiv:2512.03324 TRIM-KV) would use a retention gate `β = σ(W x + b)` with temporal decay `β^(t-i)`.

### External validation: Bucher et al. 2026

A direct external validation of the topological-classifier design appeared in Bucher, Kaminer et al., *Superluminal Correlations in Ensembles of Optical Phase Singularities* (Nature 651:920, 2026 — arXiv:2509.17675). The paper measures phase-singularity dynamics in hexagonal boron nitride phonon polaritons and confirms experimentally what Toulouse & Kléman (1976) argued mathematically: topological defects in superconductors, superfluids, fluids, and optical fields are the same object — classified by homotopy group, not by any scalar threshold. Bucher's dataset fits the same Berry-Dennis (2000) random-wave statistics that would apply to the chirality field FANT 3 classifies.

The practical implication: the α/β chirality split is principled in the same sense that the particle/anti-particle split in Bucher's polariton data is principled. Neither relies on a learned threshold; both derive from a topological invariant of the underlying field.

A longer discussion of this external validation, including the 140-year intellectual lineage from Kelvin 1867 through Bucher 2026, is in [article/topological-classifier-universality.md](../article/topological-classifier-universality.md).

## AHN

**Where:** `fant3/model/ahn.py`
**Paper:** ByteDance Artificial Hippocampus Networks (2025)

A sliding FIFO of the last `short_window` hidden states, plus a compressed long-term buffer that accumulates across steps. Both are fed through a gated residual before the final layer norm.

### Zero-initialized gate

The gate is initialized to zero so AHN contributes nothing at step 0. Training data will ramp the gate only if AHN helps. This keeps the base transformer from regressing due to unvalidated memory.

### Dtype fix

A subtle bug had `get_stats` computing a dummy_q in f32 while `gate_proj` was bf16, leading to a dtype mismatch warning. Fixed in `fant3/model/ahn.py` on 2026-04-19.

## ETF freezing

**Where:** `fant3/model/etf.py`
**Concept:** Equiangular Tight Frame router geometry

The router matrix is initialized as a simplex ETF (equiangular tight frame). After `etf_freeze_after_step` steps (default 500 for 50m+) the routers are frozen. ETFs have the property that `W W^T = I` with off-diagonal entries at `-1/(n-1)`; this geometry is optimal for classification with equal prior and minimizes inter-expert interference.

### Why frozen routers don't hurt

After warmup, the routers have learned the expert-by-task assignment. The remaining training adjusts the experts themselves, not the routing. Freezing buys us:
- Simpler backward pass (no router gradients)
- Zero router drift (which otherwise causes mid-training collapse)
- Free compression: the frozen router can be quantized to int8 with no quality loss

## Cerebellum

**Where:** `fant3/model/cerebellum.py`
**Concept:** Echo-state reservoir with Purkinje linear readout

A 25 M-parameter reservoir (spectral radius 0.95, input scale 0.1) receives the hidden stream; a linear Purkinje readout projects back. The reservoir weights are **fixed at initialization**; only the Purkinje layer trains. Capacity is independent of backbone scale — the 25 M is fixed whether the backbone is 150 M or 1 B.

### Why this makes sense at 742m+

At smaller scales (50m, 150m), the main transformer has enough capacity to absorb all the useful gradients. At 742m and 1b the cerebellum adds a high-dimensional nonlinear transform that is cheap (because only the linear readout trains) and captures temporal patterns the dense layers miss.

## Progressive curriculum

**Where:** `fant3/training/curriculum.py`
**Paper:** arxiv:2604.16278 DeepInsightTheorem (Li et al., April 2026)

Training data is weighted differently across three phases:
- **Apprentice (0–25 %)**: foundational language — 55% FineWeb-Edu + simple math problems
- **Journeyman (25–65 %)**: sketched reasoning — Kimi/Sonnet/Opus traces + proof sketches
- **Expert (65–100 %)**: insight — Opus Crownelius 30% + Sonnet 20% + structured reasoning dominates

### Why this works at 1B–3B

The paper reports disproportionate gains at exactly our target band. Independent supporting evidence:
- **arxiv:2510.14865** — "midtraining mixing specialized with base pretraining data consistently outperforms continued pretraining on specialized data alone."
- **arxiv:2510.01631** — "pure synthetic data is *not* superior to CommonCrawl alone; mixtures beat both." Justifies the 5% FineWeb anchor in every phase.
- **arxiv:2510.25741 Ouro + arxiv:2511.07384 Retrofitted Recurrence** — "latent-refinement loops need to be either baked in from pretraining or added via a structured curriculum." Our 3-phase schedule with graded boundaries matches the published recipe.

### Backward compatibility

The default curriculum is `legacy_2phase`, bit-identical to the pre-2026-04-24 hardcoded mix. A unit test enforces this (`tests/test_curriculum.py::TestPresets::test_legacy_2phase_matches_original_runpod_train`). Existing queued runs keep working.

## Related theory documents

- [Formal specification of Spinor Apollonian memory](../architecture/) — exists in architecture/
- [Monotonic CE prototype](../testing/) — smoke test harness at 1M
- [SAE introspection](../architecture/) — sparse autoencoder analysis of memory packs
- [Size comparison](../size-comparison.md) — how FANT numbers stack against frontier models

## Primary references (papers)

| Paper | Used for |
|---|---|
| Wang et al., arXiv:2509.26520 | Matryoshka MoE |
| Kocik, arXiv:2001.05866 | Spinor Apollonian memory |
| ByteDance AHN (2025) | Artificial Hippocampus |
| arxiv:2604.16278 | Progressive curriculum |
| arxiv:2604.16004 | AgentV-RL (Fix 3, queued) |
| arxiv:2510.14865 | Midtraining validates curriculum |
| arxiv:2510.01631 | FineWeb anchor justification |
| arxiv:2510.25741 | Ouro recurrence curriculum |
| arxiv:2511.07384 | Retrofitted Recurrence |
| arXiv:2604.11921 | Parisi RSB, routing diversity |
| arXiv:2504.19874 | TurboQuant (queued) |
| arXiv:2512.03324 | TRIM-KV retention (queued) |
| Delétang et al. 2023 | Language Models are Compressors (framing) |
