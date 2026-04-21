# FANT 3 Component Reference

This document walks through every file in `fant3/model/` (plus the two FANT 2 modules reused by FANT 3), explaining the algorithm, source paper, implementation location, and relevant config fields for each module.

---

## 1. `fant3/config.py` — Configuration Dataclass and Presets

### What it does

`FANT3Config` is a Python `@dataclass` that holds every architectural hyperparameter for a FANT 3 model. Preset functions (`fant3_smoke`, `fant3_20m`, `fant3_50m`, `fant3_742m`, `fant3_1b`) return pre-tuned instances. The trainer reads this object to construct the model; the Colab notebook exposes a `TARGET_SCALE` string that selects the right preset.

### Critical sizing note

FANT 3's MoE experts are stored as **full-rank** weight tensors of shape `(n_experts, dim, 2 * moe_hidden)`. The Kronecker fields (`kron_A_p`, `kron_B_p`, etc.) exist in the config but are **not yet used by the model code**. This means the dominant parameter budget is:

```
MoE params ≈ 4 × n_megapools × n_per_megapool × dim × 3 × moe_hidden
```

This formula caused a 6.6 billion-parameter preset to be committed for the `fant3_742m` name (using 128 experts, `dim=2048`, `moe_hidden=2048`). Both presets were fixed on 2026-04-19. See `CONFIGS.md` for verified counts and the full corrective note.

### Key config fields by group

**Core dimensions**

| Field | Default (1b) | Notes |
|---|---|---|
| `dim` | 1024 | Hidden dimension. Was 2048 before the preset fix. |
| `n_layers` | 20 | Total logical layers. |
| `n_dense_layers` | 3 | Dense prefix layers. |
| `n_heads` | 8 | Query heads. |
| `n_kv_heads` | 2 | KV heads (GQA-4 with n_heads=8). |
| `head_dim` | 128 | Per-head dimension. |
| `rope_partial` | 0.25 | Fraction of head_dim rotated by RoPE (Phi-4-Mini style). |
| `vocab_size` | 32768 | BPE (Byte-Pair Encoding) vocabulary size. |
| `max_seq_len` | 1024 | Maximum sequence length. |

**Matryoshka MoE**

| Field | Default (1b) | Notes |
|---|---|---|
| `n_megapools` | 4 | Number of expert pools. |
| `n_per_megapool` | 8 | Experts per pool; total = 32. Was 128 before fix. |
| `n_matryoshka_levels` | 2 | Nested band levels (1, 2 experts). |
| `top_k` | 2 | Not used by current router (router uses argmax + nested band); reserved. |
| `moe_hidden` | 2304 | Expert hidden dimension. |
| `shared_expert_hidden` | 640 | Always-active shared expert hidden size. |

**MASA attention**

| Field | Default (1b) | Notes |
|---|---|---|
| `masa_enabled` | True | Enable MASA atom sharing across layers. |
| `n_attention_atoms` | 5 | Number of shared atom matrices per projection (Q, K, V, O). |
| `masa_coef_rank` | 8 | Rank of per-layer coefficient tensor. |

**MoR (Mixture of Recursions)**

| Field | Default (1b) | Notes |
|---|---|---|
| `mor_enabled` | True | Wrap middle layers in `MoRShared`. |
| `n_recursion_depths` | 2 | Maximum number of times the shared block is applied to a token. |
| `mor_router_dim` | 128 | Bottleneck dimension of the depth router MLP. |
| `mor_depth_bias` | "alpha" | Use Apollonian chirality as a depth prior: α tokens → shallow, β → deep. |

**SpinorApollonianMemory**

| Field | Default (1b) | Notes |
|---|---|---|
| `apollonian_alpha_cap` | 10000 | FIFO capacity of the α (positive-chirality, instance) pack. |
| `apollonian_beta_cap` | 10000 | FIFO capacity of the β (negative-chirality, schema) pack. |
| `apollonian_curvature_threshold` | 0.5 | Used only by the legacy `ApollonianMemory`; ignored by spinor path. |
| `apollonian_retrieval_layers` | (18, 19) | Layer indices where memory retrieval augments attention. |
| `spinor_apollonian_enabled` | True | Use Kocik spinor classifier instead of scalar curvature. |

**AHN (Artificial Hippocampus Networks)**

| Field | Default (1b) | Notes |
|---|---|---|
| `ahn_enabled` | True | Enable the AHN gated residual. |
| `ahn_n_heads` | 4 | Attention heads inside AHN. |
| `ahn_short_window` | 256 | Short-term FIFO KV slots. |
| `ahn_long_capacity` | 512 | Long-term compressed KV slots. |
| `ahn_compress_ratio` | 0.25 | Latent dimension = `int(dim × compress_ratio)`. |

**Reservoir computing module (fixed size)**

| Field | Default | Notes |
|---|---|---|
| `cerebellum_enabled` | True | Enable the reservoir residual. |
| `cerebellum_in_dim` | 768 | Input projection dimension — deliberately NOT equal to `dim`. |
| `cerebellum_expand_dim` | 7680 | 10× fan-out granule layer. |
| `cerebellum_out_dim` | 768 | Output projection dimension. |
| `cerebellum_layers` | 4 | Reservoir recurrence iterations. |
| `cerebellum_spectral_radius` | 0.95 | Edge-of-chaos initialization target. |
| `cerebellum_sparsity` | 0.001 | Fraction of non-zero entries in the frozen reservoir matrix. |

**ETF (Equiangular Tight Frame) freezing**

| Field | Default | Notes |
|---|---|---|
| `etf_freeze_enabled` | True | Enable the ETF router freeze. |
| `etf_freeze_after_step` | 2000 | Training step at which router weights are frozen to ETF simplex. |
| `etf_freeze_layers` | range(3, 17) | Which logical layer indices have their router frozen. |

---

## 2. `fant3/model/attention.py` — MASA (Multi-head Attention with Shared Atoms)

### What it does algorithmically

Standard multi-head attention gives every layer its own independent Q, K, V, O projection matrices. For a 24-layer model with `dim=2048`, that is `4 × 24 × 2048² ≈ 402 M` parameters just for attention projections.

MASA (Multi-head Attention with Shared Atoms, arxiv:2508.04581, Zhussip et al. 2025) replaces this with a shared **atom bank**: a set of `n_attention_atoms` matrices per projection type. Each layer only stores small **coefficient tensors** of shape `(n_atoms, coef_rank)`, and assembles its weight matrix at forward time:

```
W_Q_layer = sum_i  (coef_q[i].sum() × A_Q[i])   for i in 1..n_atoms
```

The assembled `W_Q_layer` is `(dim, dim)`. For 5 atoms and rank 8, the per-layer overhead is `4 × 5 × 8 = 160` parameters instead of `4 × dim²`. Parameter savings scale with depth.

FANT 3 also uses:
- **GQA (Grouped-Query Attention)** (Ainslie et al. 2023): `n_kv_heads < n_heads` so K and V atoms are shaped `(n_atoms, dim, kv_dim)` where `kv_dim = n_kv_heads × head_dim`. At forward time K and V are repeated across query-head groups.
- **Partial RoPE (Rotary Position Embedding)** (Su et al. 2021): only the first `int(head_dim × rope_partial)` dimensions of Q and K receive sinusoidal rotations. The remainder pass through unchanged. This is the Phi-4-Mini strategy.

A dtype bug was fixed on 2026-04-19: the RoPE `cos`/`sin` tensors must be cast to `x.dtype` (bf16) before the rotation to avoid silently promoting the query to f32 and causing shape mismatches with V in PyTorch's scaled dot-product attention kernel.

### Source paper

arxiv:2508.04581 — Zhussip et al. "Share Your Attention" (2025).

### Implementation

- `fant3/model/attention.py`, class `MASAAtomBank` (lines 47–67): the shared dictionary, instantiated once per model.
- `fant3/model/attention.py`, class `MASAAttention` (lines 74–236): per-layer attention using `_assemble()` to build projection matrices on the fly.
- The atom bank is passed by reference from `FANT3Model.__init__` to every layer; it is NOT copied.

### Config fields

`n_attention_atoms`, `masa_coef_rank`, `n_heads`, `n_kv_heads`, `head_dim`, `rope_partial`, `rope_theta`.

---

## 3. `fant3/model/matryoshka_moe.py` — Matryoshka MoE (Mixture of Experts) FFN

### What it does algorithmically

Standard sparse MoE picks a fixed `top_k` experts per token. Matryoshka MoE (arxiv:2509.26520, Wang et al. 2025) trains with a **varying** number of active experts across training steps, arranged in nested bands of increasing size:

```
Level 0:  band = {expert_0}           — learns coarse behavior
Level 1:  band = {expert_0, expert_1} — adds detail to level 0
Level 2:  band = {expert_0..3}        — adds further detail
```

Because the hierarchy is nested, activating only level 0 at inference time produces a valid (if coarser) output — **elastic inference** without retraining.

The FANT 3 router is two-stage:
1. **Megapool selection**: a `dim → n_megapools` linear head with softmax selects which pool of `n_per_megapool` experts this token uses.
2. **Level selection**: a `dim → n_matryoshka_levels` linear head selects how many experts within that pool to activate.

The final aggregation is a **uniform weighted sum** over the active band, plus a smaller always-active **shared expert** (a SwiGLU FFN with a gated scalar). The shared expert gate starts at zero so it contributes nothing until the model learns to use it.

Load balancing uses DeepSeek-V3 style **bias correction buffers** (additive bias on routing logits, not auxiliary loss) and EMA load tracking for Tikkun repair. An OLMoE-style **z-loss** and an FEP (Free Energy Principle) KL prior toward uniform load are available for the trainer.

### Source paper

arxiv:2509.26520 — Wang et al. "Matryoshka Mixture of Experts" (2025).

### Implementation

- `fant3/model/matryoshka_moe.py`, class `MatryoshkaRouter` (lines 48–150): two projections + bias buffers + EMA tracking.
- `fant3/model/matryoshka_moe.py`, class `MatryoshkaMoEFFN` (lines 157–273): full block including the level-iteration dispatch loop, SwiGLU expert forward, and shared expert.
- Expert weights are stored as a `(n_experts, dim, 2 * moe_hidden)` parameter tensor. The Kronecker factorization fields in `FANT3Config` are reserved for a future upgrade and are not read by the current code.

### Config fields

`n_megapools`, `n_per_megapool`, `n_matryoshka_levels`, `moe_hidden`, `shared_expert_hidden`, `n_special`.

---

## 4. `fant3/model/recursion.py` — MoR (Mixture of Recursions)

### What it does algorithmically

MoR (arxiv:2507.10524, Bae et al. 2025, NeurIPS 2025) replaces a stack of `N` distinct transformer layers with a **single shared layer** applied **1 to K times** per token, where K is chosen by a lightweight router. Tokens that carry complex or schema-like information get more compute; tokens that are easy or recent get less.

The implementation in FANT 3 uses a simple "mask-and-accumulate" strategy: the shared block is called `max_depth` times on the full batch, and after each call a boolean mask writes back only the tokens whose chosen depth is at least that pass index. This wastes some compute on easy tokens but is GPU-friendly (no gather/scatter by group in v1).

An optional **curvature-informed depth bias** can be applied: if `mor_depth_bias == "alpha"`, tokens with Apollonian curvature above the median are shifted toward shallower depths (α = instance memory = simple tokens). This requires Phase 4 memory population to be active.

Gradient checkpointing is applied to each recursion pass independently when `use_gradient_checkpointing=True`, because without it each MoR pass stores a full activation copy — multiply by `n_recursion_depths` for total cost.

### Source paper

arxiv:2507.10524 — Bae et al. "Mixture of Recursions" (NeurIPS 2025).

### Implementation

- `fant3/model/recursion.py`, class `MoRDepthRouter` (lines 34–65): two-layer MLP, bottleneck through `mor_router_dim`.
- `fant3/model/recursion.py`, class `MoRShared` (lines 68–156): wraps one `MoEBlock` and calls it up to `max_depth` times.
- In `FANT3Model.__init__`, the MoR wrapper is instantiated at line 151: `self.mor = MoRShared(cfg, shared_middle)`.

### Config fields

`mor_enabled`, `n_recursion_depths`, `mor_router_dim`, `mor_depth_bias`.

---

## 5. `fant3/model/spinor_apollonian.py` — SpinorApollonianMemory

### What it does algorithmically

This module is the default long-term memory store for FANT 3, replacing the legacy scalar-curvature `ApollonianMemory` from FANT 2. The theoretical motivation and mathematical foundations are covered in detail in `MATH.md`; the algorithmic description here focuses on the implementation.

The module maintains two **FIFO circular buffers** (α pack and β pack), each of fixed capacity `alpha_cap` / `beta_cap`. New items are classified into α or β via the sign of the second component of a learned 2D spinor projection:

```
s = proj_spinor(h)      h ∈ ℝ^dim → s ∈ ℝ²
chirality = sign(s[1])  → α if > 0,  β if ≤ 0
```

The spinor projection is the only trainable parameter in this module (a 2×dim weight matrix). It is initialized with `std=0.01` so that the chirality split starts near 50/50.

**Store** (`store(embeddings, hidden_preRMSnorm)`): takes post-norm embeddings as the retrieval keys, but computes spinors from pre-RMSNorm hiddens for a richer signal. Items are written to their respective FIFO pack; oldest items are overwritten when the pack is full.

**Retrieve** (`retrieve(query, top_k, pool)`): combines cosine similarity (70% weight) with the Clifford bilinear form `a[0]b[0] − a[1]b[1]` (30% weight) to produce a score. Returns the top-k items from the specified pool.

**Descartes regularizer** (`descartes_loss`): an optional training signal that measures how far the local 4-spinor neighborhood of each query deviates from the Descartes circle theorem `(Σbᵢ)² = 2Σbᵢ²`. This encourages the memory to organise itself into a self-similar Apollonian packing.

**SleepGate consolidation** (`sleep_consolidate`): evicts stale entries from the α pack and greedy-merges near-duplicate embeddings (cosine > `merge_threshold`). Compatible with the N3 SleepGate training schedule (every 100 steps, +5.3pp accuracy gain validated on FANT 2).

The starvation bug fixed by this module: the legacy `ApollonianMemory` used L2 norm as a curvature proxy. At scale, all hidden-state norms cluster in `[0.99, 1.01]` (near-unit norm due to RMSNorm), so the scalar threshold classifies every item as α and the β pack starves. Chirality balance of 0.27–0.72 was observed across all scales on the scale-ladder run (2026-04-19), confirming the fix.

### Source papers

- Kocik, J. (2001). "Clifford algebras and Euclid's parametrisation of Pythagorean triples." arxiv:2001.05866 — tangency spinors, §3.
- Boyd, Lagarias, Mallows, Wilks (2003) — Apollonian circle packings.

### Implementation

- `fant3/model/spinor_apollonian.py`, class `SpinorApollonianMemory` (lines 101–589).
- `clifford_bilinear()` (lines 71–81) and `clifford_norm()` (lines 84–94) are module-level helper functions.
- Called from `FANT3Model.forward()` at lines 298–310 when `store_to_memory=True`.

### Config fields

`spinor_apollonian_enabled`, `apollonian_alpha_cap`, `apollonian_beta_cap`, `apollonian_retrieval_layers`, `apollonian_curvature_threshold` (used only by the legacy path).

---

## 6. `fant3/model/ahn.py` — AHN (Artificial Hippocampus Networks)

### What it does algorithmically

The AHN (Artificial Hippocampus Networks, ByteDance 2025) provides a **two-tier online memory** that operates as a gated residual applied just before the final RMSNorm. Unlike SpinorApollonianMemory (which stores compressed semantic representations across many training steps), AHN operates within a single forward pass over recent token positions.

**Tier 1 — Short-term sliding window**: a FIFO buffer of the most recent `short_window` (256) token K/V pairs at full dimension. The query attends to this buffer at every forward call.

**Tier 2 — Long-term compressed memory**: when a K/V pair is evicted from the short-term buffer, it is compressed by a learned linear `compressor` (dim → `latent_dim = dim × compress_ratio`) and pushed into a second FIFO of capacity `long_capacity` (512 compressed slots). At attention time the long-term keys and values are decompressed back to full dimension by a learned `decompressor`.

**Gated blend**: a 2-way softmax gate (conditioned on the mean-pooled query) controls the ratio of short-term vs. long-term output:

```
out = alpha_short × attn(q, K_short, V_short)
    + alpha_long  × attn(q, K_long_dec, V_long_dec)
```

The gate projection is zero-initialized, so the AHN contributes nothing at training start. The outer `ahn_gate` scalar parameter in `FANT3Model` is also zero-initialized, providing a second guard.

Buffer updates happen in `torch.no_grad()`. The compressor and decompressor always receive a gradient path via the current-token latents concatenated onto the buffer at attention time.

### Source paper

ByteDance "Artificial Hippocampus Networks" (2025). Internal citation; committed to MemPalace as `lab_bytedance_paper_*`.

### Implementation

- `fant3/model/ahn.py`, class `ArtificialHippocampusNetwork` (lines 55–411).
- Called from `FANT3Model.forward()` at lines 275–277.
- The outer `ahn_gate` zero-init is at `fant3_model.py` line 204.

### Config fields

`ahn_enabled`, `ahn_n_heads`, `ahn_short_window`, `ahn_long_capacity`, `ahn_compress_ratio`.

---

## 7. `fant3/model/etf.py` — ETF (Equiangular Tight Frame) Router Freezing

### What it does algorithmically

This module implements the "neural collapse + ETF freezing" trick from arxiv:2412.00884. The key insight from that paper is that modern classifiers converge during training to a configuration where the final-layer weight rows approach a **simplex ETF** — a set of `k` unit vectors with equal pairwise cosine angle `−1/(k−1)`. This is the maximally separated configuration for `k` points on the unit sphere in `ℝ^d` (with `k ≤ d+1`).

Since the router is going to converge to an ETF anyway, we can **freeze it there early** and save the gradient computation and memory for those parameters for the rest of training. In FANT 3, this applies to the Matryoshka MoE router's `megapool_proj` and `level_proj` linear heads.

`simplex_etf(k, dim)` computes the exact ETF via SVD on the centered identity matrix:
1. `M = I_k − (1/k) J_k` (centered identity; rows sum to zero, pairwise cosine = −1/(k−1))
2. Normalize rows to unit norm
3. Project onto the (k−1)-dimensional principal subspace via SVD, pad with zeros to `dim`
4. Optionally rotate by a random orthogonal matrix (so the ETF is not axis-aligned)

`freeze_linear_to_etf(linear)` overwrites the weight in-place and sets `requires_grad=False`.

### Source paper

arxiv:2412.00884 — "Leveraging Intermediate Neural Collapse with Simplex ETFs" (2024).

### Implementation

- `fant3/model/etf.py`, function `simplex_etf()` (lines 31–79).
- `fant3/model/etf.py`, function `freeze_linear_to_etf()` (lines 89–99).
- Called from `FANT3Model.freeze_intermediate_routers_to_etf()` at `fant3_model.py` lines 323–340, triggered by the trainer once `etf_freeze_after_step` is reached.

### Config fields

`etf_freeze_enabled`, `etf_freeze_after_step`, `etf_freeze_layers`.

---

## 8. `fant2/model/cerebellum.py` — Reservoir Computing Module

### What it does algorithmically

This module implements an **echo-state network** (reservoir computing, Jaeger 2001; Maass 2002) as a parallel residual path. It is reused unchanged from FANT 2 and occupies a **fixed parameter budget of ~11.8 M** regardless of the main model's `dim`, because all reservoir dimensions are hard-coded in the config (default 768→7680→768).

The architecture follows the biological cerebellum circuit:

1. **Mossy fiber projection** (`mossy_proj`, learned): lifts `in_dim → expand_dim` with a 10× fanout, applying `tanh` nonlinearity.
2. **Edge-of-chaos reservoir** (frozen sparse matrix): a random sparse weight matrix whose spectral radius is rescaled to 0.95 using power iteration. This is the Bertschinger-Natschläger (2004) "edge of chaos" region. The reservoir runs for `n_layers` (4) leaky integration steps: `h ← (1−leak)×h + leak×tanh(W_res×h + mossy)`. The leak rate is a learned scalar.
3. **Purkinje linear readout** (`purkinje`, learned): projects `expand_dim → out_dim` — the linear readout of classical echo-state network theory.
4. **Output RMSNorm**.

The reservoir matrix is stored as sparse (rows, cols, vals) buffers and applied via `index_add_`. It is a **non-parameter buffer** and does not appear in the optimizer state.

The reservoir computing module output is added as a gated residual in `FANT3Model`:

```python
x = x + torch.sigmoid(self.cereb_gate) * cereb_out
```

The `cereb_gate` scalar starts at zero.

### Source papers

- Jaeger, H. (2001). "The echo state approach to analysing and training recurrent neural networks." GMD Technical Report.
- Maass, W. et al. (2002). "Real-time computing without stable states." Neural Computation.
- Bertschinger, N. & Natschläger, T. (2004). "Real-time computation at the edge of chaos."

### Implementation

- `fant2/model/cerebellum.py`, class `CerebellumModule` (lines 155–278).
- Called from `FANT3Model.forward()` at lines 267–271.

### Config fields

`cerebellum_enabled`, `cerebellum_in_dim`, `cerebellum_expand_dim`, `cerebellum_out_dim`, `cerebellum_layers`, `cerebellum_spectral_radius`, `cerebellum_sparsity`.

---

## 9. `fant2/model/apollonian.py` — Legacy ApollonianMemory

### What it does algorithmically

This is the original FANT 2 long-term memory module, kept for A/B comparison. It uses the L2 norm of embeddings as a proxy for Apollonian curvature (`curvature = ‖e‖ / ref_norm`) and splits items into α (high curvature, above threshold) and β (low curvature, below threshold) packs using a fixed scalar threshold.

The starvation bug that motivated the spinor replacement: after RMSNorm, all hidden-state norms are near 1.0, so the fixed threshold fails to separate the populations and one pack receives almost all entries. The `SpinorApollonianMemory` module fixes this by using 2D chirality instead.

The `sleep_consolidate` method (the N3 SleepGate implementation) evicts stale entries and greedy-merges near-duplicate embeddings with cosine > 0.92. This is the mechanism that produced the +5.3pp accuracy gain in the N3 campaign.

### Implementation

- `fant2/model/apollonian.py`, class `ApollonianMemory` (lines 49–451).
- Used in `FANT3Model` only when `spinor_apollonian_enabled=False`.

---

## 10. `fant3/diagnostics/sae.py` — Sparse Autoencoder Diagnostics

This module provides a Sparse Autoencoder (SAE) for post-hoc feature extraction from hidden states, following the Anthropic Monosemanticity and Scaling Sparse Autoencoders work. It is a diagnostic/analysis tool and is not used during the main training loop. See `fant3/diagnostics/sae.py` for implementation details.
