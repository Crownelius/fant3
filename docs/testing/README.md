# Tests — Walkthrough

All test files live in the `tests/` directory. Run the full suite with:

```bash
PYTHONPATH=. pytest tests/ -v
```

Individual files can be run as standalone scripts:

```bash
PYTHONPATH=. python tests/test_spinor_apollonian.py
PYTHONPATH=. python tests/test_ahn.py
PYTHONPATH=. python tests/test_sae.py
```

---

## `tests/conftest.py`

**Purpose:** Shared pytest fixtures available to all test files in the suite.

**Fixtures defined:**

| Fixture | Scope | What it provides |
|---------|-------|-----------------|
| `tiny_cfg` | session | `fant2_tiny()` config — a minimal FANT 2 configuration for fast CPU testing |
| `tiny_model` | function | Fresh `FANT2Model(tiny_cfg)` with `torch.manual_seed(0)` — recreated for each test to prevent weight leakage |
| `device` | session | `"cpu"` — tests run on CPU for reproducibility |
| `synthetic_batch` | function | `(input_ids, target_ids)` tensor pair, shape `(2, 32)`, with `torch.manual_seed(42)` |

---

## `tests/test_smoke.py`

**Module under test:** `fant2` package — all public subpackages.

**Test count:** 11

**What it covers:**

| Test | Invariant asserted |
|------|-------------------|
| `test_import_fant2_package` | `import fant2` succeeds without error |
| `test_import_model_subpackage` | All model classes importable: `FANT2Model`, `RMSNorm`, `FractalSeedExpert`, `ZeroExpert`, `CopyExpert`, `SharedNarrowExpert`, `DenseSwiGLU`, `HierarchicalApollonianRouter`, `FractalMoELayer`, `HubAttention`, `CerebellumModule`, `ApollonianMemory`, `ApollonianRetrievalAttention`, `TransformerBlock` |
| `test_import_training_subpackage` | All training classes importable including `Muon`, `HybridOptimizer`, all loss functions, telemetry, monitors, `TrainConfig`, `FANT2Trainer` |
| `test_import_data_subpackage` | `SyntheticStream`, `HuggingFaceStream`, `LocalShardStream`, `TokenizedBatchStream`, `make_default_stream`, `SEED_CORPUS`; verifies `SEED_CORPUS` is non-empty |
| `test_import_inference_subpackage` | `FANT2Generator`, `GenerationConfig`, `ChatSession` importable |
| `test_import_bench_subpackage` | Bench functions importable; verifies `extract_gsm8k_answer("#### 42") == 42.0` |
| `test_import_tokenizer_subpackage` | `FANT2Tokenizer`, `apply_chat_template`, `GPT4_REGEX_PATTERN` importable; pattern contains `\p` (Unicode property escape) |
| `test_tiny_model_forward` | Forward pass produces `logits` shape `(B, T, vocab_size)`, finite loss in range `(5.0, 15.0)` (near ln(32768) ≈ 10.4 for random init), correct number of router outputs (one per MoE layer) |
| `test_tiny_model_backward` | Backward pass sets non-zero gradients on at least 50% of parameters |
| `test_optimizer_step` | 3 HybridOptimizer steps reduce the loss by at most +1.0 (loss must remain finite) |
| `test_router_bias_update` | `update_router_biases()` changes `expert_bias` without requiring gradients |

---

## `tests/test_router_collapse.py`

**Module under test:** `fant2.model.HierarchicalApollonianRouter`, `fant2.training.telemetry.router_jsd_pairwise`

**Test count:** 6

**Background:** The FANT 350M model suffered catastrophic router collapse — 94.5% of routing concentrated on a single expert across all input domains (mean pairwise JSD (Jensen-Shannon Divergence) ≈ 0.0). This file is the regression canary: if any of these tests fail, the collapse has recurred.

**What it covers:**

| Test | Invariant asserted |
|------|-------------------|
| `test_router_etf_init_no_collapse` | At random init, max mega-pool load < 85% for all 4 domains. The 94.5% FANT 350M signature is impossible at init. |
| `test_router_jsd_metric_sanity` | `router_jsd_pairwise` returns < 1e-6 for identical distributions, ≈ ln(2) for disjoint one-hots |
| `test_aux_loss_free_bias_balances_load` | After 50 updates with 90% load on pool 0, pool 0 bias decreases and pools 1–3 biases increase (DeepSeek aux-loss-free mechanism works) |
| `test_tikkun_repair_fires_on_skew` | When EMA load is 80%/pool 0, `tikkun_repair()` returns `True` and pool 0 bias is reduced |
| `test_tikkun_no_op_when_balanced` | Balanced router (25% per pool) does not trigger tikkun |
| `test_full_model_no_init_collapse_with_more_domains` | 6-domain test: max load < 85% per domain, at least half of all mega-pools are active (load > 0.001) for every domain |

---

## `tests/test_trainer_integration.py`

**Module under test:** `fant2.training.FANT2Trainer`

**Test count:** 7 (4 parametrised phases + 3 standalone)

**What it covers:**

| Test | Invariant asserted |
|------|-------------------|
| `test_phase_train_step[phase=1]` | 4 training steps with CE + JEPA + sigreg losses; at least 1 of the first 5 params changes; `final.pt` is saved |
| `test_phase_train_step[phase=2]` | Phase 2 losses: `ce`, `fep_kl`, `z_loss`, `total`; param change + checkpoint |
| `test_phase_train_step[phase=3]` | Phase 3 adds `calib_rank` and `calib_cond` calibration losses |
| `test_phase_train_step[phase=4]` | Phase 4 adds `succ` (success) loss |
| `test_phase5_grpo_real` | 2 outer G2RPO (Group Relative Policy Optimisation) steps with G=2 rollouts; loss is finite; `final.pt` exists. Requires `Phase5BatchStream`, `ProceduralMathStream`, and a frozen reference policy. |
| `test_phase6_simpo_kto_real` | 2 SimPO+KTO (Kahneman-Tversky Optimisation) steps with `Phase6BatchStream`; loss is finite; `final.pt` exists |
| `test_checkpoint_save_and_load` | After a 2-step phase 2 run, a fresh trainer loaded with `resume_from=final.pt` has `step >= 2` |
| `test_train_step_returns_finite_losses` | A single `train_step()` call returns a dict of floats with no NaN and no ±inf values |

---

## `tests/test_spinor_apollonian.py`

**Module under test:** `fant3.model.spinor_apollonian.SpinorApollonianMemory`

**Test count:** 10

**Background:** `SpinorApollonianMemory` is the FANT 3 replacement for the scalar-curvature Apollonian memory classifier. It splits stored tokens into chirality-positive (alpha) and chirality-negative (beta) packs using a 2D spinor projection from the Clifford algebra Cl(2,1) (Minkowski space). The chirality split is derived from Kocik's 2020 tangency-spinor classification of Apollonian packings.

**What it covers:**

| Test | Invariant asserted |
|------|-------------------|
| `test_instantiation` | Empty memory has `alpha_fill=0`, `beta_fill=0`, `chirality_balance=0.5` |
| `test_store_chirality_balance` | 32 tokens stored → chirality balance in [0.25, 0.75] (binomial std ≈ 0.088); mean over 5 seeds × 128 tokens in [0.35, 0.65] — confirms no systematic bias in `proj_spinor` init |
| `test_store_no_hidden` | Fallback to `emb` when `hidden_preRMSnorm=None` works; 32 items stored |
| `test_retrieve_shape` | `retrieve(query, top_k=4, pool="both")` returns `values` shape `(1, 4, 4, 128)` and `scores` shape `(1, 4, 4)` |
| `test_retrieve_pool_selection` | `pool="alpha"` and `pool="beta"` both return correct shapes independently |
| `test_descartes_loss` | Returns a non-negative scalar tensor |
| `test_get_stats` | Returns dict with keys `alpha_fill`, `beta_fill`, `alpha_curvature_mean`, `beta_curvature_mean`, `chirality_balance` as correct Python types; balance in [0.0, 1.0] |
| `test_autograd` | Backprop through `scores.mean()` reaches `proj_spinor.weight.grad` with correct shape `(2, 128)` |
| `test_empty_pool` | Retrieve from empty memory returns zeros without error |
| `test_clifford_helpers` | `clifford_bilinear([3, 4], [1, 2]) == -5`; `clifford_norm([3, 4]) == 25` (exact arithmetic) |

---

## `tests/test_ahn.py`

**Module under test:** `fant3.model.ahn.ArtificialHippocampusNetwork`

**Test count:** 6

**Background:** The AHN (Artificial Hippocampus Network) is a two-tier memory module: a sliding short-term window (`short_window=32` positions) and a compressed long-term store (`long_capacity=64` slots). When the short-term buffer overflows, activations are compressed and written to long-term memory. A learned gate combines short-term and long-term context before the residual add.

**What it covers:**

| Test | Invariant asserted |
|------|-------------------|
| `test_instantiation` | Module builds without error; `latent_dim` is accessible |
| `test_output_shape` | `ahn(x)` where `x.shape=(2, 40, 128)` returns shape `(2, 40, 128)` unchanged |
| `test_long_fill_increases` | After 3 sequences of length 40 (> short_window=32), `long_fill > 0` — compression fires |
| `test_gradient_flow` | All 7 learnable projections (`gate_proj`, `q_proj`, `k_proj`, `v_proj`, `out_proj`, `compressor`, `decompressor`) have non-zero gradients; buffer tensors (`short_K`, `short_V`, etc.) are registered as buffers, not parameters |
| `test_reset_memory` | `reset_memory()` zeroes all buffer fill counts and contents |
| `test_get_stats` | Returns dict with `short_fill`, `long_fill`, `gate_short`, `gate_long`; fill values in [0, 1]; gate values sum to 1.0 ± 1e-5 |

---

## `tests/test_sae.py`

**Module under test:** `fant3.diagnostics.ApollonianSAE`, `fant3.diagnostics.train_on_hidden_states`, `fant3.diagnostics.analyze_apollonian_memory`

**Test count:** 6

**Background:** `ApollonianSAE` is a Top-K Sparse Autoencoder (SAE) used to interpret FANT 3's Apollonian memory. It learns a dictionary of `n_features` features (default 512) over hidden states of dimension `d_in`. Only the top `k` features per token are active (L0 sparsity ≤ k). The analysis function computes per-feature activation histograms, discriminating features (alpha-preferring vs beta-preferring), ghost features (never activated), and spinor chirality correlation.

**What it covers:**

| Test | Invariant asserted |
|------|-------------------|
| `test_instantiation` | `W_enc` shape `(512, 128)`, `W_dec` shape `(128, 512)`, biases correct shapes |
| `test_training_loss_decreases` | 2 epochs on 2 000 synthetic hidden states: last-quarter mean loss < first-quarter mean loss |
| `test_forward_batch` | After training: `reconstruction.shape == (8, 128)`, L0 ≤ k=16, dead-feature fraction < 50%, loss is a scalar |
| `test_analyze_apollonian_memory` | With mock `alpha_bank` / `beta_bank` memory: all 7 required keys present (`pack_sizes`, `feature_activation_histograms`, `top_discriminating_features`, `ghost_features`, `ghost_feature_count`, `ghost_feature_fraction`, `chirality_correlation`); alpha=200 and beta=150 correctly read; top-20 discriminating features sorted by `abs_difference` descending |
| `test_chirality_correlation` | With spinor-style mock memory (has `chirality` attribute): `chirality_correlation` is not None; length=512 (one per feature); all values in [-1, 1] |
| `test_real_apollonian_memory_layout` | With `RealStyleMemory` (uses `alpha_emb`/`alpha_count`/`beta_emb`/`beta_count` buffer names matching actual `fant2.model.ApollonianMemory`): `pack_sizes["alpha"]=200`, `pack_sizes["beta"]=150` correctly extracted |
