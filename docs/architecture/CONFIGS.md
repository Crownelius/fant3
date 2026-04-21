# FANT 3 Scale Presets and Configuration Reference

This document lists every scale preset, its verified actual parameter count, and the recommended training recipe for each. It also documents the critical preset-naming bug that was discovered and fixed on 2026-04-19.

---

## Critical Historical Note: The 6.6B Preset Bug

Between the initial implementation and 2026-04-19, two of the five presets were severely mis-sized because of a discrepancy between the config fields and the model code:

**Root cause**: `MatryoshkaMoEFFN` stores expert weights as **full-rank** `nn.Parameter` tensors of shape `(n_experts, dim, 2 × moe_hidden)`. The Kronecker factorization fields (`kron_A_p`, `kron_A_q`, etc.) exist in `FANT3Config` but are **not read by the model code** — they are reserved for a future upgrade. This means the actual parameter count is dominated by:

```
MoE params per block ≈ n_megapools × n_per_megapool × dim × (2 × moe_hidden + moe_hidden)
                     = n_experts × dim × 3 × moe_hidden
```

**The broken `fant3_742m` preset** used `n_megapools=8`, `n_per_megapool=16` (128 experts), `dim=2048`, `moe_hidden=2048`. This produced:

```
128 experts × 2048 × 3 × 2048 × 4 MoE blocks ≈ 6.4 B params in experts alone
Total stored: ~6.6 B
```

This was ~8.9× the intended size. Three consecutive OOM errors on Colab A100 96 GB revealed the bug.

**The broken `FANT3Config()` defaults** (the 1b preset) had `dim=2048`, `n_megapools=8`, `n_per_megapool=16`, `moe_hidden=2048`, producing approximately 7 B stored.

**The fix** (2026-04-19) reduced expert count to 32 per preset and adjusted dimensions to hit the intended stored-parameter targets. The formula to check a config before training:

```python
from fant3.config import FANT3Config
from fant3.model.fant3_model import FANT3Model
cfg = FANT3Config()                    # or any preset function
n = FANT3Model(cfg).n_params()
print(f"{n/1e6:.1f}M stored params")  # Always verify this before running
```

**Lesson for future developers**: never trust a preset name alone. Always count parameters directly before committing a training run.

---

## Preset Table

The following table summarizes all five presets. Parameter counts are verified by instantiating the full model locally and calling `model.n_params()`.

| Preset | Function | dim | n_layers | n_experts | moe_hidden | Verified stored params | Cerebellum | AHN | Primary use case |
|---|---|---|---|---|---|---|---|---|---|
| smoke | `fant3_smoke()` | 512 | 8 | 16 (4×4) | 512 | 72.70 M | disabled | enabled | Smoke tests; fits in ~2 GB VRAM |
| 20m | `fant3_20m()` | 320 | 10 | 4 (2×2) | 640 | 23.5 M | disabled | enabled | 12h A100 distillation run |
| 50m | `fant3_50m()` | 384 | 12 | 8 (2×4) | 896 | 50.8 M | disabled | disabled | Heavy overtraining on distillation data |
| 742m | `fant3_742m()` | 1024 | 16 | 32 (4×8) | 1792 | 770.9 M | enabled | enabled | Validated on Colab A100 96 GB |
| 1b (default) | `FANT3Config()` or `fant3_1b()` | 1024 | 20 | 32 (4×8) | 2304 | 986.6 M | enabled | enabled | Flagship pretrain target |

Note: `fant3_1b()` is simply `return FANT3Config()` — it uses the dataclass defaults directly.

---

## Preset Details

### `fant3_smoke` — 72.70 M stored (verified by direct instantiation 2026-04-19)

```
dim=512, n_layers=8, n_dense_layers=1
n_heads=8, n_kv_heads=2, head_dim=64
n_megapools=4, n_per_megapool=4, top_k=2       → 16 experts total
moe_hidden=512, shared_expert_hidden=256
n_attention_atoms=3, masa_coef_rank=8
n_recursion_depths=2
vocab_size=32768, max_seq_len=512
cerebellum_enabled=False
apollonian_alpha_cap=1000, apollonian_beta_cap=1000
apollonian_retrieval_layers=(6, 7)
etf_freeze_after_step=100
etf_freeze_layers=(2, 3, 4)
```

**Training recipe**: batch=1, seq=512, steps=200, warmup=20, lr=2e-4. Fits in ~2 GB VRAM. Use for verifying code changes before a full run.

---

### `fant3_20m` — 23.5 M stored

```
dim=320, n_layers=10, n_dense_layers=2
n_heads=4, n_kv_heads=2, head_dim=80           → GQA-2
n_megapools=2, n_per_megapool=2, top_k=1       → 4 experts total
moe_hidden=640, shared_expert_hidden=256
n_attention_atoms=3, masa_coef_rank=4
n_recursion_depths=2
vocab_size=32768, max_seq_len=1024
cerebellum_enabled=False
apollonian_alpha_cap=1000, apollonian_beta_cap=1000
apollonian_retrieval_layers=(8, 9)
etf_freeze_after_step=3000
etf_freeze_layers=range(2, 8)
```

**Target**: 12 h training on A100 96 GB ≈ 1.4 B training tokens. Honest capability ceiling: fluent conversational English, basic arithmetic, simple code. Not MMLU-competitive.

**Training recipe**: batch=2, seq=1024, accum=4, steps=5000, warmup=500, lr=2e-4.

---

### `fant3_50m` — 50.8 M stored

```
dim=384, n_layers=12, n_dense_layers=2
n_heads=6, n_kv_heads=2, head_dim=64           → GQA-3
n_megapools=2, n_per_megapool=4, top_k=2       → 8 experts total
moe_hidden=896, shared_expert_hidden=320
n_attention_atoms=4, masa_coef_rank=6
n_recursion_depths=2
vocab_size=32768, max_seq_len=1024
cerebellum_enabled=False, ahn_enabled=False
apollonian_alpha_cap=2000, apollonian_beta_cap=2000
apollonian_retrieval_layers=(10, 11)
etf_freeze_after_step=6000
etf_freeze_layers=range(2, 10)
```

**Target**: 12 h training on A100 96 GB ≈ 2 B training tokens (~40× Chinchilla optimal). Follows the SmolLM2 / Phi over-training playbook.

**Training recipe**: batch=2, seq=1024, accum=4, steps=5000, warmup=500, lr=2e-4.

---

### `fant3_742m` — 770.9 M stored (FIXED 2026-04-19)

```
dim=1024, n_layers=16, n_dense_layers=2
n_heads=8, n_kv_heads=2, head_dim=128          → GQA-4
n_megapools=4, n_per_megapool=8, top_k=2       → 32 experts (was 128 before fix)
moe_hidden=1792, shared_expert_hidden=512       (was 2048/2048 before fix)
n_attention_atoms=4, masa_coef_rank=8
n_recursion_depths=2
vocab_size=32768, max_seq_len=1024
cerebellum_enabled=True, ahn_enabled=True
apollonian_alpha_cap=5000, apollonian_beta_cap=5000
apollonian_retrieval_layers=(14, 15)
etf_freeze_after_step=1000
etf_freeze_layers=range(2, 13)
use_gradient_checkpointing=True                 (auto-enabled for 742m+)
```

**VRAM usage (A100 96 GB, batch=1, seq=1024, gc=True)**: ~45.7 GB stable (validated on Colab A100 96 GB, 2026-04-19).

**Training recipe (Tier C, validated)**: batch=1, seq=1024, accum=8 (effective batch=8), steps=10000, warmup=1500, peak_lr=1.5e-4. Wall time: ~78.6 min on A100 96 GB. Best CE observed: 5.72 at step 7025.

**Pre-fix parameter count** (documented for audit trail): `n_megapools=8`, `n_per_megapool=16`, `dim=2048`, `moe_hidden=2048` → **6,648 M stored** (6.6 B). This was the broken value.

---

### `FANT3Config()` / `fant3_1b` — 986.6 M stored (FIXED 2026-04-19)

```
dim=1024, n_layers=20, n_dense_layers=3        (was 2048 / 24 before fix)
n_heads=8, n_kv_heads=2, head_dim=128          → GQA-4 (was 16q/4kv)
n_megapools=4, n_per_megapool=8, top_k=2       → 32 experts (was 128)
moe_hidden=2304, shared_expert_hidden=640       (was 2048/768 before fix)
n_attention_atoms=5, masa_coef_rank=8           (was 6 / 16)
n_recursion_depths=2                            (was 3)
vocab_size=32768, max_seq_len=1024              (was 2048)
cerebellum_enabled=True, ahn_enabled=True
apollonian_alpha_cap=10000, apollonian_beta_cap=10000
apollonian_retrieval_layers=(18, 19)
etf_freeze_after_step=2000
etf_freeze_layers=range(3, 17)
use_gradient_checkpointing=False                (trainer should set True for A100)
```

**Projected VRAM (A100 96 GB, batch=2, seq=1024, gc=True)**: 36–46 GB based on linear scaling from 742m result.

**Training recipe (Tier D, not yet run)**: batch=2, seq=1024, accum=4 (effective batch=8), steps=12000, warmup=1800, peak_lr=1.2e-4 ≈ 98 M training tokens. Projected wall time: 10–12 h on A100 96 GB.

**Pre-fix parameter count**: `dim=2048`, `n_megapools=8`, `n_per_megapool=16`, `moe_hidden=2048` → approximately **7,000 M** (7 B). This was the broken value.

---

## Layer Plan Summary by Preset

For a model with `n_layers=L`, `n_dense_layers=D`, the suffix is `n_suffix = min(3, max(1, L − D − 1))` and the shared middle covers the remaining `L − D − n_suffix` logical layers.

| Preset | n_layers | n_dense | n_suffix | n_middle_logical |
|---|---|---|---|---|
| smoke | 8 | 1 | 3 | 4 |
| 20m | 10 | 2 | 3 | 5 |
| 50m | 12 | 2 | 3 | 7 |
| 742m | 16 | 2 | 3 | 11 |
| 1b | 20 | 3 | 3 | 14 |

The shared middle layers pass through ONE `MoEBlock` called 1–2 times per token (via `MoRShared`). The suffix layers are distinct `MoEBlock` instances.

---

## Gradient Checkpointing VRAM Guide

Gradient checkpointing is required for 742m and 1b at sequence length 512+. The VRAM reduction factor is approximately 3–5×, at the cost of ~25–40% more compute (activations are recomputed during the backward pass).

| Preset | seq_len | gc=False peak VRAM | gc=True peak VRAM |
|---|---|---|---|
| smoke | 512 | ~2 GB | N/A (fits without gc) |
| 20m | 1024 | ~4 GB | N/A |
| 50m | 1024 | ~8 GB | N/A |
| 742m | 1024 | ~93 GB (OOM on A100 80 GB) | ~45.7 GB (validated) |
| 1b | 1024 | ~130+ GB (OOM) | ~36–46 GB (projected) |

The `use_gradient_checkpointing` config field defaults to `False`. The Colab notebook automatically sets it to `True` for 742m and 1b scales.
