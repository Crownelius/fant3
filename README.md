# FANT 3 (Fractal Atomic Neural Topology version 3)

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-TBD-lightgrey)
![Status](https://img.shields.io/badge/status-active%20research-green)

> *"Train an attention pattern to be Apollonian, train a routing distribution to be Parisi-ultrametric, and train a residual stream to be at the edge of chaos — then let the model see what kind of cognition falls out."*

FANT 3 is the third-generation **Fractal Atomic Neural Topology** language model research workspace. It explores a cluster of ideas that do not fit neatly into standard transformer recipes:

- **MoE (Mixture of Experts)** with Matryoshka nested coarse-to-fine activation for elastic inference
- **MoR (Mixture of Recursions)** for per-token adaptive compute depth
- **MASA (Multi-head Attention with Shared Atoms)** where all layers share a learned dictionary of attention basis matrices
- **Apollonian-geometry memory** — two complementary memory packs (α instance / β schema) split by Kocik tangency spinors rather than a scalar threshold
- **AHN (Artificial Hippocampus Networks)** — a sliding short-term buffer plus a compressed long-term buffer applied as a gated residual
- **ETF (Equiangular Tight Frame)** router freezing for free compression after warmup
- **Cerebellum** — an echo-state reservoir with Purkinje readout (active at 742m and 1b scales)

The workspace targets a single Ampere-class GPU (RTX 3060 12 GB for development, A100 40–96 GB for training runs), with **bf16 (bfloat16)** weights, 8-bit AdamW, and gradient checkpointing.

---

## Table of Contents

1. [Quick Facts](#quick-facts)
2. [Quickstart](#quickstart)
3. [Architecture](#architecture)
4. [Training Recipes and Data](#training-recipes-and-data)
5. [Notebooks](#notebooks)
6. [Scripts](#scripts)
7. [Testing](#testing)
8. [Evaluation](#evaluation)
9. [History](#history)
10. [Architectural Decisions](#architectural-decisions)
11. [Related Research](#related-research)
12. [License and Acknowledgments](#license-and-acknowledgments)

---

## Quick Facts

| Attribute | Value |
|---|---|
| Supported scales | 20m, 50m, 150m, 350m, 742m, 1b |
| Actual stored param counts | 23.5M / 50.8M / 96M / 263M / 770.88M / 986.62M |
| Minimum hardware (50m) | A100 40 GB (Colab Pro) |
| Minimum hardware (742m / 1b) | A100 80–96 GB |
| Training tokens (742m Tier C) | 81.92M (190x under Chinchilla-optimal for this scale) |
| Latest milestone | 742m Tier C: 78.6 min on A100 96 GB, best CE (Cross-Entropy) 5.72 at step 7025 |
| Vocabulary | 32,768 BPE (Byte-Pair Encoding) tokens, retrained on 6-source distillation mix |
| Precision | bf16 weights + 8-bit AdamW + gradient checkpointing at 742m+ |
| Key architectural novelties | Matryoshka MoE, SpinorApollonianMemory, MASA, MoR, AHN, ETF freezing |

> **Important:** The named presets (`fant3_742m`, `fant3_1b`) have been calibrated to their advertised sizes as of 2026-04-19. The original presets materialized 6.6B and ~7B parameters respectively due to full-rank expert weight matrices ignoring the Kronecker config fields. Always verify with `sum(p.numel() for p in model.parameters())` before trusting a VRAM budget.

---

## Quickstart

For a five-minute end-to-end walkthrough see **[docs/GETTING_STARTED.md](docs/GETTING_STARTED.md)**.

The fastest path:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Smoke-test (CPU, ~30 seconds)
python scripts/smoke_fant3.py

# 3. Run the full test suite
python -m pytest tests/ -v

# 4. Train (Colab notebook, recommended for GPU)
# Open notebooks/fant3_colab_train.ipynb in Google Colab
# Set TARGET_SCALE = '50m' and run top-to-bottom
```

Colab is the recommended training environment. See [docs/NOTEBOOKS.md](docs/NOTEBOOKS.md) for a cell-by-cell walkthrough.

---

## Architecture

Architecture deep-dives live in **[docs/architecture/](docs/architecture/)**.

High-level summary:

| Component | Design |
|---|---|
| Attention | MASA — all layers share `n_attention_atoms` basis matrices; per-layer coefficients of rank `masa_coef_rank` |
| Feed-forward | Matryoshka MoE — nested megapools of experts; elastic inference by level |
| Depth | MoR — lightweight router selects recursion depth per token (1–3 passes) |
| Memory | SpinorApollonianMemory — Kocik Cl(2,1) spinor chirality splits hidden states into α (instance) / β (schema) packs |
| Short-term memory | AHN — FIFO of last `short_window` tokens + compressed long-term latents; gated residual before final norm |
| Reservoir | Cerebellum — echo-state reservoir (spectral radius 0.95) + Purkinje linear readout; fixed 25M params regardless of scale |
| Router stability | Simplex ETF initialization; frozen after `etf_freeze_after_step` |
| Vocabulary | 32,768-token BPE tokenizer trained on 82K documents from the 6-source distillation mix |

FANT 3 builds on the FANT 2 lineage but is a clean-room implementation with these key differences:

- **FANT 2**: 60M stored / 200M active, HierarchicalApollonianRouter, 7-phase FEP training pipeline, RTX 3060 target
- **FANT 3**: 20M–1B stored, Matryoshka MoE, MoR, MASA, SpinorApollonianMemory, Colab A100 primary target, single-notebook training

---

## Training Recipes and Data

Dataset registry, data mixing logic, and decontamination details live in **[docs/datasets/](docs/datasets/)**.

Brief summary of the data strategy:

- **MIX v3 (NVIDIA-heavy, for 150m–1b)**: 11 sources, NVIDIA datasets at 60% weight (OpenMathReasoning, OpenCodeReasoning-2, Cascade-2 math/chat/science), FineWeb-Edu 20%, Sonnet 4.6 distillation 12%, Opus 4.6 distillation 8%
- **MIX v4 (chat-focused, for 20m–50m)**: 12 sources, Sonnet 4.6 at 22%, Cascade-2 chat/IF, FineTome, Daring-Anteater; targets fluent short-form conversation
- **Decontamination**: 13-gram SHA-1 filter against GSM8K + MATH-500 + MMLU test sets (457,910 unique hashes); worst contamination rate 1.80% (NVIDIA OpenMathInstruct-2)
- **Format**: All training targets wrapped in `<|answer|>...<|/answer|>` tags matching the evaluation extraction pattern

---

## Notebooks

Full cell-by-cell documentation: **[docs/NOTEBOOKS.md](docs/NOTEBOOKS.md)**.

The primary training notebook is `notebooks/fant3_colab_train.ipynb` — 28 cells, parameterized by a single `TARGET_SCALE` constant (`'20m'`, `'50m'`, `'150m'`, `'350m'`, `'742m'`, `'1b'`).

Quick reference for Colab A100:

| Scale | Effective batch | Sequence length | Steps | Approx wall time | VRAM (GPU RAM) |
|---|---|---|---|---|---|
| 50m | 32 | 1024 | 60,000 | ~12 h | ~8 GB |
| 150m | 8 | 512 | 2,500 | ~45 min | ~8 GB |
| 350m | 8 | 512 | 5,000 | ~2 h | ~15 GB |
| 742m | 8 | 1024 | 10,000 | ~80 min | ~46 GB |
| 1b | 8 | 1024 | 12,000 | ~95 min | ~50 GB |

---

## Scripts

The `scripts/` directory contains standalone Python scripts for training, evaluation, and diagnostics. Key scripts:

| Script | Purpose |
|---|---|
| `smoke_fant3.py` | Quick end-to-end smoke test (CPU, ~30 s) |
| `scale_ladder_smoke.py` | Validates all five scales end-to-end |
| `decontaminate.py` | Builds/queries the 13-gram benchmark contamination filter |
| `eval_benchmarks.py` | Unified GSM8K / MMLU / MATH-500 evaluator with Wilson 95% CI |
| `eval_1k.py` / `eval_1k_default.py` | 1K-problem accuracy eval with extraction |
| `retrain_tokenizer.py` | Trains `tokenizer_v2.json` from the 6-source corpus |
| `train_2b.py` | Standalone 2B-scale training (requires 24+ GB VRAM) |
| `run_campaign_n.py` | Runner for FANT 2 Campaign N ablations |

---

## Testing

```bash
python -m pytest tests/ -v        # full suite (~30 tests)
python -m pytest tests/test_smoke.py -v           # import + forward/backward
python -m pytest tests/test_spinor_apollonian.py  # SpinorApollonianMemory (10 tests)
python -m pytest tests/test_ahn.py                # AHN (6 tests)
python -m pytest tests/test_sae.py                # SAE diagnostics (6 tests)
python -m pytest tests/test_router_collapse.py    # FANT 2 router regression canary
```

The test suite covers:
- All public FANT 3 modules (config, attention, MoE, MoR, spinor memory, AHN, SAE)
- SpinorApollonianMemory chirality balance (verified unbiased at 5-seed mean 0.5188)
- AHN buffer saturation and gate initialization
- FANT 2 router-collapse regression (single expert must not exceed 85% load; FANT 350M had 94.5%)
- Trainer integration (every FANT 2 phase, loss keys, parameter updates, checkpoint round-trip)

---

## Evaluation

After training, evaluation runs through `scripts/eval_benchmarks.py`:

```bash
python scripts/eval_benchmarks.py \
    --ckpt output/742m/final.pt \
    --tokenizer output/tokenizer/tokenizer_v2.json \
    --benchmark gsm8k \
    --n 50
```

Supported benchmarks: `gsm8k`, `mmlu`, `math500`.

Expected results at current training scale (742m Tier C, 82M tokens):
- **GSM8K**: 1–4% (undertrained; Chinchilla-optimal would be 15B tokens)
- **MMLU**: ~26% (at statistical chance; 4-way multiple-choice baseline is 25%)

The Colab notebook (cell 26) runs GSM8K + MMLU automatically after training completes.

---

## History

Chronological training log: **[docs/HISTORY.md](docs/HISTORY.md)**.

Milestones in reverse chronological order:

| Date | Event |
|---|---|
| 2026-04-19 | 742m Tier C complete: 78.6 min, best CE 5.72, domain vocabulary acquired |
| 2026-04-19 | Preset size bugs fixed: 742m was secretly 6.6B; 1b was secretly ~7B |
| 2026-04-19 | Gradient checkpointing landed: 2.65x VRAM reduction, bit-exact loss |
| 2026-04-19 | NVIDIA full stack integrated (MIX v3, NeMo-style recipe) |
| 2026-04-19 | 150m validation baseline: CE 6.66 at 750 steps, "the the the" → word-frequency learning |
| 2026-04-19 | Five pre-launch fixes landed (answer-tag format, tokenizer v2, SpinorApollonian, AHN, SAE) |
| 2026-04-18 | HF archive extended to 36 months + 23 AI labs (2,286 KG triples) |
| 2026-04-16 | N3 SleepGate result: 59.9% (+5.3pp over L1.5 baseline) — new FANT 2 best |
| 2026-04-16 | FANT 3 architectural modules landed and smoke-tested |

---

## Architectural Decisions

All ADRs (Architectural Decision Records) are in **[docs/ADR/](docs/ADR/)**.

| ADR | Decision |
|---|---|
| [ADR 0001](docs/ADR/0001-matryoshka-moe-over-standard-moe.md) | Matryoshka MoE over standard top-k MoE |
| [ADR 0002](docs/ADR/0002-spinor-apollonian-over-scalar-curvature.md) | Spinor Apollonian memory over scalar curvature classifier |
| [ADR 0003](docs/ADR/0003-nvidia-datasets-over-community.md) | NVIDIA reasoning datasets as primary training signal |
| [ADR 0004](docs/ADR/0004-gradient-checkpointing-for-742m-plus.md) | Gradient checkpointing mandatory at 742m and above |

---

## Related Research

Key papers grounding the FANT 3 architecture:

| Paper | Connection |
|---|---|
| Kocik, arXiv:2001.05866 — Spinors and Descartes | Theoretical basis for SpinorApollonianMemory chirality split |
| Wang et al., arXiv:2509.26520 — Matryoshka MoE | Nested expert activation with elastic inference |
| ByteDance AHN (2025) | Artificial Hippocampus Network short+long term memory |
| Anthropic Scaling Monosemanticity (2024) | SAE introspection design |
| TurboQuant, arXiv:2504.19874 (ICLR 2026) | Post-training KV-cache compression (planned) |
| Parisi RSB / de Almeida-Thouless, arXiv:2604.11921 | Theoretical grounding for MoE routing diversity |
| Delétang et al. (2023) — Language Models are Compressors | Compression-as-intelligence framing |
| TRIM-KV, arXiv:2512.03324 | Retention gate for Apollonian memory eviction (planned) |
| Kimi k1.5 (2025) | SleepGate memory consolidation inspiration |

---

## License and Acknowledgments

License: TBD (research-only at present).

Training data sources are CC-BY-4.0 (NVIDIA datasets) and permissive open licenses (FineWeb-Edu CC-BY, FineTome Apache 2.0). The Sonnet 4.6 and Opus 4.6 distillation datasets are used under their respective HuggingFace repository terms.

This workspace is a private research project. The FANT 3 architecture is original; it draws on and cites the published papers listed above.
