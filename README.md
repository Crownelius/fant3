# FANT 3

> **Fractal Atomic Neural Topology, version 3 — a compact language model research workspace.**

[![License: Research](https://img.shields.io/badge/License-Research--Only-lightgrey)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![Status](https://img.shields.io/badge/status-active%20research-green)](./ACHIEVEMENTS.md)
[![Tests](https://img.shields.io/badge/tests-125%20passing-brightgreen)](./tests/)
[![Scale](https://img.shields.io/badge/scales-20m%20to%201b-orange)](./docs/USER_GUIDE/README.md)

FANT 3 is a Mixture-of-Experts language model designed to run the *entire* training-and-eval loop on a single Ampere-class GPU. It explores a cluster of ideas that do not fit neatly into standard transformer recipes: Matryoshka-nested expert activation, per-token recursion depth, attention built from a shared dictionary, and memory packs split by Kocik spinor chirality rather than scalar curvature.

### Size in context

- **FANT 3 (50m preset, stored):** 0.05 GB (50.79 M parameters)
- **FANT 3 (1b preset, stored):** 0.99 GB (986.62 M parameters)
- **Llama 3 70B:** 140 GB
- **GPT-4 (estimated):** 3,500 GB
- **Ratio (50m vs GPT-4):** 70,000 times smaller

At the 50m preset FANT 3 fits in Colab's free tier. At the 1b preset it fits on a single A100 40 GB. No preset in this repo requires more than one GPU.

---

## What makes this special

- **[Matryoshka Mixture of Experts](./docs/THEORY/README.md#matryoshka-moe)** — nested megapools of experts; coarse-to-fine activation lets a single trained model serve at multiple inference budgets.
- **[Mixture of Recursions](./docs/THEORY/README.md#mor)** — a lightweight router selects 1–3 recursion passes per token; simple tokens exit early, hard tokens loop.
- **[Multi-head Attention with Shared Atoms](./docs/THEORY/README.md#masa)** — all layers share a learned dictionary of attention basis matrices; per-layer rank-`r` coefficients.
- **[Spinor Apollonian Memory](./docs/THEORY/README.md#spinor-apollonian-memory)** — Kocik Cl(2,1) tangency spinors split hidden states into α-instance and β-schema packs; fixes the α/β starvation bug of scalar-curvature classifiers.
- **[Artificial Hippocampus Network](./docs/THEORY/README.md#ahn)** — short-term FIFO plus a compressed long-term buffer, applied as a zero-initialized gated residual.
- **[Equiangular Tight Frame router freezing](./docs/THEORY/README.md#etf-freezing)** — free compression after warmup; no information loss under a simplex ETF geometry.
- **[Cerebellum reservoir](./docs/THEORY/README.md#cerebellum)** — a 25M-parameter echo-state reservoir with Purkinje linear readout; capacity is fixed regardless of backbone scale.
- **[Progressive curriculum](./docs/THEORY/README.md#progressive-curriculum)** — Apprentice/Journeyman/Expert three-phase data mix per arxiv:2604.16278; disproportionate gains in the 1B–3B band.

---

## Quick start

### Prerequisites

- Python 3.10+
- One Ampere-class GPU (RTX 3060 12 GB for development, A100 40–96 GB for training runs)
- Optional: Google Colab Pro+ for A100 access

### 30-second smoke test

```bash
# Clone
git clone https://github.com/Crownelius/fant3.git
cd fant3

# Install
pip install -r requirements.txt

# Smoke test (CPU, under 30 seconds)
python scripts/smoke_curriculum.py

# Run the full test suite
python -m pytest tests/ -v
```

### What you will see

```
Checking 3 preset(s) @ total_steps=12000

=== legacy_2phase @ total_steps=12000 ===
  curriculum.name = legacy_2phase
  n_phases = 2
    phase[0]      'A'  end_frac=0.667  end_step=8000  n_datasets=6  seq_len=1024
    phase[1]      'B'  end_frac=1.000  end_step=12000 n_datasets=6  seq_len=1024
  phase walk: A@0 -> B@8160
  milestone boundaries: {8000: '_phaseA'}
  OK

=== deepinsight_3phase @ total_steps=12000 ===
  n_phases = 3
    phase[0] 'apprentice' end_frac=0.250  end_step=3000  n_datasets=5
    phase[1] 'journeyman' end_frac=0.650  end_step=7800  n_datasets=7
    phase[2]     'expert' end_frac=1.000  end_step=12000 n_datasets=7
  OK

ALL 3 preset(s) OK
```

### First real training run (Colab recommended)

```bash
# Local (for development)
python scripts/runpod_train.py --scale 50m --dry-run
```

Or open `notebooks/fant3_1b_nvidia_train.ipynb` in Colab with an A100, set `TARGET_SCALE = '50m'`, and run the notebook top-to-bottom.

---

## Learning path

### 1. Start here: understanding the design

**Read first:** [Overview](./overview/README.md)
- What problem is FANT 3 solving?
- Why compact Mixture of Experts?
- How does the training loop fit on one GPU?

### 2. User guide

**Next:** [User Guide](./docs/USER_GUIDE/README.md)
- Getting started with the notebooks
- Training recipes per scale (20m, 50m, 150m, 350m, 742m, 1b)
- Running evaluations
- Resuming checkpoints

### 3. Theory

**The physics and geometry:** [Theory Guide](./docs/THEORY/README.md)
- Matryoshka MoE and nested expert activation
- Mixture of Recursions and per-token adaptive depth
- Spinor Apollonian memory and Kocik tangency
- ETF (Equiangular Tight Frame) routing and free compression
- Progressive curriculum and why it works in 1B–3B

**For the actual equations:** [Mathematical Foundations](./docs/mathematical-foundations.md)
- Matryoshka MoE parameter-count derivation
- MoR contractive decay and Banach fixed-point
- MASA rank-r decomposition (600x attention param reduction derivation)
- Spinor Apollonian: Cl(1,2) Clifford algebra + Descartes invariant + chirality sign
- ETF and the Welch bound
- Echo-state property and Cerebellum
- Phase-weighted curriculum distribution matrix
- Compression-as-intelligence (bits-per-byte from cross-entropy)

### 4. Developer guide

**For contributors:** [Developer Guide](./docs/DEVELOPER_GUIDE/README.md)
- Architecture walkthrough (where each concept lives in code)
- Testing protocol
- Adding a new preset / dataset / curriculum
- Diagnosing router collapse, chirality starvation, NaN steps

### 5. Article

**The narrative:** [The FANT Story](./docs/article/README.md)
- How FANT 2 (60 M / 200 M active) led to FANT 3 (1 B / 100 M active)
- Why Matryoshka MoE over standard top-k
- The geometry of Apollonian memory

### 6. Walkthrough

**End-to-end:** [Walkthrough](./docs/walkthrough.md)
- From `git clone` to a trained checkpoint
- Every architectural component you encounter on the way

---

## Project structure

```
fant3/
|-- README.md                  # This file
|-- ACHIEVEMENTS.md             # Milestones and results
|-- CLAUDE.md                   # Session context for AI collaborators
|-- docs/                       # Full documentation suite
|   |-- README.md                    # Docs index
|   |-- glossary.md                  # Terms (MoE, MoR, MASA, ETF, bf16, ...)
|   |-- size-comparison.md           # FANT vs GPT-class models
|   |-- walkthrough.md               # End-to-end narrative
|   |-- USER_GUIDE/                  # For users running training + eval
|   |-- DEVELOPER_GUIDE/             # For contributors + architects
|   |-- THEORY/                      # Mathematical foundations
|   |-- article/                     # Narrative essays
|   |-- ADR/                         # Architectural Decision Records
|   |-- architecture/                # Component deep-dives
|   |-- datasets/                    # Data mix + decontamination
|   |-- evaluation/                  # Benchmark protocol
|   |-- scripts/                     # Script-level references
|   `-- testing/                     # Testing protocol
|-- overview/                   # One-page conceptual overview
|   `-- README.md
|-- fant3/                      # Current architecture (v3)
|   |-- config.py                    # Preset table (fant3_1m through fant3_1b)
|   |-- model/                       # Attention, MoE, MoR, memory, AHN, Cerebellum
|   |-- training/                    # Curriculum, schedulers, optimizer, RL (queued)
|   |-- diagnostics/                 # SAE introspection
|   `-- data/                        # Formats + registry (shared with fant2)
|-- fant2/                      # Legacy runtime (for comparison and probes)
|-- scripts/                    # Standalone training / eval / smoke scripts
|   |-- runpod_train.py              # Primary training driver
|   |-- smoke_curriculum.py          # GPU-free curriculum dry-run
|   |-- eval_benchmarks.py           # GSM8K / MMLU / MATH-500
|   `-- decontaminate.py             # 13-gram benchmark filter
|-- notebooks/                  # Colab training notebooks
|-- tests/                      # 125 passing pytest tests
|-- bendvm/                     # Spinor VM experiments (research aside)
|-- output/                     # Tokenizers, decontamination hashes, checkpoints
`-- fant_code.zip               # RunPod upload artifact (rebuilt via build_fant_zip.py)
```

---

## Key features

### Scale ladder

| Preset | Stored params | Min VRAM | Typical run | Notes |
|---|---|---|---|---|
| `fant3_1m` | 0.99 M | CPU | Laptop ISRM smoke | Copy-task training harness |
| `fant3_10m` | 9.5 M | 4 GB | Sub-Chinchilla sanity | Verified 2026-04-23 |
| `fant3_15m` | 14.6 M | 4 GB | Colab T4 | — |
| `fant3_20m` | 23.5 M | 6 GB | Colab T4 / RTX 3060 | — |
| `fant3_50m` | 50.79 M | 12 GB | Colab A100 40 GB | Current RunPod target |
| `fant3_150m` | 96 M | 12 GB | RTX 3060 | — |
| `fant3_350m` | 263 M | 24 GB | A100 40 GB | — |
| `fant3_742m` | 770.88 M | 46 GB | A100 80 GB | 742m Tier C complete 2026-04-19 |
| `fant3_1b` | 986.62 M | 50 GB | A100 80 GB | — |

The `fant3_742m` and `fant3_1b` presets were recalibrated on 2026-04-19 after a bug caused them to materialize 6.6 B and ~7 B parameters respectively. Always verify with `sum(p.numel() for p in model.parameters())` before trusting a VRAM budget.

### Training

- bf16 (bfloat16) weights + 8-bit AdamW + gradient checkpointing for 742m and 1b
- Litim compact-support LR schedule (Phys. Rev. D 64 105007; smoother than cosine at the right endpoint)
- Three data-mix curricula: `legacy_2phase` (default), `deepinsight_3phase` (per arxiv:2604.16278), `flat_1phase` (control arm)
- Checkpoint naming: rolling `step_XXXXX.pt` plus milestone `step_XXXXX_phase_{name}.pt` / `_final.pt`

### Evaluation

- 13-gram SHA-1 decontamination cache: 457,910 unique hashes across GSM8K, MATH-500, MMLU
- Unified `scripts/eval_benchmarks.py` with Wilson 95% confidence intervals
- Benchmarks: GSM8K, MMLU, MATH-500
- Worst-source contamination rate: 1.80% (NVIDIA OpenMathInstruct-2)

### Data

- 11-source MIX v3: NVIDIA 60%, FineWeb-Edu 20%, Sonnet 4.6 12%, Opus 4.6 8%
- All targets wrapped in `<|answer|>...<|/answer|>` matching the eval extraction pattern
- Tokenizer v2: 32,768-token BPE trained on 82K documents from the 6-source mix

---

## Running training on RunPod

```bash
# A/B arm: DeepInsight 3-phase curriculum
python scripts/runpod_train.py --scale 50m --curriculum deepinsight_3phase \
  --batch-size 8 --grad-accum 2 --peak-lr 3e-4 --warmup-steps 1000 \
  --total-steps 1000000 --ckpt-every 2500 --ckpt-keep-last 3 \
  --wandb-project fant3 --wandb-run-name curriculum_deepinsight --hf-login

# Control arm: legacy 2-phase (bit-identical to pre-curriculum runs)
python scripts/runpod_train.py --scale 50m \
  --phase-a-steps 60000 --total-steps 1000000 \
  --batch-size 8 --grad-accum 2 --peak-lr 3e-4 --warmup-steps 1000 \
  --ckpt-every 2500 --ckpt-keep-last 3 \
  --wandb-project fant3 --wandb-run-name curriculum_legacy --hf-login
```

Environment variables required: `WANDB_API_KEY` and `HF_TOKEN`, set via pod dashboard (never via CLI). See [User Guide](./docs/USER_GUIDE/README.md) for the full RunPod setup walkthrough.

---

## Related research

| Paper | Connection |
|---|---|
| Kocik, arXiv:2001.05866 | Theoretical basis for Spinor Apollonian memory chirality split |
| Wang et al., arXiv:2509.26520 | Matryoshka Mixture of Experts |
| arxiv:2604.16278 DeepInsightTheorem | Progressive Apprentice/Journeyman/Expert curriculum |
| arxiv:2604.16004 AgentV-RL | Agentic verifier for Fix 3 RL plan (deferred, post-pretrain) |
| ByteDance AHN (2025) | Artificial Hippocampus Network short + long term memory |
| Anthropic Scaling Monosemanticity (2024) | SAE introspection design |
| TurboQuant, arXiv:2504.19874 (ICLR 2026) | Post-training KV-cache compression (queued) |
| Parisi RSB / de Almeida-Thouless, arXiv:2604.11921 | Theoretical grounding for MoE routing diversity |
| Delétang et al. (2023) | Language Models are Compressors — the framing |
| TRIM-KV, arXiv:2512.03324 | Retention gate for Apollonian memory eviction (queued) |

---

## License and acknowledgments

Research-only at present. License TBD before any public release.

Training data sources are CC-BY-4.0 (NVIDIA datasets) or permissive open licenses (FineWeb-Edu CC-BY, FineTome Apache 2.0). The Sonnet 4.6 and Opus 4.6 distillation datasets are used under their respective HuggingFace repository terms.

The FANT 3 architecture is original. The papers listed above are the external references whose ideas it builds on or cites.

---

## See also

- [Achievements](./ACHIEVEMENTS.md) — what has been measured and verified
- [Claude context](./CLAUDE.md) — session notes for AI collaborators
- [Overview](./overview/README.md) — one-page conceptual summary
- [Glossary](./docs/glossary.md) — terms and abbreviations
- [Size comparison](./docs/size-comparison.md) — FANT vs GPT-class models in detail
- [Mathematical foundations](./docs/mathematical-foundations.md) — equations, proofs, and notation
