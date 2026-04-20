# FANT 2 — Fractal Atomic Neural Topology, Generation 2

> *"Train an attention pattern to be Apollonian, train a routing distribution to
> be Parisi-ultrametric, and train a residual stream to be at the edge of chaos —
> then let the model see what kind of cognition falls out."*

FANT 2 is the second-generation Fractal Atomic Neural Topology language model:
a 60M-stored / 200M-active fractal Mixture-of-Experts transformer designed
around three commitments:

1. **The brain-as-Apollonian-packing hypothesis.** Sparse circuits in cortex
   look like Apollonian gaskets (recursive disk packings); FANT 2 builds the
   same structure into its memory module, its routing tree, and its weight
   generation.
2. **The Free Energy Principle as a single training signal.** Cross-entropy,
   z-loss, KL-to-prior, and Parisi RSB diversity collapse into one
   variational free-energy bound that all phases optimize variants of.
3. **Hard architectural defenses against router collapse.** FANT 350M, the
   prior generation, lost 94.5% of its routing onto a single expert after one
   epoch. FANT 2 ships a hierarchical, sigmoid-gated, ETF-initialized,
   bias-balanced, Tikkun-repaired, Fanā-shuffled router whose collapse
   regression test is part of CI.

This repository is a self-contained implementation: model, tokenizer, training
loop, 7-phase pipeline, inference, evaluation, CLI, and tests.

---

## Quick start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Smoke-test the package and run the FANT 350M router-collapse regression test
python -m pytest tests/ -v

# 3. Show the locked default config + parameter count
python -m fant2 info

# 4. Train end-to-end on a tiny preset (fits on CPU in a few minutes)
PRESET=tiny DEVICE=cpu N_STEPS=200 ./train.sh all
```

Or, if you have GNU make:

```bash
make help
make test
make info PRESET=tiny
make train-all PRESET=tiny DEVICE=cpu N_STEPS=200
```

---

## What's in the box

```
fant2/
├── config.py                 # FANT2Config dataclass + presets (default, tiny)
├── constants.py              # vocab IDs, chat-template tokens
├── model/                    # the architecture
│   ├── norm.py               # RMSNorm
│   ├── rope.py               # partial RoPE (Phi-4-Mini, 25%)
│   ├── kron3.py              # 3-level Kronecker A ⊗ B ⊗ C
│   ├── experts.py            # FractalSeed, Zero, Copy, SharedNarrow, DenseSwiGLU
│   ├── router.py             # HierarchicalApollonianRouter
│   ├── moe.py                # FractalMoELayer
│   ├── hub_attention.py      # GQA-2 + 32 hub tokens + 4 sinks + window=128
│   ├── cerebellum.py         # echo-state reservoir + Purkinje readout
│   ├── apollonian.py         # α/β memory dual-pack
│   ├── memory_retrieval.py   # ApollonianRetrievalAttention
│   ├── transformer_block.py
│   └── fant2_model.py        # FANT2Model (top-level)
├── tokenizer/                # BPE wrapper around HuggingFace `tokenizers`
├── data/                     # streaming data sources
├── training/
│   ├── optimizer.py          # Muon (Newton-Schulz) + HybridOptimizer (Muon + AdamW8bit)
│   ├── losses.py             # FEP unified, LLM-JEPA, calibration, SimPO, KTO, Dr.GRPO
│   ├── telemetry.py          # 8-metric diagnostic suite
│   ├── monitors.py           # tikkun, Fanā, JSD canary
│   ├── trainer.py            # FANT2Trainer (phase dispatcher)
│   ├── phase_common.py       # shared CLI flags
│   ├── phase0_bpe.py         # BPE tokenizer training
│   ├── phase1_jepa.py        # LLM-JEPA + SIGReg pretrain
│   ├── phase2_moe.py         # MoE specialization
│   ├── phase3_calibrate.py   # active-layer calibration
│   ├── phase4_refine.py      # self-refinement + STaR + Apollonian fill
│   ├── phase5_grpo.py        # Dr.GRPO RL (stub)
│   └── phase6_simpo_kto.py   # SimPO + KTO preference (stub)
├── inference/                # FANT2Generator + GenerationConfig + ChatSession
├── bench/                    # perplexity, GSM8K, ARC, HellaSwag
├── cli/                      # `python -m fant2 ...` dispatcher
└── __main__.py
tests/
├── test_smoke.py             # imports + forward + backward + optimizer
├── test_router_collapse.py   # the FANT 350M regression canary
└── test_trainer_integration.py
```

---

## Architecture (locked spec)

| component         | spec                                                                       |
|-------------------|----------------------------------------------------------------------------|
| dimensions        | dim=768, 12 layers (2 dense + 10 MoE), GQA-2 (8 query / 2 KV heads), head_dim=96 |
| RoPE              | partial 25% (Phi-4-Mini); θ = 10000                                       |
| MoE               | 72 fractal seeds in 8 mega-pools × 9 (Parisi RSB ultrametric)             |
| router            | HierarchicalApollonianRouter: 2 stages, sigmoid gating, Simplex ETF init  |
| router defenses   | DeepSeek aux-loss-free bias, OLMoE z-loss, FEP KL-to-prior, Tikkun, Fanā  |
| Kronecker         | A(40,8) ⊗ B(32,32) ⊗ C(32,40)  →  effective (768, 1280)                   |
| attention         | Hub: 32 hubs (VEN analog), 4 sinks (StreamingLLM), window=128             |
| cerebellum        | echo-state reservoir (spectral radius 0.95) + Purkinje linear readout      |
| memory            | Apollonian α (instances) + β (schemas), retrieval at last 2 layers         |
| optimizer         | Muon (Newton-Schulz quintic) for matrices + AdamW8bit for vectors         |
| BF16 / ckpt       | enabled by default                                                         |

Approximate parameter count for the locked preset:
- **stored** (physical): ~60 M
- **active per forward** (top-k expansion): ~200 M

---

## The 7 training phases

| phase | name                              | key loss components                                       |
|-------|-----------------------------------|----------------------------------------------------------|
| 0     | BPE tokenizer                     | (no model training; trains the byte-level BPE tokenizer) |
| 1     | LLM-JEPA + SIGReg pretrain        | CE + JEPA + SIGReg                                       |
| 2     | MoE specialization                | CE + α·z_loss + β·KL(router ‖ uniform)                   |
| 3     | active-layer calibration          | Phase 2 + rank-collapse + condition-number penalties     |
| 4     | self-refinement + STaR + Apollonian fill | Phase 2 + success-estimator BCE                  |
| 5     | Dr.GRPO RL (stub)                 | (currently falls back to Phase 2 loss)                   |
| 6     | SimPO + KTO preference (stub)     | (currently falls back to Phase 2 loss)                   |

Phases 5 and 6 are placeholder loops: they run the trainer infrastructure
end-to-end with the FEP loss while the full Dr.GRPO rollout / preference-pair
logic is filled in.

---

## CLI

```bash
python -m fant2 <subcommand> [args]
```

| subcommand          | description                                            |
|---------------------|--------------------------------------------------------|
| `info`              | print config + model parameter count                   |
| `train-phase0`      | train BPE tokenizer                                    |
| `train-phase1` … `train-phase6` | train one phase of the pipeline             |
| `generate`          | one-shot text generation                               |
| `chat`              | interactive chat session (ChatML template)             |
| `eval-ppl`          | perplexity on a stream                                 |
| `eval-gsm8k`        | GSM8K accuracy                                         |
| `eval-arc`          | ARC-Easy / ARC-Challenge multichoice                    |
| `eval-hellaswag`    | HellaSwag multichoice (length-normalized)              |

Run `python -m fant2 --help` for a full list and per-command flags.

---

## Tests

The test suite is the contract: anything that ships **must** pass
`pytest tests/`. Three files cover three layers:

- `tests/test_smoke.py` — every public symbol imports cleanly; tiny model
  forward + backward + optimizer step succeed.
- `tests/test_router_collapse.py` — **the FANT 350M regression canary.** Runs
  the router on multiple synthetic "domains" and asserts no single mega-pool
  ever exceeds 85 % load (FANT 350M had 94.5 %). Also unit-tests the
  aux-loss-free bias balancer, the Tikkun repair trigger, and the JSD metric.
- `tests/test_trainer_integration.py` — runs a few steps of every phase on the
  tiny preset, asserts the loss dict has the expected keys, parameters
  actually update, the loss is finite, and the checkpoint round-trip works.

```bash
make test                 # full suite
make smoke                # imports + forward/backward only
make router-canary        # the FANT 350M regression canary alone
make integration          # the trainer integration tests alone
```

---

## Hardware notes

The locked default preset is sized for 24 GB consumer GPUs (RTX 3090 / 4090).
With BF16 + gradient checkpointing + AdamW8bit it fits comfortably in 12-16 GB
of VRAM at batch=8, seq=1024.

For CPU-only smoke tests, use `PRESET=tiny`.

---

## Training a model end-to-end

The simplest path:

```bash
# Tiny smoke test (CPU, ~minutes)
PRESET=tiny DEVICE=cpu N_STEPS=200 ./train.sh all

# Real training run (GPU, hours-to-days)
PRESET=default DEVICE=cuda N_STEPS=50000 BATCH=8 SEQ_LEN=1024 ./train.sh all
```

Or per-phase, with your own config flags:

```bash
./train.sh 0   # tokenizer
./train.sh 1   # JEPA
./train.sh 2   # MoE specialization
./train.sh 3   # calibration
./train.sh 4   # refinement
./train.sh 5   # Dr.GRPO (stub)
./train.sh 6   # preference (stub)
```

Each phase resumes from the previous phase's `final.pt` automatically.

---

## License & status

This is a research codebase. The architecture is locked and the test suite
passes; phases 5 and 6 are stubs awaiting their full implementation.
