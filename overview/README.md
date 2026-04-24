# FANT 3: one-page overview

FANT 3 is a Mixture-of-Experts language model whose entire training and evaluation loop runs on a single Ampere-class GPU. It targets the 1 M to 1 B parameter band, where design choices from frontier-scale labs often stop paying off and different ideas start to win.

## The one-sentence pitch

Nested expert activation, per-token recursion depth, shared-atom attention, and Apollonian-geometry memory — all trained end-to-end with bf16, 8-bit AdamW, gradient checkpointing, and a progressive data curriculum, on one GPU.

## Why this shape

Standard transformer recipes are designed for frontier scale. At our size band, three kinds of design choice matter disproportionately:

1. **Elastic inference.** One trained checkpoint should serve at multiple compute budgets. Matryoshka-nested MoE gives this for free — inference can take any prefix of the expert activation sequence.
2. **Per-token adaptive compute.** Not every token needs the full depth of the network. Mixture of Recursions lets easy tokens exit early and hard tokens loop.
3. **Interpretable memory.** At this scale, the memory pack split is observable. Kocik tangency spinors give a chirality-based split (α-instance vs β-schema) that a scalar curvature classifier cannot match.

## What's in the repository

- **`fant3/`** — the current architecture: attention, MoE, MoR, spinor memory, AHN, Cerebellum, training loop.
- **`fant2/`** — the legacy runtime, kept as a known-good reference and for comparison probes.
- **`scripts/`** — standalone training, eval, and smoke scripts.
- **`notebooks/`** — Colab training notebooks (28 cells, parameterized by `TARGET_SCALE`).
- **`tests/`** — 125 passing pytest tests across all public modules.
- **`docs/`** — the full documentation suite (user guide, developer guide, theory, ADRs, article).
- **`bendvm/`** — a research aside on operating programs while compressed; unrelated to the main loop.

## What has been measured

Highlights from [ACHIEVEMENTS.md](../ACHIEVEMENTS.md):

- **742m Tier C training**: 78.6 minutes on A100 96 GB, best cross-entropy 5.72, peak VRAM 45.66 GB, no NaN, no OOM.
- **Compression validation**: bf16 vs fp32 produce 99.913% bit-identical quantized probabilities at 16-bit quantization with 2.21x VRAM savings.
- **Scale ladder**: 4 of 5 scales (5m, 40m, 150m, 350m) pass end-to-end with no code changes. 742m requires 8-bit AdamW + gradient checkpointing, a known production configuration.
- **125 of 125 tests passing** across all public modules.
- **Progressive curriculum landed** 2026-04-24 (per arxiv:2604.16278 DeepInsightTheorem).

## What this repository is not

- **Not a production system.** License is research-only.
- **Not a frontier model.** 742m is 190x under Chinchilla-optimal tokens for its scale; the model has acquired domain vocabulary but not general language.
- **Not a benchmark submission.** GSM8K, MMLU, and MATH-500 are eval-only; they have been 13-gram decontaminated from every training source.

## Next pointer

- For users: [USER_GUIDE](../docs/USER_GUIDE/README.md)
- For contributors: [DEVELOPER_GUIDE](../docs/DEVELOPER_GUIDE/README.md)
- For the science: [THEORY](../docs/THEORY/README.md)
- For the story: [article/README.md](../docs/article/README.md)
- For the walkthrough: [walkthrough.md](../docs/walkthrough.md)
