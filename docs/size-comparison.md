# Size comparison

How FANT 3 stacks against frontier language models. All numbers are stored parameter count unless noted otherwise; bf16 storage at 2 bytes per parameter.

## Order of magnitude

| Model | Stored params | Storage (bf16) | Ratio vs FANT 3 50m |
|---|---:|---:|---:|
| FANT 3 `fant3_1m` | 0.99 M | 2 MB | 0.02x |
| FANT 3 `fant3_50m` | 50.79 M | 102 MB | 1x |
| FANT 3 `fant3_742m` | 770.88 M | 1.54 GB | 15.2x |
| FANT 3 `fant3_1b` | 986.62 M | 1.97 GB | 19.4x |
| SmolLM2-135M | 135 M | 270 MB | 2.7x |
| SmolLM2-360M | 360 M | 720 MB | 7.1x |
| Phi-3-mini-4k | 3.8 B | 7.6 GB | 75x |
| Qwen 2.5 1.5B | 1.5 B | 3 GB | 29x |
| Qwen 2.5 7B | 7 B | 14 GB | 138x |
| OLMoE-1B-7B | 7 B (1 B active) | 14 GB | 138x |
| Mistral 7B | 7 B | 14 GB | 138x |
| Llama 3 8B | 8 B | 16 GB | 158x |
| Llama 3 70B | 70 B | 140 GB | 1,378x |
| DeepSeek-V3 | 671 B | 1.34 TB | 13,200x |
| GPT-4 (est.) | ~1.7 T | ~3.4 TB | 33,500x |

At the 50m preset, FANT 3 is **33,000x smaller than GPT-4-class models** and fits in Colab's free tier. At the 1b preset it is **1,700x smaller than Llama 3 70B** and fits on a single A100 40 GB with room to spare.

## Active vs stored parameters

MoE models have two numbers that matter:
- **Stored** = total parameters on disk / in VRAM
- **Active** = parameters used per-token at inference

| Preset | Stored | Active (inference) |
|---|---:|---:|
| `fant3_1m` | 0.99 M | 0.99 M (no sparsity at this size) |
| `fant3_50m` | 50.79 M | ~30 M |
| `fant3_742m` | 770.88 M | ~80 M |
| `fant3_1b` | 986.62 M | ~100 M |

The active/stored ratio at 1b is roughly 1:10. Matryoshka MoE lets you trade further: at inference-time you can activate the level-0 "core" experts only, dropping active to ~50 M.

## What FANT 3 does NOT do at this size

Honesty about boundaries:

- **Does not speak broad domains well.** 742m was trained on 82 M tokens, 190x under Chinchilla-optimal. The model has domain vocabulary (NVIDIA math corpus terms like "mutex", "idempotency", "AoPS") but not general language.
- **Does not solve GSM8K, MMLU, or MATH-500.** All three are at statistical chance in recent evals. The Chinchilla-optimal budget for 742 M is ~15 B tokens; we have burned under 1% of that.
- **Does not have RL-refined reasoning.** Fix 3 (GRPO with agentic verifier) is queued for post-pretrain but not yet run.

## What FANT 3 CAN do at this size

Compression validation (2026-04-17, see [ACHIEVEMENTS](../ACHIEVEMENTS.md)):

- **FANT 2 (60 M stored) beats classical codecs** (gzip, zlib, bz2, lzma) on three Gutenberg books. The Delétang 2023 compression-as-intelligence result replicates at our scale.
- **Qwen 2.5 1.5B** on the same books hits 0.6–1.0 bits-per-byte — 3.5–5.2x better than gzip, approaching Chinchilla-70 B quality at 47x fewer parameters.
- **bf16 vs fp32** produce 99.913% bit-identical quantized probabilities at 16-bit quantization, with 2.21x VRAM savings and a bpb delta of 0.0012.

## Training cost

For one complete training run at each scale on an A100 80 GB pod at roughly $2/h:

| Preset | Total steps | Wall time | Estimated cost (USD) |
|---|---:|---:|---:|
| `fant3_50m` (60 K steps) | 60,000 | ~12 h | $24 |
| `fant3_150m` | 2,500 | ~45 min | $1.50 |
| `fant3_350m` | 5,000 | ~2 h | $4 |
| `fant3_742m` (Tier C) | 10,000 | ~80 min | $3 |
| `fant3_1b` | 12,000 | ~95 min | $3 |

The A/B curriculum experiment (three runs at `fant3_50m`) costs approximately $72 on an A100 80 GB pod.

## Hardware envelopes

What hardware each scale needs end-to-end:

| GPU | Stored envelope | Comfortable scale | Pushing it |
|---|---|---|---|
| Laptop CPU | n/a | `fant3_1m` smoke | `fant3_10m` dry-run |
| RTX 3060 12 GB | ~10 GB | `fant3_150m` | `fant3_50m` full run |
| T4 16 GB (Colab free) | ~12 GB | `fant3_50m` | — |
| V100 32 GB | ~24 GB | `fant3_350m` | — |
| A100 40 GB | ~34 GB | `fant3_742m` with grad ckpt | — |
| A100 80 GB | ~70 GB | `fant3_1b` | `fant3_2b` (hypothetical) |
| A100 96 GB | ~85 GB | `fant3_1b` comfortable | — |
| H100 80 GB | ~70 GB | `fant3_1b` | faster than A100 |

No preset in this repository requires more than one GPU. The codebase supports torchrun-aware DDP for multi-GPU pods, but all tested configurations are single-GPU.

## See also

- [ACHIEVEMENTS](../ACHIEVEMENTS.md) for measured numbers at each preset
- [USER_GUIDE](./USER_GUIDE/README.md) for how to pick a scale
- [Glossary](./glossary.md) for Chinchilla-optimal, bf16, VRAM, and other terms
