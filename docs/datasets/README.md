# Datasets — Overview

This directory documents every dataset and data-pipeline concept used in the FANT 3 project.

## Contents

| File | What it covers |
|------|---------------|
| [MIX_V4_CHAT.md](MIX_V4_CHAT.md) | 12-source chat-focused mix for 20m/50m scales. Sonnet 4.6 at 22% is the dominant signal. |
| [MIX_V3_NVIDIA.md](MIX_V3_NVIDIA.md) | 11-source reasoning-heavy mix for 150m+ scales. NVIDIA corpora comprise ~60% by weight. |
| [DECONTAMINATION.md](DECONTAMINATION.md) | The 13-gram SHA-1 filter that prevents benchmark test-set leakage into training. |
| [TOKENIZER.md](TOKENIZER.md) | The retrained `tokenizer_v2.json` BPE (Byte-Pair Encoding) vocabulary with 32 K entries. |

## Golden constraint

Public benchmarks — GSM8K (Grade-School Math 8K), MATH-500, MMLU (Massive Multitask Language Understanding) — are **eval-only**. They must never appear as training sources. The registry enforces this by keeping them in `EVAL_DATASETS`, which is a separate dict from `TRAINING_DATASETS` in `fant2/data/registry.py`. The decontamination filter provides a second line of defence at the document level.

## Supported dataset formats

Seven format classes are defined in `fant2/data/formats.py` under the `DatasetFormat` enum. All are normalised to a plain ChatML string by `extract_text()` before tokenisation.

| Format | Schema | Typical source |
|--------|--------|----------------|
| `FLAT_TEXT` | `{"text": "..."}` | FineWeb-Edu, OpenWebText |
| `MESSAGES` | `{"messages": [{role, content}]}` | Kimi K2.5, Sonnet 4.6, Cascade-2 |
| `CONVERSATIONS` | `{"conversations": [{from, value}]}` | FineTome, Daring-Anteater |
| `INPUT_OUTPUT` | `{"input": "...", "output": "..."}` | Superior Reasoning |
| `MESSAGES_JSON` | `{"messages_json": "<json string>"}` | Claude Code Traces |
| `PROBLEM_SOLUTION` | `{"problem": "...", "solution": "..."}` | NuminaMath, OpenMathInstruct-2 |
| `PROBLEM_THINK_SOLUTION` | `{"problem": "...", "thinking": "...", "solution": "..."}` | Crownelius Opus 4.6 |

`PROBLEM_SOLUTION` also falls back through `generated_solution` and `answer` key names (needed for NVIDIA's `OpenMathInstruct-2`, which uses `generated_solution`).

`PROBLEM_THINK_SOLUTION` emits the exact Phase 5 eval shape — `<|think|>...<|/think|>\n<|answer|>...<|/answer|>` — using `format_assistant_reasoning()` from the chat template module.

## Registry

`fant2/data/registry.py` contains two dicts:

- `TRAINING_DATASETS` — 27+ entries spanning HF (Hugging Face) dataset IDs, configs, splits, text keys, formats, and training phases.
- `EVAL_DATASETS` — 5 entries (GSM8K, ARC-Easy, ARC-Challenge, MMLU, HellaSwag). Never use these in training.

`get_dataset(name, streaming=True)` loads any registered dataset by short name and returns a HuggingFace `Dataset` or `IterableDataset`.
