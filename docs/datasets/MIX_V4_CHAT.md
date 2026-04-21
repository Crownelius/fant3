# MIX V4 CHAT — 12-Source Chat-Focused Mix

**Used for:** 20m and 50m training scales in `notebooks/fant3_colab_train.ipynb` (cell 12).

**Purpose:** Optimise for fluent short-form chat quality — the goal for small scales is approaching Qwen-2B chat behaviour, not reasoning benchmark scores. Sonnet 4.6 reasoning traces at 22% provide the highest-quality distillation signal. NVIDIA corpora contribute chat/instruction-following diversity. FineWeb-Edu anchors general language statistics.

**Total weight:** 1.00 (weights re-normalised if any source fails to load).

## Dataset table

| HF (Hugging Face) ID | Config | Split | Text key | Format class | Weight | License | Contamination rate | Approx size | Role in mix |
|---|---|---|---|---|---|---|---|---|---|
| `Roman1111111/claude-sonnet-4.6-120000x` | — | train | `messages` | MESSAGES | **0.22** | Mixed (research) | 0.20% | 120 K traces | Primary quality signal — Claude Sonnet 4.6 + Gemini 3.1 Pro traces, multi-domain, quality-graded |
| `crownelius/Opus-4.6-Reasoning-3300x` | — | train | `problem` | PROBLEM_THINK_SOLUTION | 0.14 | Research | 0.00% | ~2.2 K | Reasoning anchor — Claude Opus 4.6 traces with explicit `<|think|>` blocks; teaches the Phase 5 output format |
| `ianncity/KIMI-K2.5-1000000x` | General-Distillation | train | `messages` | MESSAGES | 0.12 | Research | 0.10% | 398 K | Broad reasoning diversity — Kimi K2.5 CoT (Chain of Thought) traces |
| `nvidia/Nemotron-Cascade-2-SFT-Data` | chat | train | `messages` | MESSAGES | 0.10 | CC-BY-4.0 | 1.67%* | ~50 K | Chat instruction-following — Cascade-2 SFT (Supervised Fine-Tuning) chat subset |
| `mlabonne/FineTome-100k` | — | train | `conversations` | CONVERSATIONS | 0.08 | Apache-2.0 | 0.10% | 100 K | High-quality curated SFT pairs |
| `HuggingFaceFW/fineweb-edu` | default | train | `text` | FLAT_TEXT | 0.10 | ODC-By | 0.00% | 1.3 T tokens | General language anchor — cleaned educational web text |
| `nvidia/OpenMathReasoning` | — | cot | `problem` | PROBLEM_SOLUTION | 0.05 | CC-BY-4.0 | 0.00% | 5.68 M (cot split) | Math reasoning — DeepSeek-R1-generated CoT solutions |
| `nvidia/Nemotron-Cascade-2-SFT-Data` | instruction_following | train | `messages` | MESSAGES | 0.06 | CC-BY-4.0 | ~0.80% | ~40 K | Instruction-following diversity |
| `nvidia/Daring-Anteater` | — | train | `conversations` | CONVERSATIONS | 0.05 | CC-BY-4.0 | ~0.30% | 100 K | Curated high-quality general conversations |
| `nvidia/OpenMathInstruct-2` | — | train | `problem` | PROBLEM_SOLUTION | 0.04 | CC-BY-4.0 | 0.00% | 14 M | Math breadth — NVIDIA pre-decontaminated math corpus |
| `nvidia/OpenCodeReasoning-2` | — | python | `question` | PROBLEM_SOLUTION | 0.02 | CC-BY-4.0 | 0.00% | 2.16 M (python split) | Code reasoning — Python problems with r1_generation traces |
| `nvidia/Nemotron-Cascade-2-SFT-Data` | math | train | `messages` | MESSAGES | 0.02 | CC-BY-4.0 | ~0.50% | ~30 K | Math SFT diversity |

*Cascade-2 chat is the highest-contamination source in this mix (1.67% on MATH-500 adjacent problems). The decontamination filter in the training loop drops affected documents before tokenisation.

## Why Sonnet 4.6 at 22%

The `Roman1111111/claude-sonnet-4.6-120000x` dataset provides 120 K multi-domain traces (general, code, math, psychology) from Claude Sonnet 4.6 and Gemini 3.1 Pro, pre-graded for quality. At 5M–50M parameter scales, distillation quality matters more than data volume: a high-quality teacher signal of 22% outweighs a larger but noisier corpus. Empirically, the 6% at-chance result from the pre-fix FANT 2 step_3000 run (which lacked `<|answer|>` wrapping) was reversed not by adding more data but by fixing format alignment — which Sonnet 4.6 traces provide correctly by construction.

The 22% weight is the single largest slice in MIX V4, deliberately exceeding the combined NVIDIA math/code footprint (0.04 + 0.02 + 0.02 = 0.08) to bias the small models toward conversational fluency over narrow benchmark performance.

## Format notes

- **PROBLEM_THINK_SOLUTION** (Crownelius): `extract_text()` calls `format_assistant_reasoning(thinking, solution)` to emit `<|think|>...<|/think|>\n<|answer|>...<|/answer|>`. This is the exact shape the eval extractor expects.
- **PROBLEM_SOLUTION** (NVIDIA datasets): `extract_text()` falls back through `solution` → `generated_solution` → `answer` key names to handle NVIDIA's `generated_solution` field in OpenMathInstruct-2.
- **MESSAGES** (Sonnet, Kimi, Cascade-2): both OpenAI-style `{role, content}` and ShareGPT-style `{from, value}` are handled by `_normalize_messages()`.

## Decontamination

All documents in this mix pass through `is_contaminated()` from `scripts/decontaminate.py` before tokenisation (active in notebook cell 14). Any document containing a 13-gram matching a GSM8K, MATH-500, or MMLU (Massive Multitask Language Understanding) test question is silently dropped. Observed rejection rate is under 2% across all sources.
