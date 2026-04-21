# Notebook Walkthrough: fant3_colab_train.ipynb

This document provides a cell-by-cell reference for `notebooks/fant3_colab_train.ipynb`.

The notebook has **28 cells** (14 markdown, 14 code). All training is controlled by the single constant `TARGET_SCALE` set in cell 4 (code cell 3). Every subsequent cell reads this value and selects the appropriate architecture config, data mix, LR (Learning Rate) schedule, and hardware recipe.

---

## Cell inventory

| # | Type | Section heading | Code cell # | Purpose |
|---|---|---|---|---|
| 1 | Markdown | FANT 3 — Colab training notebook | — | Top-level description; lists all 5 pre-launch fixes; Colab setup instructions |
| 2 | Markdown | 1. GPU + environment check | — | Section header |
| 3 | Code | *(GPU check)* | 1 | `nvidia-smi` query; sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` BEFORE `import torch` (critical — CUDA allocator config is no-op after first CUDA init); prints torch/CUDA version and bf16 support |
| 4 | Markdown | 2. Install deps | — | Section header |
| 5 | Code | *(install)* | 2 | `pip install bitsandbytes>=0.43 tokenizers>=0.15 datasets>=2.18`; prints version sanity |
| 6 | Markdown | 3. Mount Google Drive and locate code | — | Section header |
| 7 | Code | *(drive + imports)* | 3 | Mounts Drive; extracts `fant_code.zip` to `/content/fant`; sets `HF_HOME` cache to Drive (avoids re-downloading datasets on session restart); imports `FANT3Model`, `SpinorApollonianMemory`, `ArtificialHippocampusNetwork`, `FANT2Tokenizer`, `extract_text` |
| 8 | Markdown | 4. Pick a scale | — | Section header |
| 9 | Code | *(scale selector)* | 4 | **The single control knob.** Sets `TARGET_SCALE`; defines `cfg_20m()` through `cfg_1b()` as config builder functions; calls the appropriate builder; prints dim/layers/megapools/top_k |
| 10 | Markdown | 5. Load tokenizer | — | Section header |
| 11 | Code | *(tokenizer)* | 5 | Loads `tokenizer_v2.json` from `output/tokenizer/`; tokenizes a sample ChatML string with `<|answer|>` tags; prints vocab size and token count |
| 12 | Markdown | 6. Data pipeline | — | Section header |
| 13 | Code | *(data pipeline)* | 6 | Defines `MIX_V4_CHAT` (12 sources, chat-heavy for 20m/50m) and `MIX_V3_NVIDIA` (11 sources, reasoning-heavy for 150m+); selects mix by `TARGET_SCALE`; implements `build_iterators()` and `sample_batch()` with pad-token masking (`targets[targets == _PAD_ID] = -100`) |
| 14 | Markdown | 7. Decontamination check | — | Section header + explanation of 13-gram contamination rates per source |
| 15 | Code | *(decontamination)* | 7 | Calls `build_hash_cache()` to build/load the 457,910-hash 13-gram filter; wraps `sample_batch()` with `is_contaminated()` rejection; logs rejection count |
| 16 | Markdown | 8. Build model + optimizer | — | Section header |
| 17 | Code | *(model + optimizer)* | 8 | Auto-enables `use_gradient_checkpointing` for `742m` and `1b`; builds `FANT3Model(cfg)` in bf16 on CUDA; prints stored param count; creates `bnb.optim.AdamW8bit` (falls back to `torch.AdamW` if bitsandbytes fails) |
| 18 | Markdown | 9. Resume from the latest checkpoint (if any) | — | Section header |
| 19 | Code | *(checkpoint resume)* | 9 | Scans `CKPT_DIR/<scale>/step_*.pt` sorted numerically; loads latest if present (model state, optimizer state, step counter, loss log, chirality log); prints resume step or "starting fresh" |
| 20 | Markdown | 10. Training loop | — | Section header |
| 21 | Code | *(training loop)* | 10 | **Main training cell.** Scale-aware recipe table (20m/50m/150m/350m/742m/1b); cosine LR schedule with linear warmup; gradient accumulation (`loss / GRAD_ACCUM_STEPS` before backward; optimizer step every N micro-steps); `NaN` guard with `training_diverged` flag (aborts without saving corrupt weights); periodic checkpoint save to Drive; memory stats (α fill, β fill, chirality); throughput and VRAM logging; final save skipped on divergence |
| 22 | Markdown | 11. Loss + chirality plots | — | Section header |
| 23 | Code | *(plots)* | 11 | Matplotlib 2-panel figure: CE loss curve (with dashed CE=4 coherence threshold) + spinor chirality balance over training; no-op if no logs yet |
| 24 | Markdown | 12. Quick quality probe | — | Section header |
| 25 | Code | *(quality probe)* | 12 | Greedy 64-token generation for 5 arithmetic prompts; stops on `<|im_end|>` / `<|eos|>` / `<|pad|>`; prints prompt + completion (first 200 chars) |
| 26 | Markdown | 13. Benchmark eval | — | Section header |
| 27 | Code | *(benchmark eval)* | 13 | Finds `final.pt` or latest `step_*.pt`; calls `scripts/eval_benchmarks.py` for GSM8K (n=50) and MMLU (n=200); prints last 10 lines of each result |
| 28 | Markdown | 14. Scale up | — | Scaling guidance table (batch / seq / steps / wall time / VRAM per scale); advice on when to scale (CE below ~6, chirality non-degenerate) |

---

## Scale-aware recipes (cell 10 in detail)

Cell 10 selects the training recipe from the `TARGET_SCALE` constant. The full table:

| Scale | BATCH_SIZE | GRAD_ACCUM | Effective batch | SEQ_LEN | TOTAL_STEPS | WARMUP_STEPS | peak_lr | Training tokens |
|---|---|---|---|---|---|---|---|---|
| `20m` | 16 | 2 | 32 | 1024 | 70,000 | 7,000 | 5.0e-4 | ~2.3B |
| `50m` | 16 | 2 | 32 | 1024 | 60,000 | 6,000 | 4.0e-4 | ~2.0B |
| `150m` | 2 | 4 | 8 | 512 | 2,500 | 500 | 2.0e-4 | ~10M |
| `350m` | 2 | 4 | 8 | 512 | 5,000 | 750 | 1.8e-4 | ~20M |
| `742m` | 1 | 8 | 8 | 1024 | 10,000 | 1,500 | 1.5e-4 | ~82M |
| `1b` | 1 | 8 | 8 | 1024 | 12,000 | 1,800 | 1.2e-4 | ~98M |

The effective batch is always 8 (or 32 for chat scales) so the LR math is consistent across scales. GRAD_ACCUM_STEPS scales inversely with BATCH_SIZE.

---

## Data mixes (cell 6 in detail)

### MIX v4 — chat-focused (20m, 50m)

| Source | Format | Weight |
|---|---|---|
| Roman1111111/claude-sonnet-4.6-120000x | MESSAGES | 0.22 |
| crownelius/Opus-4.6-Reasoning-3300x | PROBLEM_THINK_SOLUTION | 0.14 |
| ianncity/KIMI-K2.5-1000000x (General-Distillation) | MESSAGES | 0.12 |
| nvidia/Nemotron-Cascade-2-SFT-Data (chat) | MESSAGES | 0.10 |
| mlabonne/FineTome-100k | CONVERSATIONS | 0.08 |
| nvidia/Nemotron-Cascade-2-SFT-Data (instruction_following) | MESSAGES | 0.06 |
| HuggingFaceFW/fineweb-edu | FLAT_TEXT | 0.10 |
| nvidia/Daring-Anteater | CONVERSATIONS | 0.05 |
| nvidia/OpenMathReasoning (cot) | PROBLEM_SOLUTION | 0.05 |
| nvidia/OpenMathInstruct-2 | PROBLEM_SOLUTION | 0.04 |
| nvidia/OpenCodeReasoning-2 (python) | PROBLEM_SOLUTION | 0.02 |
| nvidia/Nemotron-Cascade-2-SFT-Data (math) | MESSAGES | 0.02 |

### MIX v3 — NVIDIA reasoning-heavy (150m, 350m, 742m, 1b)

| Source | Format | Weight |
|---|---|---|
| nvidia/OpenMathReasoning (cot) | PROBLEM_SOLUTION | 0.18 |
| nvidia/OpenCodeReasoning-2 (python) | PROBLEM_SOLUTION | 0.08 |
| nvidia/OpenMathInstruct-2 | PROBLEM_SOLUTION | 0.08 |
| nvidia/Nemotron-Cascade-2-SFT-Data (math) | MESSAGES | 0.08 |
| nvidia/Nemotron-Cascade-2-SFT-Data (chat) | MESSAGES | 0.06 |
| nvidia/Nemotron-Cascade-2-SFT-Data (instruction_following) | MESSAGES | 0.04 |
| nvidia/Nemotron-Cascade-2-SFT-Data (science) | MESSAGES | 0.04 |
| nvidia/Daring-Anteater | CONVERSATIONS | 0.04 |
| HuggingFaceFW/fineweb-edu | FLAT_TEXT | 0.20 |
| Roman1111111/claude-sonnet-4.6-120000x | MESSAGES | 0.12 |
| crownelius/Opus-4.6-Reasoning-3300x | PROBLEM_THINK_SOLUTION | 0.08 |

---

## Config builders (cell 4 in detail)

Each config builder function returns a `FANT3Config` instance. Key architectural differences by scale:

| Scale | dim | n_layers | Experts | MoR depths | Cerebellum | AHN | grad ckpt |
|---|---|---|---|---|---|---|---|
| 20m | 320 | 10 | 4 (2×2) | 2 | Off | Off | Off |
| 50m | 384 | 12 | 8 (2×4) | 2 | Off | Off | Off |
| 150m | 768 | 10 | 8 (2×4) | 2 | Off | On | Off |
| 350m | 1024 | 14 | 16 (4×4) | 2 | Off | On | Off |
| 742m | 1024 | 16 | 32 (4×8) | 2 | Off | On | **On** |
| 1b | 1024 | 20 | 32 (4×8) | 2 | Off | On | **On** |

Cerebellum is disabled in the Colab notebook at all scales to save VRAM. It can be re-enabled locally for research.

---

## Checkpoint format

Every checkpoint (`step_NNNNN.pt` and `final.pt`) is a `torch.save` dict with these keys:

| Key | Type | Description |
|---|---|---|
| `step` | int | Global training step at save time |
| `model` | OrderedDict | `model.state_dict()` |
| `opt` | dict | `optimizer.state_dict()` (omitted in some early checkpoints) |
| `losses` | list[tuple] | `(step, ce_value)` pairs since training start |
| `chirality` | list[tuple] | `(step, chirality_balance)` pairs |
| `cfg_scale` | str | `TARGET_SCALE` string (e.g. `'742m'`) |
| `cfg_dict` | dict | `dataclasses.asdict(cfg)` — full config for model reconstruction |

The `cfg_dict` key is critical: `eval_benchmarks.py` uses it to reconstruct the exact architecture without requiring the caller to know the scale.

---

## Known issues and mitigations

| Issue | Cells affected | Mitigation |
|---|---|---|
| `PYTORCH_CUDA_ALLOC_CONF` must be set before `import torch` | Cell 3 | Already fixed — env var is set at top of cell 3 before all imports |
| Pad token bleeds into training targets without masking | Cell 6, 7 | `targets[targets == _PAD_ID] = -100` in both `sample_batch` definitions |
| `loss_val = loss.item()` must be captured before `del loss` | Cell 10 | `loss_val` captured immediately after `loss = out['loss']`; `del out, loss, ids, targets` moved to end of iteration |
| NaN from too-high LR at step 600–800 | Cell 10 | `training_diverged` flag; tight warmup clip of 0.5 during warmup; `final.pt` not saved on divergence |
| MoR side-effects called twice under gradient checkpointing | Cell 8, fant3_model.py | `self.last_router_info` writes are idempotent — same value written on forward-save and backward-recompute |
