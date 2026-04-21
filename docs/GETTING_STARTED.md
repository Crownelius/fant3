# Getting Started with FANT 3

This guide gets a new engineer to a running training loop in approximately five minutes.

---

## Prerequisites

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | 3.10 | 3.11 |
| GPU (Graphics Processing Unit) | A100 40 GB (for 50m scale) | A100 80–96 GB (for 742m) |
| Colab tier | Pro (A100 access) | Pro+ or PAYG for A100 80 GB |
| Google Drive free space | 5 GB (code + checkpoints) | 20 GB+ for multi-run history |

For local development without a GPU, the smoke test and test suite run on CPU.

---

## Step 1: Get the code onto Google Drive

On your local machine, zip the project:

```bash
# From D:\FANT_TRAINING_D_Drive\fant2\ on Windows:
# Zip: fant2/, fant3/, bendvm/, output/tokenizer/, tests/, scripts/, notebooks/
# Save as: fant_code.zip
```

Upload `fant_code.zip` to `MyDrive/fant_code.zip` in Google Drive.

The pre-built zip at `D:\FANT_TRAINING_D_Drive\fant2\fant_code.zip` (5.16 MB, 129 files) is the canonical version — upload that directly.

---

## Step 2: Open the notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. File → Open notebook → Google Drive → find `fant3_colab_train.ipynb` in `MyDrive/`  
   *(If the notebook is not in Drive, upload `notebooks/fant3_colab_train.ipynb` from this repo first)*
3. Runtime → Change runtime type → **A100** (GPU)

---

## Step 3: Set your target scale

In cell 4, change the single constant:

```python
TARGET_SCALE = '50m'   # start here for your first run
```

Supported values: `'20m'`, `'50m'`, `'150m'`, `'350m'`, `'742m'`, `'1b'`.

Everything else — architecture config, data mix, learning rate (LR) schedule, batch size, sequence length, total steps — adjusts automatically based on this one constant.

---

## Step 4: Run all cells top-to-bottom

Press `Runtime → Run all` (or `Ctrl+F9`). Expected output per cell:

| Cell | What you should see |
|---|---|
| 1 — GPU check | `torch X.Y.Z cuda A.B bf16 supported True` |
| 2 — Install deps | `bitsandbytes X.Y tokenizers X.Y datasets X.Y` |
| 3 — Mount Drive | Drive mount dialog, then `All imports OK` |
| 4 — Pick scale | `Target scale: 50m  dim=384 layers=12 megapools=2x4 topk=2` |
| 5 — Tokenizer | `vocab=32768  sample len NB -> N tokens` |
| 6 — Data pipeline | `Using MIX_V4_CHAT ...` or `MIX_V3_NVIDIA ...`, then `data pipeline ready` |
| 7 — Decontamination | `Decontamination filter active. Signatures loaded for GSM8K + MATH-500 + MMLU.` |
| 8 — Model + optimizer | `Built model: 50.79M stored params on cuda torch.bfloat16` and `Optimizer: bnb.AdamW8bit` |
| 9 — Resume checkpoint | `No checkpoint found — starting fresh` (first run) |
| 10 — Training loop | Streaming log lines like `[  25/60000] ce 10.247  α 0 β 0 chir 0.500  lr 4.17e-05  tok/s 18420  vram 4.21GB` |
| 11 — Loss plots | Matplotlib figure (CE (Cross-Entropy) curve + chirality curve) |
| 12 — Quality probe | Five arithmetic prompts and model completions |
| 13–14 — Benchmark eval | GSM8K + MMLU accuracy with Wilson 95% CI |
| 15 — Scale-up guide | Static markdown table (no code) |

---

## Step 5: Interpreting the training log

```
[  25/60000] ce 10.247  α 0 β 0 chir 0.500  lr 4.17e-05  tok/s 18420  vram 4.21GB
```

| Field | Meaning |
|---|---|
| `ce` | CE loss (Cross-Entropy loss). Starts near `ln(32768) ≈ 10.4` for random weights. Healthy descent: 10.4 → 8 → 6 → 5 → ... |
| `α` / `β` | Fill level of the Apollonian α (instance) and β (schema) memory packs |
| `chir` | Chirality balance of SpinorApollonianMemory. Should stay near 0.5; extreme values (< 0.1 or > 0.9) indicate starvation |
| `lr` | Current LR (Learning Rate) after warmup schedule |
| `tok/s` | Training throughput in tokens per second |
| `vram` | Peak VRAM (Video RAM) allocated so far |

---

## Step 6: Resume after session expiry

Checkpoints save to `MyDrive/fant_ckpts/<scale>/step_NNNNN.pt` every 250 steps. On next session:

1. Open the same notebook
2. Run all cells — cell 9 will detect the latest checkpoint and resume automatically
3. Log line will read: `Resumed from .../step_02500.pt at step 2500, 100 log entries`

---

## Local development (no GPU)

```bash
# Install
pip install -r requirements.txt

# Smoke test — CPU only, ~30 seconds
python scripts/smoke_fant3.py

# Full test suite — CPU, ~2 minutes
python -m pytest tests/ -v

# Scale-ladder smoke (validates all 5 scales without training)
python scripts/scale_ladder_smoke.py
```

The smoke config (`fant3_smoke()`) uses `dim=512, n_layers=8` and disables Cerebellum, fitting in ~2 GB.

---

## Common problems

| Symptom | Cause | Fix |
|---|---|---|
| `OutOfMemoryError` at step 1 (742m) | Activations scale linearly with sequence length; MoE gathers large slices | Ensure cell 8 prints `use_gradient_checkpointing: True`; confirm `TARGET_SCALE` is `'742m'` or `'1b'` |
| CE diverges to `nan` after warmup | LR (Learning Rate) too high or warmup too short | Reduce `peak_lr_setting`; increase `WARMUP_STEPS` |
| `<|pad|><|pad|>` loops in quality probe | Pad token leaked into training targets | Confirm cell 12 has `targets[targets == _PAD_ID] = -100` |
| Chirality collapses to 0.0 or 1.0 | SpinorApollonian not enabled or proj_spinor untrained | Confirm `cfg.spinor_apollonian_enabled = True` in cell 4 config builders |
| `tokenizer_v2.json not found` | Old zip on Drive without the retrained tokenizer | Re-upload `fant_code.zip` from `D:\FANT_TRAINING_D_Drive\fant2\` |
| `fant_code.zip` not found | Wrong Drive path | Set `CODE_ZIP = '/content/drive/MyDrive/fant_code.zip'` in cell 3 |

---

## Next steps

- Read [docs/NOTEBOOKS.md](NOTEBOOKS.md) for a complete cell-by-cell reference
- Read [docs/HISTORY.md](HISTORY.md) to understand what worked and what failed in previous runs
- Read [docs/ADR/](ADR/) for the reasoning behind each major architectural choice
- Once 50m training looks healthy (CE descending, chirality stable), scale up by changing `TARGET_SCALE` and re-running from cell 4
