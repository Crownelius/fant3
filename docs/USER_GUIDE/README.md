# FANT 3 User Guide

How to run FANT 3: training, evaluation, and the Colab/RunPod paths. This guide assumes you have a working Python 3.10+ environment and read access to the repository.

## Path 1: Local smoke test (CPU, under 2 minutes)

```bash
git clone https://github.com/Crownelius/fant3.git
cd fant3
pip install -r requirements.txt

# Confirm all three curriculum presets load and phase boundaries compute correctly
python scripts/smoke_curriculum.py

# Run the full test suite (125 tests, under a minute on CPU)
python -m pytest tests/ --ignore=tests/test_ahn.py -v
```

If both pass, the workspace is good; you can proceed to Colab or RunPod.

## Path 2: Colab (recommended for first training run)

1. Open `notebooks/fant3_1b_nvidia_train.ipynb` in Google Colab with an A100.
2. Set `TARGET_SCALE = '50m'` in cell 7 (or your preferred scale from the table below).
3. Mount Drive so checkpoints persist.
4. Run top-to-bottom. Expected wall time for 50m: ~12 h to 60,000 steps.
5. See [docs/NOTEBOOKS.md](../NOTEBOOKS.md) for the cell-by-cell walkthrough.

## Path 3: RunPod (primary training target)

### One-time setup

- Create a RunPod account; set `WANDB_API_KEY` and `HF_TOKEN` in the pod dashboard (never via CLI).
- Rebuild the code artifact locally: `python scripts/build_fant_zip.py`.
- Upload `fant_code.zip` to the pod's network volume.

### Launching a training run

```bash
# DeepInsight 3-phase curriculum (recommended)
python scripts/runpod_train.py --scale 50m --curriculum deepinsight_3phase \
  --batch-size 8 --grad-accum 2 --peak-lr 3e-4 --warmup-steps 1000 \
  --total-steps 1000000 --ckpt-every 2500 --ckpt-keep-last 3 \
  --wandb-project fant3 --wandb-run-name curriculum_deepinsight --hf-login

# Legacy 2-phase curriculum (control / backward compat)
python scripts/runpod_train.py --scale 50m \
  --phase-a-steps 60000 --total-steps 1000000 \
  --batch-size 8 --grad-accum 2 --peak-lr 3e-4 --warmup-steps 1000 \
  --ckpt-every 2500 --ckpt-keep-last 3 \
  --wandb-project fant3 --wandb-run-name curriculum_legacy --hf-login

# No-curriculum control arm
python scripts/runpod_train.py --scale 50m --curriculum flat_1phase \
  --batch-size 8 --grad-accum 2 --peak-lr 3e-4 --warmup-steps 1000 \
  --total-steps 1000000 --ckpt-every 2500 --ckpt-keep-last 3 \
  --wandb-project fant3 --wandb-run-name curriculum_flat --hf-login
```

### Resuming

Every 2,500 steps a checkpoint is saved as `step_XXXXX.pt` under `args.ckpt_dir`. Resume via:

```bash
python scripts/runpod_train.py --resume output/ckpts/step_50000.pt [all other flags]
```

Milestone checkpoints (phase boundaries and final) are saved with a `_phase_{name}.pt` or `_final.pt` suffix and are never swept by `--ckpt-keep-last`.

## Scale ladder

Pick the right scale for your hardware and token budget:

| Preset | Stored params | Min VRAM | Wall time (2500 steps) | Best fit |
|---|---|---|---|---|
| `fant3_1m` | 0.99 M | CPU | 30 s / step | Laptop smoke; copy task |
| `fant3_10m` | 9.5 M | 4 GB | 2 min | Sub-Chinchilla sanity |
| `fant3_15m` | 14.6 M | 4 GB | 3 min | Colab T4 |
| `fant3_20m` | 23.5 M | 6 GB | 5 min | Colab T4 or RTX 3060 |
| `fant3_50m` | 50.79 M | 12 GB | 12 min | RunPod, Colab A100 |
| `fant3_150m` | 96 M | 12 GB | 20 min | RTX 3060 |
| `fant3_350m` | 263 M | 24 GB | 45 min | A100 40 GB |
| `fant3_742m` | 770.88 M | 46 GB | 78 min | A100 80 GB |
| `fant3_1b` | 986.62 M | 50 GB | 95 min | A100 80 GB |

`fant3_742m` and above require 8-bit AdamW + gradient checkpointing (both default-on for those presets).

## Training curricula

Three named presets ship with the workspace:

| Preset | Phases | Purpose |
|---|---|---|
| `legacy_2phase` | 2 | Pre-curriculum default; bit-identical to hardcoded runs from before 2026-04-24 |
| `deepinsight_3phase` | 3 | Apprentice/Journeyman/Expert per arxiv:2604.16278; expected gain at 1B–3B |
| `flat_1phase` | 1 | No-curriculum control arm for A/B testing |

See [THEORY/README.md#progressive-curriculum](../THEORY/README.md#progressive-curriculum) for the mathematics and `fant3/training/curriculum.py` for the implementation.

## Evaluation

After training:

```bash
python scripts/eval_benchmarks.py \
  --ckpt output/ckpts/step_50000_final.pt \
  --tokenizer output/tokenizer/tokenizer_v2.json \
  --benchmark gsm8k \
  --n 100
```

Supported: `gsm8k`, `mmlu`, `math500`. All benchmarks run decontaminated (13-gram SHA-1 filter against test sets, 457,910 hashes). Wilson 95% confidence intervals are reported.

Expected numbers at current training scale (742m Tier C, 82 M tokens):

- **GSM8K**: 1–4% (under-trained; Chinchilla-optimal would be ~15 B tokens for 742 M params)
- **MMLU**: ~26% (statistical chance on 4-way multiple choice)

The Colab notebook runs GSM8K + MMLU automatically after training completes (cell 26).

## Monitoring

- **Weights & Biases**: set `--wandb-project fant3` to stream loss, grad_norm, LR, chirality balance, VRAM.
- **Stdout**: every `--log-every` (default 25) steps, one line with phase tag, CE, z-loss, grad norm, max logit, chirality.
- **Checkpoints**: every `--ckpt-every` (default 500) steps. Rolling trim via `--ckpt-keep-last N` keeps only the most recent N non-milestone checkpoints.

## Common first-run issues

- **"CUDA out of memory"** — drop to a smaller scale or enable gradient checkpointing (`cfg.use_gradient_checkpointing = True`; already default at 742m+).
- **"Cannot find tokenizer_v2.json"** — the tokenizer is in `output/tokenizer/`. If you cloned fresh, re-run the tokenizer retrain: `python scripts/retrain_tokenizer.py`.
- **"WARNING: HF_TOKEN env var is unset"** — the NVIDIA Nemotron-CC datasets are gated. Either skip them (the default 50m mix doesn't need gated data) or set the env var and pass `--hf-login`.

For deeper issues: see [DEVELOPER_GUIDE/README.md](../DEVELOPER_GUIDE/README.md).
