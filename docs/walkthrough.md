# FANT 3 walkthrough

From `git clone` to a trained checkpoint. Every architectural component you encounter on the way is explained inline or pointed to.

## 1. Clone and install

```bash
git clone https://github.com/Crownelius/fant3.git
cd fant3
pip install -r requirements.txt
```

The requirements are conservative: `torch`, `numpy`, `datasets`, `transformers`, `tokenizers`, `bitsandbytes`, `huggingface_hub`, `pytest`, `wandb`. Installable on any Python 3.10+ environment; the GPU-specific bits (`bitsandbytes`) require a CUDA runtime at training time but not at import time.

## 2. Run the smoke tests

```bash
python scripts/smoke_curriculum.py
python -m pytest tests/ --ignore=tests/test_ahn.py -v
```

`smoke_curriculum.py` loads all three curriculum presets, walks through their phases, and constructs `InterleavedMultiDatasetStream` objects (without iterating — that would hit HuggingFace network). Expected output: `ALL 3 preset(s) OK` in under 2 seconds.

The pytest suite runs 125 tests across the public modules:
- `test_smoke.py` — import and forward/backward at `fant3_smoke` preset
- `test_spinor_apollonian.py` — chirality balance, pack split
- `test_mor_lti.py` — Mixture of Recursions with LTI constraints
- `test_monotonic_dynamic_k.py` — 22 tests on ISRM
- `test_curriculum.py` — 49 tests on the curriculum module (just added)
- `test_router_collapse.py` — FANT 2 regression canary
- `test_sae.py` — SAE introspection
- `test_trainer_integration.py` — trainer end-to-end

`test_ahn.py` has a pre-existing fixture bug and is excluded.

## 3. Inspect the model

Build a tiny model to see the shapes:

```python
from fant3.config import fant3_1m
from fant3.model.fant3_model import FANT3Model

cfg = fant3_1m()           # ~0.99 M params, CPU-friendly
model = FANT3Model(cfg)
print(f"stored params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

# One forward pass
import torch
ids = torch.randint(0, cfg.vocab_size, (2, 32))
out = model(ids)
print(f"logits shape: {out['logits'].shape}")        # [2, 32, 2048]
print(f"router_infos: {len(out.get('router_infos', []))} entries")
```

At `fant3_1m` the model has no MoE routing (2 experts, top_k=1) and no AHN, Cerebellum, or ETF freeze. It's a bare attention + MoR harness for unit tests.

## 4. Inspect the curriculum

```python
from fant3.training import PRESETS, get_active_phase, build_curriculum

for name, curriculum in PRESETS.items():
    print(f"\n{name}: {len(curriculum.phases)} phase(s)")
    for p in curriculum.phases:
        print(f"  {p.name}: end_frac={p.end_frac:.2f}, datasets={len(p.datasets)}")

# Walk through training steps
cur = build_curriculum("deepinsight_3phase")
for step in [0, 1000, 3000, 5000, 8000, 12000]:
    phase = get_active_phase(step, 12000, cur)
    print(f"step {step:>5d}: phase = {phase.name}")
```

Expected output at `total_steps=12000`:
- step 0: apprentice
- step 1000: apprentice
- step 3000: apprentice (boundary — 0.25 * 12000)
- step 5000: journeyman
- step 8000: expert (after 0.65 boundary)
- step 12000: expert

## 5. Prepare for a real run

Three decisions before launching:

**Which scale?** See [size-comparison.md](./size-comparison.md) for hardware envelopes. For a first run: `fant3_50m` on Colab A100 or RunPod.

**Which curriculum?** Three options:
- `legacy_2phase` — backward-compatible default
- `deepinsight_3phase` — the paper-aligned curriculum (recommended)
- `flat_1phase` — no-curriculum control arm

**Where?** Colab (open the notebook, set `TARGET_SCALE`, run) or RunPod (upload `fant_code.zip`, run `runpod_train.py`). See [USER_GUIDE](./USER_GUIDE/README.md) for both paths.

## 6. Training on RunPod

Upload `fant_code.zip` (rebuild via `python scripts/build_fant_zip.py` if stale):

```bash
# Set these env vars in the pod dashboard, not CLI
export WANDB_API_KEY=...
export HF_TOKEN=...

python scripts/runpod_train.py \
  --scale 50m \
  --curriculum deepinsight_3phase \
  --batch-size 8 --grad-accum 2 \
  --peak-lr 3e-4 --warmup-steps 1000 \
  --total-steps 1000000 \
  --ckpt-every 2500 --ckpt-keep-last 3 \
  --wandb-project fant3 --wandb-run-name curriculum_deepinsight \
  --hf-login
```

The first few log lines you'll see:

```
device=cuda:0  dtype=torch.bfloat16  ckpt_dir=./output/runpod_ckpts
distributed=False  world_size=1  rank=0  local_rank=0
scale=50m  max_seq_len=1024
model built: 50.79 M params
HF authenticated via HF_TOKEN env var
wandb: initialized project=fant3 run=curriculum_deepinsight
aligning cfg.vocab_size 32768 -> 32768
tied emb/lm_head -> fp32
decontamination hashes: 457910
curriculum: deepinsight_3phase (3 phases)
  phase 0 'apprentice': end_frac=0.250 (end_step=250000) datasets=5 seq_len=1024
  phase 1 'journeyman': end_frac=0.650 (end_step=650000) datasets=7 seq_len=1024
  phase 2 'expert': end_frac=1.000 (end_step=1000000) datasets=7 seq_len=1024
```

Then, once per `--log-every` (default 25) steps:

```
[apprentice T=1024] step=   25 lr=1.50e-04 loss=10.4521 z=0.018 gn=5.2 max|logit|=8.3 max|rtr|=2.1 vram=11.2GB x1 chirality=0.487 nan_total=0 elapsed=0.4m
```

## 7. Watching training

- **Cross-entropy (loss).** Starts around `log(vocab_size) ≈ 10.4` for bf16 init. Should fall below 8.0 within 500 steps, below 6.0 within 2,500 steps at 50m scale.
- **Grad norm (gn).** Should stay below `--grad-clip` (default 10.0). If it hits the clip repeatedly, the LR is too high or the batch is too small.
- **Max logit.** If it exceeds 1,000, expect NaN next step. `--z-coef` regulates this; default 1e-4 is usually enough.
- **Chirality balance.** Healthy: 0.4–0.6. Starved: 0.0 or 1.0 (α or β pack empty).
- **VRAM.** Should stabilize after step 2 (allocator warmup). If it keeps climbing, there's a leak.

## 8. Resuming

Every 2,500 steps a checkpoint is saved. Resume:

```bash
python scripts/runpod_train.py --resume output/runpod_ckpts/step_50000.pt [all other flags]
```

Phase-boundary milestones and the final checkpoint get a suffix (e.g. `step_250000_phase_apprentice.pt`, `step_1000000_final.pt`) and are never swept by `--ckpt-keep-last`.

## 9. Evaluation

After training completes:

```bash
python scripts/eval_benchmarks.py \
  --ckpt output/runpod_ckpts/step_1000000_final.pt \
  --tokenizer output/tokenizer/tokenizer_v2.json \
  --benchmark gsm8k \
  --n 100
```

The Colab notebook (cell 26) runs GSM8K + MMLU automatically after training.

Expected at current pretraining budget (82 M tokens for 742m): GSM8K 1–4%, MMLU ~26% (statistical chance).

## 10. Next steps

- **Analyze the checkpoint.** Quality probe: `scripts/quality_probe.py`. Memory inspection: `fant3/diagnostics/sae.py`.
- **Compare curricula.** Run the `legacy_2phase` arm and compare CE curves + quality probes.
- **Queue Fix 3.** Once pretrain is stable, follow [fix3_rl_plan.md](./fix3_rl_plan.md) for verifier-distillation RL.

## See also

- [USER_GUIDE](./USER_GUIDE/README.md) — training recipes per scale
- [DEVELOPER_GUIDE](./DEVELOPER_GUIDE/README.md) — debugging, contributing
- [THEORY](./THEORY/README.md) — the mathematics per component
- [article](./article/README.md) — the narrative story
- [ACHIEVEMENTS](../ACHIEVEMENTS.md) — what has been measured
