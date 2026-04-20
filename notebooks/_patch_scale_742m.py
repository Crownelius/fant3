"""Patch the canonical notebook to default to 742m + scale-appropriate recipe."""
import json
from pathlib import Path

NB = Path(__file__).parent / "fant3_colab_train.ipynb"
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

# =============================================================================
# Cell 8 — flip TARGET_SCALE default to '742m'
# =============================================================================
cell8 = nb["cells"][8]
src8 = "".join(cell8["source"])
old_scale = "TARGET_SCALE = '150m'  # one of '50m', '150m', '350m', '742m', '1b'"
new_scale = "TARGET_SCALE = '742m'  # one of '50m', '150m', '350m', '742m', '1b'"
assert old_scale in src8, "TARGET_SCALE line not found"
src8 = src8.replace(old_scale, new_scale, 1)
cell8["source"] = src8.splitlines(keepends=True)
print("Cell 8: TARGET_SCALE default set to '742m'")

# =============================================================================
# Cell 20 — scale-appropriate training recipe for 742m on A100 80 GB
# =============================================================================
cell20 = nb["cells"][20]
src20 = "".join(cell20["source"])

old_tr = """BATCH_SIZE        = 2      # micro-batch
GRAD_ACCUM_STEPS  = 4      # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 8
SEQ_LEN           = 512    # raise to 1024 when loss is descending cleanly
TOTAL_STEPS       = 2500
LOG_EVERY         = 25
CKPT_EVERY        = 250
STORE_EVERY       = 50     # Apollonian memory store cadence
GRAD_CLIP         = 1.0
WARMUP_STEPS      = 500    # NeMo uses 2-5% of total; 20% here for MoE stability"""

new_tr = """# Scale-aware recipe (updated 2026-04-19 for 742m).  NVIDIA-style defaults;
# AdamW betas (0.9, 0.95), weight_decay 0.1, eps 1e-8 set in cell 16.
if TARGET_SCALE in ('50m', '150m'):
    BATCH_SIZE        = 2       # A100 80 GB: comfortable
    GRAD_ACCUM_STEPS  = 4       # effective batch = 8
    SEQ_LEN           = 512
    TOTAL_STEPS       = 2500
    WARMUP_STEPS      = 500     # 20% of total
    peak_lr_setting   = 2.0e-4
elif TARGET_SCALE == '350m':
    BATCH_SIZE        = 2
    GRAD_ACCUM_STEPS  = 4
    SEQ_LEN           = 512
    TOTAL_STEPS       = 5000
    WARMUP_STEPS      = 750
    peak_lr_setting   = 1.8e-4
elif TARGET_SCALE == '742m':
    BATCH_SIZE        = 1       # A100 80 GB estimate ~40 GB at B=1 T=512; raise to 2 only if you see headroom
    GRAD_ACCUM_STEPS  = 8       # effective batch preserved at 8
    SEQ_LEN           = 512
    TOTAL_STEPS       = 5000
    WARMUP_STEPS      = 750
    peak_lr_setting   = 1.5e-4
else:  # '1b' \u2014 needs A100 80 + grad checkpointing to fit
    BATCH_SIZE        = 1
    GRAD_ACCUM_STEPS  = 8
    SEQ_LEN           = 512
    TOTAL_STEPS       = 10000
    WARMUP_STEPS      = 1000
    peak_lr_setting   = 1.2e-4

LOG_EVERY           = 25
CKPT_EVERY          = 250
STORE_EVERY         = 50
GRAD_CLIP           = 1.0
print(f'Recipe: B={BATCH_SIZE} accum={GRAD_ACCUM_STEPS} seq={SEQ_LEN} '
      f'steps={TOTAL_STEPS} warmup={WARMUP_STEPS} peak_lr={peak_lr_setting:.1e}')"""

assert old_tr in src20, "old recipe block not found"
src20 = src20.replace(old_tr, new_tr, 1)

# Also replace the hardcoded peak_lr in the LR schedule loop with peak_lr_setting
old_peak = """    lr_mul = lr_at(step)
    peak_lr = 2.0e-4  # NeMo-style: bumped from 1.5e-4; still well below the 3e-4 that NaN'd
    for g in opt.param_groups:
        g['lr'] = peak_lr * lr_mul"""

new_peak = """    lr_mul = lr_at(step)
    for g in opt.param_groups:
        g['lr'] = peak_lr_setting * lr_mul"""

assert old_peak in src20, "peak_lr block not found"
src20 = src20.replace(old_peak, new_peak, 1)

cell20["source"] = src20.splitlines(keepends=True)
print("Cell 20: scale-aware recipe installed")
print("  742m defaults: B=1 accum=8 seq=512 steps=5000 warmup=750 peak_lr=1.5e-4")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"notebook saved: {NB}")
