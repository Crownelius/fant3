"""Install NVIDIA MIX v3 + NeMo-style training recipe into the canonical notebook."""
import json
from pathlib import Path

NB = Path(__file__).parent / "fant3_colab_train.ipynb"
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

# =============================================================================
# Cell 12 — NVIDIA-centric MIX v3
# =============================================================================
cell12 = nb["cells"][12]
src12 = "".join(cell12["source"])

old_mix = """# Mix v2 — added Sonnet 4.6 120K (2026-04-19) + NVIDIA OpenMathInstruct-2 (phased in)
MIX = [
    ('HuggingFaceFW/fineweb-edu',                         'default',              'train', 'text',          DatasetFormat.FLAT_TEXT,              0.30, None),
    ('Roman1111111/claude-sonnet-4.6-120000x',            None,                    'train', 'messages',      DatasetFormat.MESSAGES,               0.20, None),
    ('crownelius/Opus-4.6-Reasoning-3300x',               None,                    'train', 'problem',       DatasetFormat.PROBLEM_THINK_SOLUTION, 0.15, None),
    ('ianncity/KIMI-K2.5-1000000x',                       'General-Distillation', 'train', 'messages',      DatasetFormat.MESSAGES,               0.10, None),
    ('nvidia/OpenMathInstruct-2',                         None,                    'train', 'problem',       DatasetFormat.PROBLEM_SOLUTION,       0.08, None),
    ('AI-MO/NuminaMath-CoT',                              None,                    'train', 'problem',       DatasetFormat.PROBLEM_SOLUTION,       0.07, None),
    ('mlabonne/FineTome-100k',                            None,                    'train', 'conversations', DatasetFormat.CONVERSATIONS,          0.05, None),
    ('Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b','stage1',               'train', 'output',        DatasetFormat.INPUT_OUTPUT,           0.05, 'input'),
]"""

new_mix = """# Mix v3 (2026-04-19) — NVIDIA-centric.
# All non-gated NVIDIA datasets (CC-BY-4.0) plus complementary non-NVIDIA
# sources for web-prose diversity and Sonnet 4.6 reasoning.
# The gated Nemotron-CC-* corpora (v2.1 / Math-v1 / Code-v1) can be added
# later once HF access is granted — request on each dataset page.
#
# Columns:  (hf_id,  config,  split,  text_key,  format,  weight,  input_key)
MIX = [
    # NVIDIA reasoning / SFT backbone (60% total weight)
    ('nvidia/OpenMathReasoning',                          None,                    'cot',            'problem',       DatasetFormat.PROBLEM_SOLUTION,       0.18, None),
    ('nvidia/OpenCodeReasoning-2',                        None,                    'python',         'question',      DatasetFormat.PROBLEM_SOLUTION,       0.08, None),
    ('nvidia/OpenMathInstruct-2',                         None,                    'train',          'problem',       DatasetFormat.PROBLEM_SOLUTION,       0.08, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'math',                  'train',          'messages',      DatasetFormat.MESSAGES,               0.08, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'chat',                  'train',          'messages',      DatasetFormat.MESSAGES,               0.06, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'instruction_following', 'train',          'messages',      DatasetFormat.MESSAGES,               0.04, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'science',               'train',          'messages',      DatasetFormat.MESSAGES,               0.04, None),
    ('nvidia/Daring-Anteater',                            None,                    'train',          'conversations', DatasetFormat.CONVERSATIONS,          0.04, None),
    # Complementary external (until Nemotron-CC-v2.1 access is granted)
    ('HuggingFaceFW/fineweb-edu',                         'default',               'train',          'text',          DatasetFormat.FLAT_TEXT,              0.20, None),
    ('Roman1111111/claude-sonnet-4.6-120000x',            None,                    'train',          'messages',      DatasetFormat.MESSAGES,               0.12, None),
    ('crownelius/Opus-4.6-Reasoning-3300x',               None,                    'train',          'problem',       DatasetFormat.PROBLEM_THINK_SOLUTION, 0.08, None),
]
# Total weights: 1.00. Safety data (Aegis-2.0) deferred to Phase 6 (RLHF/alignment)."""

assert old_mix in src12, "MIX v2 block not found — maybe already patched"
src12 = src12.replace(old_mix, new_mix, 1)
cell12["source"] = src12.splitlines(keepends=True)
print("Cell 12: MIX v3 installed (11 sources, NVIDIA 0.60, fineweb 0.20, Sonnet 0.12, Opus 0.08)")

# =============================================================================
# Cell 20 — NeMo-style training recipe
# =============================================================================
cell20 = nb["cells"][20]
src20 = "".join(cell20["source"])

old_tr = """BATCH_SIZE      = 2    # MoE+MoR activation cost: B=2 fits 150m on A100 80 GB with headroom
                       # (earlier B=4 OOM\u2019d at 52 GB allocated + 22 GB fragmented)
                       # Raise to 4 only at 50m scale or if you add gradient checkpointing
SEQ_LEN         = 512  # raise to 1024 once loss is descending cleanly
TOTAL_STEPS     = 2500
LOG_EVERY       = 25
CKPT_EVERY      = 250
STORE_EVERY     = 50   # how often to push hidden states to Apollonian packs
GRAD_CLIP       = 1.0
WARMUP_STEPS    = 200"""

new_tr = """# ── NeMo / Nemotron-style recipe (2026-04-19) ───────────────────────────
# Based on Nemotron-4 and Llama-3 pretraining recipes adapted for Colab A100:
#   * Longer warmup (500 steps) — matches NVIDIA 2-5% of total, extra-safe for MoE
#   * Higher peak LR (2e-4) — halfway between NeMo's 3e-4 and our previous 1.5e-4
#   * Gradient accumulation = 4 -> effective batch 8 (closer to their 4M/step intent)
#   * AdamW betas (0.9, 0.95), weight_decay 0.1, eps 1e-8 — all NVIDIA defaults
BATCH_SIZE        = 2      # micro-batch
GRAD_ACCUM_STEPS  = 4      # effective batch = BATCH_SIZE * GRAD_ACCUM_STEPS = 8
SEQ_LEN           = 512    # raise to 1024 when loss is descending cleanly
TOTAL_STEPS       = 2500
LOG_EVERY         = 25
CKPT_EVERY        = 250
STORE_EVERY       = 50     # Apollonian memory store cadence
GRAD_CLIP         = 1.0
WARMUP_STEPS      = 500    # NeMo uses 2-5% of total; 20% here for MoE stability"""

assert old_tr in src20, "training constants block not found"
src20 = src20.replace(old_tr, new_tr, 1)

# Peak LR update
old_peak = """    lr_mul = lr_at(step)
    peak_lr = 1.5e-4  # was 3e-4 \u2014 caused NaN at step 672 in first 150m run
    for g in opt.param_groups:
        g['lr'] = peak_lr * lr_mul"""

new_peak = """    lr_mul = lr_at(step)
    peak_lr = 2.0e-4  # NeMo-style: bumped from 1.5e-4; still well below the 3e-4 that NaN'd
    for g in opt.param_groups:
        g['lr'] = peak_lr * lr_mul"""

assert old_peak in src20, "peak LR block not found"
src20 = src20.replace(old_peak, new_peak, 1)

# Gradient accumulation
old_bwd = """    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP if step >= WARMUP_STEPS else 0.5)
    opt.step()
    opt.zero_grad(set_to_none=True)"""

new_bwd = """    # Gradient accumulation: scale loss before backward so the averaged gradient
    # over GRAD_ACCUM_STEPS micro-steps equals what a single larger batch would give.
    (loss / GRAD_ACCUM_STEPS).backward()
    if (step + 1) % GRAD_ACCUM_STEPS == 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP if step >= WARMUP_STEPS else 0.5)
        opt.step()
        opt.zero_grad(set_to_none=True)"""

assert old_bwd in src20, "backward block not found"
src20 = src20.replace(old_bwd, new_bwd, 1)

cell20["source"] = src20.splitlines(keepends=True)
print("Cell 20: NeMo-style recipe installed")
print("  warmup 200 -> 500")
print("  peak LR 1.5e-4 -> 2.0e-4")
print("  grad accumulation 1 -> 4 (effective batch 2 -> 8)")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print("notebook saved:", NB)
