"""Final patch for 12h chat-focused 50m run (2026-04-19).

Makes every decision end-to-end:
  1. TARGET_SCALE defaults to '50m' (chat-optimized target)
  2. Adds cfg_50m() in cell 8 using fant3_50m() preset
  3. Adds cfg_20m() in cell 8 using fant3_20m() preset
  4. Adds '20m' and '50m' branches to cell 20 scale-aware recipe
  5. Replaces MIX v3 (NVIDIA-heavy reasoning) with MIX v4 chat-heavy for 20m/50m
  6. Keeps MIX v3 for 742m/1b (their bigger-model sweet spot is reasoning)
"""
import json
from pathlib import Path

NB = Path(__file__).parent / "fant3_colab_train.ipynb"
with open(NB, encoding="utf-8") as f:
    nb = json.load(f)

# =============================================================================
# Cell 8 — TARGET_SCALE default + add cfg_20m/cfg_50m
# =============================================================================
cell8 = nb["cells"][8]
src8 = "".join(cell8["source"])

# Flip default
old_scale = "TARGET_SCALE = '742m'  # one of '50m', '150m', '350m', '742m', '1b'"
new_scale = "TARGET_SCALE = '50m'   # one of '20m', '50m', '150m', '350m', '742m', '1b'"
assert old_scale in src8, "TARGET_SCALE line not found"
src8 = src8.replace(old_scale, new_scale, 1)

# Import the new presets
old_import = "from fant3.config       import FANT3Config, fant3_smoke, fant3_742m"
new_import = "from fant3.config       import FANT3Config, fant3_smoke, fant3_20m, fant3_50m, fant3_742m"
# Look in cell 6 for this import (it's the Drive-mount cell)
cell6 = nb["cells"][6]
src6 = "".join(cell6["source"])
if old_import in src6:
    src6 = src6.replace(old_import, new_import, 1)
    cell6["source"] = src6.splitlines(keepends=True)
    print("Cell 6: import updated with fant3_20m, fant3_50m")

# Replace inline cfg_50m definition (was ~30M) with call to fant3_50m() preset
# And add cfg_20m using fant3_20m() preset.
old_50m = """def cfg_50m():
    return FANT3Config(
        dim=384, n_layers=8, n_dense_layers=1,
        n_heads=6, n_kv_heads=2, head_dim=64,
        n_megapools=2, n_per_megapool=4, top_k=2,
        n_matryoshka_levels=2,
        shared_expert_hidden=256, moe_hidden=384,
        n_attention_atoms=3, masa_coef_rank=6,
        n_recursion_depths=2,
        kron_A_p=12, kron_A_q=6, kron_B_p=12, kron_B_q=12, kron_C_p=12, kron_C_q=16,
        max_seq_len=1024,
        n_hub_tokens=8,
        cerebellum_enabled=False,
        apollonian_alpha_cap=1000, apollonian_beta_cap=1000,
        apollonian_retrieval_layers=(6, 7),
        etf_freeze_after_step=500,
        etf_freeze_layers=(2, 3, 4, 5),
        spinor_apollonian_enabled=True,
        ahn_enabled=True,
        ahn_n_heads=2, ahn_short_window=32, ahn_long_capacity=64,
    )"""

new_50m = """def cfg_20m():
    # Uses fant3_20m() preset — ~23.5M stored, chat-focused
    return fant3_20m()

def cfg_50m():
    # Uses fant3_50m() preset — 50.79M stored, chat-optimized (12h A100 96GB)
    return fant3_50m()"""

assert old_50m in src8, "old cfg_50m block not found"
src8 = src8.replace(old_50m, new_50m, 1)

# Add '20m' to CONFIG_BUILDERS dict
old_builders = """CONFIG_BUILDERS = {
    '50m': cfg_50m, '150m': cfg_150m, '350m': cfg_350m,
    '742m': cfg_742m, '1b': cfg_1b,
}"""
new_builders = """CONFIG_BUILDERS = {
    '20m': cfg_20m, '50m': cfg_50m, '150m': cfg_150m, '350m': cfg_350m,
    '742m': cfg_742m, '1b': cfg_1b,
}"""
assert old_builders in src8, "CONFIG_BUILDERS dict not found"
src8 = src8.replace(old_builders, new_builders, 1)

cell8["source"] = src8.splitlines(keepends=True)
print("Cell 8: TARGET_SCALE='50m' default; cfg_20m and cfg_50m use presets; builder dict updated")

# =============================================================================
# Cell 12 — MIX v4 chat-heavy, scale-aware
# =============================================================================
cell12 = nb["cells"][12]
src12 = "".join(cell12["source"])

old_mix = """# Mix v3 (2026-04-19) — NVIDIA-centric."""
# Can't easily single-line find; use larger anchor.
old_mix_block_start = "# Mix v3 (2026-04-19) — NVIDIA-centric."
old_mix_block_end = "# Total weights: 1.00. Safety data (Aegis-2.0) deferred to Phase 6 (RLHF/alignment)."
assert old_mix_block_start in src12 and old_mix_block_end in src12, "MIX v3 block anchors not found"

# Build new scale-aware MIX block
new_mix_block = """# Mix v4 scale-aware (2026-04-19).
#   * For 20m/50m: CHAT-HEAVY (Sonnet 22%, Cascade-2 chat/IF, FineTome, Daring-Anteater)
#     — goal is fluent short-form chat like Qwen-2B, not reasoning benchmarks
#   * For 150m/350m/742m/1b: NVIDIA-heavy (MIX v3 preserved for reasoning-focused runs)
# All sources CC-BY-4.0 or permissive; decontaminated by 13-gram filter in cell 14.
#
# Columns:  (hf_id,  config,  split,  text_key,  format,  weight,  input_key)
MIX_V4_CHAT = [
    ('Roman1111111/claude-sonnet-4.6-120000x',            None,                    'train',  'messages',      DatasetFormat.MESSAGES,               0.22, None),
    ('crownelius/Opus-4.6-Reasoning-3300x',               None,                    'train',  'problem',       DatasetFormat.PROBLEM_THINK_SOLUTION, 0.14, None),
    ('ianncity/KIMI-K2.5-1000000x',                       'General-Distillation',  'train',  'messages',      DatasetFormat.MESSAGES,               0.12, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'chat',                  'train',  'messages',      DatasetFormat.MESSAGES,               0.10, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'instruction_following', 'train',  'messages',      DatasetFormat.MESSAGES,               0.06, None),
    ('mlabonne/FineTome-100k',                            None,                    'train',  'conversations', DatasetFormat.CONVERSATIONS,          0.08, None),
    ('nvidia/Daring-Anteater',                            None,                    'train',  'conversations', DatasetFormat.CONVERSATIONS,          0.05, None),
    ('HuggingFaceFW/fineweb-edu',                         'default',               'train',  'text',          DatasetFormat.FLAT_TEXT,              0.10, None),
    ('nvidia/OpenMathReasoning',                          None,                    'cot',    'problem',       DatasetFormat.PROBLEM_SOLUTION,       0.05, None),
    ('nvidia/OpenMathInstruct-2',                         None,                    'train',  'problem',       DatasetFormat.PROBLEM_SOLUTION,       0.04, None),
    ('nvidia/OpenCodeReasoning-2',                        None,                    'python', 'question',      DatasetFormat.PROBLEM_SOLUTION,       0.02, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'math',                  'train',  'messages',      DatasetFormat.MESSAGES,               0.02, None),
]
# Total 1.00 — Sonnet 4.6 at 22% is the dominant quality signal.

MIX_V3_NVIDIA = [
    ('nvidia/OpenMathReasoning',                          None,                    'cot',            'problem',       DatasetFormat.PROBLEM_SOLUTION,       0.18, None),
    ('nvidia/OpenCodeReasoning-2',                        None,                    'python',         'question',      DatasetFormat.PROBLEM_SOLUTION,       0.08, None),
    ('nvidia/OpenMathInstruct-2',                         None,                    'train',          'problem',       DatasetFormat.PROBLEM_SOLUTION,       0.08, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'math',                  'train',          'messages',      DatasetFormat.MESSAGES,               0.08, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'chat',                  'train',          'messages',      DatasetFormat.MESSAGES,               0.06, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'instruction_following', 'train',          'messages',      DatasetFormat.MESSAGES,               0.04, None),
    ('nvidia/Nemotron-Cascade-2-SFT-Data',                'science',               'train',          'messages',      DatasetFormat.MESSAGES,               0.04, None),
    ('nvidia/Daring-Anteater',                            None,                    'train',          'conversations', DatasetFormat.CONVERSATIONS,          0.04, None),
    ('HuggingFaceFW/fineweb-edu',                         'default',               'train',          'text',          DatasetFormat.FLAT_TEXT,              0.20, None),
    ('Roman1111111/claude-sonnet-4.6-120000x',            None,                    'train',          'messages',      DatasetFormat.MESSAGES,               0.12, None),
    ('crownelius/Opus-4.6-Reasoning-3300x',               None,                    'train',          'problem',       DatasetFormat.PROBLEM_THINK_SOLUTION, 0.08, None),
]

# Pick per-scale
if TARGET_SCALE in ('20m', '50m'):
    MIX = MIX_V4_CHAT
    print('Using MIX_V4_CHAT (Sonnet-heavy, chat-focused; for small-scale chat training)')
else:
    MIX = MIX_V3_NVIDIA
    print('Using MIX_V3_NVIDIA (NVIDIA reasoning-heavy; for larger-scale pretraining)')"""

# Replace from old block start to old block end
start_idx = src12.find(old_mix_block_start)
end_idx = src12.find(old_mix_block_end) + len(old_mix_block_end)
src12 = src12[:start_idx] + new_mix_block + src12[end_idx:]
cell12["source"] = src12.splitlines(keepends=True)
print("Cell 12: MIX v4 chat-heavy installed for 20m/50m; MIX v3 retained for bigger scales")

# =============================================================================
# Cell 20 — add '20m' and '50m' recipe branches; update existing branches slightly
# =============================================================================
cell20 = nb["cells"][20]
src20 = "".join(cell20["source"])

# Find the existing '50m'/'150m' branch and replace
old_branch = """if TARGET_SCALE in ('50m', '150m'):
    BATCH_SIZE        = 2       # A100 80 GB: comfortable
    GRAD_ACCUM_STEPS  = 4       # effective batch = 8
    SEQ_LEN           = 512
    TOTAL_STEPS       = 2500
    WARMUP_STEPS      = 500     # 20% of total
    peak_lr_setting   = 2.0e-4"""

new_branch = """if TARGET_SCALE == '20m':
    # 20M chat-optimized, 12h budget on A100 96GB.
    # ~23.5M stored. Dense-ish (small MoE), no gc needed.
    BATCH_SIZE        = 16
    GRAD_ACCUM_STEPS  = 2       # effective batch = 32
    SEQ_LEN           = 1024
    TOTAL_STEPS       = 70000   # 12h budget at ~0.6 s/step
    WARMUP_STEPS      = 7000    # 10% of total
    peak_lr_setting   = 5.0e-4  # small models tolerate higher LR
elif TARGET_SCALE == '50m':
    # 50M chat-optimized, 12h budget on A100 96GB.
    # ~51M stored. Heavy over-training (~40x Chinchilla) to approach Qwen-2B chat quality.
    BATCH_SIZE        = 16
    GRAD_ACCUM_STEPS  = 2       # effective batch = 32
    SEQ_LEN           = 1024
    TOTAL_STEPS       = 60000   # 12h budget at ~0.7 s/step
    WARMUP_STEPS      = 6000    # 10% of total
    peak_lr_setting   = 4.0e-4
elif TARGET_SCALE == '150m':
    BATCH_SIZE        = 2
    GRAD_ACCUM_STEPS  = 4
    SEQ_LEN           = 512
    TOTAL_STEPS       = 2500
    WARMUP_STEPS      = 500
    peak_lr_setting   = 2.0e-4"""

assert old_branch in src20, "old 50m/150m branch not found"
src20 = src20.replace(old_branch, new_branch, 1)

cell20["source"] = src20.splitlines(keepends=True)
print("Cell 20: added '20m' and '50m' recipe branches (12h A100 96GB chat-optimized)")

with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)
print(f"notebook saved: {NB}")
