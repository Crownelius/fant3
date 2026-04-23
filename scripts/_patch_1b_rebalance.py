"""Drop broken superior-reasoning-s1 from Phase A. Rebalance Phase B to drop
the cascade2-sft-math dead dataset (P50=20k >> seq_len cap)."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fant3_1b_nvidia_train.ipynb"

CELL3_SRC = r'''PHASE_A_DATASETS = [
    'fineweb-edu',                 # 0.35  - FLAT_TEXT web corpus
    'nvidia-openmath-reasoning',   # 0.20  - PROBLEM_SOLUTION, cot split
    'nvidia-opencode-reasoning-2', # 0.10  - PROBLEM_SOLUTION, python split
    'nvidia-openmath-2',           # 0.10  - PROBLEM_SOLUTION, NuminaMath-style
    'opus46-crownelius-3300x',     # 0.15  - PROBLEM_THINK_SOLUTION, Opus-4.6 traces (bumped +0.05 from dropped superior-s1)
    'kimi-k25-distill',            # 0.10  - MESSAGES, Kimi teacher
    # DROPPED 2026-04-22: 'superior-reasoning-s1' - HF schema is broken for pyarrow
    #   LOAD FAILED (TypeError: Couldn't cast array of type struct<training_stage: string, sampling_temperature:))
    #   InterleavedMultiDatasetStream does not per-stream try/except, so including it
    #   would crash the training loop ~step 200 when the weighted selector picks it.
]
PHASE_A_WEIGHTS = [0.35, 0.20, 0.10, 0.10, 0.15, 0.10]
assert abs(sum(PHASE_A_WEIGHTS) - 1.0) < 1e-6
stream_A = InterleavedMultiDatasetStream(PHASE_A_DATASETS, weights=PHASE_A_WEIGHTS, seed=0)
print('Phase A sources:')
for n, w in zip(PHASE_A_DATASETS, PHASE_A_WEIGHTS): print(f'  {w:.2f}  {n}')'''

CELL4_SRC = r'''# Phase B rebalanced 2026-04-22: dropped cascade2-sft-math (P50=20103 at seq_len
# cap 1024 means >95% of rows get skipped; weight was pure dead mass).
# Also scaled down cascade2-sft-chat (P50=2217 -> ~60% rows skipped) and relied
# on sonnet46-120k (P50=809, fits cleanly) for the chat signal instead.
PHASE_B_DATASETS = [
    'nvidia-cascade2-sft-if',       # 0.25  - P50=766, ~80% rows kept
    'sonnet46-120k',                # 0.30  - P50=809, ~80% rows kept, chat-heavy
    'nvidia-openmath-2',            # 0.15  - P50=356, ~95% rows kept, math SFT
    'nvidia-cascade2-sft-science',  # 0.10  - P50=1552, ~50% rows kept
    'nvidia-daring-anteater',       # 0.10  - P50=1902, ~45% rows kept
    'nvidia-cascade2-sft-chat',     # 0.10  - P50=2217, ~40% rows kept (down from 0.20)
    # DROPPED: 'nvidia-cascade2-sft-math' (P50=20103, dead at cap=1024)
]
PHASE_B_WEIGHTS = [0.25, 0.30, 0.15, 0.10, 0.10, 0.10]
assert abs(sum(PHASE_B_WEIGHTS) - 1.0) < 1e-6
stream_B = InterleavedMultiDatasetStream(PHASE_B_DATASETS, weights=PHASE_B_WEIGHTS, seed=1)
print('Phase B sources:')
for n, w in zip(PHASE_B_DATASETS, PHASE_B_WEIGHTS): print(f'  {w:.2f}  {n}')'''


def _set(c, text):
    lines = text.rstrip("\n").split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    patched = []
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "PHASE_A_DATASETS" in src and "superior-reasoning-s1" in src:
            _set(c, CELL3_SRC); patched.append(f"cell 3 Phase A (cell {i})")
        elif "PHASE_B_DATASETS" in src and "nvidia-cascade2-sft-math" in src:
            _set(c, CELL4_SRC); patched.append(f"cell 4 Phase B (cell {i})")
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("patched:")
    for p in patched: print("  -", p)
    assert len(patched) == 2, f"expected 2 patches, got {len(patched)}"


if __name__ == "__main__":
    main()
