"""Halve BATCH_SIZE, double GRAD_ACCUM_STEPS to escape Matryoshka MoE materialization OOM.
Same effective batch (8); peak MoE W_up_sel drops ~38 GB -> ~19 GB."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fant3_1b_nvidia_train.ipynb"

NEW_SRC = r'''# Recipe knobs - Tier D for 1B, calibrated for A100 80 GB
# 2026-04-22: BATCH_SIZE reduced 2 -> 1 (MoE expert W_up_sel materialization
# peaked ~38 GB at B=2 worst-case routing skew, OOM'd mid-training). Compensated
# with GRAD_ACCUM_STEPS 4 -> 8 so effective batch stays at 8.
BATCH_SIZE        = 1
GRAD_ACCUM_STEPS  = 8           # effective batch = 8 (unchanged)
SEQ_LEN_A         = SEQ_LEN_A_SUGGEST    # pretrain, concat-packing
SEQ_LEN_B         = SEQ_LEN_B_SUGGEST    # SFT, one-row-per-sample
PHASE_A_STEPS     = 8000
PHASE_B_STEPS     = 4000
TOTAL_STEPS       = PHASE_A_STEPS + PHASE_B_STEPS
WARMUP_STEPS      = 1800
PEAK_LR           = 1.2e-4
GRAD_CLIP         = 1.0
SCHEDULE_SHAPE    = 'litim'
LOG_EVERY         = 25
CKPT_EVERY        = 500
STORE_EVERY       = 50
FISHER_PRECOND    = True
print(f'total={TOTAL_STEPS}  warmup={WARMUP_STEPS}  peak_lr={PEAK_LR:.1e}  schedule={SCHEDULE_SHAPE}')
print(f'seq_len: Phase A = {SEQ_LEN_A} (concat)   Phase B = {SEQ_LEN_B} (per_row)')
print(f'batch: B={BATCH_SIZE} accum={GRAD_ACCUM_STEPS} effective={BATCH_SIZE*GRAD_ACCUM_STEPS}')'''

# Also add periodic empty_cache every LOG_EVERY in cell 6.3 (more aggressive than current post-ckpt only)
CELL63_FIND = r'''    if step % LOG_EVERY == 0 or step in (start_step + 1, PHASE_A_STEPS + 1):'''
CELL63_REPL = r'''    # Aggressive cache flush between steps to fight fragmentation at 1B scale.
    # Without this the MoE materialization peaks compound across steps.
    if DEVICE == 'cuda' and step % 4 == 0:
        torch.cuda.empty_cache()

    if step % LOG_EVERY == 0 or step in (start_step + 1, PHASE_A_STEPS + 1):'''


def _set(c, text):
    lines = text.rstrip("\n").split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    patched = []
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "Recipe knobs" in src and "BATCH_SIZE" in src:
            _set(c, NEW_SRC); patched.append(f"cell 6.1 recipe -> B=1 accum=8 (cell {i})")
        elif "Cell 6.3" in src and CELL63_FIND in src and "empty_cache() between steps" not in src:
            new_src = src.replace(CELL63_FIND, CELL63_REPL, 1)
            lines = new_src.split("\n")
            c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
            patched.append(f"cell 6.3 added empty_cache every 4 steps (cell {i})")
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("patched:")
    for p in patched: print("  -", p)


if __name__ == "__main__":
    main()
