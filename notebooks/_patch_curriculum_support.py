"""Patch script — add curriculum preset support to fant3_1b_nvidia_train.ipynb.

Replaces the hardcoded PHASE_A_DATASETS / PHASE_B_DATASETS blocks with a
thin wrapper that defaults to the legacy 2-phase mix (bit-identical to the
pre-patch values) but lets the user swap in deepinsight_3phase or any
other fant3.training.curriculum preset by setting CURRICULUM_NAME at the
top of the phase-A cell.

Matches the established `_patch_*.py` pattern: load the notebook, splice
new source into the identified cells, write it back.

Run:
    python notebooks/_patch_curriculum_support.py

Idempotent — re-running detects the marker and no-ops.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

NOTEBOOK = Path(__file__).parent / "fant3_1b_nvidia_train.ipynb"
MARKER = "# CURRICULUM_PATCH_V1"


PHASE_A_CELL_NEW = [
    "# CURRICULUM_PATCH_V1\n",
    "# Curriculum preset — switch between mixes without editing dataset lists.\n",
    "# Available presets (see fant3/training/curriculum.py):\n",
    "#   'legacy_2phase'      — the original 2-phase mix (bit-identical default)\n",
    "#   'deepinsight_3phase' — Apprentice/Journeyman/Expert, arxiv:2604.16278\n",
    "#   'flat_1phase'        — no curriculum, single mix throughout (A/B control)\n",
    "CURRICULUM_NAME = 'legacy_2phase'\n",
    "\n",
    "from fant3.training import build_curriculum\n",
    "CURRICULUM = build_curriculum(CURRICULUM_NAME)\n",
    "print(f'Curriculum: {CURRICULUM.name} with {len(CURRICULUM.phases)} phase(s)')\n",
    "for i, _p in enumerate(CURRICULUM.phases):\n",
    "    print(f'  phase {i} {_p.name!r}: end_frac={_p.end_frac:.3f} datasets={list(_p.datasets)}')\n",
    "\n",
    "# Back-compat aliases — cells below still read PHASE_A_*/PHASE_B_*.\n",
    "# For non-2-phase curricula, PHASE_B_* mirrors the LAST phase and the\n",
    "# notebook's 2-sampler architecture only sees phase 0 (A) and last (B);\n",
    "# intermediate phases are ignored unless you use scripts/runpod_train.py.\n",
    "PHASE_A_DATASETS = list(CURRICULUM.phases[0].datasets)\n",
    "PHASE_A_WEIGHTS  = list(CURRICULUM.phases[0].weights)\n",
    "PHASE_B_DATASETS = list(CURRICULUM.phases[-1].datasets)\n",
    "PHASE_B_WEIGHTS  = list(CURRICULUM.phases[-1].weights)\n",
]

PHASE_B_CELL_NEW = [
    "# Phase B is derived from CURRICULUM.phases[-1] (see patch in cell above).\n",
    "# Kept as a separate cell so per-phase audit/plot cells still execute.\n",
    "print('PHASE_B_DATASETS:', PHASE_B_DATASETS)\n",
    "print('PHASE_B_WEIGHTS: ', PHASE_B_WEIGHTS)\n",
]


def patch():
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    # Find the PHASE_A_DATASETS and PHASE_B_DATASETS cells
    phase_a_idx = phase_b_idx = None
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
        if MARKER in src:
            print(f"Notebook already patched (marker found in cell {i}); no-op.")
            return
        if phase_a_idx is None and "PHASE_A_DATASETS = [" in src:
            phase_a_idx = i
        elif phase_b_idx is None and "PHASE_B_DATASETS" in src and "=" in src:
            phase_b_idx = i

    if phase_a_idx is None or phase_b_idx is None:
        print(f"FAIL: could not find phase cells (A={phase_a_idx}, B={phase_b_idx})")
        sys.exit(1)

    print(f"Patching cell {phase_a_idx} (phase A) and cell {phase_b_idx} (phase B)")
    nb["cells"][phase_a_idx]["source"] = PHASE_A_CELL_NEW
    nb["cells"][phase_a_idx]["outputs"] = []
    nb["cells"][phase_a_idx]["execution_count"] = None
    nb["cells"][phase_b_idx]["source"] = PHASE_B_CELL_NEW
    nb["cells"][phase_b_idx]["outputs"] = []
    nb["cells"][phase_b_idx]["execution_count"] = None

    NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {NOTEBOOK} ({NOTEBOOK.stat().st_size} bytes)")


if __name__ == "__main__":
    patch()
