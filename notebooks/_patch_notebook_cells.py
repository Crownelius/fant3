"""One-shot script to insert decontamination + benchmark-eval cells into
fant3_colab_train.ipynb. Idempotent — re-run to refresh the inserted cells."""
import json
from pathlib import Path

NB = Path(__file__).parent / "fant3_colab_train.ipynb"
MARK_DECON = "# [FANT DECONTAMINATION CELL v1]"
MARK_EVAL  = "# [FANT BENCHMARK EVAL CELL v1]"

DECON_MD = (
    "## 7. Decontamination check\n\n"
    "Runs `scripts/decontaminate.py` to flag any training docs that contain a 13-gram "
    "matching any test-set question of **GSM8K + MATH-500 + MMLU**. Local testing on "
    "the 6-source mix (n=1000 per source) shows NuminaMath-CoT 0.40%, Kimi/FineTome/"
    "Superior ~0.10%, FineWeb/Opus 0%. Install the filter into the data pipeline so "
    "contaminated docs never reach the model.\n"
)

DECON_CODE = f"""{MARK_DECON}
# Build (or load from disk) the benchmark n-gram signature cache. First run
# downloads the benchmark test sets (~50 MB total) and caches them to
# output/decontamination/ngram_hashes.json — subsequent runs are instant.
from scripts.decontaminate import build_hash_cache, is_contaminated, _load_global
_ = build_hash_cache(rebuild=False)
_ = _load_global()   # warms the global set

# Wrap the sample_batch() generator with a decontamination filter: any text that
# contains a 13-gram hash-matching any benchmark test question is dropped.
_original_sample_batch = sample_batch

def sample_batch(iters, weights, batch_size, seq_len, tok, rng):
    chunks = []
    attempts = 0
    rejected = 0
    while len(chunks) < batch_size and attempts < batch_size * 80:
        attempts += 1
        src_idx = rng.choices(range(len(iters)), weights=weights, k=1)[0]
        it, kwargs = iters[src_idx]
        try:
            ex = next(it)
        except StopIteration:
            continue
        try:
            text = extract_text(ex, **kwargs)
        except Exception:
            continue
        if not text or len(text) < 16:
            continue
        if is_contaminated(text):
            rejected += 1
            continue
        ids = tok.encode(text)
        if len(ids) < 8:
            continue
        need = seq_len + 1
        if len(ids) >= need:
            start = rng.randint(0, len(ids) - need)
            ids = ids[start : start + need]
        else:
            ids = ids + [tok._tok.token_to_id('<|pad|>') if '<|pad|>' in tok._tok.get_vocab() else 0] * (need - len(ids))
        chunks.append(ids)
    if len(chunks) < batch_size:
        raise RuntimeError('data pipeline drained — check dataset IDs / HF cache')
    arr = torch.tensor(chunks, dtype=torch.long)
    return arr[:, :-1].contiguous(), arr[:, 1:].contiguous()

print(f'Decontamination filter active. Signatures loaded for GSM8K + MATH-500 + MMLU.')
"""

EVAL_MD = (
    "## 12. Benchmark eval\n\n"
    "After training, evaluate on the three benchmarks our training mix has been "
    "decontaminated against. Run on the smallest eval size that gives a meaningful "
    "signal; scale up if the accuracy looks promising.\n"
)

EVAL_CODE = f"""{MARK_EVAL}
# Run GSM8K + MMLU eval. Expect sub-chance from a 50m/150m run of 2500 steps —
# the point is to get a DIRECTION and a baseline to improve on with scale.
import subprocess, json, os

# Save a checkpoint path the eval script understands
eval_ckpt = os.path.join(scale_ckpt_dir, 'final.pt')
if not os.path.exists(eval_ckpt):
    # pick the latest step_*.pt
    ckpts = sorted([p for p in os.listdir(scale_ckpt_dir) if p.startswith('step_')],
                   key=lambda p: int(p.split('_')[1].split('.')[0]))
    eval_ckpt = os.path.join(scale_ckpt_dir, ckpts[-1]) if ckpts else None

if eval_ckpt is None:
    print('No checkpoint yet — run training first')
else:
    print(f'Evaluating {{eval_ckpt}}')
    for bench, n in [('gsm8k', 50), ('mmlu', 200)]:
        print(f'\\n--- {{bench}} ({{n}} problems) ---')
        result = !python {{WORK_DIR}}/scripts/eval_benchmarks.py \\
            --ckpt {{eval_ckpt}} \\
            --tokenizer {{TOKENIZER_PATH}} \\
            --benchmark {{bench}} \\
            --n {{n}}
        print('\\n'.join(result[-10:]))
"""


def find_cell_idx(nb, marker: str) -> int | None:
    for i, c in enumerate(nb["cells"]):
        if marker in "".join(c.get("source", [])):
            return i
    return None


def find_cell_by_prefix(nb, prefix: str) -> int | None:
    for i, c in enumerate(nb["cells"]):
        if "".join(c.get("source", [])).lstrip().startswith(prefix):
            return i
    return None


def as_cell_src(s: str) -> list[str]:
    return s.splitlines(keepends=True)


def md_cell(s: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": as_cell_src(s)}


def code_cell(s: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": as_cell_src(s)}


def ensure_cells(nb):
    # --- Decontamination: place AFTER "data pipeline" markdown/code pair ---
    idx = find_cell_idx(nb, MARK_DECON)
    if idx is None:
        # Insert after the first cell whose code contains 'build_iterators'
        anchor = None
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code" and "build_iterators" in "".join(c.get("source", [])):
                anchor = i
                break
        if anchor is None:
            raise RuntimeError("could not find data-pipeline cell (missing build_iterators)")
        nb["cells"][anchor + 1 : anchor + 1] = [md_cell(DECON_MD), code_cell(DECON_CODE)]
        print(f"Inserted decon cells at index {anchor + 1} and {anchor + 2}")
    else:
        # refresh existing code cell
        nb["cells"][idx] = code_cell(DECON_CODE)
        # ensure preceding markdown is our version
        if idx > 0 and nb["cells"][idx - 1]["cell_type"] == "markdown":
            nb["cells"][idx - 1] = md_cell(DECON_MD)
        print(f"Refreshed decon cell at index {idx}")

    # --- Benchmark eval: place AFTER the quality-probe cell ---
    idx = find_cell_idx(nb, MARK_EVAL)
    if idx is None:
        anchor = None
        for i, c in enumerate(nb["cells"]):
            if c["cell_type"] == "code" and "probes = [" in "".join(c.get("source", [])):
                anchor = i
                break
        if anchor is None:
            raise RuntimeError("could not find quality-probe cell")
        nb["cells"][anchor + 1 : anchor + 1] = [md_cell(EVAL_MD), code_cell(EVAL_CODE)]
        print(f"Inserted eval cells at index {anchor + 1} and {anchor + 2}")
    else:
        nb["cells"][idx] = code_cell(EVAL_CODE)
        if idx > 0 and nb["cells"][idx - 1]["cell_type"] == "markdown":
            nb["cells"][idx - 1] = md_cell(EVAL_MD)
        print(f"Refreshed eval cell at index {idx}")


def main():
    with open(NB, "r", encoding="utf-8") as f:
        nb = json.load(f)
    ensure_cells(nb)
    with open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print(f"Wrote {NB}  (total cells: {len(nb['cells'])})")


if __name__ == "__main__":
    main()
