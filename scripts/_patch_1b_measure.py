"""Replace cell 4.5 (measurement) with a per-dataset version in fant3_1b_nvidia_train.ipynb."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fant3_1b_nvidia_train.ipynb"

NEW_SRC = r'''# Cell 4.5 - Measure per-dataset token-length distribution
# Fresh HuggingFaceStream per registry entry so (a) one failing dataset doesn't
# crash the phase, and (b) a low-weight dataset still gets enough samples to
# report a real distribution. Reports per-dataset P50/P75/P95/P99/max, then
# a weighted phase P95 computed only from the datasets that succeeded.
from fant2.data.streaming import HuggingFaceStream
from fant2.data.registry import ALL_DATASETS
import numpy as np

def _per_dataset_lengths(ds_key, n=20):
    """Sample n docs from one registered dataset. Returns [] on any failure."""
    if ds_key not in ALL_DATASETS:
        print(f'    {ds_key:32s} NOT IN REGISTRY (skipping)')
        return []
    e = ALL_DATASETS[ds_key]
    try:
        s = HuggingFaceStream(
            dataset_name=e.hf_id, dataset_config=e.config, split=e.split,
            text_key=e.text_key, format_type=e.format,
            input_key=e.input_key, output_key=e.output_key,
        )
        lens = []
        for i, text in enumerate(s):
            if i >= n: break
            if not text: continue
            lens.append(len(tok.encode(text).ids))
        return lens
    except Exception as ex:
        print(f'    {ds_key:32s} LOAD FAILED ({type(ex).__name__}: {str(ex)[:80]})')
        return []

def _measure_phase(names, weights, tag, per_ds_n=20):
    print(f'=== {tag} ===')
    successful = []     # list of (weight, p95, lengths[]) only for datasets that produced samples
    for name, w in zip(names, weights):
        lens = _per_dataset_lengths(name, n=per_ds_n)
        if not lens:
            continue
        a = np.array(lens)
        p50, p75, p95, p99, mx = (int(np.percentile(a, q)) for q in (50, 75, 95, 99, 100))
        print(f'    {name:32s} w={w:.2f} n={len(a):3d}  P50={p50:5d}  P75={p75:5d}  P95={p95:5d}  P99={p99:5d}  max={mx:5d}')
        successful.append((w, p95, lens))

    if not successful:
        print(f'    {tag}: no successful samples')
        return None, None

    # Flat pool P95 - ignores weights, useful as a tail estimate.
    flat_pool = np.concatenate([np.array(l) for (_, _, l) in successful])
    flat_p95 = int(np.percentile(flat_pool, 95))

    # Weighted phase P95 - divide only by the successful weight mass.
    wsum = sum(w for (w, _, _) in successful)
    weighted_p95 = sum(w * p95 for (w, p95, _) in successful) / wsum
    print(f'    {tag}: flat_P95={flat_p95}  weighted_P95={weighted_p95:.0f}  n_total={len(flat_pool)}  weight_covered={wsum:.2f}')
    return flat_p95, weighted_p95

flat_A, weighted_A = _measure_phase(PHASE_A_DATASETS, PHASE_A_WEIGHTS, tag='Phase A', per_ds_n=20)
flat_B, weighted_B = _measure_phase(PHASE_B_DATASETS, PHASE_B_WEIGHTS, tag='Phase B', per_ds_n=20)

def _suggest(flat_p95, weighted_p95, cap, round_to=128):
    if flat_p95 is None:
        return cap
    # Use the larger (flat = tail estimate, weighted = typical sample) so we
    # don't clip important tokens in the bulk of batches.
    raw = max(int(flat_p95), int(weighted_p95 or 0))
    return min(cap, max(256, ((raw + round_to - 1) // round_to) * round_to))

SEQ_LEN_A_SUGGEST = _suggest(flat_A, weighted_A, cap=cfg.max_seq_len)
SEQ_LEN_B_SUGGEST = _suggest(flat_B, weighted_B, cap=cfg.max_seq_len)
print(f"\nsuggested SEQ_LEN_A = {SEQ_LEN_A_SUGGEST}   (concat-pack; pretraining)")
print(f"suggested SEQ_LEN_B = {SEQ_LEN_B_SUGGEST}   (per-row-pad; SFT)")
print("Override in the next cell (SEQ_LEN_A = ...) if you want something different.")'''


def _set_source(cell, text: str) -> None:
    lines = text.rstrip("\n").split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    hits = 0
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "# Cell 4.5" in src and "Measure per" in src:
            _set_source(c, NEW_SRC)
            hits += 1
            print(f"replaced cell {i}")
    if hits != 1:
        raise SystemExit(f"expected 1 match for cell 4.5, found {hits}")
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("notebook patched")


if __name__ == "__main__":
    main()
