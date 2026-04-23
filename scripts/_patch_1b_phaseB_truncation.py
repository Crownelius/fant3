"""Fix cell 7 NameError, add truncation warning to cell 4.5, add max_row_tokens skip to sampler."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fant3_1b_nvidia_train.ipynb"

# ---------- Cell 7 rename _ckpt -> _save ----------
CELL7_SRC = r'''# Cell 7 - final ckpt under a stable name + loss plot
# _save is defined in cell 6.3; this cell runs AFTER 6.3 so it's in scope.
final_path = os.path.join(CKPT_DIR, 'final.pt')
_save(final_path, model, optim, TOTAL_STEPS, include_optim=True,
      extra={'phase_A_steps': PHASE_A_STEPS, 'phase_B_steps': PHASE_B_STEPS,
             'nan_steps_total': NAN_STEPS_TOTAL})
print('FINAL -> ', final_path)

# Loss plot with NaN-aware masking (NaN rows drawn as gaps)
try:
    import matplotlib.pyplot as plt
    import numpy as _np
    arr = _np.array([(x if x == x else _np.nan) for x in loss_hist], dtype=float)  # NaN preserved
    plt.figure(figsize=(9, 3.5))
    plt.plot(arr, lw=0.8, alpha=0.75)
    plt.axvline(PHASE_A_STEPS, ls='--', c='grey', label='phase A -> B handoff')
    plt.xlabel('step'); plt.ylabel('CE loss')
    plt.title(f'FANT 3 1B - NVIDIA-heavy pretrain + SFT (NaN steps = {NAN_STEPS_TOTAL})')
    plt.legend(); plt.tight_layout(); plt.show()
except Exception as e:
    print('plot skipped:', e)'''

# ---------- Cell 4.5 - add prominent truncation warning ----------
CELL45_SRC = r'''# Cell 4.5 - Measure per-dataset token-length distribution
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

def _measure_phase(names, weights, tag, per_ds_n=20, cap=None):
    print(f'=== {tag} ===')
    successful = []     # (weight, p50, p95, lengths) for each successful dataset
    truncation_warnings = []
    for name, w in zip(names, weights):
        lens = _per_dataset_lengths(name, n=per_ds_n)
        if not lens:
            continue
        a = np.array(lens)
        p50, p75, p95, p99, mx = (int(np.percentile(a, q)) for q in (50, 75, 95, 99, 100))
        flag = ''
        if cap is not None:
            if p50 > cap * 2:
                flag = '  !! P50 > 2x cap: majority of rows will be severely truncated'
                truncation_warnings.append((name, w, p50, p95, cap, 'severe'))
            elif p50 > cap:
                flag = '  !  P50 > cap: at least half of rows will be truncated'
                truncation_warnings.append((name, w, p50, p95, cap, 'partial'))
        print(f'    {name:32s} w={w:.2f} n={len(a):3d}  P50={p50:5d}  P75={p75:5d}  P95={p95:5d}  P99={p99:5d}  max={mx:5d}{flag}')
        successful.append((w, p50, p95, lens))
    if not successful:
        print(f'    {tag}: no successful samples'); return None, None, truncation_warnings
    flat_pool = np.concatenate([np.array(l) for (_, _, _, l) in successful])
    flat_p95 = int(np.percentile(flat_pool, 95))
    wsum = sum(w for (w, _, _, _) in successful)
    weighted_p95 = sum(w * p95 for (w, _, p95, _) in successful) / wsum
    print(f'    {tag}: flat_P95={flat_p95}  weighted_P95={weighted_p95:.0f}  n_total={len(flat_pool)}  weight_covered={wsum:.2f}')
    return flat_p95, weighted_p95, truncation_warnings

cap = cfg.max_seq_len
flat_A, weighted_A, warn_A = _measure_phase(PHASE_A_DATASETS, PHASE_A_WEIGHTS, tag='Phase A', per_ds_n=20, cap=cap)
flat_B, weighted_B, warn_B = _measure_phase(PHASE_B_DATASETS, PHASE_B_WEIGHTS, tag='Phase B', per_ds_n=20, cap=cap)

def _suggest(flat_p95, weighted_p95, cap, round_to=128):
    if flat_p95 is None:
        return cap
    raw = max(int(flat_p95), int(weighted_p95 or 0))
    return min(cap, max(256, ((raw + round_to - 1) // round_to) * round_to))

SEQ_LEN_A_SUGGEST = _suggest(flat_A, weighted_A, cap=cap)
SEQ_LEN_B_SUGGEST = _suggest(flat_B, weighted_B, cap=cap)
print(f"\nsuggested SEQ_LEN_A = {SEQ_LEN_A_SUGGEST}   (concat-pack; pretraining)")
print(f"suggested SEQ_LEN_B = {SEQ_LEN_B_SUGGEST}   (per-row-pad; SFT)")

# Prominent warning if any SFT dataset will lose >=half its rows. For SFT this
# is especially bad because answer tokens usually live at the END of a trace,
# so truncation destroys the training signal.
all_warnings = (warn_A or []) + (warn_B or [])
if all_warnings:
    print('\n' + '='*72)
    print('TRUNCATION WARNING - the following datasets will lose most of their')
    print(f'content at SEQ_LEN cap = {cap}:')
    for (name, w, p50, p95, c, sev) in all_warnings:
        pct = int(100 * c / max(p50, 1))
        print(f'  [{sev:>7}] {name:32s} w={w:.2f}  keeps ~{pct}% of P50 row ({c}/{p50} tokens)')
    print('Options:')
    print('  1. RAISE cfg.max_seq_len (requires rebuilding model + more VRAM at same B)')
    print('  2. REDUCE weight of the long-tail dataset below (or drop entirely)')
    print('  3. ACCEPT and set MAX_ROW_TOKENS below so long rows are skipped,')
    print('     not truncated - better signal, fewer samples per epoch')
    print('='*72)

print("\nOverride in the next cell (SEQ_LEN_A = ...) if you want something different.")'''

# ---------- Cell 5 sampler - add max_row_tokens skip option ----------
CELL5_SRC = r'''def make_batch_sampler(stream, batch_size, seq_len, pad_id, eos_id,
                       contamination_filter=True, pack_mode='concat',
                       max_row_tokens=None):
    """Yields (ids, targets) tensors of shape (B, seq_len).

    pack_mode='concat'  (Phase A, pretraining):
        Concatenate documents with <|eos|> separators until the buffer reaches
        seq_len. Standard LM pretraining pattern.

    pack_mode='per_row' (Phase B, SFT):
        One document per sample. Behavior for oversized docs is controlled by
        `max_row_tokens`:
            * None (default) - truncate to seq_len-1 + <|eos|>. Simple but
              loses answer tokens when the real doc is longer than seq_len.
            * int  - skip any doc whose encoded length is > max_row_tokens.
              Preserves training signal quality at the cost of discarding
              some rows. Useful for SFT with long-tail sources (Cascade-2
              math, chat) where truncation would drop the answer.

    Both modes share the decontamination filter.
    """
    assert pack_mode in ('concat', 'per_row')
    it = iter(stream)
    def _next_clean_text():
        while True:
            text = next(it)
            if not text: continue
            if contamination_filter and is_contaminated(text):
                continue
            return text
    while True:
        batch_ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
        for b in range(batch_size):
            tokens = []
            if pack_mode == 'concat':
                while len(tokens) < seq_len:
                    text = _next_clean_text()
                    ids = tok.encode(text).ids
                    if not ids: continue
                    tokens.extend(ids)
                    tokens.append(eos_id)
                row = tokens[:seq_len]
            else:  # per_row
                while not tokens:
                    text = _next_clean_text()
                    ids = tok.encode(text).ids
                    if not ids:
                        continue
                    if max_row_tokens is not None and len(ids) > max_row_tokens:
                        # Skip docs that would be severely truncated; preserves SFT signal.
                        continue
                    tokens = ids[:seq_len - 1] + [eos_id]
                row = tokens + [pad_id] * (seq_len - len(tokens))
            batch_ids[b] = torch.tensor(row, dtype=torch.long)
        targets = batch_ids.clone()
        targets[targets == pad_id] = -100  # CE ignores pads
        yield batch_ids, targets'''

# ---------- Cell 6.3 training loop - thread max_row_tokens through ----------
CELL63_DELTA = r'''sampler_A = make_batch_sampler(stream_A, BATCH_SIZE, SEQ_LEN_A, PAD_ID, EOS_ID, pack_mode='concat')'''
CELL63_NEW = r'''# MAX_ROW_TOKENS in per_row mode: set this to SEQ_LEN_B to drop docs longer
# than the seq_len (avoids "problem setup + no answer" truncation in SFT).
# Leave None to truncate instead (keeps more rows, worse signal).
MAX_ROW_TOKENS_B = SEQ_LEN_B   # change to None if you prefer truncate-over-skip
sampler_A = make_batch_sampler(stream_A, BATCH_SIZE, SEQ_LEN_A, PAD_ID, EOS_ID, pack_mode='concat')'''
CELL63_DELTA2 = r'''sampler_B = make_batch_sampler(stream_B, BATCH_SIZE, SEQ_LEN_B, PAD_ID, EOS_ID, pack_mode='per_row')'''
CELL63_NEW2 = r'''sampler_B = make_batch_sampler(stream_B, BATCH_SIZE, SEQ_LEN_B, PAD_ID, EOS_ID, pack_mode='per_row', max_row_tokens=MAX_ROW_TOKENS_B)'''


def _set_source(cell, text: str) -> None:
    lines = text.rstrip("\n").split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    patched = []
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "Cell 4.5" in src and "Measure per" in src:
            _set_source(c, CELL45_SRC); patched.append(f"4.5 (cell {i})")
        elif "def make_batch_sampler" in src:
            _set_source(c, CELL5_SRC); patched.append(f"sampler (cell {i})")
        elif "_ckpt(final_path" in src:
            _set_source(c, CELL7_SRC); patched.append(f"cell 7 final.pt rename (cell {i})")
        elif "Cell 6.3" in src:
            if CELL63_DELTA in src and CELL63_DELTA2 in src:
                src = src.replace(CELL63_DELTA, CELL63_NEW, 1)
                src = src.replace(CELL63_DELTA2, CELL63_NEW2, 1)
                lines = src.split("\n")
                c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
                patched.append(f"cell 6.3 max_row_tokens wiring (cell {i})")
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("patched:")
    for p in patched: print("  -", p)


if __name__ == "__main__":
    main()
