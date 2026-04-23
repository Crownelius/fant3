"""One-shot patcher: add per-phase SEQ_LEN + pack_mode to fant3_1b_nvidia_train.ipynb."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fant3_1b_nvidia_train.ipynb"

SAMPLER_SRC = r'''def make_batch_sampler(stream, batch_size, seq_len, pad_id, eos_id,
                       contamination_filter=True, pack_mode='concat'):
    """Yields (ids, targets) tensors of shape (B, seq_len).

    pack_mode='concat'  (Phase A, pretraining):
        Concatenate documents with <|eos|> separators until the buffer reaches
        seq_len. Standard LM pretraining pattern; efficient but semantically
        mixes unrelated docs.

    pack_mode='per_row' (Phase B, SFT):
        One document per sample. Longer -> truncate to first seq_len-1 + <|eos|>.
        Shorter -> pad with pad_id (becomes -100 in targets so CE ignores it).
        Correct for chat/SFT where cross-doc packing would teach the model to
        predict the start of one conversation from the end of another.

    Both modes share the same decontamination filter.
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
                    if ids:
                        tokens = ids[:seq_len - 1] + [eos_id]
                row = tokens + [pad_id] * (seq_len - len(tokens))
            batch_ids[b] = torch.tensor(row, dtype=torch.long)
        targets = batch_ids.clone()
        targets[targets == pad_id] = -100  # CE ignores pads
        yield batch_ids, targets'''

MEASURE_SRC = r'''# Cell 4.5 - Measure per-phase token-length distribution
# Samples 50 docs from each phase's stream and reports token-length percentiles
# so you can pick SEQ_LEN_A / SEQ_LEN_B from real data rather than a guess.
def _measure_stream(stream, n=50, tag=''):
    import numpy as np
    lengths = []
    it = iter(stream)
    for _ in range(n):
        try: text = next(it)
        except StopIteration: break
        if not text: continue
        lengths.append(len(tok.encode(text).ids))
    if not lengths:
        print(f'{tag}: no samples collected'); return None
    a = np.array(lengths)
    print(f'{tag}: n={len(a)}  P50={int(np.percentile(a,50))}  '
          f'P95={int(np.percentile(a,95))}  max={int(a.max())}  mean={a.mean():.0f}')
    return a

stats_A = _measure_stream(InterleavedMultiDatasetStream(PHASE_A_DATASETS, weights=PHASE_A_WEIGHTS, seed=7), n=50, tag='Phase A')
stats_B = _measure_stream(InterleavedMultiDatasetStream(PHASE_B_DATASETS, weights=PHASE_B_WEIGHTS, seed=7), n=50, tag='Phase B')

import numpy as np
def _suggest(stats, cap, round_to=128):
    if stats is None: return cap
    p95 = int(np.percentile(stats, 95))
    return min(cap, max(256, ((p95 + round_to - 1) // round_to) * round_to))

SEQ_LEN_A_SUGGEST = _suggest(stats_A, cap=cfg.max_seq_len)   # typically 1024 (FineWeb is long)
SEQ_LEN_B_SUGGEST = _suggest(stats_B, cap=cfg.max_seq_len)   # typically 256-512 (SFT is short)
print(f"\nsuggested SEQ_LEN_A = {SEQ_LEN_A_SUGGEST}   (concat-pack; pretraining)")
print(f"suggested SEQ_LEN_B = {SEQ_LEN_B_SUGGEST}   (per-row-pad; SFT)")'''

RECIPE_SRC = r'''# Recipe knobs - Tier D for 1B, calibrated for A100 80 GB
# SEQ_LEN_A/B default to the measurement-suggested values; override manually below if needed.
BATCH_SIZE        = 2
GRAD_ACCUM_STEPS  = 4           # effective batch = 8
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
print(f'seq_len: Phase A = {SEQ_LEN_A} (concat)   Phase B = {SEQ_LEN_B} (per_row)')'''

TRAIN_SRC = r'''# Cell 6.3 - training loop (phase-aware seq_len + pack_mode)
import gc as _gc
sampler_A = make_batch_sampler(stream_A, BATCH_SIZE, SEQ_LEN_A, PAD_ID, EOS_ID, pack_mode='concat')
sampler_B = make_batch_sampler(stream_B, BATCH_SIZE, SEQ_LEN_B, PAD_ID, EOS_ID, pack_mode='per_row')
fisher_state = {}
loss_hist = []

def _ckpt(path, model, optim, step, extra=None):
    payload = {'model': model.state_dict(), 'optim': optim.state_dict(),
               'step': step, 'cfg': cfg.__dict__,
               'extra': extra or {}}
    torch.save(payload, path)
    print(f'  [ckpt] step={step} -> {path}  size={os.path.getsize(path)/1e9:.2f} GB')

start = time.time()
if DEVICE == 'cuda':
    torch.cuda.reset_peak_memory_stats()
for step in range(1, TOTAL_STEPS + 1):
    sampler = sampler_A if step <= PHASE_A_STEPS else sampler_B
    phase_tag = 'A' if step <= PHASE_A_STEPS else 'B'

    cur_lr = lr_at(step)
    for g in optim.param_groups: g['lr'] = cur_lr

    model.train(); step_loss = 0.0
    optim.zero_grad(set_to_none=True)
    out = None
    router_info_snapshot = None
    for micro in range(GRAD_ACCUM_STEPS):
        ids, targets = next(sampler)
        ids, targets = ids.to(DEVICE), targets.to(DEVICE)
        store_now = (step % STORE_EVERY == 0) and (micro == 0)
        out = model(ids, targets=targets, store_to_memory=store_now)
        loss = out['loss'] / GRAD_ACCUM_STEPS
        if torch.isfinite(loss):
            loss.backward()
            step_loss += float(loss) * GRAD_ACCUM_STEPS
        ri = out.get('router_infos') or []
        if ri and 'mp_replicon' in ri[0]:
            router_info_snapshot = float(ri[0]['mp_replicon'])
        del ids, targets, loss

    if FISHER_PRECOND:
        precondition_router_grads_(model, fisher_state)
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
    optim.step()
    loss_hist.append(step_loss / max(GRAD_ACCUM_STEPS, 1))

    if step % LOG_EVERY == 0 or step in (1, PHASE_A_STEPS + 1):
        elapsed = time.time() - start
        vram   = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == 'cuda' else 0.0
        mstats = model.memory.get_stats() if hasattr(model, 'memory') else {}
        cur_sl = SEQ_LEN_A if phase_tag == 'A' else SEQ_LEN_B
        print(f'[{phase_tag} T={cur_sl}] step={step:5d} lr={cur_lr:.2e} loss={loss_hist[-1]:.4f} '
              f'vram={vram:.1f}GB chirality={mstats.get("chirality_balance", 0.0):.3f} '
              f'replicon={router_info_snapshot if router_info_snapshot is not None else 0.0:+.2f} '
              f'elapsed={elapsed/60:.1f}m')
        if DEVICE == 'cuda':
            torch.cuda.reset_peak_memory_stats()

    if step % CKPT_EVERY == 0 or step == PHASE_A_STEPS or step == TOTAL_STEPS:
        _ckpt(os.path.join(CKPT_DIR, f'step_{step:05d}.pt'),
              model, optim, step,
              extra={'loss_hist': loss_hist[-CKPT_EVERY:], 'phase': phase_tag})
        _gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

print(f'training complete in {(time.time()-start)/3600:.2f} h')'''


def _set_source(cell, text: str) -> None:
    lines = text.rstrip("\n").split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    patched = []
    insert_after = None
    already_have_measure = False
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "def make_batch_sampler" in src:
            _set_source(c, SAMPLER_SRC); patched.append(f"sampler (cell {i})")
        elif "Recipe knobs" in src and "BATCH_SIZE" in src:
            _set_source(c, RECIPE_SRC); patched.append(f"recipe (cell {i})")
        elif "Cell 6.3" in src:
            _set_source(c, TRAIN_SRC); patched.append(f"training loop (cell {i})")
        elif "PHASE_B_DATASETS" in src:
            insert_after = i
        if "Measure per-phase token-length distribution" in src:
            already_have_measure = True
    if insert_after is None:
        raise SystemExit("Phase B dataset cell not found")
    if not already_have_measure:
        md = {"cell_type": "markdown", "metadata": {}, "source": [
            "## 4.5 \u00b7 Measure per-phase token-length distribution\n",
            "\n",
            "Samples 50 docs from each phase's stream. Run this before the recipe-knobs cell so `SEQ_LEN_A` / `SEQ_LEN_B` come from real data, not a hardcoded guess.",
        ]}
        code = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []}
        _set_source(code, MEASURE_SRC)
        nb["cells"].insert(insert_after + 1, code)
        nb["cells"].insert(insert_after + 1, md)
        patched.append(f"inserted measurement at {insert_after + 1}")
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("patched:")
    for p in patched:
        print("  -", p)


if __name__ == "__main__":
    main()
