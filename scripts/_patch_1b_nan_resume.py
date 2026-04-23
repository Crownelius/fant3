"""Cell 6.3: weights_only=False on torch.load + NaN guard + early stop on consecutive NaN."""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fant3_1b_nvidia_train.ipynb"

NEW_SRC = r'''# Cell 6.3 - training loop (two-tier ckpt + resume + NaN guard + phase-aware seq_len)
import gc as _gc
import glob as _glob
import math as _math

sampler_A = make_batch_sampler(stream_A, BATCH_SIZE, SEQ_LEN_A, PAD_ID, EOS_ID, pack_mode='concat')
sampler_B = make_batch_sampler(stream_B, BATCH_SIZE, SEQ_LEN_B, PAD_ID, EOS_ID, pack_mode='per_row')
fisher_state = {}
loss_hist = []

# -------------------- checkpoint destinations --------------------
LOCAL_CKPT_DIR  = '/content/ckpts_local' if IN_COLAB else os.path.join(CKPT_DIR, '_local')
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
ROLLING_KEEP    = 3
DRIVE_MILESTONE_EVERY = 2000
RESUME_FROM     = None   # set to a .pt path to resume; None = fresh start

# -------------------- NaN guard --------------------
MAX_CONSECUTIVE_NAN_STEPS = 3   # break after this many in a row
NAN_STEPS_TOTAL           = 0
CONSECUTIVE_NAN           = 0

def _save(path, model, optim, step, include_optim=True, extra=None):
    payload = {'model': model.state_dict(), 'step': step, 'cfg': cfg.__dict__,
               'extra': extra or {}}
    if include_optim:
        payload['optim'] = optim.state_dict()
    torch.save(payload, path)
    sz = os.path.getsize(path) / 1e9
    print(f'  [ckpt] step={step} -> {path}  size={sz:.2f} GB  {"(+optim)" if include_optim else "(weights only)"}')

def _rolling_trim(local_dir, keep=ROLLING_KEEP):
    files = sorted(_glob.glob(os.path.join(local_dir, 'step_*.pt')), key=os.path.getmtime)
    for old in files[:-keep]:
        try:
            os.remove(old)
            print(f'  [rolling] removed {os.path.basename(old)}')
        except OSError:
            pass

# -------------------- optional resume --------------------
start_step = 0
if RESUME_FROM is not None and os.path.exists(RESUME_FROM):
    print(f'resuming from {RESUME_FROM}')
    # weights_only=False required on torch >= 2.4: the payload has cfg.__dict__
    # and a bitsandbytes optim state_dict, neither of which are in the safe
    # allowlist. We trust this file because we wrote it ourselves.
    state = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state['model'])
    if 'optim' in state:
        optim.load_state_dict(state['optim'])
    start_step = int(state.get('step', 0))
    loss_hist = list(state.get('extra', {}).get('loss_hist', []))
    print(f'  loaded model + optim, resume at step {start_step+1}')

# -------------------- the loop --------------------
start = time.time()
if DEVICE == 'cuda':
    torch.cuda.reset_peak_memory_stats()

for step in range(start_step + 1, TOTAL_STEPS + 1):
    sampler = sampler_A if step <= PHASE_A_STEPS else sampler_B
    phase_tag = 'A' if step <= PHASE_A_STEPS else 'B'

    cur_lr = lr_at(step)
    for g in optim.param_groups: g['lr'] = cur_lr

    model.train(); step_loss = 0.0
    optim.zero_grad(set_to_none=True)
    out = None
    router_info_snapshot = None
    n_nan_micros = 0
    n_ok_micros  = 0
    for micro in range(GRAD_ACCUM_STEPS):
        ids, targets = next(sampler)
        ids, targets = ids.to(DEVICE), targets.to(DEVICE)
        store_now = (step % STORE_EVERY == 0) and (micro == 0)
        out = model(ids, targets=targets, store_to_memory=store_now)
        loss = out['loss'] / GRAD_ACCUM_STEPS
        if torch.isfinite(loss):
            loss.backward()
            step_loss += float(loss) * GRAD_ACCUM_STEPS
            n_ok_micros += 1
        else:
            n_nan_micros += 1
        ri = out.get('router_infos') or []
        if ri and 'mp_replicon' in ri[0]:
            router_info_snapshot = float(ri[0]['mp_replicon'])
        del ids, targets, loss

    # NaN-handling policy:
    #   - all-NaN step: zero grads, skip optim.step, record NaN in loss_hist
    #   - mixed step:   step anyway (partial grads), but log a warning
    #   - all-OK step:  normal
    if n_ok_micros == 0:
        optim.zero_grad(set_to_none=True)
        loss_hist.append(float('nan'))
        NAN_STEPS_TOTAL += 1
        CONSECUTIVE_NAN += 1
        print(f'  [NaN] step={step} all {GRAD_ACCUM_STEPS} micros NaN; skipped optim.step ({CONSECUTIVE_NAN}/{MAX_CONSECUTIVE_NAN_STEPS} consecutive)')
    else:
        if n_nan_micros > 0:
            print(f'  [NaN-mix] step={step} {n_nan_micros}/{GRAD_ACCUM_STEPS} micros NaN; applying partial grads')
            NAN_STEPS_TOTAL += 1
        if FISHER_PRECOND:
            precondition_router_grads_(model, fisher_state)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optim.step()
        loss_hist.append(step_loss / max(n_ok_micros, 1))
        CONSECUTIVE_NAN = 0   # reset — we got at least one clean micro

    if CONSECUTIVE_NAN >= MAX_CONSECUTIVE_NAN_STEPS:
        print(f'STOP: {CONSECUTIVE_NAN} consecutive all-NaN steps at step {step}. '
              f'Lower PEAK_LR or check for bad data; resume via RESUME_FROM.')
        break

    if step % LOG_EVERY == 0 or step in (start_step + 1, PHASE_A_STEPS + 1):
        elapsed = time.time() - start
        vram   = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == 'cuda' else 0.0
        mstats = model.memory.get_stats() if hasattr(model, 'memory') else {}
        cur_sl = SEQ_LEN_A if phase_tag == 'A' else SEQ_LEN_B
        loss_display = loss_hist[-1] if not (isinstance(loss_hist[-1], float) and _math.isnan(loss_hist[-1])) else float('nan')
        print(f'[{phase_tag} T={cur_sl}] step={step:5d} lr={cur_lr:.2e} loss={loss_display:.4f} '
              f'vram={vram:.1f}GB chirality={mstats.get("chirality_balance", 0.0):.3f} '
              f'replicon={router_info_snapshot if router_info_snapshot is not None else 0.0:+.2f} '
              f'nan_total={NAN_STEPS_TOTAL} elapsed={elapsed/60:.1f}m')
        if DEVICE == 'cuda':
            torch.cuda.reset_peak_memory_stats()

    # -------------------- two-tier ckpt --------------------
    is_milestone = (step == PHASE_A_STEPS) or (step == TOTAL_STEPS) or (step % DRIVE_MILESTONE_EVERY == 0)
    is_rolling   = (step % CKPT_EVERY == 0)

    if is_rolling:
        _save(os.path.join(LOCAL_CKPT_DIR, f'step_{step:05d}.pt'),
              model, optim, step, include_optim=True,
              extra={'loss_hist': loss_hist[-CKPT_EVERY:], 'phase': phase_tag,
                     'nan_steps_total': NAN_STEPS_TOTAL})
        _rolling_trim(LOCAL_CKPT_DIR, keep=ROLLING_KEEP)

    if is_milestone:
        _save(os.path.join(CKPT_DIR, f'step_{step:05d}.pt'),
              model, optim, step, include_optim=True,
              extra={'loss_hist': loss_hist[-DRIVE_MILESTONE_EVERY:], 'phase': phase_tag,
                     'milestone': 'phase_A_end' if step == PHASE_A_STEPS else ('final' if step == TOTAL_STEPS else 'periodic'),
                     'nan_steps_total': NAN_STEPS_TOTAL})
        _gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()

    del out
    router_info_snapshot = None

print(f'training complete in {(time.time()-start)/3600:.2f} h')
print(f'NaN steps: {NAN_STEPS_TOTAL} / {TOTAL_STEPS}')
print(f'local ckpts:  {LOCAL_CKPT_DIR}  (rolling, last {ROLLING_KEEP})')
print(f'drive ckpts:  {CKPT_DIR}         (milestones only)')'''


def _set_source(cell, text: str) -> None:
    lines = text.rstrip("\n").split("\n")
    cell["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    hits = 0
    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))
        if "Cell 6.3" in src and "training loop" in src:
            _set_source(c, NEW_SRC)
            hits += 1
            print(f"replaced cell {i}")
    if hits != 1:
        raise SystemExit(f"expected 1 match for cell 6.3, found {hits}")
    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("notebook patched")


if __name__ == "__main__":
    main()
