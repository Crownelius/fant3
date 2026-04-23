"""Tier 1+2 CE stabilisation patches for fant3_1b_nvidia_train.ipynb.

Changes:
  - cell 1.1 cfg: set lm_head_logit_cap = 30.0 + apollonian_channel_warmup_steps = 500
  - cell 6.1 recipe: GRAD_CLIP 1.0 -> 0.5, WARMUP 1800 -> 2400, + Z_LOSS_COEF,
                      + FISHER_WARMUP_STEPS, + GRAD_NORM_ALERT_X
  - NEW cell 6.1b (markdown + code): build CE probe batches per Phase-A/B dataset
  - cell 6.3 training loop: sum z_loss, gate Fisher by step, log grad_norm +
    max_logit + max_router_logit, periodic CE probe eval, running-median alert
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "fant3_1b_nvidia_train.ipynb"

# ---------- cell 1.1 cfg (extend the existing opt-in block) ----------
CELL11_FIND = r'''cfg.mor_isrm_contractive        = True   # Banach contractive refinement
# spinor_apollonian_enabled already True by default in fant3_1b()
print(cfg)'''
CELL11_NEW = r'''cfg.mor_isrm_contractive        = True   # Banach contractive refinement
# spinor_apollonian_enabled already True by default in fant3_1b()

# Tier 1/2 CE stabilisation (landed 2026-04-22) - see investigation memo
cfg.lm_head_logit_cap               = 30.0   # Gemma-2 style bf16 overflow guard
cfg.apollonian_channel_warmup_steps = 500    # delay MoR C-channel until memory embeddings meaningful

print(cfg)'''

# ---------- cell 6.1 recipe (GRAD_CLIP, WARMUP, new coefficients) ----------
CELL61_NEW = r'''# Recipe knobs - Tier D for 1B, calibrated for A100 80 GB
# 2026-04-22: BATCH_SIZE reduced 2 -> 1 (MoE expert W_up_sel materialization
# peaked ~38 GB at B=2 worst-case routing skew, OOM'd mid-training). Compensated
# with GRAD_ACCUM_STEPS 4 -> 8 so effective batch stays at 8.
# Tier 1/2 CE stabilisation: GRAD_CLIP tightened 1.0 -> 0.5, WARMUP 1800 -> 2400,
# added z-loss wiring + Fisher-preconditioner warmup + grad-norm monitoring.
BATCH_SIZE        = 1
GRAD_ACCUM_STEPS  = 8           # effective batch = 8 (unchanged)
SEQ_LEN_A         = SEQ_LEN_A_SUGGEST    # pretrain, concat-packing
SEQ_LEN_B         = SEQ_LEN_B_SUGGEST    # SFT, one-row-per-sample
PHASE_A_STEPS     = 8000
PHASE_B_STEPS     = 4000
TOTAL_STEPS       = PHASE_A_STEPS + PHASE_B_STEPS
WARMUP_STEPS      = 2400        # 20% of total (was 1800 = 15%); extra safety at 1B
PEAK_LR           = 1.2e-4
GRAD_CLIP         = 0.5         # was 1.0; tighter clip at 1B given MoE grad variance
SCHEDULE_SHAPE    = 'litim'
LOG_EVERY         = 25
CKPT_EVERY        = 500
STORE_EVERY       = 50
FISHER_PRECOND    = True
FISHER_WARMUP_STEPS = 100       # skip precondition for first N steps (EMA warmup)
Z_LOSS_COEF       = 1e-4        # OLMoE-style router z-loss weight; kills router drift
GRAD_NORM_ALERT_X = 3.0         # log warning when grad_norm > X * running median
CE_PROBE_EVERY    = 200         # run per-domain CE probe every N steps (set 0 to disable)
print(f'total={TOTAL_STEPS}  warmup={WARMUP_STEPS}  peak_lr={PEAK_LR:.1e}  schedule={SCHEDULE_SHAPE}')
print(f'seq_len: Phase A = {SEQ_LEN_A} (concat)   Phase B = {SEQ_LEN_B} (per_row)')
print(f'batch: B={BATCH_SIZE} accum={GRAD_ACCUM_STEPS} effective={BATCH_SIZE*GRAD_ACCUM_STEPS}')
print(f'stability: clip={GRAD_CLIP}  z_loss={Z_LOSS_COEF}  fisher_warmup={FISHER_WARMUP_STEPS}  ce_probe_every={CE_PROBE_EVERY}')'''

# ---------- NEW cell 6.1b: CE probe builder ----------
CELL61B_MD = [
    "## 6.1b \u00b7 Per-domain CE probe (held-out batches)\n",
    "\n",
    "Samples a fixed held-out batch from each phase dataset so the training loop can measure CE per source on identical samples over time. This is what tells you whether CE is actually moving toward 1.x on clean prose (FineWeb) versus getting stuck on reasoning/chat traces where CE is intrinsically higher."
]
CELL61B_SRC = r'''# Cell 6.1b - build per-domain CE probe (held-out samples, not in training path)
from fant2.data.streaming import HuggingFaceStream
from fant2.data.registry import ALL_DATASETS

def _build_probe(ds_key, batch_size=4, seq_len=SEQ_LEN_A, pad_id=PAD_ID, eos_id=EOS_ID):
    """Grab `batch_size` rows from one dataset, tokenize, pad/trunc to seq_len,
    return (ids, targets) on DEVICE. Returns None if dataset unavailable."""
    if ds_key not in ALL_DATASETS:
        return None
    e = ALL_DATASETS[ds_key]
    try:
        s = HuggingFaceStream(dataset_name=e.hf_id, dataset_config=e.config,
                              split=e.split, text_key=e.text_key, format_type=e.format,
                              input_key=e.input_key, output_key=e.output_key)
    except Exception:
        return None
    rows = []
    it = iter(s)
    for i in range(batch_size * 3):  # overhead for skip of empty/contaminated
        try: text = next(it)
        except StopIteration: break
        if not text: continue
        ids = tok.encode(text).ids
        if not ids: continue
        ids = ids[:seq_len - 1] + [eos_id]
        ids = ids + [pad_id] * (seq_len - len(ids))
        rows.append(ids)
        if len(rows) >= batch_size: break
    if len(rows) < batch_size:
        return None
    batch = torch.tensor(rows, dtype=torch.long, device=DEVICE)
    tgt = batch.clone()
    tgt[tgt == pad_id] = -100
    return (batch, tgt)

CE_PROBES = {}
for ds_key in set(PHASE_A_DATASETS) | set(PHASE_B_DATASETS):
    probe = _build_probe(ds_key, batch_size=4, seq_len=min(SEQ_LEN_A, 1024))
    if probe is not None:
        CE_PROBES[ds_key] = probe
print(f'CE probe built for {len(CE_PROBES)} domains: {sorted(CE_PROBES.keys())}')

@torch.no_grad()
def ce_probe_eval(model):
    """Return {ds_key: ce_float} for every probe. Sets model.eval() temporarily."""
    was_training = model.training
    model.eval()
    out = {}
    for k, (ids, tgt) in CE_PROBES.items():
        try:
            r = model(ids, targets=tgt)
            l = float(r['loss'])
            if l == l:  # not NaN
                out[k] = l
        except Exception as e:
            out[k] = float('nan')
    if was_training: model.train()
    return out'''

# ---------- cell 6.3 training loop — sum z_loss, gate Fisher, log grad_norm, periodic CE probe ----------
CELL63_NEW = r'''# Cell 6.3 - training loop (Tier 1/2 CE stabilisation + two-tier ckpt + resume + NaN guard)
import gc as _gc
import glob as _glob
import math as _math

# MAX_ROW_TOKENS in per_row mode: set this to SEQ_LEN_B to drop docs longer
# than the seq_len (avoids "problem setup + no answer" truncation in SFT).
MAX_ROW_TOKENS_B = SEQ_LEN_B   # change to None if you prefer truncate-over-skip
sampler_A = make_batch_sampler(stream_A, BATCH_SIZE, SEQ_LEN_A, PAD_ID, EOS_ID, pack_mode='concat')
sampler_B = make_batch_sampler(stream_B, BATCH_SIZE, SEQ_LEN_B, PAD_ID, EOS_ID, pack_mode='per_row', max_row_tokens=MAX_ROW_TOKENS_B)
fisher_state = {}
loss_hist  = []
grad_norm_hist = []   # for running-median alert

# -------------------- checkpoint destinations --------------------
LOCAL_CKPT_DIR  = '/content/ckpts_local' if IN_COLAB else os.path.join(CKPT_DIR, '_local')
os.makedirs(LOCAL_CKPT_DIR, exist_ok=True)
ROLLING_KEEP    = 3
DRIVE_MILESTONE_EVERY = 2000
RESUME_FROM     = None

# -------------------- NaN guard --------------------
MAX_CONSECUTIVE_NAN_STEPS = 3
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
            os.remove(old); print(f'  [rolling] removed {os.path.basename(old)}')
        except OSError: pass

# -------------------- optional resume --------------------
start_step = 0
if RESUME_FROM is not None and os.path.exists(RESUME_FROM):
    print(f'resuming from {RESUME_FROM}')
    state = torch.load(RESUME_FROM, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state['model'])
    if 'optim' in state: optim.load_state_dict(state['optim'])
    start_step = int(state.get('step', 0))
    loss_hist = list(state.get('extra', {}).get('loss_hist', []))
    print(f'  loaded model + optim, resume at step {start_step+1}')

# -------------------- the loop --------------------
start = time.time()
if DEVICE == 'cuda':
    torch.cuda.reset_peak_memory_stats()

for step in range(start_step + 1, TOTAL_STEPS + 1):
    # Propagate step to submodules (MoR Apollonian-channel warmup gate)
    model.set_global_step(step)

    sampler = sampler_A if step <= PHASE_A_STEPS else sampler_B
    phase_tag = 'A' if step <= PHASE_A_STEPS else 'B'
    cur_lr = lr_at(step)
    for g in optim.param_groups: g['lr'] = cur_lr

    model.train(); step_loss = 0.0; step_z_loss = 0.0
    optim.zero_grad(set_to_none=True)
    out = None; router_info_snapshot = None
    n_nan_micros = 0; n_ok_micros = 0
    max_logit_abs = 0.0; max_router_logit_abs = 0.0

    for micro in range(GRAD_ACCUM_STEPS):
        ids, targets = next(sampler)
        ids, targets = ids.to(DEVICE), targets.to(DEVICE)
        store_now = (step % STORE_EVERY == 0) and (micro == 0)
        out = model(ids, targets=targets, store_to_memory=store_now)

        # Sum router z-loss from every suffix MoE block + the MoR shared block
        z_sum = 0.0
        for ri in (out.get('router_infos') or []):
            z = ri.get('z_loss')
            if z is not None: z_sum = z_sum + z
            mp_lg = ri.get('mp_logits')
            if mp_lg is not None:
                mla = float(mp_lg.abs().max())
                if mla > max_router_logit_abs: max_router_logit_abs = mla

        total_loss = out['loss'] + Z_LOSS_COEF * z_sum
        loss_scaled = total_loss / GRAD_ACCUM_STEPS

        if torch.isfinite(loss_scaled):
            loss_scaled.backward()
            step_loss   += float(out['loss'])
            step_z_loss += float(z_sum) if isinstance(z_sum, torch.Tensor) else 0.0
            n_ok_micros += 1
            # Track max |logit| post-soft-cap if enabled
            if 'logits' in out:
                mla = float(out['logits'].abs().max())
                if mla > max_logit_abs: max_logit_abs = mla
        else:
            n_nan_micros += 1

        ri = out.get('router_infos') or []
        if ri and 'mp_replicon' in ri[0]:
            router_info_snapshot = float(ri[0]['mp_replicon'])
        del ids, targets, total_loss, loss_scaled

    # --------- step policy: NaN guard + Fisher + clip + optim.step ---------
    if n_ok_micros == 0:
        optim.zero_grad(set_to_none=True)
        loss_hist.append(float('nan'))
        NAN_STEPS_TOTAL += 1; CONSECUTIVE_NAN += 1
        print(f'  [NaN] step={step} all {GRAD_ACCUM_STEPS} micros NaN; skipped optim.step ({CONSECUTIVE_NAN}/{MAX_CONSECUTIVE_NAN_STEPS} consecutive)')
    else:
        if n_nan_micros > 0:
            print(f'  [NaN-mix] step={step} {n_nan_micros}/{GRAD_ACCUM_STEPS} micros NaN; applying partial grads')
            NAN_STEPS_TOTAL += 1
        # Fisher precondition only after warmup (EMA needs ~100 steps to stabilise)
        if FISHER_PRECOND and step >= FISHER_WARMUP_STEPS:
            precondition_router_grads_(model, fisher_state)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        grad_norm_hist.append(float(gn))
        optim.step()
        loss_hist.append(step_loss / max(n_ok_micros, 1))
        CONSECUTIVE_NAN = 0

        # Running-median alert: warn if grad_norm suddenly spikes
        if len(grad_norm_hist) >= 50:
            import statistics as _stats
            med = _stats.median(grad_norm_hist[-50:])
            if gn > GRAD_NORM_ALERT_X * med and med > 1e-6:
                print(f'  [grad-spike] step={step} grad_norm={gn:.3f} = {gn/med:.1f}x running-median (median_50={med:.3f})')

    if CONSECUTIVE_NAN >= MAX_CONSECUTIVE_NAN_STEPS:
        print(f'STOP: {CONSECUTIVE_NAN} consecutive all-NaN steps at step {step}. Lower PEAK_LR or check for bad data; resume via RESUME_FROM.')
        break

    # Aggressive cache flush every 4 steps to fight 1B-scale fragmentation
    if DEVICE == 'cuda' and step % 4 == 0:
        torch.cuda.empty_cache()

    # ------------------- logging ---------------------
    if step % LOG_EVERY == 0 or step in (start_step + 1, PHASE_A_STEPS + 1):
        elapsed = time.time() - start
        vram = torch.cuda.max_memory_allocated() / 1e9 if DEVICE == 'cuda' else 0.0
        mstats = model.memory.get_stats() if hasattr(model, 'memory') else {}
        cur_sl = SEQ_LEN_A if phase_tag == 'A' else SEQ_LEN_B
        cur_loss = loss_hist[-1]
        loss_display = cur_loss if not (isinstance(cur_loss, float) and _math.isnan(cur_loss)) else float('nan')
        cur_gn = grad_norm_hist[-1] if grad_norm_hist else 0.0
        print(f'[{phase_tag} T={cur_sl}] step={step:5d} lr={cur_lr:.2e} loss={loss_display:.4f} z={step_z_loss:.3f} '
              f'gn={cur_gn:.2f} max|logit|={max_logit_abs:.1f} max|rtr|={max_router_logit_abs:.1f} '
              f'vram={vram:.1f}GB chirality={mstats.get("chirality_balance", 0.0):.3f} '
              f'replicon={router_info_snapshot if router_info_snapshot is not None else 0.0:+.2f} '
              f'nan_total={NAN_STEPS_TOTAL} elapsed={elapsed/60:.1f}m')
        if DEVICE == 'cuda': torch.cuda.reset_peak_memory_stats()

    # ------------------- per-domain CE probe ---------------------
    if CE_PROBE_EVERY > 0 and step % CE_PROBE_EVERY == 0 and len(CE_PROBES) > 0:
        probe_ce = ce_probe_eval(model)
        # print as one compact line, sorted by CE ascending (best first)
        items = sorted(probe_ce.items(), key=lambda kv: kv[1] if kv[1] == kv[1] else 99.0)
        probe_line = '  [probe] ' + '  '.join(f'{k[:16]}={v:.2f}' for k, v in items[:6])
        print(probe_line)

    # -------------------- two-tier ckpt --------------------
    is_milestone = (step == PHASE_A_STEPS) or (step == TOTAL_STEPS) or (step % DRIVE_MILESTONE_EVERY == 0)
    is_rolling   = (step % CKPT_EVERY == 0)
    if is_rolling:
        _save(os.path.join(LOCAL_CKPT_DIR, f'step_{step:05d}.pt'), model, optim, step, include_optim=True,
              extra={'loss_hist': loss_hist[-CKPT_EVERY:], 'phase': phase_tag, 'nan_steps_total': NAN_STEPS_TOTAL,
                     'grad_norm_hist': grad_norm_hist[-CKPT_EVERY:]})
        _rolling_trim(LOCAL_CKPT_DIR, keep=ROLLING_KEEP)
    if is_milestone:
        _save(os.path.join(CKPT_DIR, f'step_{step:05d}.pt'), model, optim, step, include_optim=True,
              extra={'loss_hist': loss_hist[-DRIVE_MILESTONE_EVERY:], 'phase': phase_tag,
                     'milestone': 'phase_A_end' if step == PHASE_A_STEPS else ('final' if step == TOTAL_STEPS else 'periodic'),
                     'nan_steps_total': NAN_STEPS_TOTAL})
        _gc.collect()
        if DEVICE == 'cuda': torch.cuda.empty_cache()

    del out; router_info_snapshot = None

print(f'training complete in {(time.time()-start)/3600:.2f} h')
print(f'NaN steps: {NAN_STEPS_TOTAL} / {TOTAL_STEPS}')
print(f'local ckpts:  {LOCAL_CKPT_DIR}  (rolling, last {ROLLING_KEEP})')
print(f'drive ckpts:  {CKPT_DIR}         (milestones only)')'''


def _set(c, text):
    lines = text.rstrip("\n").split("\n")
    c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]


def main() -> None:
    nb = json.loads(NB_PATH.read_text(encoding="utf-8"))
    patched = []
    insert_after_61 = None
    have_61b = False

    for i, c in enumerate(nb["cells"]):
        src = "".join(c.get("source", []))

        # cell 1.1 cfg
        if "cfg.mor_isrm_contractive" in src and "CERN-inspired opt-ins" in src:
            if CELL11_FIND in src and "lm_head_logit_cap" not in src:
                new = src.replace(CELL11_FIND, CELL11_NEW, 1)
                lines = new.split("\n")
                c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
                patched.append(f"cell 1.1 (cell {i}) added lm_head_logit_cap + apollonian warmup")

        # cell 6.1 recipe knobs
        elif "Recipe knobs" in src and "BATCH_SIZE" in src:
            _set(c, CELL61_NEW); patched.append(f"cell 6.1 (cell {i}) knobs")
            insert_after_61 = i

        # detect existing 6.1b so we don't insert twice
        elif "Cell 6.1b" in src or "build per-domain CE probe" in src:
            have_61b = True

        # cell 6.3
        elif "Cell 6.3" in src and "training loop" in src:
            _set(c, CELL63_NEW); patched.append(f"cell 6.3 (cell {i}) loop")

    # insert cell 6.1b right after the recipe knob cell
    if insert_after_61 is not None and not have_61b:
        md_cell = {"cell_type": "markdown", "metadata": {}, "source": CELL61B_MD}
        code_cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": []}
        _set(code_cell, CELL61B_SRC)
        nb["cells"].insert(insert_after_61 + 1, code_cell)
        nb["cells"].insert(insert_after_61 + 1, md_cell)
        patched.append(f"inserted cell 6.1b CE probe after index {insert_after_61}")

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    print("patched:")
    for p in patched: print("  -", p)


if __name__ == "__main__":
    main()
