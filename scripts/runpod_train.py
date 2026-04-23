#!/usr/bin/env python
"""Standalone RunPod training script — a headless distillation of the
fant3_1b_nvidia_train.ipynb training loop.

Reads the same config, datasets, and hyperparameters as the notebook, but
runs without JupyterLab so it survives SSH disconnects under `nohup` / tmux.

Usage:
    python scripts/runpod_train.py --resume /workspace/ckpts/step_00500.pt

Key differences vs the notebook:
  * CKPT_DIR defaults to ./output/runpod_ckpts (local disk on the pod);
    pass --ckpt-dir to override (e.g. to a network volume).
  * No Drive mount.
  * No CE probe by default (CE_PROBE_EVERY=0).
  * Logs to stdout every LOG_EVERY steps; redirect to a file when running
    under nohup.
"""
from __future__ import annotations

import argparse
import gc as _gc
import glob as _glob
import math as _math
import os
import statistics as _stats
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Ensure the repo root is importable
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from fant3.config import fant3_1b, fant3_20m, fant3_10m, fant3_15m, fant3_80m, fant3_50m
from fant3.model.fant3_model import FANT3Model
from fant3.training import precondition_router_grads_, schedule_multiplier
from fant2.data.streaming import InterleavedMultiDatasetStream


# ---------------------------------------------------------------------------
# Recipe — match the notebook cell 6.1 knobs so resuming is bit-compatible
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--resume", type=Path, default=None,
                   help="path to step_XXXXX.pt to resume from")
    p.add_argument("--ckpt-dir", type=Path, default=_ROOT / "output" / "runpod_ckpts")
    p.add_argument("--total-steps", type=int, default=12000)
    p.add_argument("--phase-a-steps", type=int, default=8000)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--peak-lr", type=float, default=2e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len-a", type=int, default=1024)
    p.add_argument("--seq-len-b", type=int, default=1024)
    p.add_argument("--grad-clip", type=float, default=10.0)
    p.add_argument("--z-coef", type=float, default=1e-4)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--ckpt-every", type=int, default=500)
    p.add_argument("--ckpt-keep-last", type=int, default=0,
                   help="if >0, keep only the N most recent checkpoints (by mtime) — "
                        "plus phase_A_end and total_steps milestones which are always kept. "
                        "Prevents volume fill on long unlimited runs. Default 0 = keep all.")
    p.add_argument("--store-every", type=int, default=50)
    p.add_argument("--max-nan-steps", type=int, default=3)
    p.add_argument("--no-fp32-tied", action="store_true",
                   help="skip fp32 promotion of tied tok_emb/lm_head (default: promote)")
    p.add_argument("--scale", choices=["1b", "80m", "50m", "25m", "15m", "10m"], default="1b",
                   help="model preset: 1b=fant3_1b (default, ~1.0B stored), "
                        "80m=fant3_80m (88.69M stored), 50m=fant3_50m (50.79M stored), "
                        "25m=fant3_20m (23.5M stored), 15m=fant3_15m (14.6M stored), "
                        "10m=fant3_10m (9.5M stored)")
    p.add_argument("--max-seq-len", type=int, default=None,
                   help="override cfg.max_seq_len (default: preset value)")
    p.add_argument("--dry-run", action="store_true",
                   help="build model + load ckpt, print shapes, don't train")

    # Experiment tracking + secrets
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name. If set, wandb.init is called and loss/gn/lr are logged every step. "
                        "Requires WANDB_API_KEY env var (set via RunPod pod secrets, NOT in clear text).")
    p.add_argument("--wandb-entity", default=None,
                   help="W&B entity (user or team). Optional; defaults to the API key's default.")
    p.add_argument("--wandb-run-name", default=None,
                   help="W&B run name. Optional; auto-generated if omitted.")
    p.add_argument("--hf-login", action="store_true",
                   help="Call huggingface_hub.login() with the HF_TOKEN env var. Required for "
                        "gated datasets like nvidia/Nemotron-CC-v2.1.")
    return p.parse_args()


def build_cfg(scale="1b", max_seq_len=None):
    preset_map = {"1b": fant3_1b, "80m": fant3_80m, "50m": fant3_50m,
                  "25m": fant3_20m, "15m": fant3_15m, "10m": fant3_10m}
    cfg = preset_map[scale]()
    if max_seq_len is not None:
        cfg.max_seq_len = max_seq_len
    cfg.use_gradient_checkpointing  = True
    cfg.mor_lti_injection_enabled   = True
    cfg.mor_spectral_constraint     = True
    cfg.mor_loop_index_enabled      = True
    cfg.mor_lti_apollonian_channel  = True
    cfg.mor_adaptive_depth          = True
    cfg.mor_isrm_contractive        = True
    cfg.lm_head_logit_cap               = 30.0
    cfg.apollonian_channel_warmup_steps = 500
    return cfg


PHASE_A_DATASETS = [
    'fineweb-edu',
    'nvidia-openmath-reasoning',
    'nvidia-opencode-reasoning-2',
    'nvidia-openmath-2',
    'opus46-crownelius-3300x',
    'kimi-k25-distill',
]
PHASE_A_WEIGHTS = [0.35, 0.20, 0.10, 0.10, 0.15, 0.10]

PHASE_B_DATASETS = [
    'nvidia-cascade2-sft-if',
    'sonnet46-120k',
    'nvidia-openmath-2',
    'nvidia-cascade2-sft-science',
    'nvidia-daring-anteater',
    'nvidia-cascade2-sft-chat',
]
PHASE_B_WEIGHTS = [0.25, 0.30, 0.15, 0.10, 0.10, 0.10]


def make_batch_sampler(stream, tok, batch_size, seq_len, pad_id, eos_id,
                       pack_mode='per_row', max_row_tokens=None, is_contaminated=None):
    assert pack_mode in ('concat', 'per_row')
    it = iter(stream)
    def _next_clean():
        while True:
            text = next(it)
            if not text: continue
            if is_contaminated is not None and is_contaminated(text):
                continue
            return text
    while True:
        batch = torch.full((batch_size, seq_len), pad_id, dtype=torch.long)
        for b in range(batch_size):
            tokens = []
            if pack_mode == 'concat':
                while len(tokens) < seq_len:
                    ids = tok.encode(_next_clean()).ids
                    if not ids: continue
                    tokens.extend(ids); tokens.append(eos_id)
                row = tokens[:seq_len]
            else:
                while not tokens:
                    ids = tok.encode(_next_clean()).ids
                    if not ids: continue
                    if max_row_tokens is not None and len(ids) > max_row_tokens:
                        continue
                    tokens = ids[:seq_len - 1] + [eos_id]
                row = tokens + [pad_id] * (seq_len - len(tokens))
            batch[b] = torch.tensor(row, dtype=torch.long)
        targets = batch.clone()
        targets[targets == pad_id] = -100
        yield batch, targets


def _init_distributed():
    """Detect torchrun environment and initialise NCCL process group.
    Returns (rank, local_rank, world_size, is_main). Single-GPU path
    returns (0, 0, 1, True)."""
    if "RANK" in os.environ and "LOCAL_RANK" in os.environ:
        import torch.distributed as dist
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = dist.get_world_size()
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size, (rank == 0)
    return 0, 0, 1, True


def main():
    args = parse_args()

    rank, local_rank, world_size, is_main = _init_distributed()
    is_distributed = world_size > 1

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}" if is_distributed else "cuda"
    else:
        device = "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    def log(*a, **kw):
        if is_main:
            print(*a, **kw, flush=True)

    log(f"device={device}  dtype={dtype}  ckpt_dir={args.ckpt_dir}")
    log(f"distributed={is_distributed}  world_size={world_size}  rank={rank}  local_rank={local_rank}")
    if is_main:
        args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    # --- HuggingFace auth (optional) ---
    # Enables gated datasets (nvidia/Nemotron-CC-v2.1, Math-v1, Code-v1, etc).
    # Token MUST come from HF_TOKEN env var; never passed as CLI arg so it
    # doesn't appear in shell history or process listing.
    if args.hf_login:
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not hf_token:
            log("WARN --hf-login passed but HF_TOKEN env var is unset; continuing without auth.")
        else:
            try:
                from huggingface_hub import login as hf_login_fn
                hf_login_fn(token=hf_token, add_to_git_credential=False)
                log("HF authenticated via HF_TOKEN env var")
            except Exception as e:
                log(f"WARN HF login failed: {type(e).__name__}: {e}")

    # --- Weights & Biases init (rank-0 only) ---
    # API key MUST come from WANDB_API_KEY env var.
    wandb_run = None
    if is_main and args.wandb_project is not None:
        try:
            import wandb
            if os.environ.get("WANDB_API_KEY") is None:
                log("WARN --wandb-project set but WANDB_API_KEY env var is unset; skipping wandb init.")
            else:
                wandb_run = wandb.init(
                    project=args.wandb_project,
                    entity=args.wandb_entity,
                    name=args.wandb_run_name,
                    config={
                        "scale": args.scale,
                        "peak_lr": args.peak_lr,
                        "batch_size": args.batch_size,
                        "grad_accum": args.grad_accum,
                        "seq_len_a": args.seq_len_a,
                        "seq_len_b": args.seq_len_b,
                        "warmup_steps": args.warmup_steps,
                        "total_steps": args.total_steps,
                        "phase_a_steps": args.phase_a_steps,
                        "z_coef": args.z_coef,
                        "grad_clip": args.grad_clip,
                        "max_seq_len": args.max_seq_len,
                        "world_size": world_size,
                        "resume": str(args.resume) if args.resume else None,
                    },
                )
                log(f"wandb: initialized project={args.wandb_project} run={wandb_run.name}")
        except ImportError:
            log("WARN wandb not installed; skipping. Install via: pip install wandb")
        except Exception as e:
            log(f"WARN wandb init failed: {type(e).__name__}: {e}")

    # Config + model
    cfg = build_cfg(scale=args.scale, max_seq_len=args.max_seq_len)
    log(f"scale={args.scale}  max_seq_len={cfg.max_seq_len}")
    torch.manual_seed(0); np.random.seed(0)
    model = FANT3Model(cfg).to(dtype=dtype, device=device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model built: {n_params/1e6:.2f} M params")

    # Tokenizer + vocab alignment
    from tokenizers import Tokenizer
    tok_path = _ROOT / "output" / "tokenizer" / "tokenizer_v2.json"
    tok = Tokenizer.from_file(str(tok_path))
    V = tok.get_vocab_size()
    pad_id = tok.token_to_id("<|pad|>") or 0
    eos_id = tok.token_to_id("<|eos|>") or 1
    if V != cfg.vocab_size:
        log(f"aligning cfg.vocab_size {cfg.vocab_size} -> {V}")
        cfg.vocab_size = V
        torch.manual_seed(0); np.random.seed(0)
        model = FANT3Model(cfg).to(dtype=dtype, device=device)

    # Tier 3 fp32 tied tok_emb/lm_head promotion — MUST happen BEFORE DDP wrap
    # because DDP bucketizes params by dtype at construction.
    if not args.no_fp32_tied:
        with torch.no_grad():
            model.tok_emb.weight.data = model.tok_emb.weight.data.float()
        assert model.lm_head.weight.data_ptr() == model.tok_emb.weight.data_ptr()
        log(f"tied emb/lm_head -> fp32  ({model.tok_emb.weight.dtype})")

    # Resume BEFORE DDP wrap so state_dict maps to bare module params.
    start_step = 0
    loss_hist = []
    if args.resume is not None and args.resume.exists():
        log(f"resuming from {args.resume}")
        state = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(state["model"])
        start_step = int(state.get("step", 0))
        loss_hist = list(state.get("extra", {}).get("loss_hist", []))
        log(f"  loaded model state, resume at step {start_step+1}")

    # Wrap in DDP if distributed
    if is_distributed:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)   # MoE routing makes some params unused per-batch
        inner = model.module
    else:
        inner = model

    # Decontamination filter
    from scripts.decontaminate import is_contaminated, build_hash_cache
    _cache = build_hash_cache(rebuild=False)
    log(f"decontamination hashes: {sum(len(v) for v in _cache.values())}")

    # Streams + samplers — each rank gets a different seed so they see different
    # interleaved sequences. InterleavedMultiDatasetStream is a stateless iterator
    # over HF streaming datasets; there's no cross-rank coordination needed.
    stream_A = InterleavedMultiDatasetStream(PHASE_A_DATASETS, weights=PHASE_A_WEIGHTS, seed=rank)
    stream_B = InterleavedMultiDatasetStream(PHASE_B_DATASETS, weights=PHASE_B_WEIGHTS, seed=rank + 1000)
    sampler_A = make_batch_sampler(stream_A, tok, args.batch_size, args.seq_len_a,
                                   pad_id, eos_id, pack_mode="per_row",
                                   max_row_tokens=args.seq_len_a, is_contaminated=is_contaminated)
    sampler_B = make_batch_sampler(stream_B, tok, args.batch_size, args.seq_len_b,
                                   pad_id, eos_id, pack_mode="per_row",
                                   max_row_tokens=args.seq_len_b, is_contaminated=is_contaminated)

    # Optimiser (bf16 params + 8-bit Adam state) — construct AFTER DDP wrap so
    # it walks model.parameters() correctly.
    import bitsandbytes as bnb
    optim = bnb.optim.AdamW8bit(model.parameters(), lr=args.peak_lr,
                                betas=(0.9, 0.95), weight_decay=0.1, eps=1e-8)

    # Optim-state resume happens after optimiser construction
    if args.resume is not None and args.resume.exists():
        state = torch.load(args.resume, map_location=device, weights_only=False)
        if "optim" in state:
            optim.load_state_dict(state["optim"])
            log("  loaded optimiser state")

    def lr_at(step):
        return args.peak_lr * schedule_multiplier(step, args.warmup_steps, args.total_steps, "litim")

    if args.dry_run:
        log("dry run: skipping training loop")
        if is_distributed:
            import torch.distributed as dist
            dist.destroy_process_group()
        return

    # Training loop
    nan_total = 0; consec_nan = 0
    grad_norm_hist = []
    start = time.time()
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    for step in range(start_step + 1, args.total_steps + 1):
        inner.set_global_step(step)
        sampler = sampler_A if step <= args.phase_a_steps else sampler_B
        phase_tag = 'A' if step <= args.phase_a_steps else 'B'
        cur_lr = lr_at(step)
        for g in optim.param_groups: g['lr'] = cur_lr

        model.train()
        optim.zero_grad(set_to_none=True)
        step_loss = 0.0; step_z = 0.0
        n_ok = 0; n_nan = 0
        max_logit = 0.0; max_rtr = 0.0

        for micro_i in range(args.grad_accum):
            ids, targets = next(sampler)
            ids = ids.to(device); targets = targets.to(device)
            store_now = (step % args.store_every == 0)
            # Under DDP, only sync gradients on the final micro of a grad-accum
            # group to avoid O(grad_accum) all-reduces per step.
            if is_distributed and micro_i < args.grad_accum - 1:
                sync_ctx = model.no_sync()
            else:
                from contextlib import nullcontext
                sync_ctx = nullcontext()
            with sync_ctx:
                out = model(ids, targets=targets, store_to_memory=store_now)

                z_sum = 0.0
                for ri in (out.get('router_infos') or []):
                    z = ri.get('z_loss')
                    if z is not None: z_sum = z_sum + z
                    mp_lg = ri.get('mp_logits')
                    if mp_lg is not None:
                        mla = float(mp_lg.abs().max())
                        if mla > max_rtr: max_rtr = mla

                total = out['loss'] + args.z_coef * z_sum
                loss_scaled = total / args.grad_accum

                if torch.isfinite(loss_scaled):
                    loss_scaled.backward()
                    step_loss += float(out['loss'])
                    step_z += float(z_sum) if isinstance(z_sum, torch.Tensor) else 0.0
                    n_ok += 1
                    if 'logits' in out:
                        mla = float(out['logits'].abs().max())
                        if mla > max_logit: max_logit = mla
                else:
                    n_nan += 1
            del ids, targets, total, loss_scaled

        if n_ok == 0:
            optim.zero_grad(set_to_none=True)
            loss_hist.append(float('nan'))
            nan_total += 1; consec_nan += 1
            log(f'  [NaN] step={step} all {args.grad_accum} micros NaN ({consec_nan}/{args.max_nan_steps})')
            if consec_nan >= args.max_nan_steps:
                log(f'STOP: {consec_nan} consecutive NaN at step {step}')
                break
        else:
            if n_nan > 0:
                log(f'  [NaN-mix] step={step} {n_nan}/{args.grad_accum} micros NaN')
                nan_total += 1
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm_hist.append(float(gn))
            optim.step()
            loss_hist.append(step_loss / max(n_ok, 1))
            consec_nan = 0

        if step % 4 == 0 and device.startswith("cuda"):
            torch.cuda.empty_cache()

        if step % args.log_every == 0 or step == start_step + 1:
            elapsed = time.time() - start
            vram = torch.cuda.max_memory_allocated() / 1e9 if device.startswith("cuda") else 0.0
            cur_gn = grad_norm_hist[-1] if grad_norm_hist else 0.0
            mstats = inner.memory.get_stats() if hasattr(inner, 'memory') else {}
            log(f'[{phase_tag} T={args.seq_len_a if phase_tag=="A" else args.seq_len_b}] '
                f'step={step:5d} lr={cur_lr:.2e} loss={loss_hist[-1]:.4f} z={step_z:.3f} '
                f'gn={cur_gn:.2f} max|logit|={max_logit:.1f} max|rtr|={max_rtr:.1f} '
                f'vram={vram:.1f}GB x{world_size} chirality={mstats.get("chirality_balance", 0.0):.3f} '
                f'nan_total={nan_total} elapsed={elapsed/60:.1f}m')
            if wandb_run is not None:
                wandb_run.log({
                    "step": step,
                    "phase": phase_tag,
                    "loss": loss_hist[-1] if loss_hist[-1] == loss_hist[-1] else float('nan'),
                    "lr": cur_lr,
                    "z_loss": step_z,
                    "grad_norm": cur_gn,
                    "max_logit": max_logit,
                    "max_router_logit": max_rtr,
                    "vram_gb": vram,
                    "chirality_balance": mstats.get("chirality_balance", 0.0),
                    "pq_overlap_mean": mstats.get("pq_overlap_mean", 0.0),
                    "chsh_s": mstats.get("chsh_S", 0.0),
                    "nan_total": nan_total,
                    "elapsed_min": elapsed / 60.0,
                }, step=step)
            if device.startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()

        # Only rank 0 saves. Every rank has identical weights post-step under DDP,
        # so rank-0 state_dict is representative. Use inner.state_dict() (not the
        # DDP wrapper) so resume works under any world_size.
        is_milestone = (step == args.phase_a_steps or step == args.total_steps)
        is_rolling   = (step % args.ckpt_every == 0)
        if is_main and (is_rolling or is_milestone):
            path = args.ckpt_dir / f'step_{step:05d}.pt'
            if is_milestone:
                # stash milestones in a named copy so rolling-trim doesn't delete them
                milestone_suffix = "_phaseA" if step == args.phase_a_steps else "_final"
                path = args.ckpt_dir / f'step_{step:05d}{milestone_suffix}.pt'
            payload = {'model': inner.state_dict(), 'optim': optim.state_dict(),
                       'step': step, 'cfg': cfg.__dict__,
                       'extra': {'loss_hist': loss_hist[-args.ckpt_every:], 'phase': phase_tag,
                                 'nan_total': nan_total, 'world_size': world_size}}
            torch.save(payload, path)
            sz = path.stat().st_size / 1e9
            print(f'  [ckpt] step={step} -> {path}  size={sz:.2f} GB', flush=True)

            # Rolling-keep trim on the non-milestone ckpts only (milestones are named
            # with _phaseA or _final suffix and are never deleted by this sweep).
            if args.ckpt_keep_last > 0:
                import glob as _glob
                rolling = [p for p in _glob.glob(str(args.ckpt_dir / 'step_*.pt'))
                           if not (p.endswith('_phaseA.pt') or p.endswith('_final.pt'))]
                rolling.sort(key=lambda p: os.path.getmtime(p))
                for old in rolling[:-args.ckpt_keep_last]:
                    try:
                        os.remove(old)
                        print(f'  [rolling] removed {os.path.basename(old)}', flush=True)
                    except OSError as e:
                        print(f'  [rolling] couldn\'t remove {old}: {e}', flush=True)

            _gc.collect()
            if device.startswith("cuda"): torch.cuda.empty_cache()

    log(f'training complete in {(time.time()-start)/3600:.2f} h')
    log(f'NaN steps: {nan_total} / {args.total_steps}')
    if wandb_run is not None:
        wandb_run.finish()
    if is_distributed:
        import torch.distributed as dist
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
