"""
Option C — First real GPU run on the default 60M-stored / ~85M-total preset.

Goal: prove the FANT 2 default architecture trains end-to-end on a single
RTX 3060 (12 GB) with BF16 + gradient checkpointing + 8-bit AdamW. This is
the FIRST time the default preset has been instantiated outside `info` and
the FIRST time anything has touched a GPU.

What this does:
  1. Pin GPU 0 (RTX 3060)
  2. Build the default preset model and report VRAM cost
  3. Reuse the Option B BPE tokenizer (or train a fresh one if missing)
  4. Stream real tokenized text via SyntheticStream + TokenizedBatchStream
  5. Run Phase 2 (FEP MoE) for 300 steps with batch=4, seq=512, bf16
  6. Report loss curve, throughput (tokens/sec), peak VRAM
  7. Verify NO NaN/inf, loss DROPS, peak VRAM < 11 GB
  8. Run the post-training router-collapse canary on multiple synthetic domains
     and verify the FANT 350M failure mode (>85% on a single mega-pool) is
     avoided AT FULL SCALE

Run:
    PYTHONPATH=. python scripts/option_c_gpu_smoke.py
"""

from __future__ import annotations

import math
import os
import time

import torch

from fant2.config import fant2_default
from fant2.data import SEED_CORPUS, SyntheticStream, TokenizedBatchStream
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training import TrainConfig, FANT2Trainer
from fant2.training.telemetry import router_jsd_pairwise


# -----------------------------------------------------------------------------
# 1. GPU setup
# -----------------------------------------------------------------------------

def setup_gpu() -> str:
    if not torch.cuda.is_available():
        print("  no CUDA available — Option C requires a GPU.")
        raise SystemExit(2)
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = torch.cuda.mem_get_info(0)[0] / 1e9
    print(f"  GPU 0: {name}")
    print(f"  total VRAM: {total:.2f} GB,  free VRAM: {free:.2f} GB")
    torch.cuda.reset_peak_memory_stats(0)
    return "cuda:0"


# -----------------------------------------------------------------------------
# 2. Tokenizer (reuse or train fresh)
# -----------------------------------------------------------------------------

def get_tokenizer(path: str) -> FANT2Tokenizer:
    if os.path.exists(path):
        print(f"  reusing tokenizer at {path}")
        return FANT2Tokenizer.load(path)
    print(f"  training a fresh BPE tokenizer at {path}")
    def gen():
        for i in range(5000):
            yield SEED_CORPUS[i % len(SEED_CORPUS)]
    tok = FANT2Tokenizer.train_from_iterator(
        iterator=gen(),
        vocab_size=32768,
        min_frequency=2,
        show_progress=False,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tok.save(path)
    return tok


# -----------------------------------------------------------------------------
# 3. Build trainer
# -----------------------------------------------------------------------------

def build_trainer(tokenizer, out_dir: str, device: str, n_steps: int):
    print()
    print("  building default preset model on GPU...")
    torch.manual_seed(0)
    cfg = fant2_default()
    model = FANT2Model(cfg)

    # Move to GPU
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    vram_after_model = torch.cuda.memory_allocated(0) / 1e9
    print(f"  model on GPU: {n_params/1e6:.2f} M params, VRAM = {vram_after_model:.3f} GB")

    text_stream = SyntheticStream(seed=2026)
    batch_stream = TokenizedBatchStream(
        text_stream=text_stream,
        tokenizer=tokenizer,
        batch_size=4,
        seq_len=512,
        device=device,
    )

    train_cfg = TrainConfig(
        phase=2,
        n_steps=n_steps,
        batch_size=4,
        seq_len=512,
        muon_lr=1e-3,
        adam_lr=3e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.1,
        fep_kl_beta_max=1.0,
        fep_kl_anneal_steps=n_steps,
        telemetry_every=100,
        tikkun_every=50,
        fana_every=200,
        log_every=20,
        save_every=10000,
        out_dir=out_dir,
        device=device,
        bf16=True,
        grad_checkpoint=True,
        use_8bit_adam=True,
    )
    trainer = FANT2Trainer(model, train_cfg, batch_stream)
    return trainer, model, cfg


# -----------------------------------------------------------------------------
# 4. Loss curve capture (override the trainer's logging)
# -----------------------------------------------------------------------------

class LossHistory:
    def __init__(self):
        self.steps = []
        self.ce = []

    def hook(self, trainer):
        original_train_step = trainer.train_step
        def wrapped(batch):
            losses = original_train_step(batch)
            if "ce" in losses and (trainer.step % 20 == 0 or trainer.step <= 5):
                self.steps.append(trainer.step)
                self.ce.append(losses["ce"])
            return losses
        trainer.train_step = wrapped


# -----------------------------------------------------------------------------
# 5. Post-training router canary
# -----------------------------------------------------------------------------

def post_train_router_canary(model, tokenizer, device: str, n_domains: int = 6) -> bool:
    print()
    print("  -- post-training router-collapse canary --")
    cfg = model.config
    n_megapools = cfg.n_megapools
    domain_routings = {}
    model.eval()
    with torch.no_grad():
        for d in range(n_domains):
            counts = torch.zeros(n_megapools)
            n_tokens = 0
            stream = SyntheticStream(seed=5000 + d)
            text_iter = iter(stream)
            for _ in range(8):
                text = next(text_iter)
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                if len(ids) < 4:
                    continue
                ids = ids[:128]
                inp = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
                fwd = model(inp)
                for ro in fwd["router_outputs"]:
                    bins = torch.bincount(ro.megapool_idx.cpu(), minlength=n_megapools).float()
                    counts += bins
                    n_tokens += int(ro.megapool_idx.numel())
            counts = counts / max(n_tokens, 1)
            domain_routings[f"domain_{d}"] = counts
            print(f"    domain_{d}: max_load = {counts.max().item():.3f}, "
                  f"top3 = {sorted(counts.tolist(), reverse=True)[:3]}")

    jsd = router_jsd_pairwise(domain_routings)
    print(f"  mean pairwise JSD = {jsd['mean_jsd']:.4f}")

    failed = False
    for name, dist in domain_routings.items():
        if float(dist.max().item()) >= 0.85:
            print(f"  ✗ FAIL: {name} max_load = {float(dist.max().item()):.3f}")
            failed = True
        if int((dist > 1e-3).sum().item()) < n_megapools // 2:
            print(f"  ✗ FAIL: {name} too few active pools")
            failed = True
    return not failed


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print(" FANT 2 — Option C: first real GPU run on default preset")
    print("=" * 64)
    device = setup_gpu()

    # Reuse Option B's tokenizer if it's there
    tok_path = "output/option_b/tokenizer.json"
    tokenizer = get_tokenizer(tok_path)
    print(f"  tokenizer vocab: {tokenizer.vocab_size}")

    out_dir = "output/option_c"
    os.makedirs(out_dir, exist_ok=True)

    n_steps = 300
    trainer, model, cfg = build_trainer(tokenizer, out_dir, device, n_steps)

    history = LossHistory()
    history.hook(trainer)

    print()
    print(f"  -- training {n_steps} Phase 2 steps on default preset --")
    t0 = time.time()
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print(f"  ✗ FAIL: CUDA OOM — {e}")
        return 1
    elapsed = time.time() - t0

    peak_vram = torch.cuda.max_memory_allocated(0) / 1e9
    n_tokens = n_steps * 4 * 512  # batch * seq

    print()
    print("  -- result --")
    print(f"  steps          : {n_steps}")
    print(f"  elapsed        : {elapsed:.1f} s")
    print(f"  throughput     : {n_tokens/elapsed:,.0f} tok/s")
    print(f"  peak VRAM      : {peak_vram:.2f} GB / 12.0 GB")
    if history.ce:
        print(f"  CE first->last : {history.ce[0]:.4f} → {history.ce[-1]:.4f}  "
              f"(Δ = {history.ce[0]-history.ce[-1]:.3f} nats)")

    # ----- Sanity gates -----
    failed = False
    if not history.ce:
        print("  ✗ FAIL: no loss history captured")
        failed = True
    else:
        # Any NaN or inf?
        for s, c in zip(history.steps, history.ce):
            if math.isnan(c) or math.isinf(c):
                print(f"  ✗ FAIL: non-finite CE at step {s}: {c}")
                failed = True
                break
        # Did the loss drop?
        if history.ce[-1] >= history.ce[0]:
            print(f"  ✗ FAIL: CE did not drop ({history.ce[0]:.3f} -> {history.ce[-1]:.3f})")
            failed = True
        # VRAM budget
        if peak_vram > 11.0:
            print(f"  ✗ FAIL: peak VRAM {peak_vram:.2f} GB exceeds budget")
            failed = True

    canary_ok = post_train_router_canary(model, tokenizer, device, n_domains=6)
    if not canary_ok:
        print("  ✗ FAIL: post-training router canary failed")
        failed = True

    print()
    if failed:
        print("  ✗ Option C FAILED")
        return 1
    print(f"  ✓ Option C PASSED")
    print(f"    - default preset trains on GPU without OOM/NaN")
    print(f"    - peak VRAM = {peak_vram:.2f} GB (well under 12 GB)")
    print(f"    - throughput = {n_tokens/elapsed:,.0f} tok/s")
    print(f"    - router did not collapse at full scale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
