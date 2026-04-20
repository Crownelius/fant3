"""
Option B — Real-text training on the tiny preset (~10-20 min on CPU).

What this does:
  1. Train a real BPE tokenizer on the SEED_CORPUS (20 sentences, repeated)
  2. Run Phase 1 (LLM-JEPA + SIGReg) for 500 steps on real tokenized text
  3. Run Phase 2 (FEP MoE specialization) for 500 steps, resuming from Phase 1
  4. Run a POST-TRAINING router-collapse canary across 6 synthetic "domains"
     and verify the FANT 350M failure mode (>85% load on a single mega-pool)
     does not occur after real training.

This is the first end-to-end pass over real text and the first time the
router-collapse canary is checked AFTER training (not just at init).

Run:
    PYTHONPATH=. python scripts/option_b_real_text.py
"""

from __future__ import annotations

import os
import time

import torch

from fant2.config import fant2_tiny
from fant2.data import SEED_CORPUS, SyntheticStream, TokenizedBatchStream
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training import TrainConfig, FANT2Trainer
from fant2.training.telemetry import router_jsd_pairwise


# -----------------------------------------------------------------------------
# Step 1: Train the BPE tokenizer
# -----------------------------------------------------------------------------

def train_tokenizer(out_path: str, n_docs: int = 5000, vocab_size: int = 32768):
    print("=" * 64)
    print(" Step 1: Training BPE tokenizer on SEED_CORPUS")
    print("=" * 64)

    def _seed_repeat():
        for i in range(n_docs):
            yield SEED_CORPUS[i % len(SEED_CORPUS)]

    t0 = time.time()
    tok = FANT2Tokenizer.train_from_iterator(
        iterator=_seed_repeat(),
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=False,
    )
    dt = time.time() - t0

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    tok.save(out_path)
    print(f"  trained in {dt:.1f}s, vocab={tok.vocab_size}, saved to {out_path}")

    # Round-trip check
    sample = SEED_CORPUS[0]
    ids = tok.encode(sample, add_bos=True, add_eos=True)
    rt = tok.decode(ids, skip_special_tokens=True)
    print(f"  round-trip: {sample!r}")
    print(f"     ids[:20]: {ids[:20]}")
    print(f"     decoded : {rt!r}")
    return tok


# -----------------------------------------------------------------------------
# Step 2: Phase 1 — LLM-JEPA + SIGReg
# -----------------------------------------------------------------------------

def run_phase1(tokenizer, out_dir: str, n_steps: int = 500) -> str:
    print()
    print("=" * 64)
    print(f" Step 2: Phase 1 — LLM-JEPA + SIGReg, {n_steps} steps")
    print("=" * 64)

    torch.manual_seed(0)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)

    text_stream = SyntheticStream(seed=42)
    batch_stream = TokenizedBatchStream(
        text_stream=text_stream,
        tokenizer=tokenizer,
        batch_size=4,
        seq_len=64,
        device="cpu",
    )

    train_cfg = TrainConfig(
        phase=1,
        n_steps=n_steps,
        batch_size=4,
        seq_len=64,
        muon_lr=1e-3,
        adam_lr=3e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.05,
        fep_kl_beta_max=0.5,
        fep_kl_anneal_steps=n_steps,
        telemetry_every=200,
        tikkun_every=200,
        fana_every=10000,        # off — too short to need
        log_every=50,
        save_every=10000,        # save only at end
        out_dir=out_dir,
        device="cpu",
        bf16=False,
        grad_checkpoint=False,
        use_8bit_adam=False,
    )
    trainer = FANT2Trainer(model, train_cfg, batch_stream)

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    print(f"  Phase 1 done in {dt:.1f}s ({dt/n_steps*1000:.1f} ms/step)")

    final_path = os.path.join(out_dir, "final.pt")
    return final_path


# -----------------------------------------------------------------------------
# Step 3: Phase 2 — FEP MoE specialization, resuming from Phase 1
# -----------------------------------------------------------------------------

def run_phase2(tokenizer, out_dir: str, resume_from: str, n_steps: int = 500) -> tuple:
    print()
    print("=" * 64)
    print(f" Step 3: Phase 2 — FEP MoE, {n_steps} steps (resume from Phase 1)")
    print("=" * 64)

    torch.manual_seed(1)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)

    text_stream = SyntheticStream(seed=99)
    batch_stream = TokenizedBatchStream(
        text_stream=text_stream,
        tokenizer=tokenizer,
        batch_size=4,
        seq_len=64,
        device="cpu",
    )

    train_cfg = TrainConfig(
        phase=2,
        n_steps=n_steps,
        batch_size=4,
        seq_len=64,
        muon_lr=1e-3,
        adam_lr=3e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.1,
        fep_kl_beta_max=1.0,
        fep_kl_anneal_steps=n_steps,
        telemetry_every=200,
        tikkun_every=100,
        fana_every=300,
        log_every=50,
        save_every=10000,
        out_dir=out_dir,
        device="cpu",
        bf16=False,
        grad_checkpoint=False,
        use_8bit_adam=False,
        resume_from=resume_from,
    )
    trainer = FANT2Trainer(model, train_cfg, batch_stream)

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    print(f"  Phase 2 done in {dt:.1f}s ({dt/n_steps*1000:.1f} ms/step)")

    final_path = os.path.join(out_dir, "final.pt")
    return model, final_path


# -----------------------------------------------------------------------------
# Step 4: Post-training router-collapse canary
# -----------------------------------------------------------------------------

def post_training_router_canary(model, tokenizer, n_domains: int = 6) -> bool:
    print()
    print("=" * 64)
    print(" Step 4: POST-TRAINING router-collapse canary")
    print("=" * 64)
    print(f"  feeding {n_domains} distinct synthetic domains through the trained model")

    cfg = model.config
    n_megapools = cfg.n_megapools

    # Generate n_domains different "domains" by varying random seed
    domain_routings = {}
    model.eval()
    with torch.no_grad():
        for d in range(n_domains):
            counts = torch.zeros(n_megapools)
            n_tokens = 0
            stream = SyntheticStream(seed=1000 + d)
            text_iter = iter(stream)
            # Pull a few hundred tokens worth of real text
            for _ in range(8):
                text = next(text_iter)
                ids = tokenizer.encode(text, add_bos=True, add_eos=True)
                if len(ids) < 4:
                    continue
                ids = ids[:64]  # cap
                inp = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
                fwd = model(inp)
                for ro in fwd["router_outputs"]:
                    bins = torch.bincount(ro.megapool_idx, minlength=n_megapools).float()
                    counts += bins
                    n_tokens += int(ro.megapool_idx.numel())
            counts = counts / max(n_tokens, 1)
            domain_routings[f"domain_{d}"] = counts
            print(f"    domain_{d}: load = {[f'{x:.3f}' for x in counts.tolist()]}, "
                  f"max = {counts.max().item():.3f}")

    # Pairwise JSD
    jsd = router_jsd_pairwise(domain_routings)
    mean_jsd = jsd["mean_jsd"]
    print()
    print(f"  mean pairwise JSD = {mean_jsd:.4f}")

    # Pass criteria:
    #  1. No domain has a single mega-pool > 0.85 (FANT 350M had 0.945)
    #  2. All domains have at least half the mega-pools active (load > 1e-3)
    #  3. Mean pairwise JSD > 0 (the routes are not all identical)
    failed = False
    for name, dist in domain_routings.items():
        max_load = float(dist.max().item())
        n_active = int((dist > 1e-3).sum().item())
        if max_load >= 0.85:
            print(f"  ✗ FAIL: {name} has max_load = {max_load:.3f} (≥ 0.85)")
            failed = True
        if n_active < n_megapools // 2:
            print(f"  ✗ FAIL: {name} has only {n_active}/{n_megapools} active mega-pools")
            failed = True
    if mean_jsd < 1e-4:
        print(f"  ✗ FAIL: mean JSD ≈ 0 — all domains routed identically")
        failed = True

    if not failed:
        print(f"  ✓ PASS: router did not collapse after training.")
        print(f"  ✓ Option B complete.")
    return not failed


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    base = "output/option_b"
    os.makedirs(base, exist_ok=True)
    tok_path = os.path.join(base, "tokenizer.json")
    p1_dir = os.path.join(base, "phase1")
    p2_dir = os.path.join(base, "phase2")

    tokenizer = train_tokenizer(tok_path, n_docs=5000)
    p1_ckpt = run_phase1(tokenizer, p1_dir, n_steps=500)
    model, p2_ckpt = run_phase2(tokenizer, p2_dir, resume_from=p1_ckpt, n_steps=500)
    ok = post_training_router_canary(model, tokenizer, n_domains=6)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
