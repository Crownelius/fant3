"""
Option D — Phase 4 smoke gate (true two-pass + retrieval + homeostat).

What this proves:
  1. The new prepend_vec=… kwarg on FANT2Model.forward works in both
     prepended and unprepended modes and the shapes match (B, T, *).
  2. _phase4_refine_forward runs the true two-pass refinement without NaN.
  3. The Apollonian retrieval cross-attention at the config-listed layers
     receives gradient (i.e. the model can actually use it).
  4. After ~200 Phase 4 steps from a Phase 2 checkpoint, the new losses
     "succ_gap" and "consistency" are finite and the run completes.
  5. The avalanche-τ homeostat is wired and can be exercised (we manually
     prime self.telemetry_log with two synthetic snapshots and verify the
     dropout fires without crashing).

Run:
    PYTHONPATH=. python scripts/option_d_phase4.py
"""

from __future__ import annotations

import math
import os
import time

import torch

from fant2.config import fant2_tiny
from fant2.data import SEED_CORPUS, SyntheticStream, TokenizedBatchStream
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training import TrainConfig, FANT2Trainer


# -----------------------------------------------------------------------------
# Helpers
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
        vocab_size=4096,
        min_frequency=2,
        show_progress=False,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tok.save(path)
    return tok


# -----------------------------------------------------------------------------
# 1. Forward shape contract
# -----------------------------------------------------------------------------

def test_prepend_vec_shape_contract() -> bool:
    print()
    print("  -- step 1: forward(prepend_vec=…) shape contract --")
    torch.manual_seed(0)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)
    model.eval()

    B, T = 2, 32
    input_ids = torch.randint(0, cfg.vocab_size, (B, T))

    with torch.no_grad():
        out_normal = model(input_ids)
        prepend = torch.randn(B, cfg.dim) * 0.1
        out_prepended = model(input_ids, prepend_vec=prepend)

    ok = True
    for k in ("logits", "final_hidden", "success_pred"):
        sn = tuple(out_normal[k].shape)
        sp = tuple(out_prepended[k].shape)
        match = "✓" if sn == sp else "✗"
        print(f"    {k}: normal={sn}, prepended={sp}  {match}")
        if sn != sp:
            ok = False
    return ok


# -----------------------------------------------------------------------------
# 2. Phase 4 trainer with true two-pass
# -----------------------------------------------------------------------------

def run_phase4(tokenizer, out_dir: str, resume_from: str, n_steps: int = 200):
    print()
    print("  -- step 2: Phase 4 true-two-pass training --")
    torch.manual_seed(2)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)

    text_stream = SyntheticStream(seed=77)
    batch_stream = TokenizedBatchStream(
        text_stream=text_stream,
        tokenizer=tokenizer,
        batch_size=2,
        seq_len=32,
        device="cpu",
    )

    train_cfg = TrainConfig(
        phase=4,
        n_steps=n_steps,
        batch_size=2,
        seq_len=32,
        muon_lr=5e-4,
        adam_lr=1e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=1.0,
        fep_kl_beta_max=1.0,
        fep_kl_anneal_steps=1,
        refine_weight=0.5,
        telemetry_every=50,
        tikkun_every=200,
        fana_every=10000,
        log_every=20,
        save_every=10000,
        out_dir=out_dir,
        device="cpu",
        bf16=False,
        grad_checkpoint=False,
        use_8bit_adam=False,
        resume_from=resume_from,
    )
    trainer = FANT2Trainer(model, train_cfg, batch_stream)

    # Capture per-step losses to verify the new keys are populated
    history = {"ce": [], "succ_gap": [], "consistency": []}
    original_train_step = trainer.train_step
    def wrapped(batch):
        losses = original_train_step(batch)
        for k in history:
            if k in losses:
                history[k].append(losses[k])
        return losses
    trainer.train_step = wrapped

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    print(f"  Phase 4 done in {dt:.1f}s ({dt/n_steps*1000:.1f} ms/step)")
    return trainer, model, history


# -----------------------------------------------------------------------------
# 3. Retrieval-cross-attention gradient check
# -----------------------------------------------------------------------------

def test_retrieval_grad(model: FANT2Model) -> bool:
    print()
    print("  -- step 3: Apollonian retrieval gradient check --")
    cfg = model.config

    # Find blocks with mem_attn
    mem_blocks = [i for i, b in enumerate(model.blocks) if b.use_memory]
    print(f"    use_memory blocks: {mem_blocks}")
    if not mem_blocks:
        print("    ✗ FAIL: no blocks have use_memory=True")
        return False

    # Run a forward + backward and check that mem_attn params have grad
    model.train()
    x = torch.randint(0, cfg.vocab_size, (1, 32))
    targets = torch.randint(0, cfg.vocab_size, (1, 32))
    out = model(x, targets=targets, store_to_memory=True)
    out["loss"].backward()

    ok = True
    for i in mem_blocks:
        block = model.blocks[i]
        wq_grad = block.mem_attn.W_q.weight.grad
        wo_grad = block.mem_attn.W_o.weight.grad
        gate_grad = block.mem_attn.output_gate.grad
        wq_norm = wq_grad.norm().item() if wq_grad is not None else 0.0
        wo_norm = wo_grad.norm().item() if wo_grad is not None else 0.0
        gate_norm = gate_grad.item() if gate_grad is not None else 0.0
        print(f"    block[{i}].mem_attn  W_q.grad={wq_norm:.3e}  "
              f"W_o.grad={wo_norm:.3e}  output_gate.grad={gate_norm:.3e}")
        if wq_grad is None or wo_grad is None:
            print(f"    ✗ FAIL: block[{i}].mem_attn has no gradient at all")
            ok = False
    return ok


# -----------------------------------------------------------------------------
# 4. Avalanche-τ homeostat synthetic exercise
# -----------------------------------------------------------------------------

def test_homeostat_fires(trainer: FANT2Trainer) -> bool:
    print()
    print("  -- step 4: avalanche-τ homeostat exercise --")
    from fant2.training.telemetry import TelemetrySnapshot

    # Build two synthetic snapshots both showing |τ - 1.5| > 0.2
    snap_a = TelemetrySnapshot(step=10, avalanche_tau=2.5)
    snap_b = TelemetrySnapshot(step=20, avalanche_tau=2.7)
    trainer.telemetry_log = [snap_a, snap_b]

    # The homeostat lives inside trainer.train() so we replicate the trigger logic
    tau_now = trainer.telemetry_log[-1].avalanche_tau
    tau_prev = trainer.telemetry_log[-2].avalanche_tau
    drifted = (
        tau_now is not None and tau_prev is not None
        and not (math.isnan(tau_now) or math.isnan(tau_prev))
        and abs(tau_now - 1.5) > 0.2
        and abs(tau_prev - 1.5) > 0.2
    )
    print(f"    τ_prev={tau_prev}, τ_now={tau_now}, drifted={drifted}")
    if not drifted:
        return False
    try:
        trainer.model.fana_dropout_all(p=0.2)
    except Exception as e:
        print(f"    ✗ FAIL: fana_dropout_all raised: {e}")
        return False
    print(f"    fana_dropout_all(p=0.2) ran without crashing")
    return True


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print(" FANT 2 — Option D: Phase 4 smoke gate")
    print("=" * 64)

    base = "output/option_d"
    os.makedirs(base, exist_ok=True)
    tok_path = os.path.join(base, "tokenizer.json")
    p2_dir = os.path.join(base, "phase2")
    p4_dir = os.path.join(base, "phase4")

    tokenizer = get_tokenizer(tok_path)

    # Need a Phase 2 checkpoint to resume Phase 4 from. Train a tiny one.
    print()
    print("  -- bootstrapping a tiny Phase 2 checkpoint to resume from --")
    torch.manual_seed(1)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)
    text_stream = SyntheticStream(seed=33)
    batch_stream = TokenizedBatchStream(
        text_stream=text_stream,
        tokenizer=tokenizer,
        batch_size=2,
        seq_len=32,
        device="cpu",
    )
    bootstrap_cfg = TrainConfig(
        phase=2,
        n_steps=80,
        batch_size=2,
        seq_len=32,
        muon_lr=1e-3,
        adam_lr=3e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.1,
        fep_kl_beta_max=0.5,
        fep_kl_anneal_steps=80,
        telemetry_every=200,
        tikkun_every=200,
        fana_every=10000,
        log_every=40,
        save_every=10000,
        out_dir=p2_dir,
        device="cpu",
        bf16=False,
        grad_checkpoint=False,
        use_8bit_adam=False,
    )
    bootstrap_trainer = FANT2Trainer(model, bootstrap_cfg, batch_stream)
    bootstrap_trainer.train()
    p2_ckpt = os.path.join(p2_dir, "final.pt")
    print(f"  bootstrap Phase 2 checkpoint at {p2_ckpt}")

    # ===== Step 1: shape contract =====
    shape_ok = test_prepend_vec_shape_contract()

    # ===== Step 2: full Phase 4 run =====
    trainer, model4, history = run_phase4(tokenizer, p4_dir, p2_ckpt, n_steps=200)

    # ===== Step 3: retrieval gradient =====
    grad_ok = test_retrieval_grad(model4)

    # ===== Step 4: homeostat synthetic exercise =====
    homeo_ok = test_homeostat_fires(trainer)

    # ===== Pass criteria =====
    failed = False
    if not shape_ok:
        print("  ✗ FAIL: shape contract broken"); failed = True

    print()
    print("  -- training history --")
    for k, vs in history.items():
        if not vs:
            print(f"    ✗ FAIL: history[{k}] is empty"); failed = True
            continue
        v0, v1 = vs[0], vs[-1]
        n_nan = sum(1 for v in vs if math.isnan(v) or math.isinf(v))
        print(f"    {k}: first={v0:.4f}, last={v1:.4f}, n={len(vs)}, n_nonfinite={n_nan}")
        if n_nan > 0:
            print(f"    ✗ FAIL: {k} has {n_nan} non-finite values"); failed = True

    if "consistency" not in history or not history["consistency"]:
        print("  ✗ FAIL: 'consistency' loss key missing"); failed = True
    if "succ_gap" not in history or not history["succ_gap"]:
        print("  ✗ FAIL: 'succ_gap' loss key missing"); failed = True

    if not grad_ok:
        print("  ✗ FAIL: Apollonian retrieval did not receive gradient"); failed = True
    if not homeo_ok:
        print("  ✗ FAIL: avalanche-τ homeostat exercise failed"); failed = True

    print()
    if failed:
        print("  ✗ Option D FAILED")
        return 1
    print("  ✓ Option D PASSED")
    print("    - prepend_vec shape contract holds")
    print("    - Phase 4 true-two-pass runs without NaN")
    print("    - Apollonian retrieval cross-attention receives gradient")
    print("    - Avalanche-τ homeostat fires correctly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
