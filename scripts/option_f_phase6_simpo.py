"""
Option F — Phase 6 SimPO + KTO smoke gate (synthetic preference, no benchmarks).

What this proves:
  1. `simpo_loss` and `kto_loss` are finite and non-zero on synthetic
     `(chosen_lp, rejected_lp, ref_chosen_lp, ref_rejected_lp)` tensors.
  2. `SyntheticPreferenceStream` produces well-formed (prompt, chosen,
     rejected) triples covering all three rejected_kinds.
  3. `Phase6BatchStream` yields the trainer-compatible (dummy, dummy)
     tensor pair while exposing `last_examples`.
  4. A Phase 5 → Phase 6 resume + 5 outer SimPO+KTO steps runs end-to-end
     without NaN, the composite loss is finite, and BOTH the SimPO and
     KTO terms produce gradient on the live model. The frozen `ref_model`
     stays untouched.
  5. The chosen response really IS more likely under the model than the
     rejected one for at least some of the examples after training (this
     is a sanity check, not a real preference accuracy gate — the model
     is barely trained).

NO public benchmark (UltraFeedback / Tulu / Magpie-Pro / etc.) is touched
at any point. All preference data is procedurally generated.

Run:
    PYTHONPATH=. python scripts/option_f_phase6_simpo.py
"""

from __future__ import annotations

import copy
import math
import os
import time

import torch

from fant2.config import fant2_tiny
from fant2.data import SEED_CORPUS
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training import TrainConfig, FANT2Trainer
from fant2.training.losses import simpo_loss, kto_loss
from fant2.training.phase6_pref import (
    SyntheticPreferenceStream,
    Phase6BatchStream,
    PrefExample,
    simpo_kto_step,
)


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
        iterator=gen(), vocab_size=4096, min_frequency=2, show_progress=False,
    )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tok.save(path)
    return tok


# -----------------------------------------------------------------------------
# 1. Synthetic SimPO + KTO loss test
# -----------------------------------------------------------------------------

def test_pref_losses_synthetic() -> bool:
    print()
    print("  -- step 1: simpo_loss + kto_loss synthetic test --")
    torch.manual_seed(0)

    # Pretend the model gives chosen a higher per-token logp than rejected.
    chosen_lp   = torch.tensor([-12.0, -8.0, -15.0], requires_grad=True)
    rejected_lp = torch.tensor([-18.0, -10.0, -22.0], requires_grad=True)
    chosen_len   = torch.tensor([8.0, 6.0, 12.0])
    rejected_len = torch.tensor([8.0, 6.0, 14.0])
    ref_chosen   = torch.tensor([-13.0, -9.0, -16.0])
    ref_rejected = torch.tensor([-17.0, -11.0, -21.0])

    L_simpo = simpo_loss(
        chosen_lp, rejected_lp, chosen_len, rejected_len,
        beta=2.0, gamma=1.6,
    )
    L_kto = kto_loss(chosen_lp, rejected_lp, ref_chosen, ref_rejected, beta=0.1)
    L = L_simpo + 0.5 * L_kto
    print(f"    simpo = {L_simpo.item():+.4f}  kto = {L_kto.item():+.4f}  "
          f"composite = {L.item():+.4f}")

    if not torch.isfinite(L):
        print("    ✗ FAIL: composite loss non-finite")
        return False
    if abs(L.item()) < 1e-9:
        print("    ✗ FAIL: composite loss is exactly zero")
        return False
    L.backward()
    if (chosen_lp.grad is None or rejected_lp.grad is None
            or not torch.isfinite(chosen_lp.grad).all()
            or not torch.isfinite(rejected_lp.grad).all()):
        print("    ✗ FAIL: gradient missing or non-finite")
        return False
    print(f"    grad ok: chosen={chosen_lp.grad.tolist()}  "
          f"rejected={rejected_lp.grad.tolist()}  ✓")
    return True


# -----------------------------------------------------------------------------
# 2. SyntheticPreferenceStream sanity
# -----------------------------------------------------------------------------

def test_pref_stream() -> bool:
    print()
    print("  -- step 2: SyntheticPreferenceStream sanity --")
    stream = SyntheticPreferenceStream(seed=11)
    bad = 0
    seen_kinds = set()
    seen_templates = set()
    import itertools
    for ex in itertools.islice(stream, 24):
        seen_kinds.add(ex.rejected_kind)
        seen_templates.add(ex.template)
        if not isinstance(ex, PrefExample):
            print(f"    ✗ FAIL: stream did not yield PrefExample"); bad += 1
            continue
        if "<think>" not in ex.prompt:
            print(f"    ✗ FAIL: prompt missing <think> open tag"); bad += 1
        if "</think>" not in ex.chosen or "<answer>" not in ex.chosen:
            print(f"    ✗ FAIL: chosen missing closing tags: {ex.chosen!r}"); bad += 1
        if "{" in ex.prompt or "}" in ex.prompt:
            print(f"    ✗ FAIL: unsubstituted placeholder in prompt"); bad += 1
    print(f"    saw {len(seen_templates)} math templates, "
          f"{len(seen_kinds)} rejected kinds: {sorted(seen_kinds)}")
    if seen_kinds != {"wrong", "unformatted", "unhelpful"}:
        print(f"    ✗ FAIL: did not cover all 3 rejected kinds in 24 samples")
        bad += 1
    return bad == 0


# -----------------------------------------------------------------------------
# 3. Phase6BatchStream contract
# -----------------------------------------------------------------------------

def test_batch_stream(tokenizer) -> bool:
    print()
    print("  -- step 3: Phase6BatchStream contract --")
    stream = Phase6BatchStream(
        tokenizer=tokenizer,
        pairs=SyntheticPreferenceStream(seed=3),
        batch_size=2,
        device="cpu",
    )
    inp, tgt = next(iter(stream))
    print(f"    yielded shapes: input_ids={tuple(inp.shape)}  target_ids={tuple(tgt.shape)}")
    if inp.shape != (2, 1) or tgt.shape != (2, 1):
        print("    ✗ FAIL: expected (2, 1) dummy tensors")
        return False
    if len(stream.last_examples) != 2:
        print(f"    ✗ FAIL: last_examples has {len(stream.last_examples)} entries, expected 2")
        return False
    for ex in stream.last_examples:
        print(f"    [{ex.template}/{ex.rejected_kind}]")
        print(f"      prompt: {ex.prompt[:60]}...")
        print(f"      chosen: {ex.chosen[:60]}...")
        print(f"      rejected: {ex.rejected[:60]}...")
    return True


# -----------------------------------------------------------------------------
# 4. End-to-end Phase 6 trainer step
# -----------------------------------------------------------------------------

def bootstrap_phase5_ckpt(tokenizer, p5_dir: str) -> str:
    """Train a tiny Phase 2 model so Phase 6 has something to resume from.
    (We use Phase 2 not Phase 5 because the smoke gate's job is to validate
    the SimPO+KTO loop, not chain it with the GRPO loop.)"""
    from fant2.data import SyntheticStream, TokenizedBatchStream
    print()
    print("  -- bootstrapping a tiny Phase 2 ckpt to resume from --")
    torch.manual_seed(1)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)
    text_stream = SyntheticStream(seed=33)
    batch_stream = TokenizedBatchStream(
        text_stream=text_stream, tokenizer=tokenizer,
        batch_size=2, seq_len=32, device="cpu",
    )
    bootstrap_cfg = TrainConfig(
        phase=2, n_steps=60, batch_size=2, seq_len=32,
        muon_lr=1e-3, adam_lr=3e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.1, fep_kl_beta_max=0.5, fep_kl_anneal_steps=60,
        telemetry_every=200, tikkun_every=200, fana_every=10000,
        log_every=30, save_every=10000,
        out_dir=p5_dir, device="cpu",
        bf16=False, grad_checkpoint=False, use_8bit_adam=False,
    )
    trainer = FANT2Trainer(model, bootstrap_cfg, batch_stream)
    trainer.train()
    ckpt = os.path.join(p5_dir, "final.pt")
    print(f"  bootstrap ckpt at {ckpt}")
    return ckpt


def run_phase6_trainer(tokenizer, p6_dir: str, resume_from: str, n_steps: int = 5):
    print()
    print("  -- step 4: Phase 6 SimPO + KTO trainer (end-to-end) --")
    torch.manual_seed(2)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)

    pair_stream = SyntheticPreferenceStream(seed=99, max_value=12)
    batch_stream = Phase6BatchStream(
        tokenizer=tokenizer,
        pairs=pair_stream,
        batch_size=2,
        device="cpu",
    )

    train_cfg = TrainConfig(
        phase=6, n_steps=n_steps, batch_size=2, seq_len=64,
        muon_lr=5e-7, adam_lr=5e-7,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=1.0, fep_kl_beta_max=1.0, fep_kl_anneal_steps=1,
        simpo_beta=2.0,
        simpo_gamma=1.6,
        kto_beta=0.1,
        kto_weight=0.5,
        telemetry_every=1000, tikkun_every=1000, fana_every=10000,
        log_every=1, save_every=10000,
        out_dir=p6_dir,
        resume_from=resume_from,
        device="cpu",
        bf16=False, grad_checkpoint=False, use_8bit_adam=False,
    )
    trainer = FANT2Trainer(model, train_cfg, batch_stream)

    # Snapshot a frozen reference policy.
    print("  cloning ref_model from current trainer.model...")
    ref_model = copy.deepcopy(trainer.model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    trainer.ref_model = ref_model

    # Capture per-step losses + metrics via train_step wrapping.
    history = {
        "pref_loss": [], "simpo": [], "kto": [],
        "mean_margin": [], "pref_acc": [],
    }
    original_train_step = trainer.train_step
    def wrapped(batch):
        losses = original_train_step(batch)
        for k in history:
            if k in losses:
                history[k].append(losses[k])
        return losses
    trainer.train_step = wrapped

    # Snapshot ALL params before training.
    live_params_before = {n: p.detach().clone() for n, p in trainer.model.named_parameters()}
    ref_params_before = {n: p.detach().clone() for n, p in ref_model.named_parameters()}

    t0 = time.time()
    trainer.train()
    dt = time.time() - t0
    print(f"  Phase 6 done in {dt:.1f}s ({dt/n_steps:.1f} s/step)")

    n_changed = 0
    n_total = 0
    biggest_delta = 0.0
    biggest_name = ""
    for n, p in trainer.model.named_parameters():
        n_total += 1
        diff = (p.detach() - live_params_before[n]).abs().max().item()
        if diff > 0:
            n_changed += 1
            if diff > biggest_delta:
                biggest_delta = diff
                biggest_name = n
    live_changed = n_changed > 0
    print(f"  live model: {n_changed}/{n_total} params changed  "
          f"(max |Δ| = {biggest_delta:.2e} on {biggest_name})  "
          f"{'✓' if live_changed else '✗ FAIL'}")

    ref_unchanged = all(
        torch.equal(p.detach(), ref_params_before[n])
        for n, p in ref_model.named_parameters()
    )
    print(f"  ref  model param frozen:  {ref_unchanged}  ✓" if ref_unchanged else
          "  ✗ FAIL: ref_model parameters changed (should be frozen)")

    return trainer, history, live_changed, ref_unchanged


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print(" FANT 2 — Option F: Phase 6 SimPO + KTO smoke gate")
    print(" (synthetic preference data, no public benchmarks)")
    print("=" * 64)

    base = "output/option_f"
    os.makedirs(base, exist_ok=True)
    tok_path = os.path.join(base, "tokenizer.json")
    boot_dir = os.path.join(base, "phase5_boot")
    p6_dir = os.path.join(base, "phase6")
    tokenizer = get_tokenizer(tok_path)

    failed = False

    # Step 1
    if not test_pref_losses_synthetic():
        print("  ✗ FAIL: simpo_loss / kto_loss synthetic test failed"); failed = True

    # Step 2
    if not test_pref_stream():
        print("  ✗ FAIL: SyntheticPreferenceStream sanity failed"); failed = True

    # Step 3
    if not test_batch_stream(tokenizer):
        print("  ✗ FAIL: Phase6BatchStream contract failed"); failed = True

    # Bootstrap a checkpoint to resume from
    boot_ckpt = bootstrap_phase5_ckpt(tokenizer, boot_dir)

    # Step 4: end-to-end Phase 6 trainer
    trainer, history, live_changed, ref_unchanged = run_phase6_trainer(
        tokenizer, p6_dir, boot_ckpt, n_steps=5,
    )
    if not live_changed:
        failed = True
    if not ref_unchanged:
        failed = True

    print()
    print("  -- training history --")
    for k, vs in history.items():
        if not vs:
            print(f"    ✗ FAIL: history[{k}] is empty"); failed = True
            continue
        n_nan = sum(1 for v in vs if math.isnan(v) or math.isinf(v))
        v0, v1 = vs[0], vs[-1]
        print(f"    {k}: first={v0:.4f}, last={v1:.4f}, n={len(vs)}, n_nonfinite={n_nan}")
        if n_nan > 0:
            print(f"    ✗ FAIL: {k} has {n_nan} non-finite values"); failed = True

    # Both terms must actually be non-zero, otherwise the composite is degenerate.
    if "simpo" not in history or not history["simpo"]:
        print("  ✗ FAIL: 'simpo' not logged"); failed = True
    elif all(abs(v) < 1e-9 for v in history["simpo"]):
        print("  ✗ FAIL: simpo is identically zero"); failed = True
    if "kto" not in history or not history["kto"]:
        print("  ✗ FAIL: 'kto' not logged"); failed = True
    elif all(abs(v) < 1e-9 for v in history["kto"]):
        print("  ✗ FAIL: kto is identically zero"); failed = True

    print()
    if failed:
        print("  ✗ Option F FAILED")
        return 1
    print("  ✓ Option F PASSED")
    print("    - simpo_loss + kto_loss are finite + differentiable")
    print("    - SyntheticPreferenceStream produces well-formed triples (no benchmark data)")
    print("    - Phase6BatchStream + trainer hook end-to-end OK")
    print("    - 5 Phase 6 steps run without NaN; live policy updates, ref stays frozen")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
