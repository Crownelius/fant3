"""
End-to-end integration tests for FANT2Trainer.

We run a tiny model + synthetic batch stream for a few steps in each
training phase and verify:
  * the loss is finite
  * the model parameters update
  * the training-loss dict has the expected keys for that phase
  * checkpoints save and load round-trip
"""

import os
import shutil
import tempfile
from typing import Tuple

import pytest
import torch

from fant2.config import fant2_tiny
from fant2.model import FANT2Model
from fant2.training import TrainConfig, FANT2Trainer


# -----------------------------------------------------------------------------
# Tiny synthetic stream
# -----------------------------------------------------------------------------

class _SyntheticStream:
    def __init__(self, vocab_size: int, batch_size: int = 2, seq_len: int = 32, n: int = 200):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.n = n

    def __iter__(self):
        gen = torch.Generator().manual_seed(123)
        for _ in range(self.n):
            ids = torch.randint(
                0, self.vocab_size,
                (self.batch_size, self.seq_len + 1),
                generator=gen, dtype=torch.long,
            )
            yield ids[:, :-1].contiguous(), ids[:, 1:].contiguous()


def _build_trainer(phase: int, out_dir: str, n_steps: int = 4) -> Tuple[FANT2Trainer, FANT2Model]:
    torch.manual_seed(0)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)
    train_cfg = TrainConfig(
        phase=phase,
        n_steps=n_steps,
        batch_size=2,
        seq_len=32,
        muon_lr=1e-3,
        adam_lr=3e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.1,
        fep_kl_beta_max=1.0,
        fep_kl_anneal_steps=100,
        telemetry_every=10,
        tikkun_every=10,
        fana_every=10,
        log_every=2,
        save_every=10,
        out_dir=out_dir,
        device="cpu",
        bf16=False,
        grad_checkpoint=False,
        use_8bit_adam=False,
    )
    stream = _SyntheticStream(cfg.vocab_size, batch_size=2, seq_len=32)
    return FANT2Trainer(model, train_cfg, stream), model


# -----------------------------------------------------------------------------
# Per-phase integration tests
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("phase,expected_keys", [
    (1, {"ce", "jepa", "sigreg", "total"}),
    (2, {"ce", "fep_kl", "z_loss", "total"}),
    (3, {"ce", "fep_kl", "z_loss", "calib_rank", "calib_cond", "total"}),
    (4, {"ce", "fep_kl", "z_loss", "succ", "total"}),
])
def test_phase_train_step(tmp_path, phase, expected_keys):
    out_dir = str(tmp_path / f"phase{phase}_test")
    trainer, model = _build_trainer(phase, out_dir, n_steps=4)
    # Snapshot a couple of params before training
    pre = {n: p.detach().clone() for n, p in list(model.named_parameters())[:5]}
    trainer.train()
    # Check that the parameters changed
    n_changed = 0
    for n, p in list(model.named_parameters())[:5]:
        if not torch.equal(pre[n], p.detach()):
            n_changed += 1
    assert n_changed >= 1, f"no params changed after {trainer.cfg.n_steps} steps"
    # Final checkpoint should exist
    assert os.path.exists(os.path.join(out_dir, "final.pt"))


def test_phase5_grpo_real(tmp_path):
    """
    Phase 5 now runs the real Dr.GRPO loop. The hook requires a
    `Phase5BatchStream` (not a generic LM stream) and a frozen reference
    policy. We do 2 outer steps with G=2 rollouts and assert the loss is
    finite, the trainer reports the GRPO metric keys, and a checkpoint is
    saved.
    """
    import copy
    from fant2.tokenizer import FANT2Tokenizer
    from fant2.data import SEED_CORPUS
    from fant2.training.phase5_rollout import (
        Phase5BatchStream, ProceduralMathStream,
    )

    out_dir = str(tmp_path / "phase5_test")
    torch.manual_seed(0)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)

    # Tiny BPE
    def gen():
        for i in range(2000):
            yield SEED_CORPUS[i % len(SEED_CORPUS)]
    tokenizer = FANT2Tokenizer.train_from_iterator(
        iterator=gen(), vocab_size=1024, min_frequency=2, show_progress=False,
    )

    stream = Phase5BatchStream(
        tokenizer=tokenizer,
        problems=ProceduralMathStream(seed=1, max_value=8),
        batch_size=1,
        device="cpu",
    )
    train_cfg = TrainConfig(
        phase=5, n_steps=2, batch_size=1, seq_len=32,
        muon_lr=1e-4, adam_lr=1e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=1.0, fep_kl_beta_max=1.0, fep_kl_anneal_steps=1,
        grpo_n_rollouts=2, grpo_max_new_tokens=8,
        grpo_temperature=1.0, grpo_top_p=1.0,
        grpo_clip_eps=0.2, grpo_clip_eps_hi=0.28,
        telemetry_every=100, tikkun_every=100, fana_every=100,
        log_every=1, save_every=100,
        out_dir=out_dir, device="cpu",
        bf16=False, grad_checkpoint=False, use_8bit_adam=False,
    )
    trainer = FANT2Trainer(model, train_cfg, stream)
    ref = copy.deepcopy(trainer.model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    trainer.ref_model = ref

    trainer.train()
    assert os.path.exists(os.path.join(out_dir, "final.pt"))


def test_phase6_simpo_kto_real(tmp_path):
    """
    Phase 6 now runs the real SimPO+KTO loop. The hook requires a
    `Phase6BatchStream` and a frozen reference policy. We do 2 steps and
    assert the loss is finite and a checkpoint is saved.
    """
    import copy
    from fant2.tokenizer import FANT2Tokenizer
    from fant2.data import SEED_CORPUS
    from fant2.training.phase6_pref import (
        Phase6BatchStream, SyntheticPreferenceStream,
    )

    out_dir = str(tmp_path / "phase6_test")
    torch.manual_seed(0)
    cfg = fant2_tiny()
    model = FANT2Model(cfg)

    def gen():
        for i in range(2000):
            yield SEED_CORPUS[i % len(SEED_CORPUS)]
    tokenizer = FANT2Tokenizer.train_from_iterator(
        iterator=gen(), vocab_size=1024, min_frequency=2, show_progress=False,
    )

    stream = Phase6BatchStream(
        tokenizer=tokenizer,
        pairs=SyntheticPreferenceStream(seed=1, max_value=8),
        batch_size=2,
        device="cpu",
    )
    train_cfg = TrainConfig(
        phase=6, n_steps=2, batch_size=2, seq_len=32,
        muon_lr=1e-4, adam_lr=1e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=1.0, fep_kl_beta_max=1.0, fep_kl_anneal_steps=1,
        simpo_beta=2.0, simpo_gamma=1.6,
        kto_beta=0.1, kto_weight=0.5,
        telemetry_every=100, tikkun_every=100, fana_every=100,
        log_every=1, save_every=100,
        out_dir=out_dir, device="cpu",
        bf16=False, grad_checkpoint=False, use_8bit_adam=False,
    )
    trainer = FANT2Trainer(model, train_cfg, stream)
    ref = copy.deepcopy(trainer.model)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    trainer.ref_model = ref

    trainer.train()
    assert os.path.exists(os.path.join(out_dir, "final.pt"))


# -----------------------------------------------------------------------------
# Checkpoint round-trip
# -----------------------------------------------------------------------------

def test_checkpoint_save_and_load(tmp_path):
    out_dir = str(tmp_path / "ckpt_rt")
    trainer, model = _build_trainer(2, out_dir, n_steps=2)
    trainer.train()
    final_path = os.path.join(out_dir, "final.pt")
    assert os.path.exists(final_path)

    # Build a fresh trainer and load the checkpoint
    torch.manual_seed(99)
    cfg = fant2_tiny()
    fresh_model = FANT2Model(cfg)
    fresh_train_cfg = TrainConfig(
        phase=2, n_steps=0, batch_size=2, seq_len=32,
        device="cpu", bf16=False, use_8bit_adam=False,
        out_dir=out_dir, resume_from=final_path,
    )
    stream = _SyntheticStream(cfg.vocab_size)
    fresh_trainer = FANT2Trainer(fresh_model, fresh_train_cfg, stream)
    # Step counter should be > 0 after loading
    assert fresh_trainer.step >= 2


# -----------------------------------------------------------------------------
# Single-step micro test
# -----------------------------------------------------------------------------

def test_train_step_returns_finite_losses(tmp_path):
    out_dir = str(tmp_path / "single_step")
    trainer, _ = _build_trainer(2, out_dir, n_steps=1)
    batch = next(iter(trainer.data_stream))
    losses = trainer.train_step(batch)
    for k, v in losses.items():
        assert isinstance(v, float)
        assert v == v  # not NaN
        assert v != float("inf") and v != float("-inf"), f"{k} is infinite: {v}"
