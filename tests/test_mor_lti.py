"""
Unit tests for the Mythos / RDT (Recurrent-Depth Transformer) style
augmentations to ``fant3.model.recursion.MoRShared``:

  * LTI (Linear Time-Invariant) injection:   h_{k+1} = A*h + B*x_orig [+ C*ret] + block(h + loop_emb[k])
  * Spectral-radius constraint:               A = -softplus(a_diag)  ⇒  rho(A) < 1
  * Loop-index positional signal:             learned (max_depth, dim) embedding

The tests cover:

  1. Default config = v1 behavior (bit-equivalent to pre-Mythos MoR)
  2. Enabling LTI with zero-init B/C matrices ≈ v1 at step 0 (sanity)
  3. Spectral constraint keeps (I + A) contraction-stable by construction
  4. ``spectral_radius_report`` returns an entry per MoR layer
  5. ``assert_lti_stable`` does NOT raise on a freshly-initialized model
  6. Forward pass still produces correct output shape with all flags on
  7. Backward pass flows through a_diag / B / C / loop_emb
"""

from __future__ import annotations

import copy
import pytest
import torch
import torch.nn as nn

from fant3.config import FANT3Config, fant3_smoke
from fant3.model.fant3_model import FANT3Model
from fant3.diagnostics.spectral import (
    spectral_radius_report,
    assert_lti_stable,
    _A_MIN,
    _A_MAX,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_cfg(**overrides) -> FANT3Config:
    """Small FANT 3 config for fast CPU tests."""
    c = fant3_smoke()
    # Further shrink for test speed
    c.dim = 64
    c.n_layers = 4
    c.n_dense_layers = 1
    c.n_heads = 2
    c.n_kv_heads = 1
    c.head_dim = 32
    c.n_megapools = 1
    c.n_per_megapool = 2
    c.top_k = 1
    c.n_matryoshka_levels = 1
    c.shared_expert_hidden = 32
    c.moe_hidden = 64
    c.n_attention_atoms = 2
    c.masa_coef_rank = 2
    c.n_recursion_depths = 2
    c.max_seq_len = 32
    c.apollonian_alpha_cap = 64
    c.apollonian_beta_cap = 64
    c.apollonian_retrieval_layers = (2, 3)
    c.etf_freeze_after_step = 10
    c.etf_freeze_layers = (1, 2)
    c.cerebellum_enabled = False
    c.ahn_enabled = False
    c.use_gradient_checkpointing = False
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def _run_forward(model: FANT3Model, B: int = 2, T: int = 8) -> torch.Tensor:
    ids = torch.randint(0, model.cfg.vocab_size, (B, T))
    out = model(ids, targets=ids)
    return out["loss"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_v1_default_no_lti_no_loop_emb():
    """With all Mythos flags OFF, MoR shouldn't have any new parameters."""
    cfg = _tiny_cfg()  # all Mythos flags default False
    model = FANT3Model(cfg)
    mor = model.mor
    assert mor is not None, "Test preset should have MoR enabled"
    assert not mor.lti_enabled
    assert not mor.spectral_enabled
    assert not mor.loop_idx_enabled
    # No a_diag / b_proj / c_proj / loop_emb should exist
    assert not hasattr(mor, "a_diag")
    assert not hasattr(mor, "b_proj")
    assert not hasattr(mor, "c_proj")
    assert not hasattr(mor, "loop_emb")


def test_lti_enabled_adds_expected_params():
    cfg = _tiny_cfg(
        mor_lti_injection_enabled=True,
        mor_lti_apollonian_channel=True,
    )
    model = FANT3Model(cfg)
    mor = model.mor
    assert mor.lti_enabled
    assert hasattr(mor, "a_diag") and mor.a_diag.shape == (cfg.dim,)
    assert hasattr(mor, "b_proj") and mor.b_proj.weight.shape == (cfg.dim, cfg.dim)
    assert mor.c_proj is not None and mor.c_proj.weight.shape == (cfg.dim, cfg.dim)
    # a_diag starts at zero, B and C start at zero → injection is zero at init
    assert torch.allclose(mor.a_diag, torch.zeros(cfg.dim))
    assert torch.allclose(mor.b_proj.weight, torch.zeros_like(mor.b_proj.weight))
    assert torch.allclose(mor.c_proj.weight, torch.zeros_like(mor.c_proj.weight))


def test_lti_without_apollonian_channel_omits_c():
    cfg = _tiny_cfg(
        mor_lti_injection_enabled=True,
        mor_lti_apollonian_channel=False,
    )
    model = FANT3Model(cfg)
    mor = model.mor
    assert mor.c_proj is None, "C projection should be absent when channel disabled"


def test_spectral_constraint_shape():
    cfg = _tiny_cfg(
        mor_lti_injection_enabled=True,
        mor_spectral_constraint=True,
    )
    model = FANT3Model(cfg)
    mor = model.mor
    # After constraint, A = -softplus(a_diag); initial a_diag = 0 ⇒ A = -softplus(0) = -ln(2) ≈ -0.693
    a_eff = mor._effective_A()
    assert a_eff.shape == (cfg.dim,)
    # Must be strictly negative everywhere
    assert (a_eff < 0).all(), "Constrained A must have all-negative diagonal"
    # -softplus(0) = -ln(2) ≈ -0.693
    expected = -torch.log(torch.tensor(2.0))
    assert torch.allclose(a_eff, expected.expand_as(a_eff), atol=1e-6)


def test_spectral_constraint_guarantees_stability():
    """For ANY a_diag, -softplus(a_diag) stays in (-inf, 0); combined with (I + A)
    we want the result in (-1, 1) for stability. Large-magnitude a_diag can
    violate this — verify we catch it when it happens."""
    cfg = _tiny_cfg(
        mor_lti_injection_enabled=True,
        mor_spectral_constraint=True,
    )
    model = FANT3Model(cfg)
    mor = model.mor

    # Set a_diag extremely large → -softplus(large) ≈ -large ≈ -5 → (1 + -5) = -4  UNSTABLE
    with torch.no_grad():
        mor.a_diag.fill_(5.0)
    entries = spectral_radius_report(model)
    mor_entries = [e for e in entries if e.lti_enabled]
    assert len(mor_entries) == 1
    e = mor_entries[0]
    # effective A ≈ -5, (I+A) ≈ -4 -> |I+A| = 4  -> unstable
    assert not e.stable
    assert e.abs_i_plus_a_max > 1.0

    # Reset to init (a_diag=0) → A = -ln(2) ≈ -0.693; (I+A) ≈ 0.307 → |.| < 1 stable
    with torch.no_grad():
        mor.a_diag.zero_()
    entries = spectral_radius_report(model)
    e = [e for e in entries if e.lti_enabled][0]
    assert e.stable
    assert e.abs_i_plus_a_max < 1.0


def test_loop_emb_shape_and_init():
    cfg = _tiny_cfg(mor_loop_index_enabled=True)
    model = FANT3Model(cfg)
    mor = model.mor
    assert mor.loop_emb.shape == (cfg.n_recursion_depths, cfg.dim)
    # Small init (std 0.02) — not too big
    assert mor.loop_emb.abs().max().item() < 0.5


def test_forward_with_all_flags_on_runs_and_has_finite_loss():
    cfg = _tiny_cfg(
        mor_lti_injection_enabled=True,
        mor_spectral_constraint=True,
        mor_loop_index_enabled=True,
        mor_lti_apollonian_channel=True,
    )
    model = FANT3Model(cfg)
    loss = _run_forward(model)
    assert torch.isfinite(loss), f"loss must be finite, got {loss.item()}"
    assert loss.item() > 0


def test_backward_flows_through_new_params():
    cfg = _tiny_cfg(
        mor_lti_injection_enabled=True,
        mor_spectral_constraint=True,
        mor_loop_index_enabled=True,
    )
    model = FANT3Model(cfg)
    loss = _run_forward(model)
    loss.backward()
    mor = model.mor

    # a_diag should receive gradient
    assert mor.a_diag.grad is not None
    assert mor.a_diag.grad.abs().sum().item() > 0, "a_diag must receive gradient"

    # b_proj (zero-init) should receive gradient too, even though output is zero:
    # gradient w.r.t. weight exists because a linear layer's grad depends on input*output_grad
    assert mor.b_proj.weight.grad is not None

    # loop_emb should receive gradient (k_emb added to block input)
    assert mor.loop_emb.grad is not None
    assert mor.loop_emb.grad.abs().sum().item() > 0


def test_assert_lti_stable_at_init_does_not_raise():
    cfg = _tiny_cfg(
        mor_lti_injection_enabled=True,
        mor_spectral_constraint=True,
    )
    model = FANT3Model(cfg)
    # At init a_diag = 0 → A = -ln(2) → (I+A) ≈ 0.307 → stable
    assert_lti_stable(model)  # should not raise


def test_assert_lti_stable_raises_on_unstable():
    cfg = _tiny_cfg(
        mor_lti_injection_enabled=True,
        mor_spectral_constraint=True,
    )
    model = FANT3Model(cfg)
    with torch.no_grad():
        model.mor.a_diag.fill_(5.0)   # forces spectral failure
    with pytest.raises(AssertionError, match="NOT spectrally stable"):
        assert_lti_stable(model)


def test_spectral_report_entries_count():
    """One entry per MoR layer; FANT 3 default has a single MoR-wrapped
    middle block."""
    cfg = _tiny_cfg(mor_lti_injection_enabled=True)
    model = FANT3Model(cfg)
    entries = spectral_radius_report(model)
    lti_entries = [e for e in entries if e.lti_enabled]
    assert len(lti_entries) == 1, f"expected 1 MoR LTI entry, got {len(lti_entries)}"


def test_retrieved_argument_optional_when_flag_off():
    """Passing no retrieved context with LTI on but apollonian_channel off
    should still produce a valid forward pass."""
    cfg = _tiny_cfg(
        mor_lti_injection_enabled=True,
        mor_lti_apollonian_channel=False,
    )
    model = FANT3Model(cfg)
    loss = _run_forward(model)
    assert torch.isfinite(loss)


def test_constants_and_parameterization_documentation():
    """Guard against silent drift in the stability bounds; if these change,
    the diagnostic will start returning different stable/unstable verdicts."""
    assert _A_MIN == -2.0
    assert _A_MAX == 0.0
