"""
Smoke tests for two ISRM-derived training improvements to FANT 3's MoR:

  1. Contractive decay (mor_isrm_contractive)
     – The alpha schedule is provably decreasing: alpha_k → 0.
     – Forward / backward work at any K.

  2. Monotonic improvement loss
     – Penalises any MoR pass whose CE is worse than the previous pass.
     – Zero when losses are non-increasing (good trajectory).
     – Positive and differentiable when any pass regresses.

  3. Dynamic-K smoke
     – Forward succeeds for K ∈ {1 … n_recursion_depths}.
     – Output changes with K (K actually matters).
     – inference_k_override > n_recursion_depths both runs AND changes output,
       verifying the implementation's K-extrapolation path is not a silent
       no-op. (See recursion.py: when k_cap > max_depth we promote depth to
       k_cap so extra passes actually write to `current`.)

Named: monotonic_dynamic-k
"""

from __future__ import annotations

import math
from typing import List

import pytest
import torch
import torch.nn.functional as F

from fant3.config import FANT3Config, fant3_smoke
from fant3.model.fant3_model import FANT3Model


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _tiny_cfg(**overrides) -> FANT3Config:
    c = fant3_smoke()
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
    c.n_recursion_depths = 3      # 3 so we can test k=1,2,3
    c.max_seq_len = 32
    c.apollonian_alpha_cap = 32
    c.apollonian_beta_cap = 32
    c.apollonian_retrieval_layers = (2, 3)
    c.etf_freeze_after_step = 10
    c.etf_freeze_layers = (1, 2)
    c.cerebellum_enabled = False
    c.ahn_enabled = False
    c.use_gradient_checkpointing = False
    for k, v in overrides.items():
        setattr(c, k, v)
    return c


def _ids(cfg: FANT3Config, B: int = 2, T: int = 16) -> torch.Tensor:
    return torch.randint(0, cfg.vocab_size, (B, T))


def _alpha(k: int) -> float:
    """ISRM contractive decay schedule (same formula as recursion.py)."""
    return (0.15 / (1.0 + 0.15 * k)) * (0.97 ** k)


def compute_monotonic_ce_loss(step_losses: List[torch.Tensor]) -> torch.Tensor:
    """
    Penalise any MoR pass that regresses CE vs the previous pass.

    step_losses: list of scalar tensors, one per MoR pass (1..K).
    Returns a non-negative scalar loss; zero iff losses are non-increasing.

    Preserves the autograd graph even in the single-step and no-violation
    cases, so callers can safely `.backward()` through the result.

    Standalone prototype — not yet wired into FANT3Model.forward.
    """
    if len(step_losses) < 2:
        return step_losses[0] * 0.0  # preserves the graph
    total = step_losses[0] * 0.0
    for i in range(1, len(step_losses)):
        violation = F.relu(step_losses[i] - step_losses[i - 1])
        total = total + violation ** 2
    return total / (len(step_losses) - 1)


# ---------------------------------------------------------------------------
# 1. Contractive alpha schedule
# ---------------------------------------------------------------------------

class TestContractiveAlphaSchedule:
    def test_alpha_strictly_decreasing(self):
        """alpha_1 > alpha_2 > … > alpha_32 — mathematical guarantee."""
        alphas = [_alpha(k) for k in range(1, 33)]
        for i in range(len(alphas) - 1):
            assert alphas[i] > alphas[i + 1], (
                f"alpha not decreasing at k={i+1}: {alphas[i]:.6f} vs {alphas[i+1]:.6f}"
            )

    def test_alpha_bounded_by_max(self):
        """alpha_k ≤ 0.15 for all k (can't exceed the base rate)."""
        for k in range(1, 64):
            assert _alpha(k) <= 0.15 + 1e-9

    def test_alpha_approaches_zero(self):
        """At large k, alpha should be negligible."""
        assert _alpha(128) < 1e-3

    def test_alpha_k1_is_dominant(self):
        """First pass makes the largest correction."""
        assert _alpha(1) > _alpha(2) * 1.05


# ---------------------------------------------------------------------------
# 2. Contractive MoR forward / backward
# ---------------------------------------------------------------------------

class TestContractiveForwardBackward:
    def test_forward_no_error(self):
        cfg = _tiny_cfg(mor_isrm_contractive=True)
        model = FANT3Model(cfg)
        ids = _ids(cfg)
        out = model(ids, targets=ids)
        assert torch.isfinite(out["loss"]), "loss must be finite with contractive MoR"

    def test_backward_no_error(self):
        cfg = _tiny_cfg(mor_isrm_contractive=True)
        model = FANT3Model(cfg)
        ids = _ids(cfg)
        out = model(ids, targets=ids)
        out["loss"].backward()
        # The shared MoR block MUST receive gradient (attn + FFN are in the
        # compute graph at every pass). NOTE: the router itself uses argmax
        # over its logits for depth selection, which is non-differentiable,
        # so router.fc2 legitimately has no gradient from CE loss — that's
        # expected, not a bug. We check block params, which ARE graph-connected.
        block_grad = sum(
            p.grad.abs().sum().item()
            for p in model.mor.block.parameters()
            if p.grad is not None
        )
        assert block_grad > 0, "Shared MoR block received no gradient from CE loss"

    def test_contractive_output_differs_from_default(self):
        """Contractive mode must produce different hidden states than raw replacement."""
        torch.manual_seed(0)
        cfg_base = _tiny_cfg(mor_isrm_contractive=False)
        cfg_ctr  = _tiny_cfg(mor_isrm_contractive=True)

        ids = _ids(cfg_base)

        model_base = FANT3Model(cfg_base)
        model_ctr  = FANT3Model(cfg_ctr)

        # Same weights in both models; `isrm_contractive` is a plain Python
        # attribute (not a buffer or parameter), so load_state_dict does NOT
        # overwrite it — each model keeps the behavior from its own cfg.
        model_ctr.load_state_dict(model_base.state_dict())
        assert model_base.mor.isrm_contractive is False
        assert model_ctr.mor.isrm_contractive is True

        model_base.eval(); model_ctr.eval()
        with torch.no_grad():
            out_base = model_base(ids)
            out_ctr  = model_ctr(ids)

        logits_base = out_base["logits"]
        logits_ctr  = out_ctr["logits"]
        assert not torch.allclose(logits_base, logits_ctr, atol=1e-5), (
            "Contractive and default MoR produced identical logits — flag has no effect"
        )


# ---------------------------------------------------------------------------
# 3. Monotonic CE loss
# ---------------------------------------------------------------------------

class TestMonotonicCELoss:
    def test_zero_on_non_increasing_losses(self):
        """Perfect improvement trajectory → loss == 0."""
        losses = [torch.tensor(v, requires_grad=True) for v in [5.0, 4.5, 4.0, 3.8]]
        loss = compute_monotonic_ce_loss(losses)
        assert loss.item() == pytest.approx(0.0), f"Expected 0, got {loss.item()}"

    def test_zero_on_flat_losses(self):
        """Plateau (no regression) also yields 0."""
        losses = [torch.tensor(4.0, requires_grad=True) for _ in range(4)]
        loss = compute_monotonic_ce_loss(losses)
        assert loss.item() == pytest.approx(0.0)

    def test_positive_on_single_violation(self):
        """One regression among otherwise-improving steps gives positive loss."""
        losses = [
            torch.tensor(5.0),
            torch.tensor(4.5),
            torch.tensor(4.8),  # regression here
            torch.tensor(4.2),
        ]
        loss = compute_monotonic_ce_loss(losses)
        assert loss.item() > 0.0

    def test_larger_regression_gives_larger_loss(self):
        """Bigger violation → bigger loss (quadratic)."""
        small_reg = [torch.tensor(5.0), torch.tensor(5.1)]   # +0.1
        large_reg = [torch.tensor(5.0), torch.tensor(5.5)]   # +0.5
        loss_small = compute_monotonic_ce_loss(small_reg)
        loss_large = compute_monotonic_ce_loss(large_reg)
        assert loss_large.item() > loss_small.item()

    def test_single_step_returns_zero_and_backward_works(self):
        """One pass → zero loss, and loss must still be backward-safe (graph preserved)."""
        l0 = torch.tensor(5.0, requires_grad=True)
        loss = compute_monotonic_ce_loss([l0])
        assert loss.item() == pytest.approx(0.0)
        loss.backward()  # must NOT raise: requires a preserved graph
        assert l0.grad is not None
        assert l0.grad.item() == pytest.approx(0.0)

    def test_gradient_flows(self):
        """Loss must be differentiable through the violating loss tensor."""
        l0 = torch.tensor(4.0, requires_grad=True)
        l1 = torch.tensor(4.5, requires_grad=True)  # violation
        l2 = torch.tensor(4.2, requires_grad=True)  # ok
        loss = compute_monotonic_ce_loss([l0, l1, l2])
        loss.backward()
        # l1 caused the violation; it must receive a gradient
        assert l1.grad is not None and l1.grad.item() != 0.0
        # l0 is the reference for the first violation; it also gets gradient
        assert l0.grad is not None

    def test_gradient_zero_at_no_violation(self):
        """No violation → loss is 0 → all gradients are zero."""
        tensors = [torch.tensor(v, requires_grad=True) for v in [5.0, 4.0, 3.0]]
        loss = compute_monotonic_ce_loss(tensors)
        loss.backward()
        for t in tensors:
            # After backward(), .grad is populated (not None) with zeros.
            assert t.grad is not None
            assert t.grad.item() == pytest.approx(0.0)

    def test_integration_with_model_grads(self):
        """
        REAL integration: run the model in training mode at K=1 and K=max_depth,
        keep the CE losses IN THE GRAPH (no detach), construct a GRAPH-CONNECTED
        violation, compute monotonic loss, and verify its backward actually
        flows gradient to a parameter that sits inside the MoR block.

        Key detail: adding a Python float to a graph tensor like `loss + 5.0`
        creates a constant difference in the pair (loss, loss+5.0), so the
        violation has no graph connection. Instead we use MULTIPLICATIVE
        inflation (`loss * 2.0`) — the pair (loss, 2*loss) has violation =
        loss itself, which is graph-connected.
        """
        torch.manual_seed(1)
        cfg = _tiny_cfg(mor_isrm_contractive=True, n_recursion_depths=3)
        model = FANT3Model(cfg)
        model.train()

        ids = _ids(cfg)

        model.mor.inference_k_override = 1
        loss_k1 = model(ids, targets=ids)["loss"]

        model.mor.inference_k_override = cfg.n_recursion_depths
        loss_kN = model(ids, targets=ids)["loss"]

        model.mor.inference_k_override = None

        # Multiplicative violation: pair is (loss_kN, 2*loss_kN), violation
        # = loss_kN, which IS graph-connected and depends on model params.
        step_losses = [loss_k1, loss_kN, loss_kN * 2.0]
        mono = compute_monotonic_ce_loss(step_losses)
        assert mono.item() > 0.0, "Multiplicative violation should give positive loss"

        # Target a MoR-block parameter (guaranteed graph-connected, unlike
        # the router which uses argmax). wo is the attention output projection.
        param = None
        for name, p in model.mor.block.named_parameters():
            if p.requires_grad:
                param = p
                break
        assert param is not None, "Could not find a trainable block parameter"
        assert param.grad is None, "Test precondition failed: param should have no grad yet"

        mono.backward()
        assert param.grad is not None, "monotonic loss did not reach MoR block parameters"
        assert param.grad.abs().sum().item() > 0, (
            "monotonic loss produced zero gradient on a block parameter "
            "inside both forward graphs"
        )


# ---------------------------------------------------------------------------
# 4. Dynamic K
# ---------------------------------------------------------------------------

class TestDynamicK:
    @pytest.mark.parametrize("k", [1, 2, 3])
    def test_forward_at_each_k(self, k: int):
        """Model produces a finite loss at every K value."""
        cfg = _tiny_cfg(n_recursion_depths=3)
        model = FANT3Model(cfg)
        model.eval()
        model.mor.inference_k_override = k
        ids = _ids(cfg)
        with torch.no_grad():
            out = model(ids, targets=ids)
        assert torch.isfinite(out["loss"]), f"Loss not finite at K={k}"

    def test_output_varies_with_k(self):
        """Logits at K=1 must differ from K=3 — K is doing real work."""
        cfg = _tiny_cfg(n_recursion_depths=3)
        model = FANT3Model(cfg)
        model.eval()
        ids = _ids(cfg)

        try:
            with torch.no_grad():
                model.mor.inference_k_override = 1
                out1 = model(ids)["logits"]
                model.mor.inference_k_override = 3
                out3 = model(ids)["logits"]
        finally:
            model.mor.inference_k_override = None

        assert not torch.allclose(out1, out3, atol=1e-5), (
            "K=1 and K=3 produced identical logits — recursion depth has no effect"
        )

    def test_inference_k_override_doesnt_crash(self):
        """inference_k_override can exceed n_recursion_depths without crashing."""
        cfg = _tiny_cfg(n_recursion_depths=2)
        model = FANT3Model(cfg)
        model.eval()
        ids = _ids(cfg)
        try:
            model.mor.inference_k_override = 6   # 3× the training max
            with torch.no_grad():
                out = model(ids, targets=ids)
            assert torch.isfinite(out["loss"]), "Extrapolated K crashed or produced NaN"
        finally:
            model.mor.inference_k_override = None

    def test_k_extrapolation_actually_extends_compute(self):
        """
        CRITICAL: verify K > n_recursion_depths isn't a silent no-op. Output
        at K=max_depth must differ from K=3*max_depth, i.e. the extra passes
        must actually write back to `current`. Without the router-depth
        promotion in recursion.py this test fails (extra passes are masked
        out and output is bit-identical to K=max_depth).
        """
        cfg = _tiny_cfg(mor_isrm_contractive=True, n_recursion_depths=2)
        model = FANT3Model(cfg)
        model.eval()
        ids = _ids(cfg)
        try:
            with torch.no_grad():
                model.mor.inference_k_override = cfg.n_recursion_depths  # at-training-max
                out_max = model(ids)["logits"].clone()
                model.mor.inference_k_override = cfg.n_recursion_depths * 3  # extrapolated
                out_ext = model(ids)["logits"].clone()
        finally:
            model.mor.inference_k_override = None

        assert not torch.allclose(out_max, out_ext, atol=1e-6), (
            "K extrapolation produced identical logits to K=max_depth — "
            "extra recursion passes were silently no-op'd"
        )

    def test_dynamic_k_training_loop_with_graph_connected_mono(self):
        """
        Simulate a real training step: sample K per batch, collect per-pass
        losses THROUGH THE GRAPH (no detach), add monotonic loss, and verify
        it CONTRIBUTES gradient distinct from CE-only.
        """
        import random
        torch.manual_seed(0)

        cfg = _tiny_cfg(mor_isrm_contractive=True, n_recursion_depths=3)
        model = FANT3Model(cfg)
        ids = _ids(cfg)

        # Capture per-parameter grad fingerprint for CE-only vs CE+mono.
        def run_step(include_mono: bool) -> float:
            torch.manual_seed(42)  # reset RNG between runs
            m = FANT3Model(cfg)
            # Sync weights so only the loss-shape differs
            m.load_state_dict(model.state_dict())
            m.train()
            optimizer = torch.optim.AdamW(m.parameters(), lr=1e-4)

            k = random.Random(123).randint(2, cfg.n_recursion_depths)

            step_losses: List[torch.Tensor] = []
            for pass_k in range(1, k + 1):
                m.mor.inference_k_override = pass_k
                step_losses.append(m(ids, targets=ids)["loss"])
            m.mor.inference_k_override = None

            ce_loss = m(ids, targets=ids)["loss"]

            if include_mono:
                # Force a violation via MULTIPLICATIVE inflation so the
                # violation remains graph-connected. Additive constants
                # (e.g. +5.0) produce constant differences that carry no
                # gradient to model params — the test would pass vacuously.
                synthetic = step_losses + [step_losses[-1] * 2.0]
                mono_loss = compute_monotonic_ce_loss(synthetic)
                total = ce_loss + 0.5 * mono_loss
            else:
                total = ce_loss

            optimizer.zero_grad()
            total.backward()

            grad_norm = sum(
                p.grad.data.norm().item() ** 2
                for p in m.parameters()
                if p.grad is not None
            ) ** 0.5
            return grad_norm

        grad_ce_only = run_step(include_mono=False)
        grad_with_mono = run_step(include_mono=True)

        assert math.isfinite(grad_ce_only), "CE-only produced non-finite gradient"
        assert math.isfinite(grad_with_mono), "CE+mono produced non-finite gradient"
        # With a forced violation the mono loss is ~25 (= 5**2), so the total
        # gradient norm MUST differ materially from CE-only. If they match,
        # the monotonic branch is a no-op (e.g. accidentally detached).
        assert abs(grad_with_mono - grad_ce_only) > 1e-5, (
            f"Adding monotonic loss changed grad norm by <1e-5 "
            f"({grad_ce_only=}, {grad_with_mono=}) — mono branch likely detached"
        )
