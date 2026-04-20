"""
Smoke test for SpinorApollonianMemory.

Run with:
    python tests/test_spinor_apollonian.py
  or
    python -m pytest tests/test_spinor_apollonian.py -v
"""

import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import pytest

from fant3.model.spinor_apollonian import SpinorApollonianMemory, clifford_bilinear, clifford_norm


# ---------------------------------------------------------------------------
# Test 1: instantiation
# ---------------------------------------------------------------------------

def test_instantiation():
    mem = SpinorApollonianMemory(dim=128, alpha_cap=100, beta_cap=100)
    assert mem.dim == 128
    assert mem.alpha_cap == 100
    assert mem.beta_cap == 100
    stats = mem.get_stats()
    assert stats["alpha_fill"] == 0
    assert stats["beta_fill"] == 0
    assert stats["chirality_balance"] == 0.5  # default when both empty
    print("[PASS] instantiation")


# ---------------------------------------------------------------------------
# Test 2: store — chirality_balance near 0.5
# ---------------------------------------------------------------------------

def test_store_chirality_balance():
    """
    With N=32 random tokens and std=0.01 spinor init, the natural binomial
    std is ~0.088, so we allow [0.25, 0.75].  We also run a larger N=512
    sweep across 5 seeds and require the mean balance is in [0.35, 0.65],
    confirming the projection starts unbiased.
    """
    torch.manual_seed(42)
    mem = SpinorApollonianMemory(dim=128, alpha_cap=100, beta_cap=100)

    emb  = torch.randn(2, 16, 128)   # [B=2, T=16, dim=128]
    hid  = torch.randn(2, 16, 128)   # pre-RMSNorm hidden

    result = mem.store(emb, hidden_preRMSnorm=hid)

    assert "alpha_stored" in result and "beta_stored" in result
    total = result["alpha_stored"] + result["beta_stored"]
    assert total == 32, f"expected 32 items stored, got {total}"

    stats = mem.get_stats()
    bal = stats["chirality_balance"]
    print(f"  chirality_balance (N=32, seed=42) = {bal:.4f}")
    # Wide tolerance for N=32; binomial std ≈ 0.088 → ±2σ window
    assert 0.25 <= bal <= 0.75, (
        f"chirality_balance {bal:.4f} outside [0.25, 0.75] — large bias in "
        "spinor projection init"
    )

    # Larger-N check: average over 5 seeds × 128 tokens should be near 0.5
    balances = []
    for seed in range(5):
        torch.manual_seed(seed)
        m = SpinorApollonianMemory(dim=128, alpha_cap=200, beta_cap=200)
        e = torch.randn(4, 32, 128)
        h = torch.randn(4, 32, 128)
        m.store(e, hidden_preRMSnorm=h)
        balances.append(m.get_stats()["chirality_balance"])
    mean_bal = sum(balances) / len(balances)
    print(f"  mean chirality_balance (N=128, 5 seeds) = {mean_bal:.4f}")
    assert 0.35 <= mean_bal <= 0.65, (
        f"mean balance {mean_bal:.4f} shows systematic bias in proj_spinor init"
    )
    print("[PASS] store chirality balance")


# ---------------------------------------------------------------------------
# Test 3: store fallback (no hidden given)
# ---------------------------------------------------------------------------

def test_store_no_hidden():
    torch.manual_seed(7)
    mem = SpinorApollonianMemory(dim=128, alpha_cap=100, beta_cap=100)
    emb = torch.randn(2, 16, 128)
    result = mem.store(emb)  # hidden_preRMSnorm=None → fallback to emb
    total = result["alpha_stored"] + result["beta_stored"]
    assert total == 32
    print("[PASS] store without hidden (fallback)")


# ---------------------------------------------------------------------------
# Test 4: retrieve — output shape
# ---------------------------------------------------------------------------

def test_retrieve_shape():
    torch.manual_seed(42)
    mem = SpinorApollonianMemory(dim=128, alpha_cap=100, beta_cap=100)

    # First store some items so pools are non-empty
    emb  = torch.randn(2, 16, 128)
    hid  = torch.randn(2, 16, 128)
    mem.store(emb, hidden_preRMSnorm=hid)

    # Retrieve
    query = torch.randn(1, 4, 128)
    out = mem.retrieve(query, top_k=4, pool="both")

    assert "values" in out and "scores" in out
    assert out["values"].shape == (1, 4, 4, 128), f"got {out['values'].shape}"
    assert out["scores"].shape == (1, 4, 4),       f"got {out['scores'].shape}"
    print(f"  values shape: {out['values'].shape}")
    print(f"  scores shape: {out['scores'].shape}")
    print("[PASS] retrieve shape")


# ---------------------------------------------------------------------------
# Test 5: retrieve from alpha-only and beta-only pools
# ---------------------------------------------------------------------------

def test_retrieve_pool_selection():
    torch.manual_seed(0)
    mem = SpinorApollonianMemory(dim=128, alpha_cap=100, beta_cap=100)
    emb = torch.randn(4, 32, 128)
    mem.store(emb)

    q = torch.randn(1, 2, 128)
    out_a = mem.retrieve(q, top_k=2, pool="alpha")
    out_b = mem.retrieve(q, top_k=2, pool="beta")

    assert out_a["values"].shape == (1, 2, 2, 128)
    assert out_b["values"].shape == (1, 2, 2, 128)
    print("[PASS] retrieve pool='alpha' and pool='beta'")


# ---------------------------------------------------------------------------
# Test 6: descartes_loss returns a scalar tensor
# ---------------------------------------------------------------------------

def test_descartes_loss():
    torch.manual_seed(42)
    mem = SpinorApollonianMemory(dim=128, alpha_cap=100, beta_cap=100)

    # Populate memory
    emb = torch.randn(2, 16, 128)
    mem.store(emb)

    query_spinors = torch.randn(32, 2)
    loss = mem.descartes_loss(query_spinors)

    assert loss.dim() == 0, f"expected scalar, got shape {loss.shape}"
    assert float(loss.item()) >= 0.0, "Descartes violation must be ≥ 0"
    print(f"  descartes_loss = {float(loss.item()):.6f}")
    print("[PASS] descartes_loss scalar")


# ---------------------------------------------------------------------------
# Test 7: get_stats keys and types
# ---------------------------------------------------------------------------

def test_get_stats():
    torch.manual_seed(42)
    mem = SpinorApollonianMemory(dim=128, alpha_cap=100, beta_cap=100)
    emb = torch.randn(2, 16, 128)
    mem.store(emb)

    stats = mem.get_stats()
    print(f"  get_stats() = {stats}")

    required_keys = {
        "alpha_fill", "beta_fill",
        "alpha_curvature_mean", "beta_curvature_mean",
        "chirality_balance",
    }
    assert required_keys <= set(stats.keys()), f"missing keys: {required_keys - set(stats.keys())}"
    assert isinstance(stats["alpha_fill"], int)
    assert isinstance(stats["beta_fill"], int)
    assert isinstance(stats["chirality_balance"], float)
    assert 0.0 <= stats["chirality_balance"] <= 1.0
    print("[PASS] get_stats")


# ---------------------------------------------------------------------------
# Test 8: autograd through retrieve scores
# ---------------------------------------------------------------------------

def test_autograd():
    torch.manual_seed(42)
    mem = SpinorApollonianMemory(dim=128, alpha_cap=100, beta_cap=100)

    # Populate memory (no_grad, as always)
    emb = torch.randn(2, 16, 128)
    mem.store(emb)

    # Create a differentiable query
    query = torch.randn(1, 4, 128, requires_grad=True)
    out = mem.retrieve(query, top_k=4, pool="both")

    # Backprop through the mean retrieval score — this exercises proj_spinor
    loss = out["scores"].mean()
    loss.backward()

    spinor_weight_grad = mem.proj_spinor.weight.grad
    assert spinor_weight_grad is not None, "proj_spinor.weight should receive a gradient"
    assert spinor_weight_grad.shape == (2, 128)
    print(f"  proj_spinor.weight.grad norm = {spinor_weight_grad.norm().item():.6f}")
    print("[PASS] autograd through retrieve")


# ---------------------------------------------------------------------------
# Test 9: empty-pool edge case
# ---------------------------------------------------------------------------

def test_empty_pool():
    mem = SpinorApollonianMemory(dim=128, alpha_cap=100, beta_cap=100)
    q = torch.randn(1, 4, 128)
    out = mem.retrieve(q, top_k=4, pool="both")
    assert out["values"].shape == (1, 4, 4, 128)
    assert out["scores"].sum().item() == 0.0
    print("[PASS] empty pool returns zeros")


# ---------------------------------------------------------------------------
# Test 10: Clifford helpers
# ---------------------------------------------------------------------------

def test_clifford_helpers():
    a = torch.tensor([[3.0, 4.0]])
    b = torch.tensor([[1.0, 2.0]])
    # bilinear: 3*1 − 4*2 = 3 − 8 = −5
    assert float(clifford_bilinear(a, b).item()) == pytest.approx(-5.0)
    # norm: 3² + 4² = 25
    assert float(clifford_norm(a).item()) == pytest.approx(25.0)
    print("[PASS] Clifford helpers")


# ---------------------------------------------------------------------------
# Main (standalone runner)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("SpinorApollonianMemory smoke test")
    print("=" * 60)

    test_clifford_helpers()
    test_instantiation()
    test_store_chirality_balance()
    test_store_no_hidden()
    test_retrieve_shape()
    test_retrieve_pool_selection()
    test_descartes_loss()
    test_get_stats()
    test_autograd()
    test_empty_pool()

    print("=" * 60)
    print("All tests passed.")
    print("=" * 60)

    # Print final summary stats for the report
    torch.manual_seed(42)
    mem = SpinorApollonianMemory(dim=128, alpha_cap=100, beta_cap=100)
    emb = torch.randn(2, 16, 128)
    hid = torch.randn(2, 16, 128)
    mem.store(emb, hidden_preRMSnorm=hid)
    stats = mem.get_stats()
    qs = torch.randn(32, 2)
    dl = mem.descartes_loss(qs)
    print(f"\nFinal stats: {stats}")
    print(f"descartes_loss (lambda=1e-4 weight not applied): {float(dl.item()):.6f}")
