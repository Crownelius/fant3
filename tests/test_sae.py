"""
Smoke tests for fant3.diagnostics.sae — ApollonianSAE.

Run with:
    python tests/test_sae.py

or via pytest:
    pytest tests/test_sae.py -v
"""

import sys
import os

# Allow running from any working directory inside the project tree.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn as nn

from fant3.diagnostics import ApollonianSAE, train_on_hidden_states, analyze_apollonian_memory


# ─────────────────────────────────────────────────────────────────────────────
#  Test 1: basic instantiation + shapes
# ─────────────────────────────────────────────────────────────────────────────

def test_instantiation():
    sae = ApollonianSAE(d_in=128, n_features=512, k=16)
    assert sae.W_enc.shape == (512, 128), f"W_enc shape wrong: {sae.W_enc.shape}"
    assert sae.W_dec.shape == (128, 512), f"W_dec shape wrong: {sae.W_dec.shape}"
    assert sae.b_enc.shape == (512,)
    assert sae.b_dec.shape == (128,)
    print("PASS  test_instantiation")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 2: training — loss should decrease over epochs
# ─────────────────────────────────────────────────────────────────────────────

def test_training_loss_decreases():
    torch.manual_seed(42)
    sae = ApollonianSAE(d_in=128, n_features=512, k=16)
    H = torch.randn(2000, 128)

    history = train_on_hidden_states(sae, H, n_epochs=2, batch_size=64, lr=1e-3)

    losses = history["losses"]
    assert len(losses) > 0, "No loss values recorded"

    # Compare first-quarter mean vs last-quarter mean
    q = max(len(losses) // 4, 1)
    early_loss = sum(losses[:q]) / q
    late_loss  = sum(losses[-q:]) / q
    assert late_loss < early_loss, (
        f"Loss did not decrease: early={early_loss:.4f}  late={late_loss:.4f}"
    )
    print(f"PASS  test_training_loss_decreases  "
          f"early={early_loss:.4f}  late={late_loss:.4f}")
    return history, losses[-1]


# ─────────────────────────────────────────────────────────────────────────────
#  Test 3: forward — shapes, L0 sparsity, dead-feature fraction
# ─────────────────────────────────────────────────────────────────────────────

def test_forward_batch():
    torch.manual_seed(7)
    sae = ApollonianSAE(d_in=128, n_features=512, k=16)

    # Brief warmup training so dead-feature tracking has data
    H = torch.randn(2000, 128)
    history = train_on_hidden_states(sae, H, n_epochs=2, batch_size=64)

    sae.eval()
    x = torch.randn(8, 128)
    with torch.no_grad():
        out = sae(x)

    # Reconstruction shape
    assert out["reconstruction"].shape == (8, 128), (
        f"reconstruction shape wrong: {out['reconstruction'].shape}"
    )

    # L0 sparsity <= k
    l0 = float(out["l0"].item())
    assert l0 <= 16, f"L0 sparsity {l0:.1f} exceeds k=16"

    # Dead-feature fraction < 50% after 2 epochs of training
    dead_frac = sae.dead_feature_fraction()
    assert dead_frac < 0.5, (
        f"Dead-feature fraction {dead_frac:.2%} >= 50% after training — "
        "something is wrong with activation"
    )

    # Loss is a scalar
    assert out["loss"].ndim == 0, "loss should be a scalar"

    print(f"PASS  test_forward_batch  "
          f"recon={out['reconstruction'].shape}  "
          f"l0={l0:.1f}  dead={dead_frac:.2%}")
    return out, dead_frac


# ─────────────────────────────────────────────────────────────────────────────
#  Test 4: analyze_apollonian_memory — mock memory object, expected keys
# ─────────────────────────────────────────────────────────────────────────────

class MockMemory(nn.Module):
    """
    Minimal stand-in for ApollonianMemory or SpinorApollonianMemory.
    Uses the alpha_bank / beta_bank layout accepted by _extract_memory_banks.
    """
    def __init__(self, n_alpha: int, n_beta: int, dim: int):
        super().__init__()
        torch.manual_seed(99)
        # alpha_bank: high-norm embeddings (simulating high curvature / recent)
        self.alpha_bank = torch.randn(n_alpha, dim) * 2.0
        # beta_bank: low-norm embeddings (simulating low curvature / schema)
        self.beta_bank  = torch.randn(n_beta,  dim) * 0.5


def test_analyze_apollonian_memory():
    torch.manual_seed(0)
    sae = ApollonianSAE(d_in=128, n_features=512, k=16)

    # Brief training so that not all features are dead
    H = torch.randn(2000, 128)
    train_on_hidden_states(sae, H, n_epochs=2, batch_size=64)

    mock_mem = MockMemory(n_alpha=200, n_beta=150, dim=128)

    diag = analyze_apollonian_memory(sae, mock_mem, top_n_features=20)

    # --- Check required keys ---
    required_keys = {
        "pack_sizes",
        "feature_activation_histograms",
        "top_discriminating_features",
        "ghost_features",
        "ghost_feature_count",
        "ghost_feature_fraction",
        "chirality_correlation",
    }
    missing = required_keys - set(diag.keys())
    assert not missing, f"Missing keys in diagnostics: {missing}"

    # --- pack sizes ---
    assert diag["pack_sizes"]["alpha"] == 200
    assert diag["pack_sizes"]["beta"]  == 150

    # --- histogram lengths ---
    assert len(diag["feature_activation_histograms"]["alpha"]) == 512
    assert len(diag["feature_activation_histograms"]["beta"])  == 512

    # --- top discriminating features ---
    assert len(diag["top_discriminating_features"]) == 20
    for entry in diag["top_discriminating_features"]:
        assert "feature_index"  in entry
        assert "mean_alpha"     in entry
        assert "mean_beta"      in entry
        assert "abs_difference" in entry
        assert entry["prefers"] in ("alpha", "beta")

    # --- ghost-feature fraction is in [0, 1] ---
    gf = diag["ghost_feature_fraction"]
    assert 0.0 <= gf <= 1.0, f"ghost_feature_fraction out of range: {gf}"

    # --- chirality_correlation is None for this mock (no chirality buffer) ---
    assert diag["chirality_correlation"] is None

    # --- top discriminating should be SORTED by abs_difference descending ---
    diffs = [e["abs_difference"] for e in diag["top_discriminating_features"]]
    assert diffs == sorted(diffs, reverse=True), (
        "top_discriminating_features not sorted by abs_difference"
    )

    print(f"PASS  test_analyze_apollonian_memory  "
          f"ghost_frac={gf:.2%}  "
          f"top_diff={diffs[0]:.4f}")
    return diag


# ─────────────────────────────────────────────────────────────────────────────
#  Test 5: chirality correlation path (SpinorApollonianMemory layout)
# ─────────────────────────────────────────────────────────────────────────────

class MockSpinorMemory(nn.Module):
    """Mock with chirality attribute for testing the spinor path."""
    def __init__(self, n_alpha: int, n_beta: int, dim: int):
        super().__init__()
        torch.manual_seed(13)
        # Chirality: first half +1, second half -1
        chir = torch.ones(n_alpha)
        chir[n_alpha // 2:] = -1.0
        # Store chirality and alpha_bank at the same size
        self.alpha_bank = torch.randn(n_alpha, dim)
        self.beta_bank  = torch.randn(n_beta,  dim)
        self.chirality  = chir


def test_chirality_correlation():
    torch.manual_seed(3)
    sae = ApollonianSAE(d_in=128, n_features=512, k=16)
    H = torch.randn(2000, 128)
    train_on_hidden_states(sae, H, n_epochs=2, batch_size=64)

    mock = MockSpinorMemory(n_alpha=100, n_beta=80, dim=128)
    diag = analyze_apollonian_memory(sae, mock, top_n_features=5)

    corr = diag["chirality_correlation"]
    assert corr is not None, "chirality_correlation should not be None for SpinorMemory"
    assert len(corr) == 512, f"Expected 512 correlation values, got {len(corr)}"
    # All correlations should be in [-1, 1]
    assert all(-1.0 <= c <= 1.0 for c in corr), "Some correlations out of [-1, 1]"

    print(f"PASS  test_chirality_correlation  "
          f"max_corr={max(abs(c) for c in corr):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 6: ApollonianMemory layout (FIFO-buffer style, matching real fant2 API)
# ─────────────────────────────────────────────────────────────────────────────

def test_real_apollonian_memory_layout():
    """Verify compatibility with the actual ApollonianMemory buffer layout."""
    torch.manual_seed(5)
    sae = ApollonianSAE(d_in=128, n_features=512, k=16)
    H = torch.randn(2000, 128)
    train_on_hidden_states(sae, H, n_epochs=2, batch_size=64)

    # Simulate the real ApollonianMemory buffers (without importing fant2)
    class RealStyleMemory(nn.Module):
        def __init__(self):
            super().__init__()
            cap = 500
            torch.manual_seed(17)
            self.register_buffer("alpha_emb",   torch.randn(cap, 128))
            self.register_buffer("alpha_count", torch.tensor(200, dtype=torch.long))
            self.register_buffer("beta_emb",    torch.randn(cap, 128) * 0.3)
            self.register_buffer("beta_count",  torch.tensor(150, dtype=torch.long))

    mem = RealStyleMemory()
    diag = analyze_apollonian_memory(sae, mem, top_n_features=10)

    assert diag["pack_sizes"]["alpha"] == 200
    assert diag["pack_sizes"]["beta"]  == 150
    assert len(diag["top_discriminating_features"]) == 10

    print(f"PASS  test_real_apollonian_memory_layout  "
          f"ghost_frac={diag['ghost_feature_fraction']:.2%}")


# ─────────────────────────────────────────────────────────────────────────────
#  Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("ApollonianSAE smoke tests")
    print("=" * 60)

    test_instantiation()

    history, final_loss = test_training_loss_decreases()
    out, dead_frac = test_forward_batch()
    diag = test_analyze_apollonian_memory()
    test_chirality_correlation()
    test_real_apollonian_memory_layout()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print(f"  Final SAE loss:         {final_loss:.4f}")
    print(f"  L0 sparsity:            {float(out['l0'].item()):.1f}  (k=16)")
    print(f"  Dead-feature fraction:  {dead_frac:.2%}")
    print(f"  Ghost-feature fraction: {diag['ghost_feature_fraction']:.2%}")
    print("=" * 60)
