"""
Smoke tests for ArtificialHippocampusNetwork (AHN).

Run:
    python tests/test_ahn.py
    # or via pytest: pytest tests/test_ahn.py -v
"""

import sys
import os

# Allow running from repo root without installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from fant3.model.ahn import ArtificialHippocampusNetwork


# ─────────────────────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────────────────────

DIM          = 128
SHORT_WINDOW = 32
LONG_CAP     = 64
BATCH        = 2
SEQ_LEN      = 40   # > short_window → short-term overflows; long-term accumulates
N_SEQS       = 3


# ─────────────────────────────────────────────────────────────────────────────
#  Test 1: instantiation
# ─────────────────────────────────────────────────────────────────────────────

def test_instantiation():
    ahn = ArtificialHippocampusNetwork(
        dim=DIM,
        n_heads=4,
        short_window=SHORT_WINDOW,
        long_capacity=LONG_CAP,
        compress_ratio=0.25,
    )
    print(f"[PASS] Instantiation — latent_dim={ahn.latent_dim}")
    return ahn


# ─────────────────────────────────────────────────────────────────────────────
#  Test 2: output shape
# ─────────────────────────────────────────────────────────────────────────────

def test_output_shape(ahn: ArtificialHippocampusNetwork):
    x = torch.randn(BATCH, SEQ_LEN, DIM)
    y = ahn(x)
    assert y.shape == (BATCH, SEQ_LEN, DIM), f"Expected {(BATCH, SEQ_LEN, DIM)}, got {y.shape}"
    print(f"[PASS] Output shape: {tuple(y.shape)}")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 3: long-term fill increases after multiple sequences
# ─────────────────────────────────────────────────────────────────────────────

def test_long_fill_increases(ahn: ArtificialHippocampusNetwork):
    # Reset so we start clean
    ahn.reset_memory()
    lf_before = int(ahn.long_fill.item())

    for i in range(N_SEQS):
        x = torch.randn(BATCH, SEQ_LEN, DIM)
        _ = ahn(x)

    lf_after = int(ahn.long_fill.item())
    assert lf_after > lf_before, (
        f"Long-term fill did not increase: before={lf_before}, after={lf_after}. "
        f"Need SEQ_LEN ({SEQ_LEN}) > short_window ({SHORT_WINDOW}) to trigger compression."
    )
    print(f"[PASS] Long-term fill: {lf_before} → {lf_after}  "
          f"(short_fill={ahn.short_fill.item()})")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 4: backward pass — gradients flow through learnable params
#          but NOT through buffer contents
# ─────────────────────────────────────────────────────────────────────────────

def test_gradient_flow(ahn: ArtificialHippocampusNetwork):
    ahn.reset_memory()
    x = torch.randn(BATCH, SEQ_LEN, DIM, requires_grad=False)
    y = ahn(x)

    # Dummy scalar loss
    loss = y.mean()
    loss.backward()

    # Params that MUST have gradients
    must_have_grad = {
        "gate_proj":    ahn.gate_proj.weight,
        "q_proj":       ahn.q_proj.weight,
        "k_proj":       ahn.k_proj.weight,
        "v_proj":       ahn.v_proj.weight,
        "out_proj":     ahn.out_proj.weight,
        "compressor":   ahn.compressor.weight,
        "decompressor": ahn.decompressor.weight,
    }

    failed = []
    for name, param in must_have_grad.items():
        if param.grad is None:
            failed.append(f"  {name}: grad=None")
        elif param.grad.abs().sum().item() == 0.0:
            failed.append(f"  {name}: grad all-zero")

    assert not failed, "Missing gradients:\n" + "\n".join(failed)

    # Buffers must NOT be Parameters (i.e. not in parameters())
    param_names = {n for n, _ in ahn.named_parameters()}
    buffers_that_should_not_be_params = [
        "short_K", "short_V", "long_K", "long_V",
        "short_ptr", "short_fill", "long_ptr", "long_fill",
    ]
    for buf_name in buffers_that_should_not_be_params:
        assert buf_name not in param_names, f"Buffer {buf_name} is mistakenly a Parameter!"

    print("[PASS] Gradient flow: all learnable params have non-zero grads; buffers are not Parameters")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 5: reset_memory clears all buffers
# ─────────────────────────────────────────────────────────────────────────────

def test_reset_memory(ahn: ArtificialHippocampusNetwork):
    # Fill buffers first
    for _ in range(N_SEQS):
        _ = ahn(torch.randn(BATCH, SEQ_LEN, DIM))

    sf_before = int(ahn.short_fill.item())
    lf_before = int(ahn.long_fill.item())
    assert sf_before > 0 or lf_before > 0, "Buffers should be non-empty before reset"

    ahn.reset_memory()

    assert ahn.short_fill.item() == 0,  f"short_fill not cleared: {ahn.short_fill.item()}"
    assert ahn.long_fill.item()  == 0,  f"long_fill not cleared: {ahn.long_fill.item()}"
    assert ahn.short_ptr.item()  == 0,  f"short_ptr not cleared: {ahn.short_ptr.item()}"
    assert ahn.long_ptr.item()   == 0,  f"long_ptr not cleared: {ahn.long_ptr.item()}"
    assert ahn.short_K.abs().sum().item() == 0.0, "short_K not zeroed"
    assert ahn.long_K.abs().sum().item()  == 0.0, "long_K not zeroed"

    print(f"[PASS] reset_memory: buffers cleared (short_fill was {sf_before}, long_fill was {lf_before})")


# ─────────────────────────────────────────────────────────────────────────────
#  Test 6: get_stats diagnostic
# ─────────────────────────────────────────────────────────────────────────────

def test_get_stats(ahn: ArtificialHippocampusNetwork):
    ahn.reset_memory()
    for _ in range(N_SEQS):
        _ = ahn(torch.randn(BATCH, SEQ_LEN, DIM))

    stats = ahn.get_stats()
    required_keys = {"short_fill", "long_fill", "gate_short", "gate_long"}
    assert required_keys == set(stats.keys()), f"Missing keys: {required_keys - set(stats.keys())}"

    assert 0.0 <= stats["short_fill"] <= 1.0, f"short_fill out of range: {stats['short_fill']}"
    assert 0.0 <= stats["long_fill"]  <= 1.0, f"long_fill out of range: {stats['long_fill']}"

    gate_sum = stats["gate_short"] + stats["gate_long"]
    assert abs(gate_sum - 1.0) < 1e-5, f"Gate weights don't sum to 1: {gate_sum}"

    print(f"[PASS] get_stats: {stats}")
    return stats


# ─────────────────────────────────────────────────────────────────────────────
#  Main runner
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("ArtificialHippocampusNetwork smoke test")
    print(f"  dim={DIM}  short_window={SHORT_WINDOW}  long_cap={LONG_CAP}")
    print(f"  batch={BATCH}  seq_len={SEQ_LEN}  n_seqs={N_SEQS}")
    print("=" * 60)

    ahn = test_instantiation()
    test_output_shape(ahn)

    ahn.reset_memory()
    test_long_fill_increases(ahn)

    ahn.reset_memory()
    test_gradient_flow(ahn)

    ahn.reset_memory()
    # Fill again before reset test
    for _ in range(N_SEQS):
        _ = ahn(torch.randn(BATCH, SEQ_LEN, DIM))
    test_reset_memory(ahn)

    final_stats = test_get_stats(ahn)

    print("=" * 60)
    print("All tests PASSED")
    print(f"Final stats after {N_SEQS} sequences of length {SEQ_LEN}:")
    for k, v in final_stats.items():
        print(f"  {k:12s}: {v:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
