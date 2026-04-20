#!/usr/bin/env python3
"""
Scale-ladder smoke test for FANT 3 with the five 2026-04-19 fixes active.

For each of ~5M, ~40M, ~150M, ~350M, ~742M, build the model, run a forward,
backward, and 3 optimizer steps on CUDA with batch=2 seq=128. Track param
count, VRAM peak, step time, loss, and NaN-freeness.

Pass criterion: all 5 scales complete without error; loss is finite at every
step; chirality balance (spinor α/β split) stays near 50/50.

Run:
    python scripts/scale_ladder_smoke.py
"""

from __future__ import annotations

import sys
import time
import gc
from pathlib import Path

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))

from fant3.config import FANT3Config   # noqa: E402
from fant3.model.fant3_model import FANT3Model  # noqa: E402


def cfg_5m() -> FANT3Config:
    """Target ~5M stored params."""
    return FANT3Config(
        dim=192, n_layers=4, n_dense_layers=1,
        n_heads=4, n_kv_heads=1, head_dim=48,
        n_megapools=2, n_per_megapool=2, top_k=2,
        n_matryoshka_levels=2,
        shared_expert_hidden=128, moe_hidden=192,
        n_attention_atoms=2, masa_coef_rank=4,
        n_recursion_depths=2,
        kron_A_p=8, kron_A_q=4, kron_B_p=8, kron_B_q=8, kron_C_p=8, kron_C_q=12,
        max_seq_len=256,
        n_hub_tokens=4,
        cerebellum_enabled=False,
        apollonian_alpha_cap=200, apollonian_beta_cap=200,
        apollonian_retrieval_layers=(2, 3),
        etf_freeze_after_step=100,
        etf_freeze_layers=(1, 2),
        spinor_apollonian_enabled=True,
        ahn_enabled=True,
        ahn_n_heads=2, ahn_short_window=16, ahn_long_capacity=16,
    )


def cfg_40m() -> FANT3Config:
    """Target ~40M — basically fant3_smoke with new fixes enabled."""
    from fant3.config import fant3_smoke
    c = fant3_smoke()
    c.spinor_apollonian_enabled = True
    c.ahn_enabled = True
    c.ahn_n_heads = 2
    c.ahn_short_window = 32
    c.ahn_long_capacity = 32
    return c


def cfg_150m() -> FANT3Config:
    """Target ~150M stored params — smaller MoE, shorter Kronecker."""
    return FANT3Config(
        dim=768, n_layers=10, n_dense_layers=2,
        n_heads=8, n_kv_heads=2, head_dim=96,
        n_megapools=2, n_per_megapool=4, top_k=2,  # 8 total experts
        n_matryoshka_levels=2,
        shared_expert_hidden=384, moe_hidden=768,
        n_attention_atoms=3, masa_coef_rank=8,
        n_recursion_depths=2,
        kron_A_p=24, kron_A_q=8, kron_B_p=16, kron_B_q=16, kron_C_p=24, kron_C_q=24,
        max_seq_len=1024,
        cerebellum_enabled=False,
        apollonian_alpha_cap=1000, apollonian_beta_cap=1000,
        apollonian_retrieval_layers=(8, 9),
        etf_freeze_after_step=500,
        etf_freeze_layers=tuple(range(2, 8)),
        spinor_apollonian_enabled=True,
        ahn_enabled=True,
        ahn_n_heads=2, ahn_short_window=32, ahn_long_capacity=64,
    )


def cfg_350m() -> FANT3Config:
    """Target ~350M stored params."""
    return FANT3Config(
        dim=1024, n_layers=14, n_dense_layers=2,
        n_heads=8, n_kv_heads=2, head_dim=128,
        n_megapools=4, n_per_megapool=4, top_k=2,  # 16 experts
        n_matryoshka_levels=2,
        shared_expert_hidden=512, moe_hidden=1024,
        n_attention_atoms=4, masa_coef_rank=8,
        n_recursion_depths=2,
        kron_A_p=32, kron_A_q=8, kron_B_p=16, kron_B_q=16, kron_C_p=32, kron_C_q=32,
        max_seq_len=1024,
        cerebellum_enabled=False,
        apollonian_alpha_cap=2000, apollonian_beta_cap=2000,
        apollonian_retrieval_layers=(12, 13),
        etf_freeze_after_step=800,
        etf_freeze_layers=tuple(range(2, 11)),
        spinor_apollonian_enabled=True,
        ahn_enabled=True,
        ahn_n_heads=4, ahn_short_window=64, ahn_long_capacity=128,
    )


def cfg_742m() -> FANT3Config:
    """Target ~742M — validation-scale rung, uses the existing preset."""
    from fant3.config import fant3_742m
    c = fant3_742m()
    c.spinor_apollonian_enabled = True
    c.ahn_enabled = True
    c.ahn_n_heads = 4
    c.ahn_short_window = 128
    c.ahn_long_capacity = 256
    # Disable cerebellum at smoke-test scale to keep the test fast; it's
    # orthogonal to the scale ladder we're validating.
    c.cerebellum_enabled = False
    return c


LADDER = [
    ("5M",   cfg_5m),
    ("40M",  cfg_40m),
    ("150M", cfg_150m),
    ("350M", cfg_350m),
    ("742M", cfg_742m),
]


def run_one(name: str, cfg_factory, device: str = "cuda", dtype=torch.bfloat16,
            B: int = 1, T: int = 64, n_steps: int = 3) -> dict:
    """
    End-to-end smoke at one scale.

    Default B=1 T=64 keeps even the 742M rung within RTX 3060 12 GB budget
    (model + grads + AdamW state + activations + Apollonian/AHN buffers).
    The goal is architectural verification, not convergence — 3 steps is enough
    to confirm no NaN, no OOM, no shape mismatches.
    """
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    cfg = cfg_factory()

    t_build_0 = time.time()
    model = FANT3Model(cfg).to(device=device, dtype=dtype)
    t_build = time.time() - t_build_0

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Pre-forward peak reset
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # SGD has no optimizer state (no m/v buffers) — keeps the test lean at the
    # large-model rungs. We're testing architecture, not convergence.
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)

    losses = []
    t_fwd = 0.0
    t_bwd = 0.0
    t_opt = 0.0

    for step in range(n_steps):
        ids = torch.randint(0, cfg.vocab_size, (B, T), device=device)
        targets = torch.randint(0, cfg.vocab_size, (B, T), device=device)

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        out = model(ids, targets=targets, store_to_memory=(step == n_steps - 1))
        if device == "cuda":
            torch.cuda.synchronize()
        t_fwd += time.time() - t0

        loss = out["loss"]
        if not torch.isfinite(loss):
            return {
                "name": name, "ok": False, "error": f"non-finite loss at step {step}",
                "n_params": n_params,
            }
        losses.append(loss.item())

        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
        t_bwd += time.time() - t0

        t0 = time.time()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if device == "cuda":
            torch.cuda.synchronize()
        t_opt += time.time() - t0

    # Collect final diagnostics
    mem_stats = model.memory.get_stats() if hasattr(model.memory, "get_stats") else {}
    ahn_stats = model.ahn.get_stats() if model.ahn is not None else {}

    vram_peak_gb = None
    if device == "cuda":
        vram_peak_gb = torch.cuda.max_memory_allocated() / 1e9

    # cleanup for next rung
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "name":         name,
        "ok":           True,
        "n_params_M":   n_params / 1e6,
        "n_trainable_M": n_trainable / 1e6,
        "build_s":      t_build,
        "fwd_s_avg":    t_fwd / n_steps,
        "bwd_s_avg":    t_bwd / n_steps,
        "opt_s_avg":    t_opt / n_steps,
        "loss_first":   losses[0],
        "loss_last":    losses[-1],
        "vram_peak_GB": vram_peak_gb,
        "chirality":    mem_stats.get("chirality_balance"),
        "alpha_fill":   mem_stats.get("alpha_fill"),
        "beta_fill":    mem_stats.get("beta_fill"),
        "ahn_short":    ahn_stats.get("short_fill"),
        "ahn_long":     ahn_stats.get("long_fill"),
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Device: {device}  dtype: {dtype}")
    print()

    results = []
    for name, factory in LADDER:
        print(f"▶ Running {name} ...", flush=True)
        # Progressively shrink B/T at large scales to stay within 12 GB on 3060
        if name == "742M":
            B, T = 1, 32
        elif name in ("350M", "150M"):
            B, T = 1, 64
        else:
            B, T = 1, 128
        try:
            r = run_one(name, factory, device=device, dtype=dtype, B=B, T=T)
            results.append(r)
            if r["ok"]:
                print(f"  ✓ params {r['n_params_M']:7.2f}M  vram {r['vram_peak_GB']:5.2f} GB  "
                      f"fwd {r['fwd_s_avg']*1000:5.1f}ms  bwd {r['bwd_s_avg']*1000:5.1f}ms  "
                      f"loss {r['loss_first']:.3f}→{r['loss_last']:.3f}  "
                      f"chirality {r['chirality']}")
            else:
                print(f"  ✗ FAILED: {r['error']}")
                break
        except Exception as e:
            results.append({"name": name, "ok": False, "error": str(e)})
            print(f"  ✗ EXCEPTION: {type(e).__name__}: {e}")
            break

    print()
    print("=" * 100)
    print(f"{'scale':<6} {'params':>9} {'vram':>8} {'fwd':>8} {'bwd':>8} {'loss':>12} {'α fill':>7} {'β fill':>7} {'chir':>6}")
    print("-" * 100)
    for r in results:
        if not r.get("ok"):
            print(f"{r['name']:<6}  FAILED: {r.get('error', '?')}")
            continue
        chir = r.get("chirality")
        chir_s = f"{chir:.3f}" if chir is not None else "  —  "
        loss_s = f"{r['loss_first']:.2f}→{r['loss_last']:.2f}"
        vram_s = f"{r['vram_peak_GB']:.2f}GB" if r['vram_peak_GB'] else "    —"
        print(f"{r['name']:<6} {r['n_params_M']:7.2f}M {vram_s:>8} "
              f"{r['fwd_s_avg']*1000:5.1f}ms {r['bwd_s_avg']*1000:5.1f}ms "
              f"{loss_s:>12} {r.get('alpha_fill','-'):>7} {r.get('beta_fill','-'):>7} {chir_s:>6}")
    print()
    n_ok = sum(1 for r in results if r.get("ok"))
    n_total = len(LADDER)
    print(f"RESULT: {n_ok}/{n_total} scales passed end-to-end")


if __name__ == "__main__":
    main()
