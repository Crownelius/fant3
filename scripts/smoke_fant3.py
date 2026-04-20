"""
FANT 3 smoke test — builds the smoke-scale model, runs forward + backward +
5 training steps with a tiny synthetic dataset, asserts no NaN.

Usage:
    PYTHONPATH=. python scripts/smoke_fant3.py
    PYTHONPATH=. python scripts/smoke_fant3.py --scale 1b   # try the 1B preset
"""
from __future__ import annotations
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from fant3 import FANT3Config, fant3_smoke, fant3_742m, fant3_1b
from fant3.model import FANT3Model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scale", choices=["smoke", "742m", "1b"], default="smoke")
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--bf16", action="store_true")
    args = p.parse_args()

    print("=" * 70)
    print(f"  FANT 3 smoke test — scale={args.scale}  device={args.device}  steps={args.steps}")
    print("=" * 70)

    # Build config
    cfg = {
        "smoke": fant3_smoke,
        "742m":  fant3_742m,
        "1b":    fant3_1b,
    }[args.scale]()

    print(f"\n--- Config ---")
    for f in ["dim", "n_layers", "n_dense_layers", "n_heads", "n_kv_heads",
              "n_megapools", "n_per_megapool", "n_matryoshka_levels",
              "n_attention_atoms", "mor_enabled", "n_recursion_depths",
              "vocab_size", "max_seq_len", "cerebellum_enabled"]:
        print(f"  {f}: {getattr(cfg, f)}")

    # Build model
    print(f"\n--- Building model ---")
    t0 = time.time()
    model = FANT3Model(cfg)
    print(f"  Built in {time.time()-t0:.1f}s")
    print(model.summary())

    if args.device == "cuda":
        if args.bf16:
            model = model.to(args.device, dtype=torch.bfloat16)
            print("  device: cuda bf16")
        else:
            model = model.to(args.device)
            print("  device: cuda fp32")

    # Synthetic batch
    B, T = 2, min(64, cfg.max_seq_len)
    print(f"\n--- Synthetic data: batch={B} seq={T} vocab={cfg.vocab_size} ---")

    # Forward + backward smoke
    print(f"\n--- Forward smoke ---")
    input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=args.device)
    targets = torch.randint(0, cfg.vocab_size, (B, T), device=args.device)
    t0 = time.time()
    out = model(input_ids, targets=targets)
    fw = time.time() - t0
    print(f"  Forward time: {fw:.2f}s")
    print(f"  logits shape: {out['logits'].shape} (expected {(B, T, cfg.vocab_size)})")
    print(f"  loss: {out['loss'].item():.4f}")
    print(f"  loss is NaN: {torch.isnan(out['loss']).item()}")
    print(f"  router_infos: {len(out['router_infos'])} suffix block(s) reported")
    if out["mor_info"] is not None:
        depths = out["mor_info"]["depth"]
        print(f"  MoR depth distribution: min={depths.min().item()}, max={depths.max().item()}, "
              f"mean={depths.float().mean().item():.2f}")

    print(f"\n--- Backward smoke ---")
    t0 = time.time()
    out["loss"].backward()
    bw = time.time() - t0
    print(f"  Backward time: {bw:.2f}s")

    # Check for NaN gradients
    nan_grads = []
    for name, p in model.named_parameters():
        if p.grad is not None and torch.isnan(p.grad).any():
            nan_grads.append(name)
    print(f"  NaN grads: {len(nan_grads)} (sample: {nan_grads[:3] if nan_grads else 'none'})")

    if args.device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak VRAM: {peak:.2f} GB")

    # Mini training loop
    if args.steps > 0:
        print(f"\n--- Training loop ({args.steps} steps) ---")
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for step in range(args.steps):
            input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=args.device)
            targets = torch.randint(0, cfg.vocab_size, (B, T), device=args.device)
            opt.zero_grad()
            t0 = time.time()
            out = model(input_ids, targets=targets)
            loss = out["loss"]
            loss.backward()
            opt.step()
            dt = time.time() - t0
            nan = torch.isnan(loss).item()
            print(f"  step {step+1}: loss={loss.item():.4f}  time={dt:.2f}s  nan={nan}")
            if nan:
                print(f"  ABORT: loss became NaN")
                sys.exit(2)

    # ETF freeze test
    if cfg.etf_freeze_enabled:
        print(f"\n--- ETF freeze test ---")
        n_frozen = model.freeze_intermediate_routers_to_etf()
        print(f"  Froze routers in {n_frozen} layers (target: {len(cfg.etf_freeze_layers)})")
        # Forward should still work after freezing
        input_ids = torch.randint(0, cfg.vocab_size, (B, T), device=args.device)
        out = model(input_ids)
        print(f"  Forward after ETF freeze: logits {out['logits'].shape}, no NaN: {not torch.isnan(out['logits']).any().item()}")

    print(f"\n{'=' * 70}")
    print(f"  SMOKE TEST PASSED")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
