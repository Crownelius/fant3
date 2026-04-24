"""
Sweep n_recursion_depths to find the K-depth frontier for FANT 3 MoR.

Two questions we're answering:

  1. Training-time ceiling — at what K does the model stop converging?
  2. Inference-time ceiling — how far past the trained K range can you push
     and still see benefit (or at least not regress)?

Method:
  For each n_recursion_depths N in the sweep:
    - Build a fresh fant3_1m() model with cfg.n_recursion_depths = N
    - Train with dynamic K ~ Uniform[1, N], contractive decay ON, no monotonic
      loss (single forward per step for fair compute comparison)
    - After training, sweep inference K across {1, N/2, N, 2*N, 4*N} and
      record copy-half accuracy + CE loss at each

Note on compute: per-step FLOPs scale linearly with K_sampled (avg N/2).
The training step budget is held fixed, so bigger N gets more compute.
This is intentional — we want to see if extra depth CAPACITY pays off.

Run:
    python scripts/find_max_k_depth.py                # default sweep
    python scripts/find_max_k_depth.py --sweep 3 6 12 # custom values
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import math
import time
from typing import List, Dict

# Path bootstrap so scripts/ can import fant3/
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch

from fant3.config import fant3_1m
from fant3.model.fant3_model import FANT3Model
from scripts.train_1m_local import sample_batch


def train_one(n_depths: int, steps: int, batch_size: int, seq_len: int,
              lr: float, seed: int, device: torch.device) -> Dict:
    """Train a fresh fant3_1m with the given n_recursion_depths, dynamic K on,
    contractive on, single forward per step. Return the trained model + stats."""
    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed)

    cfg = fant3_1m()
    cfg.n_recursion_depths = n_depths
    cfg.mor_isrm_contractive = True
    cfg.apollonian_retrieval_layers = (2, 3)   # last 2 of 4 layers — unchanged

    model = FANT3Model(cfg).to(device)
    n_params = model.n_params()
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    first_ce = None
    losses: List[float] = []
    t0 = time.time()

    model.train()
    for step in range(1, steps + 1):
        ids, targets = sample_batch(batch_size, seq_len, cfg.vocab_size, rng, device)

        # Dynamic K: sample uniformly over the trained range [1, n_depths].
        k = rng.randint(1, n_depths)
        model.mor.inference_k_override = k
        out = model(ids, targets=targets)
        model.mor.inference_k_override = None
        loss = out["loss"]

        optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.step()

        if first_ce is None:
            first_ce = loss.item()
        losses.append(loss.item())

    dt = time.time() - t0

    # Tail average — more robust than the single final step (which is stochastic
    # because K varies per step).
    tail_n  = max(10, steps // 10)
    tail_ce = sum(losses[-tail_n:]) / tail_n

    return {
        "n_depths": n_depths,
        "n_params": n_params,
        "steps":    steps,
        "first_ce": first_ce,
        "final_ce": losses[-1],
        "tail_ce":  tail_ce,
        "best_ce":  min(losses),
        "walltime": dt,
        "losses":   losses,
        "model":    model,
        "cfg":      cfg,
    }


def eval_k_sweep(model: FANT3Model, cfg, k_values: List[int],
                 batch_size: int, seq_len: int, seed: int,
                 device: torch.device) -> List[Dict]:
    """Evaluate the trained model at each K in k_values. Returns per-K metrics."""
    model.eval()
    rng = random.Random(seed)
    ids, targets = sample_batch(batch_size, seq_len, cfg.vocab_size, rng, device)

    results = []
    for k in k_values:
        model.mor.inference_k_override = k
        with torch.no_grad():
            out = model(ids, targets=targets)
        logits = out["logits"]
        preds  = logits.argmax(dim=-1)
        valid  = (targets != -100)
        correct = (preds == targets) & valid
        acc = correct.sum().item() / max(1, valid.sum().item())
        results.append({
            "k":    k,
            "loss": out["loss"].item(),
            "acc":  acc,
        })
    model.mor.inference_k_override = None
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep",     type=int, nargs="+",
                    default=[3, 6, 10, 16, 24],
                    help="n_recursion_depths values to sweep")
    ap.add_argument("--steps",     type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--seq-len",   type=int, default=15)
    ap.add_argument("--lr",        type=float, default=1e-3)
    ap.add_argument("--seed",      type=int, default=0)
    ap.add_argument("--eval-batch", type=int, default=16,
                    help="Batch size for K-extrapolation eval (more samples = less noise)")
    ap.add_argument("--device",    type=str, default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    print("=" * 78)
    print(f"K-depth sweep: n_recursion_depths ∈ {args.sweep}")
    print(f"  {args.steps} steps, batch {args.batch_size}, seq {args.seq_len}, "
          f"lr {args.lr}, seed {args.seed}, device {device}")
    print("=" * 78)

    trained = []
    for n_depths in args.sweep:
        print()
        print(f"--- training with n_recursion_depths={n_depths} ---")
        t0 = time.time()
        result = train_one(
            n_depths=n_depths,
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            seed=args.seed,
            device=device,
        )
        dt = time.time() - t0
        print(f"  first CE = {result['first_ce']:.3f}   "
              f"tail CE = {result['tail_ce']:.3f}   "
              f"best CE = {result['best_ce']:.3f}   "
              f"({dt:.1f}s, {result['n_params']/1e6:.2f}M params)")

        # K-extrapolation eval — sweep K from 1 up to 4x training max.
        k_values = sorted(set([1, max(2, n_depths // 2), n_depths,
                               n_depths * 2, n_depths * 4]))
        eval_res = eval_k_sweep(
            result["model"], result["cfg"], k_values,
            batch_size=args.eval_batch,
            seq_len=args.seq_len,
            seed=args.seed + 999,
            device=device,
        )
        result["eval"] = eval_res
        trained.append(result)

        print("  K-extrapolation eval:")
        for r in eval_res:
            note = ""
            if r["k"] == n_depths:                note = "  (training max)"
            elif r["k"] == 1:                     note = "  (training min)"
            elif r["k"] > n_depths:               note = f"  ({r['k']/n_depths:.0f}× extrapolation)"
            print(f"    K={r['k']:>3d}: loss={r['loss']:6.3f}  acc={r['acc']*100:5.2f}%{note}")

    # Summary table
    print()
    print("=" * 78)
    print("Summary")
    print("=" * 78)
    print(f"{'n_depths':>10s}  {'params':>8s}  {'first CE':>8s}  {'tail CE':>8s}  "
          f"{'best CE':>8s}  {'K=1 acc':>8s}  {'K=N acc':>8s}  {'K=2N acc':>8s}  {'K=4N acc':>8s}")
    print("-" * 96)
    for r in trained:
        k1  = next((e for e in r["eval"] if e["k"] == 1), None)
        kN  = next((e for e in r["eval"] if e["k"] == r["n_depths"]), None)
        k2N = next((e for e in r["eval"] if e["k"] == r["n_depths"] * 2), None)
        k4N = next((e for e in r["eval"] if e["k"] == r["n_depths"] * 4), None)
        fmt_acc = lambda e: f"{e['acc']*100:6.2f}%" if e else "   -   "
        print(f"{r['n_depths']:>10d}  "
              f"{r['n_params']/1e6:>7.2f}M  "
              f"{r['first_ce']:>8.3f}  "
              f"{r['tail_ce']:>8.3f}  "
              f"{r['best_ce']:>8.3f}  "
              f"{fmt_acc(k1):>8s}  "
              f"{fmt_acc(kN):>8s}  "
              f"{fmt_acc(k2N):>8s}  "
              f"{fmt_acc(k4N):>8s}")

    # Diagnosis
    print()
    print("Diagnosis:")
    best = min(trained, key=lambda r: r["tail_ce"])
    print(f"  Best training CE at n_depths={best['n_depths']} (tail CE {best['tail_ce']:.3f})")

    # Find largest n_depths where K-extrapolation doesn't catastrophically regress
    ok_extrap = []
    for r in trained:
        k_train = next(e for e in r["eval"] if e["k"] == r["n_depths"])
        k_2x    = next((e for e in r["eval"] if e["k"] == r["n_depths"] * 2), None)
        if k_2x is not None and k_2x["acc"] >= k_train["acc"] - 0.05:
            ok_extrap.append(r["n_depths"])
    if ok_extrap:
        print(f"  2× K-extrapolation safe (acc within 5pp of training-K) at n_depths ≤ "
              f"{max(ok_extrap)}")
    else:
        print("  2× K-extrapolation degrades > 5pp at every n_depths tried — "
              "model isn't learning the 'more passes = better' invariant")


if __name__ == "__main__":
    main()
