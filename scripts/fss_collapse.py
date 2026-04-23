#!/usr/bin/env python
"""
Finite-size-scaling (FSS) collapse over the FANT scale ladder.

Reference: Privman (ed.), "Finite Size Scaling and Numerical Simulation
of Statistical Systems" (World Scientific, 1990), CERN CDS record 207748.

The MoE-aware ansatz for cross-entropy loss is

    CE(N, D)  =  N^{-alpha}  *  g(D * N^{-beta})

where N is the stored-parameter count, D is the training-token count, and
g is a universal scaling function that is the same across all scales once
the data are plotted in the collapsed coordinates x = D * N^{-beta} and
y = CE * N^{alpha}.

Usage
-----
    python scripts/fss_collapse.py \
        --ladder fant_ladder.json \
        --predict-1b   # recommend tokens for the 1B flagship run

where ``fant_ladder.json`` is a list of records of the form

    [
      {"name": "5m",   "N_stored": 8_330_000,   "D_tokens": 40_000_000,  "CE_final": 5.10},
      {"name": "40m",  "N_stored": 72_700_000,  "D_tokens": 80_000_000,  "CE_final": 4.05},
      ...
    ]

If ``--ladder`` is omitted the script runs on a synthetic toy ladder so
the caller can verify the collapse math on a device without the real logs.
"""

from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np


def _synthetic_ladder(seed: int = 0) -> List[dict]:
    # Toy ladder with known alpha=0.1, beta=0.2 so the optimizer must recover
    # them. CE = N^{-alpha} * (1 + (D * N^{-beta}) ** -0.3)  — arbitrary but
    # Chinchilla-shaped.
    rng = np.random.default_rng(seed)
    alpha_true, beta_true = 0.1, 0.2
    recs = []
    for name, N in [("5m", 8.33e6), ("40m", 7.27e7), ("150m", 9.6e7),
                    ("350m", 2.63e8), ("770m", 7.71e8)]:
        D = float(10 * N)
        x = D * N ** -beta_true
        ce_raw = N ** -alpha_true * (1.0 + x ** -0.3)
        ce = ce_raw * (1.0 + 0.005 * rng.standard_normal())
        recs.append({"name": name, "N_stored": N, "D_tokens": D, "CE_final": ce})
    return recs


def _collapse_residual(params: Tuple[float, float], recs: List[dict]) -> float:
    alpha, beta = params
    # Bound: collapsed coordinates must stay finite on the given ladder.
    if not (0.01 <= alpha <= 2.0 and 0.01 <= beta <= 2.0):
        return 1e6
    log_xs = [math.log(r["D_tokens"]) - beta * math.log(r["N_stored"]) for r in recs]
    log_ys = [math.log(r["CE_final"]) + alpha * math.log(r["N_stored"]) for r in recs]
    order = np.argsort(log_xs)
    lxs = np.array(log_xs)[order]
    lys = np.array(log_ys)[order]
    # Residual from a straight-line fit in log-log (simplest universal g):
    # a good collapse makes log y a smooth function of log x, so the
    # sum of squared residuals from the best linear fit measures scatter.
    slope, intercept = np.polyfit(lxs, lys, 1)
    pred = slope * lxs + intercept
    return float(np.mean((lys - pred) ** 2))


def fit_fss(recs: List[dict]) -> Tuple[float, float, float]:
    from scipy.optimize import minimize  # type: ignore
    best = minimize(
        _collapse_residual,
        x0=np.array([0.1, 0.2]),
        args=(recs,),
        method="Nelder-Mead",
        options={"xatol": 1e-5, "fatol": 1e-7},
    )
    alpha, beta = float(best.x[0]), float(best.x[1])
    return alpha, beta, float(best.fun)


def predict_1b_tokens(recs: List[dict], alpha: float, beta: float,
                      target_ce: float | None = None) -> dict:
    # At a new N* (1B), pick D* such that y* = CE* * N*^alpha equals the
    # mean y of the largest 2 ladder points (assumes asymptotic collapse).
    recs_sorted = sorted(recs, key=lambda r: r["N_stored"])
    top_two = recs_sorted[-2:]
    y_target = np.mean([r["CE_final"] * r["N_stored"] ** alpha for r in top_two])
    N_star = 1e9
    if target_ce is not None:
        ce_predicted = target_ce
    else:
        ce_predicted = y_target * N_star ** -alpha
    # From y* = CE* * N*^alpha, and the collapsed curve g(x), solve g(x*) = y*.
    # Simplest inversion: fit g as a loglog polynomial on (xs, ys) then invert.
    xs = np.array([r["D_tokens"] * r["N_stored"] ** -beta for r in recs_sorted])
    ys = np.array([r["CE_final"] * r["N_stored"] ** alpha for r in recs_sorted])
    coef = np.polyfit(np.log(xs), np.log(ys), deg=min(3, len(xs) - 1))
    # Invert: find x such that poly(log x) = log y_target
    log_y = math.log(max(y_target, 1e-9))
    roots = np.roots(np.concatenate([coef[:-1], [coef[-1] - log_y]]))
    real_pos = [float(r.real) for r in roots if abs(r.imag) < 1e-6 and r.real > 0]
    log_x_star = min(real_pos) if real_pos else float(np.log(xs.mean()))
    x_star = math.exp(log_x_star)
    D_star = x_star * N_star ** beta
    return {
        "N_flagship":        N_star,
        "D_recommended":     D_star,
        "D_per_param":       D_star / N_star,
        "CE_predicted":      ce_predicted,
        "x_target":          x_star,
        "alpha_fit":         alpha,
        "beta_fit":          beta,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ladder", type=Path, default=None)
    p.add_argument("--predict-1b", action="store_true")
    p.add_argument("--target-ce", type=float, default=None)
    args = p.parse_args()

    if args.ladder and args.ladder.exists():
        recs = json.loads(args.ladder.read_text())
        print(f"loaded {len(recs)} ladder points from {args.ladder}")
    else:
        recs = _synthetic_ladder()
        print("using synthetic ladder (toy alpha=0.1 beta=0.2)")

    alpha, beta, residual = fit_fss(recs)
    print(f"FSS fit:  alpha={alpha:.4f}  beta={beta:.4f}  residual={residual:.3e}")

    if args.predict_1b:
        pred = predict_1b_tokens(recs, alpha, beta, target_ce=args.target_ce)
        print("1B flagship projection:")
        for k, v in pred.items():
            print(f"  {k:20s} = {v:.3e}" if isinstance(v, float) else f"  {k:20s} = {v}")


if __name__ == "__main__":
    main()
