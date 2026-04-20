"""
Run all 7 Campaign N variants sequentially and summarize results.

Usage:
    PYTHONPATH=. python -u scripts/run_all_campaign_n.py

This script calls run_campaign_n.py for each variant in turn, then
prints a comparison table at the end. Each variant takes ~1-2 hours
on CPU (2500 steps + 1K eval), so the full suite takes ~7-14 hours.

If a run fails, the script logs the error and continues to the next variant.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time


VARIANTS = ["n3", "n6", "n7", "n3_n6", "n3_n7", "n6_n7", "n3_n6_n7"]
BASEDIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASEDIR)


def run_variant(variant: str) -> dict | None:
    """Run one variant and return the results dict (or None on failure)."""
    print(f"\n{'='*64}")
    print(f"  STARTING: {variant}")
    print(f"{'='*64}\n")

    env = os.environ.copy()
    env["PYTHONPATH"] = PROJECT_ROOT
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, "-u",
             os.path.join(BASEDIR, "run_campaign_n.py"),
             "--variant", variant],
            env=env,
            cwd=PROJECT_ROOT,
            timeout=7200,  # 2 hour max per variant
        )
        dt = time.time() - t0
        if result.returncode != 0:
            print(f"  FAILED: {variant} (exit code {result.returncode}) "
                  f"after {dt/60:.1f} min")
            return None
    except subprocess.TimeoutExpired:
        dt = time.time() - t0
        print(f"  TIMEOUT: {variant} after {dt/60:.1f} min")
        return None
    except Exception as exc:
        dt = time.time() - t0
        print(f"  ERROR: {variant}: {exc} after {dt/60:.1f} min")
        return None

    # Read results JSON
    results_json = os.path.join(PROJECT_ROOT, "output", f"campaign_{variant}", "results.json")
    if not os.path.exists(results_json):
        print(f"  WARNING: results.json not found for {variant}")
        return None
    with open(results_json) as f:
        return json.load(f)


def main():
    print("=" * 64)
    print(" FANT 2 — Campaign N: Full 7-variant suite")
    print("=" * 64)
    print(f"  Variants: {', '.join(VARIANTS)}")
    print(f"  Estimated time: 7-14 hours total")
    print()

    results = {}
    t_total = time.time()

    for variant in VARIANTS:
        res = run_variant(variant)
        results[variant] = res

    dt_total = time.time() - t_total

    # Print comparison table
    print()
    print("=" * 72)
    print(" CAMPAIGN N — COMPARISON TABLE")
    print("=" * 72)
    print(f"  {'Variant':<12s}  {'Correct':>7s}  {'Total':>5s}  {'Accuracy':>8s}  {'CI_lo':>6s}  {'CI_hi':>6s}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}")

    # Baselines
    print(f"  {'L1.5 base':<12s}  {'546':>7s}  {'1000':>5s}  {'54.6%':>8s}  {'0.515':>6s}  {'0.577':>6s}")
    print(f"  {'N1 ortho':<12s}  {'76':>7s}  {'1000':>5s}  {' 7.6%':>8s}  {'0.061':>6s}  {'0.094':>6s}")
    print(f"  {'-'*12}  {'-'*7}  {'-'*5}  {'-'*8}  {'-'*6}  {'-'*6}")

    for variant in VARIANTS:
        res = results.get(variant)
        if res is None:
            print(f"  {variant:<12s}  {'FAILED':>7s}")
            continue
        pr = res.get("post_ramp", {})
        c = pr.get("correct", 0)
        t = pr.get("total", 0)
        acc = pr.get("accuracy", 0)
        ci = pr.get("wilson_ci_95", [0, 0])
        print(f"  {variant:<12s}  {c:>7d}  {t:>5d}  {acc*100:>7.1f}%  {ci[0]:>6.3f}  {ci[1]:>6.3f}")

    print()
    print(f"  Total wall time: {dt_total/3600:.1f} hours")
    print()

    # Save summary
    summary_path = os.path.join(PROJECT_ROOT, "output", "campaign_n_summary.json")
    summary = {
        "variants": list(results.keys()),
        "results": {},
        "wall_hours": dt_total / 3600,
    }
    for v, r in results.items():
        if r is not None:
            summary["results"][v] = {
                "accuracy": r.get("post_ramp", {}).get("accuracy"),
                "wilson_ci_95": r.get("post_ramp", {}).get("wilson_ci_95"),
                "correct": r.get("post_ramp", {}).get("correct"),
                "total": r.get("post_ramp", {}).get("total"),
            }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
