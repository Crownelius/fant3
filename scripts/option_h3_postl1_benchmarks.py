"""
Option H3 — Re-bench against the Option L1 checkpoint.

Same harness as `option_h2_postk_benchmarks.py`, only the checkpoint path
changes. Option L1 = Option I -> Phase 4 procedural-math ramp (two-pass
refinement + Apollonian memory writes activated). Option H3 measures
whether activating the dormant memory subsystem produced any benchmark
delta vs Option H2 (Option K = Option I -> Phase 2 procedural-math ramp).

  CKPT  : output/option_l1/math_ramp/final.pt   (Option I -> Option L1)
  TOK   : output/option_i/tokenizer.json         (32K BPE on FineWeb-Edu)

Expected delta vs Option H2 (Option K, phase=2):
  * GSM8K:   may move off 1% if two-pass + memory recovers some capacity
  * MMLU:    still ~25% — 5M params is too small for 57-subject knowledge
  * ARC-E/C: depends on whether two-pass gives enough effective depth to
             move off the un-normed-scorer artifact

Run:
    PYTHONPATH=. python scripts/option_h3_postl1_benchmarks.py
"""

from __future__ import annotations

import os
import time
import json
import argparse

import torch

from datasets import load_dataset

from fant2.bench import (
    evaluate_gsm8k,
    evaluate_mmlu,
    evaluate_arc_multichoice,
)
from fant2.config import fant2_tiny
from fant2.inference import FANT2Generator
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer


# -----------------------------------------------------------------------------
# Configuration — Option I tokenizer + Option L1 checkpoint
# -----------------------------------------------------------------------------

CKPT_PATH = "output/option_l1/math_ramp/final.pt"
TOK_PATH  = "output/option_i/tokenizer.json"
OUT_DIR   = "output/option_h3"
RESULTS_JSON = os.path.join(OUT_DIR, "results.json")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--gsm8k-n", type=int, default=100,
                   help="Number of GSM8K problems (default 100)")
    p.add_argument("--mmlu-n", type=int, default=200,
                   help="Number of MMLU problems (default 200)")
    p.add_argument("--arc-easy-n", type=int, default=200,
                   help="Number of ARC-Easy problems (default 200)")
    p.add_argument("--arc-challenge-n", type=int, default=200,
                   help="Number of ARC-Challenge problems (default 200)")
    p.add_argument("--gsm8k-max-new", type=int, default=96,
                   help="Max new tokens for GSM8K generation (default 96)")
    p.add_argument("--device", type=str, default="cpu",
                   help="cpu or cuda (default cpu — tiny model is fast on CPU)")
    return p.parse_args()


def load_model_and_tokenizer(device: str):
    print(f"  loading tokenizer from {TOK_PATH}")
    tokenizer = FANT2Tokenizer.load(TOK_PATH)
    print(f"    vocab_size = {tokenizer.vocab_size}")

    print(f"  building model (fant2_tiny preset)")
    cfg = fant2_tiny()
    model = FANT2Model(cfg)

    print(f"  loading checkpoint from {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        state = ckpt["model"]
    else:
        state = ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"    WARNING: {len(missing)} missing keys (e.g. {missing[:3]})")
    if unexpected:
        print(f"    WARNING: {len(unexpected)} unexpected keys (e.g. {unexpected[:3]})")
    print(f"    step in ckpt: {ckpt.get('step', 'unknown') if isinstance(ckpt, dict) else 'unknown'}")

    model.to(device)
    model.eval()
    return model, tokenizer


def run_gsm8k(model, tokenizer, n: int, max_new: int, device: str) -> dict:
    print()
    print("  ===== GSM8K =====")
    print(f"    loading gsm8k test split (streaming)")
    ds = load_dataset("gsm8k", "main", split="test", streaming=True)
    gen = FANT2Generator(model, tokenizer, device=device)
    t0 = time.time()
    res = evaluate_gsm8k(
        gen, ds,
        max_problems=n,
        max_new_tokens=max_new,
        temperature=0.0,
        verbose=True,
    )
    res["wall_seconds"] = time.time() - t0
    print(f"    GSM8K: {res['correct']}/{res['total']} = {res['accuracy']:.1%} "
          f"({res['wall_seconds']:.0f}s)")
    return res


def run_mmlu(model, tokenizer, n: int, device: str) -> dict:
    print()
    print("  ===== MMLU =====")
    print(f"    loading mmlu/all test split (streaming)")
    ds = load_dataset("cais/mmlu", "all", split="test", streaming=True)
    t0 = time.time()
    res = evaluate_mmlu(
        model, tokenizer, ds,
        max_problems=n,
        device=device,
        verbose=True,
    )
    res["wall_seconds"] = time.time() - t0
    print(f"    MMLU: {res['correct']}/{res['total']} = {res['accuracy']:.1%} "
          f"({res['wall_seconds']:.0f}s)")
    return res


def run_arc(model, tokenizer, name: str, n: int, device: str) -> dict:
    print()
    print(f"  ===== {name} =====")
    print(f"    loading {name} test split (streaming)")
    ds = load_dataset("allenai/ai2_arc", name, split="test", streaming=True)
    t0 = time.time()
    res = evaluate_arc_multichoice(
        model, tokenizer, ds,
        max_problems=n,
        device=device,
        verbose=True,
    )
    res["wall_seconds"] = time.time() - t0
    print(f"    {name}: {res['correct']}/{res['total']} = {res['accuracy']:.1%} "
          f"({res['wall_seconds']:.0f}s)")
    return res


def main() -> int:
    args = parse_args()

    print("=" * 64)
    print(" FANT 2 — Option H3: re-bench against Option L1 checkpoint")
    print(" (eval-only — no training touches these benchmarks)")
    print("=" * 64)

    if not os.path.exists(CKPT_PATH):
        print(f"  ✗ checkpoint not found at {CKPT_PATH}")
        print(f"    run scripts/option_l1_phase4_ramp.py first")
        return 1
    if not os.path.exists(TOK_PATH):
        print(f"  ✗ tokenizer not found at {TOK_PATH}")
        print(f"    run scripts/option_i_real_pretrain.py first")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(args.device)

    results: dict = {
        "ckpt": CKPT_PATH,
        "tokenizer": TOK_PATH,
        "vocab_size": tokenizer.vocab_size,
        "device": args.device,
    }

    for name, fn in [
        ("gsm8k", lambda: run_gsm8k(
            model, tokenizer, args.gsm8k_n, args.gsm8k_max_new, args.device,
        )),
        ("mmlu", lambda: run_mmlu(
            model, tokenizer, args.mmlu_n, args.device,
        )),
        ("arc_easy", lambda: run_arc(
            model, tokenizer, "ARC-Easy", args.arc_easy_n, args.device,
        )),
        ("arc_challenge", lambda: run_arc(
            model, tokenizer, "ARC-Challenge", args.arc_challenge_n, args.device,
        )),
    ]:
        try:
            results[name] = fn()
        except Exception as exc:
            print(f"    ✗ {name} failed: {exc!r}")
            results[name] = {"error": repr(exc)}

    print()
    print("=" * 64)
    print(" RESULTS — Option H3 (Option L1 checkpoint)")
    print("=" * 64)
    print(f"  ckpt           : {CKPT_PATH}")
    print(f"  tokenizer      : {TOK_PATH}")
    print(f"  vocab_size     : {tokenizer.vocab_size}")
    print(f"  device         : {args.device}")
    print()
    print(f"  {'benchmark':<18} {'score':>10}   {'n':>6}   {'time':>8}")
    print(f"  {'-'*18} {'-'*10}   {'-'*6}   {'-'*8}")
    for label, key in [
        ("GSM8K (greedy)",       "gsm8k"),
        ("MMLU (all subjects)",  "mmlu"),
        ("ARC-Easy",             "arc_easy"),
        ("ARC-Challenge",        "arc_challenge"),
    ]:
        r = results.get(key, {})
        if "error" in r:
            print(f"  {label:<18} {'ERROR':>10}   {'-':>6}   {'-':>8}    ({r['error'][:60]})")
            continue
        acc = r.get("accuracy", float("nan"))
        n   = r.get("total",    0)
        wt  = r.get("wall_seconds", 0.0)
        print(f"  {label:<18} {acc*100:>9.1f}%   {n:>6d}   {wt:>7.0f}s")
    print()

    print("  Option H2 baseline (Option K = phase=2 ramp):")
    print("    GSM8K          1.0%")
    print("    MMLU          24.5%")
    print("    ARC-Easy      24.5%")
    print("    ARC-Challenge 25.5%")
    print()

    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  full results JSON: {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
