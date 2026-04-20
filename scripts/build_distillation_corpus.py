"""
Build a sequence-level distillation corpus from openrouter/elephant-alpha.

Sources prompts from the existing training registry (NuminaMath problems,
Superior Reasoning stage-1 inputs, OpenR1 logic/puzzles) and asks the teacher
model to produce <think>...</think><answer>...</answer> responses with the
FANT-style chat schema.

The generated (prompt, response) pairs land in a JSONL cache that the trainer
consumes via ``LocalJSONLStream`` (registered as dataset name
``elephant-alpha-distill``).

Usage:
    # 1. make a throwaway inference key (uses OPENROUTER_PROVISIONING_KEY)
    python scripts/openrouter_keys.py create --label overnight-distill --out-file .openrouter_key

    # 2. generate, e.g., 150 samples (free tier daily cap is ~200)
    PYTHONPATH=. python scripts/build_distillation_corpus.py \\
        --n-samples 150 \\
        --sources numina,superior1,logic \\
        --cache data/distill_cache/elephant_alpha.jsonl

    # 3. when done, revoke the key
    python scripts/openrouter_keys.py delete --hash <hash-from-step-1>
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from typing import Iterator, List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fant2.data.openrouter_teacher import (
    DistillCache,
    TeacherClient,
    TeacherConfig,
    drive_distillation,
)


# --------------------------------------------------------------------------- #
# Prompt sources (all return a stream of raw user-facing questions)           #
# --------------------------------------------------------------------------- #

def _iter_numina(limit: Optional[int] = None) -> Iterator[str]:
    """NuminaMath-CoT problems."""
    try:
        from datasets import load_dataset
        ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", streaming=True)
    except Exception as e:
        print(f"  [numina] load failed: {e}")
        return
    n = 0
    for row in ds:
        problem = (row.get("problem") or "").strip()
        if len(problem) < 20 or len(problem) > 1500:
            continue
        yield problem
        n += 1
        if limit and n >= limit:
            break


def _iter_superior1(limit: Optional[int] = None) -> Iterator[str]:
    """Superior Reasoning stage-1 inputs."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "Alibaba-Apsara/Superior-Reasoning-SFT-gpt-oss-120b",
            "stage1", split="train", streaming=True,
        )
    except Exception as e:
        print(f"  [superior1] load failed: {e}")
        return
    n = 0
    for row in ds:
        q = (row.get("input") or "").strip()
        if len(q) < 20 or len(q) > 1500:
            continue
        yield q
        n += 1
        if limit and n >= limit:
            break


def _iter_logic(limit: Optional[int] = None) -> Iterator[str]:
    """OpenR1 logic and puzzles."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "sunyiyou/openr1_logic_and_puzzles_1k_lg",
            split="train", streaming=True,
        )
    except Exception as e:
        print(f"  [logic] load failed: {e}")
        return
    n = 0
    for row in ds:
        q = (row.get("problem") or "").strip()
        if len(q) < 20 or len(q) > 2000:
            continue
        yield q
        n += 1
        if limit and n >= limit:
            break


def _iter_kimi_science(limit: Optional[int] = None) -> Iterator[str]:
    """Kimi K2.5 PhD-science user turns (question only)."""
    try:
        from datasets import load_dataset
        ds = load_dataset(
            "ianncity/KIMI-K2.5-1000000x",
            "PHD-Science", split="train", streaming=True,
        )
    except Exception as e:
        print(f"  [kimi-sci] load failed: {e}")
        return
    n = 0
    for row in ds:
        msgs = row.get("messages") or []
        user_q: Optional[str] = None
        for m in msgs:
            role = m.get("role") or m.get("from", "")
            if role in ("user", "human"):
                user_q = (m.get("content") or m.get("value") or "").strip()
                break
        if not user_q or len(user_q) < 20 or len(user_q) > 1500:
            continue
        yield user_q
        n += 1
        if limit and n >= limit:
            break


SOURCES = {
    "numina":     _iter_numina,
    "superior1":  _iter_superior1,
    "logic":      _iter_logic,
    "kimi-sci":   _iter_kimi_science,
}


# --------------------------------------------------------------------------- #
# Interleaved prompt source                                                   #
# --------------------------------------------------------------------------- #

def interleaved_prompts(source_names: List[str], *, per_source: int,
                        seed: int = 42) -> Iterator[str]:
    """Round-robin through the named sources, each contributing up to N prompts."""
    iterators: List[Iterator[str]] = []
    for name in source_names:
        fn = SOURCES.get(name)
        if fn is None:
            print(f"  WARNING: unknown source '{name}', skipping")
            continue
        iterators.append(fn(limit=per_source))
    if not iterators:
        return

    rng = random.Random(seed)
    active = list(range(len(iterators)))
    while active:
        idx = rng.choice(active)
        try:
            yield next(iterators[idx])
        except StopIteration:
            active.remove(idx)


# --------------------------------------------------------------------------- #
# CLI                                                                         #
# --------------------------------------------------------------------------- #

def main() -> None:
    p = argparse.ArgumentParser(description="Build Elephant-Alpha distillation corpus")
    p.add_argument("--sources", type=str, default="numina,superior1,logic,kimi-sci",
                   help=f"comma-separated prompt sources; choose from {list(SOURCES.keys())}")
    p.add_argument("--per-source", type=int, default=200,
                   help="max prompts per source (hard cap before dedup)")
    p.add_argument("--n-samples", type=int, default=150,
                   help="total NEW samples to generate (stops once reached)")
    p.add_argument("--cache", type=str,
                   default="data/distill_cache/elephant_alpha.jsonl")
    p.add_argument("--key-file", type=str, default=".openrouter_key",
                   help="path to the JSON file holding the inference key")
    p.add_argument("--model", type=str, default="openrouter/elephant-alpha")
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--per-minute", type=int, default=18,
                   help="req/min cap (free tier = 20)")
    p.add_argument("--per-day", type=int, default=195,
                   help="req/day cap (free tier = 200)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    print("=" * 60)
    print("  FANT 2 — Elephant Alpha distillation corpus builder")
    print("=" * 60)
    print(f"  model:       {args.model}")
    print(f"  sources:     {sources}")
    print(f"  per-source:  {args.per_source}")
    print(f"  target:      {args.n_samples} NEW samples")
    print(f"  cache:       {args.cache}")
    print(f"  rate cap:    {args.per_minute}/min, {args.per_day}/day")
    print()

    cache = DistillCache(args.cache)
    client = TeacherClient(
        TeacherConfig(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            per_minute=args.per_minute,
            per_day=args.per_day,
        ),
        key_path=args.key_file,
    )

    prompts = interleaved_prompts(sources, per_source=args.per_source, seed=args.seed)

    try:
        added = drive_distillation(
            prompts, cache=cache, client=client,
            max_samples=args.n_samples,
        )
    except KeyboardInterrupt:
        print("\n  interrupted by user; cache is safe on disk")
        added = -1

    print()
    print(f"  cache now holds {len(cache)} total samples at {args.cache}")
    print(f"  added this run: {added}")


if __name__ == "__main__":
    main()
