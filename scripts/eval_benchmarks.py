#!/usr/bin/env python3
"""
Unified benchmark eval for FANT 3.

Supported:
  * gsm8k   — 1319 grade-school math problems, extract final number, exact match
  * mmlu    — 14042 multi-choice knowledge, eval via letter-logit comparison
  * math500 — 500 competition math, extract answer in \\boxed{...}

All three are DECONTAMINATED from the training mix via scripts/decontaminate.py.

Usage:
    python scripts/eval_benchmarks.py \
        --ckpt output/overnight_opus46/final.pt \
        --tokenizer output/tokenizer/tokenizer_v2.json \
        --benchmark gsm8k \
        --n 100

    python scripts/eval_benchmarks.py ... --benchmark mmlu --n 500
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_HERE))


# -----------------------------------------------------------------------------
# Common utilities
# -----------------------------------------------------------------------------

def _wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson CI for a binomial proportion — honest under small n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def _load_model_and_tok(ckpt_path: str, tok_path: str, device: str, dtype):
    """Load FANT3Model + tokenizer.
    Checkpoint may be:
      (a) bare state_dict
      (b) {'model': state_dict, 'cfg_scale': str}               # older
      (c) {'model': state_dict, 'cfg_dict': dict, ...}          # current — preferred
    """
    from fant3.config import (
        FANT3Config, fant3_smoke, fant3_742m,
    )
    from fant3.model.fant3_model import FANT3Model
    from fant2.tokenizer.bpe import FANT2Tokenizer

    tok = FANT2Tokenizer.load(tok_path)

    ck = torch.load(ckpt_path, map_location=device)
    if isinstance(ck, dict) and "model" in ck:
        state = ck["model"]
        scale = ck.get("cfg_scale", "1b")
        cfg_dict = ck.get("cfg_dict")
    else:
        state = ck
        scale = "1b"
        cfg_dict = None

    # Config reconstruction — prefer the saved cfg_dict, fall back to scale presets
    if cfg_dict is not None:
        cfg = FANT3Config(**cfg_dict)
        print(f"  rebuilt cfg from saved cfg_dict (dim={cfg.dim}, layers={cfg.n_layers})")
    else:
        # Fall-through for older checkpoints that only have cfg_scale
        builders = {
            "smoke": fant3_smoke,
            "742m":  fant3_742m,
            "1b":    lambda: FANT3Config(),
        }
        if scale not in builders:
            raise RuntimeError(
                f"Checkpoint has cfg_scale={scale!r} but no cfg_dict. "
                f"Known presets are {list(builders)}. This checkpoint is from "
                "the Colab notebook's inline cfg_50m/150m/350m builders — "
                "those configs aren't in fant3.config. Re-train with the current "
                "notebook, which saves cfg_dict alongside the model state."
            )
        cfg = builders[scale]()
    # Enable the new fixes (matches what we trained with)
    cfg.spinor_apollonian_enabled = getattr(cfg, "spinor_apollonian_enabled", True)
    cfg.ahn_enabled               = getattr(cfg, "ahn_enabled", True)

    model = FANT3Model(cfg)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (first 3): {missing[:3]}")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (first 3): {unexpected[:3]}")
    model.to(device=device, dtype=dtype)
    model.eval()
    return model, tok, cfg


@torch.no_grad()
def _greedy(model, tok, prompt_text: str, device, max_new: int = 256,
            stop_token_ids: list[int] | None = None) -> str:
    ids = torch.tensor([tok.encode(prompt_text)], device=device)
    prompt_len = ids.shape[1]
    stop_set = set(stop_token_ids or [])
    eos_id = tok._tok.token_to_id("<|im_end|>") or tok._tok.token_to_id("<|eos|>")
    if eos_id is not None:
        stop_set.add(eos_id)
    for _ in range(max_new):
        out = model(ids)
        nxt = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if nxt.item() in stop_set:
            break
    return tok.decode(ids[0, prompt_len:].tolist())


def _build_prompt(question: str) -> str:
    """Consistent user/assistant wrapping (matches training format)."""
    from fant2.tokenizer.chat_template import apply_chat_template
    return apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        add_bos=True,
    )


# -----------------------------------------------------------------------------
# GSM8K
# -----------------------------------------------------------------------------

_FINAL_NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")
_BOXED_RE       = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER_TAG_RE  = re.compile(r"<\|answer\|>\s*([^<]+?)\s*<\|/answer\|>", re.DOTALL)
_GSM8K_ANS_RE   = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

def _extract_number(s: str) -> str | None:
    """Extract the predicted final number from a completion.
    Priority: <|answer|>…<|/answer|> > \\boxed{…} > last plain number."""
    m = _ANSWER_TAG_RE.search(s)
    if m:
        inner = m.group(1)
        m2 = _FINAL_NUMBER_RE.search(inner)
        if m2:
            return m2.group(0).replace(",", "")
    m = _BOXED_RE.search(s)
    if m:
        m2 = _FINAL_NUMBER_RE.search(m.group(1))
        if m2:
            return m2.group(0).replace(",", "")
    nums = _FINAL_NUMBER_RE.findall(s)
    if nums:
        return nums[-1].replace(",", "")
    return None


def _gsm8k_gold(answer_field: str) -> str:
    """GSM8K's answer field ends with '#### N' — pull N."""
    m = _GSM8K_ANS_RE.search(answer_field)
    return (m.group(1) if m else answer_field.strip()).replace(",", "")


def eval_gsm8k(model, tok, device, n: int) -> dict:
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test")
    n = min(n, len(ds))
    correct = 0
    seen = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i >= n:
            break
        question = ex["question"]
        gold = _gsm8k_gold(ex["answer"])
        prompt = _build_prompt(question)
        completion = _greedy(model, tok, prompt, device, max_new=256)
        pred = _extract_number(completion)
        seen += 1
        if pred is not None and pred == gold:
            correct += 1
        if seen % 20 == 0:
            dt = time.time() - t0
            print(f"  [{seen}/{n}]  acc={correct/seen:.3f}  "
                  f"last: gold={gold} pred={pred}  ({dt:.0f}s)", flush=True)
    lo, hi = _wilson_ci(correct, seen)
    return {"benchmark": "gsm8k", "n": seen, "correct": correct,
            "accuracy": correct / seen, "ci95": (lo, hi)}


# -----------------------------------------------------------------------------
# MMLU
# -----------------------------------------------------------------------------

@torch.no_grad()
def _mmlu_letter_logits(model, tok, prompt_text: str, device) -> list[float]:
    """Return logits for the next-token A/B/C/D after a prompt ending in
    'The answer is ('. More efficient than greedy generation."""
    ids = torch.tensor([tok.encode(prompt_text)], device=device)
    out = model(ids)
    next_logits = out["logits"][0, -1]
    result = []
    for letter in ["A", "B", "C", "D"]:
        tid = tok._tok.token_to_id(letter)
        if tid is None:
            # BPE may keep a leading-space variant
            tid = tok._tok.token_to_id(" " + letter)
        if tid is None:
            # Fall back: encode the single char and take first id
            tid = tok.encode(letter)[0]
        result.append(float(next_logits[tid]))
    return result


def eval_mmlu(model, tok, device, n: int) -> dict:
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test")
    n = min(n, len(ds))
    correct = 0
    seen = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i >= n:
            break
        q = ex["question"]
        choices = ex["choices"]
        gold_idx = int(ex["answer"])
        prompt = (
            f"Question: {q}\n"
            f"A. {choices[0]}\n"
            f"B. {choices[1]}\n"
            f"C. {choices[2]}\n"
            f"D. {choices[3]}\n"
            f"Answer: "
        )
        logits = _mmlu_letter_logits(model, tok, prompt, device)
        pred_idx = int(max(range(4), key=lambda k: logits[k]))
        seen += 1
        if pred_idx == gold_idx:
            correct += 1
        if seen % 100 == 0:
            dt = time.time() - t0
            print(f"  [{seen}/{n}]  acc={correct/seen:.3f}  ({dt:.0f}s)", flush=True)
    lo, hi = _wilson_ci(correct, seen)
    return {"benchmark": "mmlu", "n": seen, "correct": correct,
            "accuracy": correct / seen, "ci95": (lo, hi)}


# -----------------------------------------------------------------------------
# MATH-500
# -----------------------------------------------------------------------------

def _math500_gold(answer: str) -> str:
    """MATH answers live in \\boxed{...}."""
    m = _BOXED_RE.search(answer)
    return (m.group(1) if m else answer).strip().replace(",", "")


def eval_math500(model, tok, device, n: int) -> dict:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    n = min(n, len(ds))
    correct = 0
    seen = 0
    t0 = time.time()
    for i, ex in enumerate(ds):
        if i >= n:
            break
        question = ex["problem"]
        gold = _math500_gold(ex["answer"])
        prompt = _build_prompt(question)
        completion = _greedy(model, tok, prompt, device, max_new=512)
        # For math500, try boxed first then number
        m = _BOXED_RE.search(completion)
        if m:
            pred = m.group(1).strip().replace(",", "")
        else:
            pred = _extract_number(completion)
        seen += 1
        if pred is not None and pred == gold:
            correct += 1
        if seen % 10 == 0:
            dt = time.time() - t0
            print(f"  [{seen}/{n}]  acc={correct/seen:.3f}  "
                  f"gold={gold} pred={pred}  ({dt:.0f}s)", flush=True)
    lo, hi = _wilson_ci(correct, seen)
    return {"benchmark": "math500", "n": seen, "correct": correct,
            "accuracy": correct / seen, "ci95": (lo, hi)}


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

BENCHES = {"gsm8k": eval_gsm8k, "mmlu": eval_mmlu, "math500": eval_math500}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="path to .pt checkpoint")
    ap.add_argument("--tokenizer", default="output/tokenizer/tokenizer_v2.json")
    ap.add_argument("--benchmark", choices=list(BENCHES), required=True)
    ap.add_argument("--n", type=int, default=100, help="number of problems to eval")
    ap.add_argument("--device", default=None, help="cuda or cpu (auto if omitted)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    print(f"Loading checkpoint: {args.ckpt}")
    model, tok, cfg = _load_model_and_tok(args.ckpt, args.tokenizer, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params/1e6:.1f}M params on {device} {dtype}")
    print(f"  vocab: {cfg.vocab_size}")
    print()
    print(f"Running {args.benchmark} on {args.n} problems ...")

    result = BENCHES[args.benchmark](model, tok, device, args.n)

    lo, hi = result["ci95"]
    print()
    print(f"=== RESULT: {result['benchmark']} ===")
    print(f"  n         {result['n']}")
    print(f"  correct   {result['correct']}")
    print(f"  accuracy  {result['accuracy']*100:.2f}%  [95% CI {lo*100:.2f}% – {hi*100:.2f}%]")


if __name__ == "__main__":
    main()
