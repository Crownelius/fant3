"""
Quick sanity-test of output/overnight_default/step_500.pt — generates
completions on a few prompts so we can eyeball whether the model has
learned the ChatML structure before we throw it away and retrain.

Usage:
    PYTHONPATH=. python scripts/test_step500.py
"""
from __future__ import annotations

import os
import sys

import torch

# project root -> path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fant2.config import fant2_default
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.tokenizer.chat_template import apply_chat_template
from fant2.inference import FANT2Generator


CKPT = "output/overnight_default/step_500.pt"
TOKENIZER = "output/option_i/tokenizer.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


PROMPTS = [
    # ChatML-wrapped reasoning prompt
    {
        "label": "ChatML math",
        "messages": [
            {"role": "user", "content": "What is 7 * 8 + 12?"},
        ],
    },
    # ChatML-wrapped reasoning prompt
    {
        "label": "ChatML reasoning",
        "messages": [
            {"role": "user", "content": "If I have 3 apples and eat 1, how many are left?"},
        ],
    },
    # ChatML open-ended
    {
        "label": "ChatML open-ended",
        "messages": [
            {"role": "user", "content": "Explain why the sky is blue."},
        ],
    },
    # Raw text continuation (no ChatML)
    {
        "label": "raw continuation",
        "raw": "The quick brown fox",
    },
    # Raw text math prompt
    {
        "label": "raw math",
        "raw": "Problem: Solve 2x + 3 = 7.\nSolution:",
    },
]


def main() -> None:
    print("=" * 60)
    print(f"  Loading checkpoint: {CKPT}")
    print(f"  Device: {DEVICE}")
    print("=" * 60)

    # Tokenizer
    tok = FANT2Tokenizer.load(TOKENIZER)
    print(f"  Tokenizer vocab: {tok.vocab_size}")

    # Model
    cfg = fant2_default()
    model = FANT2Model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model stored: {n_params/1e6:.1f}M")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} missing keys (first 3: {missing[:3]})")
    if unexpected:
        print(f"  [warn] {len(unexpected)} unexpected keys (first 3: {unexpected[:3]})")

    step = ckpt.get("step", "?") if isinstance(ckpt, dict) else "?"
    print(f"  Loaded step: {step}")

    model.eval()
    if DEVICE == "cuda":
        model = model.to(DEVICE, dtype=torch.bfloat16)

    # Generator
    gen = FANT2Generator(model, tok, device=DEVICE)

    print("\n" + "=" * 60)
    print("  Generations")
    print("=" * 60)

    for i, item in enumerate(PROMPTS):
        print(f"\n[{i+1}/{len(PROMPTS)}] {item['label']}")
        print("-" * 60)

        if "messages" in item:
            prompt_text = apply_chat_template(
                item["messages"],
                add_generation_prompt=True,
                add_bos=True,
            )
            print(f"PROMPT (repr):\n  {prompt_text!r}")
        else:
            prompt_text = item["raw"]
            print(f"PROMPT:\n  {prompt_text!r}")

        # Two temperatures for each prompt: greedy + sampled
        for label, kwargs in [
            ("greedy",  dict(greedy=True, max_new_tokens=64)),
            ("t=0.8",   dict(greedy=False, temperature=0.8, top_k=50, top_p=0.95,
                             max_new_tokens=64)),
        ]:
            try:
                out = gen.generate(prompt_text, **kwargs)
            except Exception as e:
                out = f"[ERROR: {type(e).__name__}: {e}]"

            # If gen returns full text, strip the prompt
            if isinstance(out, str) and out.startswith(prompt_text):
                out = out[len(prompt_text):]

            # Truncate long outputs
            shown = out if len(str(out)) < 300 else str(out)[:300] + "..."
            print(f"\n  [{label}] {shown!r}")

    print("\n" + "=" * 60)
    print("  Done")
    print("=" * 60)


if __name__ == "__main__":
    main()
