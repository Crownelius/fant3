"""
Qualitative sample — 5 procedural-math prompts through step_2000.pt on CPU.
No GPU contention with training (PID 25472). Takes ~3-5 min.
"""
from __future__ import annotations
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fant2.config import fant2_default
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.inference import FANT2Generator
from fant2.training.phase5_rollout import ProceduralMathStream, format_prompt


CKPT = "output/overnight_opus46/step_2000.pt"
TOK  = "output/option_i/tokenizer.json"


def main():
    print(f"Loading {CKPT} on CPU...")
    tok = FANT2Tokenizer.load(TOK)
    cfg = fant2_default()
    model = FANT2Model(cfg)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt, strict=False)
    model.eval()
    # bump output gates + curvature threshold like eval_1k.py does
    for m in model.modules():
        if hasattr(m, "output_gate") and isinstance(m.output_gate, torch.nn.Parameter):
            with torch.no_grad():
                m.output_gate.fill_(0.1)
    model.memory.curvature_threshold = 1.0

    gen = FANT2Generator(model, tok, device="cpu")
    stream = ProceduralMathStream(seed=9999, max_value=12)
    it = iter(stream)

    print("\n" + "=" * 70)
    for i in range(5):
        ex = next(it)
        prompt = format_prompt(ex.question)
        print(f"\n[{i+1}/5] Q: {ex.question!r}")
        print(f"         gold: {ex.gold_answer!r}")
        print(f"         prompt (repr): {prompt[:200]!r}")
        t0 = time.time()
        out = gen.generate(prompt, max_new_tokens=64, greedy=True, return_full_text=False)
        dt = time.time() - t0
        print(f"         [{dt:.1f}s] completion: {out[:400]!r}", flush=True)


if __name__ == "__main__":
    main()
