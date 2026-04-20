"""
Qwen-2.5-1.5B-Instruct compression benchmark on the same Gutenberg books we
ran FANT 2 on. Head-to-head: 1.5 B params (Qwen) vs 84.8 M params (FANT 2).

Measures:
  1. Cross-entropy bpb (theoretical compression rate)
  2. Wall time
  3. VRAM consumption
  4. Precision-invariance: same Qwen checkpoint in F32 vs BF16 → same bpb
     when probs are rounded to a shared integer grid (BendVM-compatible),
     demonstrating that low-precision inference is sufficient for compression.

Usage:
    PYTHONPATH=. python scripts/compress_qwen.py \
        --dtype bf16 --n-bytes 20000

    # For the precision-invariance test:
    PYTHONPATH=. python scripts/compress_qwen.py \
        --precision-test --n-bytes 5000
"""
from __future__ import annotations
import argparse
import os, sys, time, math, gzip, bz2, lzma
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

MODEL_NAME = "Qwen/Qwen2.5-1.5B"  # base model; -Instruct would also work

BOOKS = {
    "alice":     "data/gutenberg/alice.txt",
    "twocities": "data/gutenberg/tale_two_cities.txt",
    "frank":     "data/gutenberg/frankenstein.txt",
}


# --------------------------------------------------------------------------- #
#  Book loading  (copied from compress_book.py for self-containment)
# --------------------------------------------------------------------------- #

def strip_gutenberg_header(text: str) -> str:
    for m in ["*** START OF THE PROJECT GUTENBERG EBOOK",
              "*** START OF THIS PROJECT GUTENBERG EBOOK"]:
        i = text.find(m)
        if i >= 0:
            nl = text.find("\n", i)
            if nl >= 0:
                text = text[nl + 1:]
            break
    for m in ["*** END OF THE PROJECT GUTENBERG EBOOK",
              "*** END OF THIS PROJECT GUTENBERG EBOOK"]:
        i = text.find(m)
        if i >= 0:
            text = text[:i]
            break
    return text.strip()


def load_book(path: str, n_bytes: int) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc); break
        except UnicodeDecodeError:
            continue
    text = strip_gutenberg_header(text)
    if n_bytes > 0 and len(text) > n_bytes:
        text = text[:n_bytes]
    return text


def classical_bpb(raw: bytes) -> dict:
    n = len(raw)
    out = {"raw_bytes": n}
    for name, fn in [
        ("gzip", lambda x: gzip.compress(x, compresslevel=9)),
        ("bz2",  lambda x: bz2.compress(x, compresslevel=9)),
        ("lzma", lambda x: lzma.compress(x, preset=9 | lzma.PRESET_EXTREME)),
    ]:
        c = fn(raw)
        out[name] = {"bytes": len(c), "bpb": 8 * len(c) / n}
    return out


# --------------------------------------------------------------------------- #
#  Qwen loading
# --------------------------------------------------------------------------- #

def load_qwen(dtype_str: str, device: str = "cuda"):
    """Load Qwen 2.5 1.5B with the requested dtype. First load downloads (~3 GB)."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dtype = {"f32": torch.float32, "fp32": torch.float32,
             "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
             "fp16": torch.float16, "f16": torch.float16}[dtype_str.lower()]
    print(f"  Loading {MODEL_NAME} in dtype={dtype}...")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=dtype, trust_remote_code=True
    )
    model.to(device).eval()
    dt = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters())
    vram_gb = torch.cuda.memory_allocated() / 1e9 if device == "cuda" else 0
    print(f"  Loaded {n_params/1e6:.1f}M params in {dt:.1f}s  "
          f"VRAM={vram_gb:.2f} GB  dtype={dtype}")
    return model, tok, n_params, vram_gb


# --------------------------------------------------------------------------- #
#  Cross-entropy bpb (Qwen)
# --------------------------------------------------------------------------- #

def qwen_bpb(text: str, model, tok, device: str = "cuda", chunk: int = 1024) -> dict:
    raw = text.encode("utf-8")
    n_bytes = len(raw)

    enc = tok(text, return_tensors="pt").to(device)
    ids = enc.input_ids[0]
    n_tokens = ids.shape[0]
    if n_tokens < 2:
        return {"error": "too short"}

    total_nll = 0.0
    total_pred = 0
    t0 = time.time()
    idx = 1
    while idx < n_tokens:
        end = min(n_tokens, idx + chunk - 1)
        ctx_start = max(0, end - chunk)
        ctx_ids = ids[ctx_start:end].unsqueeze(0)
        with torch.no_grad():
            out = model(ctx_ids)
        logits = out.logits[0]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        for local_i in range(ctx_ids.shape[1] - 1):
            global_pos = ctx_start + local_i + 1
            if global_pos < idx:
                continue
            tgt = ids[global_pos].item()
            total_nll += -log_probs[local_i, tgt].item()
            total_pred += 1
        idx = end
        if end >= n_tokens:
            break
    dt = time.time() - t0
    nll_bits = total_nll / math.log(2.0)
    bpb = nll_bits / n_bytes
    return {
        "n_bytes": n_bytes,
        "n_tokens": int(n_tokens),
        "bpb": bpb,
        "bits_per_token": nll_bits / max(total_pred, 1),
        "token_per_byte": total_pred / n_bytes,
        "wall_s": dt,
    }


# --------------------------------------------------------------------------- #
#  Precision invariance test
# --------------------------------------------------------------------------- #

def precision_invariance_test(text: str, device: str = "cuda", n_quant_levels: int = 65536):
    """
    Show that Qwen-2.5 in BF16 vs F32 produces the SAME compressed bitstream
    when softmax probabilities are quantized to a common N-level integer grid
    before being fed to an arithmetic coder. This is the BendVM pairing: the
    coder uses exact integer range arithmetic, the LLM can be low-precision.

    We don't build the full arithmetic coder here — we just show that the
    quantized probability distributions are EQUAL bit-for-bit across precisions.
    """
    # ---- Load once in BF16, measure bpb ----
    model_bf16, tok, n_params, vram_bf16 = load_qwen("bf16", device)
    bpb_bf16 = qwen_bpb(text, model_bf16, tok, device=device)["bpb"]
    # Capture BF16 probabilities on a short prefix for byte-exact comparison
    ids_short = tok(text[:1000], return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        log_probs_bf16 = F.log_softmax(
            model_bf16(ids_short).logits[0].float(), dim=-1
        )
    # Quantize: map log-probs → integer ranges
    probs_bf16 = log_probs_bf16.exp()
    probs_bf16_int = (probs_bf16 * n_quant_levels).round().long()
    del model_bf16
    torch.cuda.empty_cache()

    # ---- Load in F32 ----
    model_f32, tok, _, vram_f32 = load_qwen("f32", device)
    bpb_f32 = qwen_bpb(text, model_f32, tok, device=device)["bpb"]
    with torch.no_grad():
        log_probs_f32 = F.log_softmax(
            model_f32(ids_short).logits[0].float(), dim=-1
        )
    probs_f32 = log_probs_f32.exp()
    probs_f32_int = (probs_f32 * n_quant_levels).round().long()
    del model_f32
    torch.cuda.empty_cache()

    # ---- Compare ----
    equal = (probs_bf16_int == probs_f32_int).all().item()
    n_diff = (probs_bf16_int != probs_f32_int).sum().item()
    total = probs_bf16_int.numel()
    print(f"\n{'='*66}")
    print(f"  PRECISION INVARIANCE  (Qwen 2.5 1.5B, {n_quant_levels}-level quant)")
    print(f"{'='*66}")
    print(f"  bpb (BF16):  {bpb_bf16:.4f}   VRAM: {vram_bf16:.2f} GB")
    print(f"  bpb (F32):   {bpb_f32:.4f}   VRAM: {vram_f32:.2f} GB")
    print(f"  bpb delta:   {abs(bpb_bf16 - bpb_f32):.4f}   (both measure same text)")
    print(f"  VRAM ratio:  F32 / BF16 = {vram_f32 / max(vram_bf16, 1e-9):.2f}×")
    print()
    print(f"  Quantized-prob equality on 1000-char prefix:")
    print(f"    integer-equal entries: {total - n_diff} / {total} "
          f"({100 * (1 - n_diff/total):.3f}%)")
    if equal:
        print(f"    ✓ BF16 and F32 produce BIT-IDENTICAL quantized probs")
        print(f"    → arithmetic coder output would match exactly")
    else:
        print(f"    ~ {n_diff} entries differ in their quantized bin")
        print(f"    → output would match for most tokens but not all at {n_quant_levels}-level")
        print(f"    (raising n_quant_levels decreases the match; lowering tightens it)")


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dtype", default="bf16", choices=["f32", "bf16", "fp16"])
    p.add_argument("--n-bytes", type=int, default=20000)
    p.add_argument("--precision-test", action="store_true",
                   help="Run the BF16-vs-F32 precision-invariance test")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.precision_test:
        # Use a short excerpt for the precision test (need to hold TWO copies)
        text = load_book(BOOKS["alice"], args.n_bytes)
        precision_invariance_test(text, device=device)
        return

    model, tok, n_params, vram_gb = load_qwen(args.dtype, device)

    rows = []
    for label, path in BOOKS.items():
        if not os.path.exists(path):
            continue
        text = load_book(path, args.n_bytes)
        raw = text.encode("utf-8")
        n = len(raw)
        print(f"\n{'='*78}")
        print(f"  {label.upper():<10}  {n} bytes ({len(text)} chars)")
        print(f"{'='*78}")
        cl = classical_bpb(raw)
        qb = qwen_bpb(text, model, tok, device=device)
        rows.append((label, n, qb["bpb"], cl["gzip"]["bpb"], cl["lzma"]["bpb"],
                     qb["wall_s"]))
        order = [("Qwen-2.5-1.5B", int(math.ceil(qb["bpb"] * n / 8)), qb["bpb"])]
        for c in ("gzip", "bz2", "lzma"):
            order.append((c, cl[c]["bytes"], cl[c]["bpb"]))
        order.sort(key=lambda r: r[2])
        print(f"  {'codec':<18} {'bytes':>8}  {'bpb':>6}   note")
        for name, byt, bpb in order:
            note = ""
            if name.startswith("Qwen"):
                note = f"wall {qb['wall_s']:.1f}s  {qb['token_per_byte']:.3f} tok/byte"
            print(f"  {name:<18} {byt:>8}  {bpb:>6.3f}   {note}")

    print(f"\n{'='*78}")
    print(f"  SUMMARY  —  Qwen-2.5-1.5B ({args.dtype.upper()}, {n_params/1e6:.0f}M params, "
          f"{vram_gb:.1f} GB) vs classical")
    print(f"{'='*78}")
    print(f"  {'book':<12} {'bytes':>7}  {'Qwen':>6}  {'gzip':>6}  {'lzma':>6}  {'wall':>5}")
    for lab, n, q, g, l, w in rows:
        print(f"  {lab:<12} {n:>7}  {q:>6.3f}  {g:>6.3f}  {l:>6.3f}  {w:>4.1f}s")

    print(f"\n  Compare to earlier FANT 2 (84.8M, 0.2 GB VRAM) run:")
    print(f"    alice      2.836       twocities 2.857     frank 2.606")


if __name__ == "__main__":
    main()
