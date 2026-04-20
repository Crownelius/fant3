"""
Book-compression benchmark — strictly out-of-distribution test.

Runs the FANT 2 step_3000 cross-entropy benchmark on the first N bytes of a
Project Gutenberg book (default 20 KB) and compares to gzip/bz2/lzma. The
books are 19th-century English — vocabulary and style are nothing like the
Opus 4.6 / Kimi K2.5 reasoning data FANT 2 was trained on, so this settles
the training-contamination question from Phase 0.

Run:
    PYTHONPATH=. python scripts/compress_book.py
"""
from __future__ import annotations
import argparse
import os, sys, time, math, gzip, zlib, bz2, lzma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from fant2.config import fant2_default
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer


CKPT = "output/overnight_opus46/step_3000.pt"
TOK  = "output/option_i/tokenizer.json"
BOOKS = {
    "alice":     "data/gutenberg/alice.txt",
    "twocities": "data/gutenberg/tale_two_cities.txt",
    "frank":     "data/gutenberg/frankenstein.txt",
}


def classical_bpb(text_bytes: bytes) -> dict:
    n = len(text_bytes)
    out = {"raw_bytes": n}
    for name, fn in [
        ("gzip", lambda x: gzip.compress(x, compresslevel=9)),
        ("zlib", lambda x: zlib.compress(x, level=9)),
        ("bz2",  lambda x: bz2.compress(x, compresslevel=9)),
        ("lzma", lambda x: lzma.compress(x, preset=9 | lzma.PRESET_EXTREME)),
    ]:
        c = fn(text_bytes)
        out[name] = {"bytes": len(c), "bpb": 8 * len(c) / n}
    return out


def fant_bpb(text: str, model, tok, cfg, device: str, chunk: int) -> dict:
    raw = text.encode("utf-8")
    n_bytes = len(raw)
    ids = tok.encode(text, add_bos=True, add_eos=False)
    n_tokens = len(ids)
    if n_tokens < 2:
        return {"error": "too short"}

    max_seq = min(chunk, cfg.max_seq_len)
    total_nll = 0.0
    total_pred = 0
    idx = 1
    t0 = time.time()
    while idx < n_tokens:
        end = min(n_tokens, idx + max_seq - 1)
        ctx_start = max(0, end - max_seq)
        ctx_ids = ids[ctx_start:end]
        x = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x)
        logits = out["logits"][0]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        for local_i in range(len(ctx_ids) - 1):
            global_pos = ctx_start + local_i + 1
            if global_pos < idx:
                continue
            tgt = ids[global_pos]
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
        "n_tokens": n_tokens,
        "bpb": bpb,
        "bits_per_token": nll_bits / max(total_pred, 1),
        "token_per_byte": total_pred / n_bytes,
        "wall_s": dt,
    }


def strip_gutenberg_header(text: str) -> str:
    """Gutenberg texts have boilerplate headers/footers — cut them out."""
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
    ]
    for m in start_markers:
        i = text.find(m)
        if i >= 0:
            nl = text.find("\n", i)
            if nl >= 0:
                text = text[nl + 1:]
            break
    for m in end_markers:
        i = text.find(m)
        if i >= 0:
            text = text[:i]
            break
    return text.strip()


def load_book(path: str, n_bytes: int) -> str:
    """Read the first `n_bytes` of stripped text from a Gutenberg file."""
    with open(path, "rb") as f:
        raw = f.read()
    # Try UTF-8, fall back with BOM stripping
    for enc in ("utf-8-sig", "utf-8", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    text = strip_gutenberg_header(text)
    if n_bytes > 0 and len(text) > n_bytes:
        text = text[:n_bytes]
    return text


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-bytes", type=int, default=20_000,
                   help="trim each book to this many chars (0 = full)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--ckpt", default=CKPT)
    args = p.parse_args()

    device = args.device
    print(f"Loading FANT 2 from {args.ckpt} on {device}...")
    tok = FANT2Tokenizer.load(TOK)
    cfg = fant2_default()
    model = FANT2Model(cfg)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck["model"] if "model" in ck else ck
    model.load_state_dict(state, strict=False)
    model.eval()
    if device == "cuda":
        model = model.to(device, dtype=torch.bfloat16)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  loaded step={ck.get('step','?')}  {n_params/1e6:.1f}M params  vocab={tok.vocab_size}")

    if device == "cuda":
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

    # Summary row accumulators
    summary = []

    for label, path in BOOKS.items():
        if not os.path.exists(path):
            print(f"\n[{label}] file missing: {path}")
            continue

        text = load_book(path, args.n_bytes)
        raw = text.encode("utf-8")
        n = len(raw)

        print(f"\n{'='*78}")
        print(f"  {label.upper():<10}  {path}  —  {n} bytes ({len(text)} chars)")
        print(f"{'='*78}")

        cl = classical_bpb(raw)
        fb = fant_bpb(text, model, tok, cfg, device=device,
                      chunk=cfg.max_seq_len)

        # Pretty table
        rows = [("FANT2-default", int(math.ceil(fb["bpb"] * n / 8)), fb["bpb"])]
        for codec in ("gzip", "zlib", "bz2", "lzma"):
            rows.append((codec, cl[codec]["bytes"], cl[codec]["bpb"]))
        rows.sort(key=lambda r: r[2])

        print(f"  {'codec':<15} {'bytes':>8}  {'bpb':>6}   note")
        print(f"  {'-'*15} {'-'*8}  {'-'*6}   {'-'*35}")
        for name, byt, bpb in rows:
            note = ""
            if name == "FANT2-default":
                note = f"wall {fb['wall_s']:.1f}s  {fb['token_per_byte']:.3f} tok/byte"
            print(f"  {name:<15} {byt:>8}  {bpb:>6.3f}   {note}")

        # First 200 chars of the book for context
        preview = text[:200].replace("\n", " ").replace("\r", " ")
        print(f"\n  preview: {preview!r}")

        summary.append((label, n, fb["bpb"], cl["gzip"]["bpb"],
                        cl["lzma"]["bpb"], fb["wall_s"]))

    # Summary
    if summary:
        print(f"\n{'='*78}")
        print(f"  SUMMARY  —  FANT2-default (84.8M) vs classical on Gutenberg OOD text")
        print(f"{'='*78}")
        print(f"  {'book':<12} {'bytes':>7}  {'FANT':>6}  {'gzip':>6}  {'lzma':>6}  {'wall':>5}  verdict")
        for lab, n, f, g, l, w in summary:
            if f < g and f < l:
                v = "FANT beats both"
            elif f < g:
                v = "FANT beats gzip but not lzma"
            elif f < l:
                v = "FANT beats lzma but not gzip"
            else:
                v = "both classical win"
            print(f"  {lab:<12} {n:>7}  {f:>6.3f}  {g:>6.3f}  {l:>6.3f}  {w:>4.1f}s  {v}")


if __name__ == "__main__":
    main()
