"""
Phase-0 compression test — how good a probability model is our FANT 2 checkpoint?

The theoretical compression rate of an arithmetic coder driven by a language
model is exactly its cross-entropy in bits-per-byte:

    bpb(text) = ─Σᵢ log₂ p(xᵢ | x_<i)  /  len(text, in bytes)

So we can measure achievable compression WITHOUT building the arithmetic
coder — we just measure the model's log-likelihood on a held-out string.

Baselines: gzip, zlib, bz2, lzma.

The FANT 2 step_3000 checkpoint was trained on Kimi K2.5 + Opus 4.6 + web
reasoning data; it was NOT trained to compress. If its bpb beats gzip on
text it didn't see, we have something. If it can't, we know the ceiling.

Run:
    PYTHONPATH=. python scripts/compress_test.py
"""
from __future__ import annotations
import os
import sys
import math
import time
import gzip
import zlib
import bz2
import lzma

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from fant2.config import fant2_default
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer


CKPT = "output/overnight_opus46/step_3000.pt"
TOK  = "output/option_i/tokenizer.json"


# ------------------------------------------------------------------------
#  Test corpora (short + long). Both are out-of-distribution — hand-written
#  paragraphs about topics the model didn't see, mixing prose and code.
# ------------------------------------------------------------------------

TEST_PROSE = """The history of the Apollonian circle packing begins with the Greek geometer
Apollonius of Perga, who in the third century BC posed the problem of constructing
a circle tangent to three given mutually tangent circles. Descartes rediscovered
the problem in 1643 and derived the remarkable algebraic identity relating the
four curvatures, which today bears his name. If k1, k2, k3, k4 are the bends of
four mutually tangent circles, then (k1 + k2 + k3 + k4)^2 = 2(k1^2 + k2^2 + k3^2 + k4^2).
This quadratic form has signature (1, 3), matching the Minkowski metric of 3+1
dimensional spacetime, and the Apollonian group is a discrete subgroup of the
Lorentz group preserving the integer lattice Z^4. The integral Apollonian packings
were classified by Kocik in 2020 using tangency spinors of the Clifford algebra
associated to Minkowski space. Each pair of tangent circles defines a 2D integer
vector, and pairs of these vectors form Pauli spinors. Modern interest in the
packing comes from its appearance in number theory: the distribution of curvatures
obeys a power law with exponent approximately 1.305, and the set of realizable
curvatures modulo 24 satisfies the local-global conjecture proven by Bourgain,
Fuchs, and others. Spectral gap theorems for thin subgroups of SL(2, Z) give
quantitative bounds on how densely the packing fills the plane."""


TEST_CODE = '''def arithmetic_encode(symbols, probs):
    """Classic range coder. Narrows the interval [lo, hi) by p(symbol)
    for each symbol in the stream. Returns a bit string of the final interval."""
    lo, hi = 0, 1 << 32
    out = []
    for sym in symbols:
        p_lo, p_hi = probs[sym]
        rng = hi - lo
        hi = lo + (rng * p_hi) // 1
        lo = lo + (rng * p_lo) // 1
        # Renormalize: pull out common prefix bits
        while (lo >> 31) == (hi >> 31):
            out.append((lo >> 31) & 1)
            lo = (lo << 1) & ((1 << 32) - 1)
            hi = (hi << 1) & ((1 << 32) - 1) | 1
    out.append(1)
    return out

def arithmetic_decode(bits, n_symbols, probs_fn):
    """Inverse of above. Requires the SAME probability model as encode."""
    lo, hi = 0, 1 << 32
    code = 0
    for b in bits[:32]:
        code = (code << 1) | b
    symbols = []
    for i in range(n_symbols):
        rng = hi - lo
        offset = ((code - lo + 1) * 1 - 1) // rng
        sym = probs_fn(offset, symbols)
        symbols.append(sym)
    return symbols
'''


TEST_STRUCTURED = """{"model":"fant2-default","scale":"84.8M","dim":768,"layers":12,
"n_kv_heads":2,"head_dim":96,"n_experts":74,"top_k":4,"moe_hidden":1280,
"shared_expert_hidden":256,"vocab_size":32768,"max_seq_len":1024,
"cerebellum":{"in":768,"expand":7680,"out":768,"spectral_radius":0.95},
"apollonian":{"alpha_cap":5000,"beta_cap":5000,"curvature_threshold":0.5},
"optimizer":"HybridMuonAdamW","muon_lr":5e-4,"adam_lr":1.5e-4,
"training_phase":2,"n_steps":3000,"batch_size":4,"seq_len":512,"grad_accum":2,
"bf16":true,"use_8bit_adam":true,"grad_checkpoint":false,
"data_mix":{"kimi_distill":0.25,"kimi_math":0.10,"superior_s1":0.10,
"crownelius_opus46":0.15,"teichai_opus46":0.10,"numina_cot":0.10,
"finetome":0.10,"fineweb_edu":0.10}}"""


def classical_bpb(text: str) -> dict:
    """Bits-per-byte for each classical codec."""
    raw = text.encode("utf-8")
    n_bytes = len(raw)
    out = {"raw_bytes": n_bytes}
    for name, fn in [
        ("gzip", lambda x: gzip.compress(x, compresslevel=9)),
        ("zlib", lambda x: zlib.compress(x, level=9)),
        ("bz2",  lambda x: bz2.compress(x, compresslevel=9)),
        ("lzma", lambda x: lzma.compress(x, preset=9 | lzma.PRESET_EXTREME)),
    ]:
        c = fn(raw)
        out[name] = {
            "compressed_bytes": len(c),
            "bpb": 8 * len(c) / n_bytes,
            "ratio": len(c) / n_bytes,
        }
    return out


def model_bpb(text: str, model, tokenizer, cfg, device: str = "cpu", chunk: int = 512) -> dict:
    """
    Cross-entropy in bits-per-byte of the FANT model on `text`.

    We tokenize the full string, then compute  Σ ─log₂ p(xᵢ | x_<i)
    by running the model on sliding windows of `chunk` tokens. This
    matches what an arithmetic coder driven by the model would achieve.
    """
    raw = text.encode("utf-8")
    n_bytes = len(raw)

    tok_ids = tokenizer.encode(text, add_bos=True, add_eos=False)
    n_tokens = len(tok_ids)

    # Need at least 2 tokens to predict anything
    if n_tokens < 2:
        return {"error": "too short", "n_tokens": n_tokens}

    t0 = time.time()
    total_nll_nats = 0.0
    total_predicted_tokens = 0

    # Sliding window, always conditioning on full prefix up to `chunk` tokens
    # For simplicity: single-pass, covering as many tokens as fit in max_seq_len
    max_seq = min(chunk, cfg.max_seq_len)

    idx = 1  # start predicting from token index 1 (needs 1+ context)
    while idx < n_tokens:
        end = min(n_tokens, idx + max_seq - 1)
        ctx_start = max(0, end - max_seq)
        ctx_ids = tok_ids[ctx_start:end]
        targets = tok_ids[ctx_start + 1:end + 1] if end < n_tokens else tok_ids[ctx_start + 1:end]

        # Predictions for positions [ctx_start+1 .. end-1]
        # We only score positions that we haven't already scored
        x = torch.tensor([ctx_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            out = model(x)
        logits = out["logits"][0]                           # (T, V)
        log_probs = F.log_softmax(logits.float(), dim=-1)   # (T, V)

        # Score positions [ctx_start + 1 .. end - 1]  (index in global tok_ids)
        # These correspond to local positions [1 .. T-1] predicting [2 .. T]
        # Actually careful: out["logits"][i] predicts token at position i+1
        # So log_probs[i, tok[i+1]] is the right number.
        # We want positions where (global index > idx - 1) — i.e. haven't scored yet
        for local_i in range(len(ctx_ids) - 1):
            global_pos = ctx_start + local_i + 1  # global index of predicted token
            if global_pos < idx:
                continue  # already scored
            tgt = tok_ids[global_pos]
            total_nll_nats += -log_probs[local_i, tgt].item()
            total_predicted_tokens += 1

        idx = end
        if end >= n_tokens:
            break

    dt = time.time() - t0
    nll_bits = total_nll_nats / math.log(2.0)
    bpb = nll_bits / n_bytes

    return {
        "n_bytes":               n_bytes,
        "n_tokens_scored":       total_predicted_tokens,
        "avg_bits_per_token":    nll_bits / max(total_predicted_tokens, 1),
        "bpb":                   bpb,
        "ratio_vs_raw":          bpb / 8.0,  # fraction of original size
        "wall_seconds":          dt,
        "token_per_byte":        total_predicted_tokens / n_bytes,
    }


def print_report(label: str, text: str, classical: dict, model_result: dict):
    n = classical["raw_bytes"]
    print(f"\n{'='*66}")
    print(f"  {label}  ({n} bytes / {len(text)} chars)")
    print(f"{'='*66}")

    rows = []
    for codec in ("gzip", "zlib", "bz2", "lzma"):
        c = classical[codec]
        rows.append((codec, c["compressed_bytes"], c["bpb"]))
    if "error" not in model_result:
        rows.append(("FANT2-default",
                     int(math.ceil(model_result["bpb"] * n / 8)),
                     model_result["bpb"]))
    rows.sort(key=lambda r: r[2])

    print(f"  {'codec':<20} {'bytes':>8}  {'bpb':>6}   {'notes':<40}")
    print(f"  {'-'*20} {'-'*8}  {'-'*6}   {'-'*40}")
    for name, bytes_, bpb in rows:
        note = ""
        if name == "FANT2-default" and "error" not in model_result:
            note = (f"wall {model_result['wall_seconds']:.1f}s  "
                    f"{model_result['token_per_byte']:.3f} tok/byte")
        print(f"  {name:<20} {bytes_:>8}  {bpb:>6.3f}   {note:<40}")
    if "error" in model_result:
        print(f"  FANT2-default         SKIP   — {model_result['error']}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {CKPT} on {device}...")

    tok = FANT2Tokenizer.load(TOK)
    cfg = fant2_default()
    model = FANT2Model(cfg)
    ckpt = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    if device == "cuda":
        model = model.to(device, dtype=torch.bfloat16)
    print(f"  loaded step={ckpt.get('step', '?')}  vocab={tok.vocab_size}  device={device}")

    # Run each test corpus
    for label, text in [
        ("PROSE (Apollonian packing math, out-of-distribution)", TEST_PROSE),
        ("CODE (toy arithmetic coder, mixed English+Python)",    TEST_CODE),
        ("STRUCTURED (JSON of our own training config)",          TEST_STRUCTURED),
    ]:
        cl = classical_bpb(text)
        mb = model_bpb(text, model, tok, cfg, device=device, chunk=cfg.max_seq_len)
        print_report(label, text, cl, mb)

    print(f"\n{'='*66}")
    print(f"  Note: model bpb = cross-entropy H(x|ctx), the theoretical limit an")
    print(f"  arithmetic coder driven by this model would achieve (Delétang 2023).")
    print(f"  If FANT2 bpb beats classical codecs, Phase 1 (real arithmetic")
    print(f"  encoder) is worth building. Current model was trained for reasoning,")
    print(f"  not compression — so any win is structural, not optimized.")
    print(f"{'='*66}")


if __name__ == "__main__":
    main()
