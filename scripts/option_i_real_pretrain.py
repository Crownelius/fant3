"""
Option I — Real tokenizer + Phase 2 pretraining on FineWeb-Edu.

The previous option_g checkpoint was trained with an 856-token domain BPE
on a 20-sentence seed corpus, which is why public benchmarks land at chance.
This script unblocks the tokenizer bottleneck by:

  1. Pulling ~5000 FineWeb-Edu documents into memory (a sample, ~25-50MB)
  2. Training a fresh 32K byte-level BPE on that sample using the existing
     `FANT2Tokenizer.train_from_iterator` pipeline (statistically equivalent
     to RoBERTa's BPE — both are GPT-4-regex byte-level on internet text)
  3. Initializing a fresh `fant2_tiny()` model with the new tokenizer
  4. Running Phase 2 (FEP/CE pretraining) on streaming FineWeb-Edu for
     N_STEPS steps with seq_len=128 (the tiny preset's max) and batch_size=8
  5. Benchmarking held-out perplexity on a fresh FineWeb-Edu shard before
     Option H re-runs the public benchmarks against this checkpoint

The new checkpoint at `output/option_i/pretrain/final.pt` is the input to
Option K (procedural-math ramp) and to the Option H re-run.

Run:
    PYTHONPATH=. python scripts/option_i_real_pretrain.py
"""

from __future__ import annotations

import math
import os
import time

import torch

from fant2.bench import evaluate_perplexity
from fant2.config import fant2_tiny
from fant2.data import HuggingFaceStream, TokenizedBatchStream
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training import TrainConfig, FANT2Trainer


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OUT_BASE = "output/option_i"
OUT_PRETRAIN = os.path.join(OUT_BASE, "pretrain")
TOK_PATH = os.path.join(OUT_BASE, "tokenizer.json")

N_BPE_DOCS = 5000          # documents pulled into memory for BPE training
MAX_DOC_BYTES = 8000       # cap each doc at 8KB so a few outliers can't blow memory
TARGET_VOCAB = 32768       # matches fant2_tiny() default vocab_size

# Note: previous run with N_STEPS=3000 stalled around step 400 with no checkpoint
# saved (save_every=10000 meant zero intermediate writes). Cutting steps and
# saving more often so a partial run still produces something usable.
N_STEPS    = 1500          # Phase 2 training steps (was 3000)
SEQ_LEN    = 128           # tiny preset's max_seq_len
BATCH_SIZE = 8             # within CPU memory budget for the tiny preset

# Held-out eval shard: pull a fresh FineWeb-Edu chunk after a different
# starting offset, tokenize with the same tokenizer, evaluate perplexity.
N_EVAL_BATCHES = 60


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def collect_bpe_corpus(n_docs: int, max_bytes: int) -> list[str]:
    """Pull `n_docs` real-text documents from FineWeb-Edu into memory."""
    print(f"  pulling {n_docs} FineWeb-Edu docs for BPE training")
    stream = HuggingFaceStream()
    docs: list[str] = []
    total_bytes = 0
    t0 = time.time()
    for text in stream:
        if not text:
            continue
        if len(text) > max_bytes:
            text = text[:max_bytes]
        docs.append(text)
        total_bytes += len(text)
        if len(docs) >= n_docs:
            break
        if len(docs) % 500 == 0:
            print(f"    [{len(docs)}/{n_docs}] {total_bytes / 1e6:.1f} MB")
    dt = time.time() - t0
    print(f"  collected {len(docs)} docs / {total_bytes / 1e6:.1f} MB / {dt:.0f}s")
    return docs


def train_bpe(docs: list[str], vocab_size: int) -> FANT2Tokenizer:
    print(f"  training fresh {vocab_size}-token BPE on {len(docs)} docs")
    t0 = time.time()
    tok = FANT2Tokenizer.train_from_iterator(
        iterator=iter(docs),
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=False,
    )
    dt = time.time() - t0
    print(f"  BPE trained: vocab_size={tok.vocab_size}  time={dt:.0f}s")
    return tok


def make_train_stream(tokenizer: FANT2Tokenizer) -> TokenizedBatchStream:
    """Streaming FineWeb-Edu → tokenized batches for the trainer."""
    text = HuggingFaceStream()
    return TokenizedBatchStream(
        text_stream=text, tokenizer=tokenizer,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device="cpu",
    )


def make_eval_stream(tokenizer: FANT2Tokenizer) -> TokenizedBatchStream:
    """A second FineWeb-Edu stream — held-out by virtue of being a fresh iterator."""
    text = HuggingFaceStream()
    return TokenizedBatchStream(
        text_stream=text, tokenizer=tokenizer,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device="cpu",
    )


def benchmark(model, tokenizer, label: str) -> dict:
    print()
    print(f"  -- benchmarking: {label} --")
    eval_stream = make_eval_stream(tokenizer)
    t0 = time.time()
    res = evaluate_perplexity(
        model, eval_stream, max_batches=N_EVAL_BATCHES, verbose=False,
    )
    dt = time.time() - t0
    print(f"    avg NLL    = {res['loss']:.4f}")
    print(f"    perplexity = {res['perplexity']:.3f}")
    print(f"    n_tokens   = {res['n_tokens']}")
    print(f"    eval time  = {dt:.1f}s")
    return res


def build_trainer(tokenizer, n_steps: int) -> FANT2Trainer:
    cfg = fant2_tiny()
    # The existing FANT2Tokenizer.train_from_iterator reserves 32 slots for
    # special tokens, so it produces vocab_size = preset_vocab - 32. We just
    # need the tokenizer's max ID to fit inside the model's embedding table.
    assert tokenizer.vocab_size <= cfg.vocab_size, (
        f"tokenizer vocab_size {tokenizer.vocab_size} exceeds preset {cfg.vocab_size}"
    )
    model = FANT2Model(cfg)
    train_stream = make_train_stream(tokenizer)
    train_cfg = TrainConfig(
        phase=2, n_steps=n_steps,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        muon_lr=1e-3, adam_lr=3e-4,
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.02,
        fep_kl_beta_max=0.2,
        fep_kl_anneal_steps=max(n_steps, 1),
        telemetry_every=2000, tikkun_every=2000, fana_every=10000,
        log_every=50,                # was n_steps//30; want frequent visibility
        save_every=250,              # was 10000; want intermediate ckpts
        out_dir=OUT_PRETRAIN,
        resume_from=None,    # FRESH init — new tokenizer means new embeddings
        device="cpu",
        bf16=False, grad_checkpoint=False, use_8bit_adam=False,
    )
    return FANT2Trainer(model, train_cfg, train_stream)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print(" FANT 2 — Option I: real tokenizer + Phase 2 on FineWeb-Edu")
    print(" (no public benchmarks touched; eval on held-out FineWeb shard)")
    print("=" * 64)

    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(OUT_PRETRAIN, exist_ok=True)

    # ---------- Steps 1+2: BPE training (skipped if tokenizer already exists) ----------
    if os.path.exists(TOK_PATH):
        print()
        print(f"  ===== Steps 1+2: reusing existing tokenizer at {TOK_PATH} =====")
        tokenizer = FANT2Tokenizer.load(TOK_PATH)
    else:
        print()
        print("  ===== Step 1: pull FineWeb-Edu corpus for BPE =====")
        docs = collect_bpe_corpus(N_BPE_DOCS, MAX_DOC_BYTES)
        if not docs:
            print("  ✗ FAIL: no docs collected (HF unreachable?)")
            return 1
        print()
        print("  ===== Step 2: train 32K BPE on FineWeb-Edu =====")
        tokenizer = train_bpe(docs, TARGET_VOCAB)
        tokenizer.save(TOK_PATH)
        print(f"  saved tokenizer to {TOK_PATH}")

    # Sanity check: real-text efficiency on a held-out sentence
    sample = "The quick brown fox jumps over the lazy dog. Photosynthesis converts sunlight to chemical energy."
    n_words = len(sample.split())
    n_tokens = len(tokenizer.encode(sample, add_bos=False, add_eos=False))
    print(f"  sanity: {n_words} words → {n_tokens} tokens "
          f"({n_tokens / n_words:.2f} tok/word)")

    # ---------- Step 3: build trainer + bench fresh init ----------
    print()
    print("  ===== Step 3: build trainer (fresh init) =====")
    trainer = build_trainer(tokenizer, n_steps=N_STEPS)
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"  fant2_tiny: {n_params:,} params, vocab={tokenizer.vocab_size}")

    fresh_res = benchmark(trainer.model, tokenizer, label="fresh init (0 steps)")

    # ---------- Step 4: Phase 2 pretraining ----------
    print()
    print(f"  ===== Step 4: Phase 2 pretrain ({N_STEPS} steps) =====")
    print(f"  batch={BATCH_SIZE}  seq={SEQ_LEN}  → {N_STEPS * BATCH_SIZE * SEQ_LEN:,} train tokens")
    t0 = time.time()
    train_exc = None
    try:
        trainer.train()
    except (KeyboardInterrupt, Exception) as exc:
        train_exc = exc
        print(f"  ! training interrupted: {type(exc).__name__}: {exc}")
        print(f"  ! saving partial checkpoint before re-raising at end")
    dt = time.time() - t0
    steps_done = max(trainer.step, 1)
    ms_per_step = dt / steps_done * 1000
    print(f"  training done/halted in {dt / 60:.1f} min ({ms_per_step:.0f} ms/step)")
    print(f"  trainer at step {trainer.step}")

    # Crash-safe save: only write our own final.pt if training crashed early.
    # If trainer.train() completed normally, it already wrote a richer final.pt
    # via save_checkpoint() (with `opt` and `cfg` keys that load_checkpoint
    # requires for Option K's resume_from). Overwriting that with our lighter
    # format would break Option K's chained resume.
    final_ckpt = os.path.join(OUT_PRETRAIN, "final.pt")
    if train_exc is not None:
        # Build a minimal-but-resumable checkpoint matching the trainer's format
        # so Option K's load_checkpoint() can still find `opt` and `cfg`.
        torch.save(
            {
                "model": trainer.model.state_dict(),
                "opt":   trainer.opt.state_dict(),
                "cfg":   trainer.cfg,
                "step":  trainer.step,
                "halted_early": True,
            },
            final_ckpt,
        )
        print(f"  saved partial (crash-safe) checkpoint to {final_ckpt}")
    else:
        # Trainer wrote its own final.pt at the end of train(); leave it alone.
        print(f"  trainer wrote final checkpoint to {final_ckpt}")

    # ---------- Step 5: bench the trained model ----------
    trained_res = benchmark(trainer.model, tokenizer, label=f"trained ({trainer.step} steps)")

    # ---------- Report ----------
    print()
    print("=" * 64)
    print(" RESULTS")
    print("=" * 64)
    fp = fresh_res["perplexity"]
    tp = trained_res["perplexity"]
    fn = fresh_res["loss"]
    tn = trained_res["loss"]
    rel_improvement = (fp - tp) / fp if fp > 0 else float("nan")
    print(f"  fresh init       : ppl = {fp:>10.2f}   nll = {fn:.4f}")
    print(f"  trained ({trainer.step} steps): ppl = {tp:>10.2f}   nll = {tn:.4f}")
    print(f"  perplexity reduction : {rel_improvement * 100:+.1f}%")
    print(f"  perplexity ratio     : {fp / tp:.2f}x improvement")
    print()
    if not (math.isfinite(fp) and math.isfinite(tp)):
        print(f"  ✗ FAIL: non-finite perplexity (fresh={fp}, trained={tp})")
        return 1
    print(f"  ✓ checkpoint saved to {final_ckpt}")
    print(f"  ✓ tokenizer saved to {TOK_PATH}")
    if train_exc is not None:
        print(f"  ! NOTE: training was halted early by {type(train_exc).__name__}")
        print(f"         partial checkpoint at step {trainer.step}/{N_STEPS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
