"""
Local CPU smoke training for a ~1M FANT 3 model with ISRM-derived features.

What this validates end-to-end in a real training loop:
  1. Contractive alpha decay (mor_isrm_contractive)      — config flag
  2. Dynamic K sampling per batch (K ~ U[1, n_depths])   — training loop
  3. Monotonic CE loss penalty across MoR passes         — training loop
  4. K-extrapolation at eval (K > n_depths)              — eval section

Toy task: copy-then-repeat. Each sample is [a, b, c, d, SEP, a, b, c, d].
The model must attend to the prompt half to predict the completion half.
Easy for a 1M model to learn; tight enough that we can see loss move in
~100 steps on a laptop CPU.

Run:
    python scripts/train_1m_local.py                # full run, all features on
    python scripts/train_1m_local.py --ablate       # also runs a CE-only baseline

Usage notes:
  - Prints one row every LOG_EVERY steps with: step / ce / mono / k_sampled / grad_norm.
  - After training, evaluates copy accuracy at K=1, K=max, K=extrapolated.
  - Writes a short summary to stdout; no checkpoints saved.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import math
import time
from typing import List, Tuple

# Make `fant3` importable whether run from repo root or elsewhere.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
import torch.nn.functional as F

from fant3.config import fant3_1m
from fant3.model.fant3_model import FANT3Model


# ---------------------------------------------------------------------------
# Data: copy-then-repeat toy task
# ---------------------------------------------------------------------------

SEP_TOKEN = 1   # reserved separator
PAD_TOKEN = 0   # reserved pad (kept clear of the copy payload)


def sample_batch(batch_size: int, seq_len: int, vocab_size: int,
                 rng: random.Random, device: torch.device
                 ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Layout (seq_len must be odd — we put SEP in the middle):
        prompt:       [x0, x1, ..., x_{L-1}]        L = (seq_len - 1) // 2
        sep:          [SEP]
        target_half:  [x0, x1, ..., x_{L-1}]

    Input  = full sequence
    Target = same sequence, but we only mark the copy-half positions with
             real token ids; everything else gets -100 (ignored by CE).
    """
    half = (seq_len - 1) // 2
    assert 2 * half + 1 == seq_len

    # Payload vocab: everything except the reserved 0 (pad) and 1 (sep).
    lo = 2
    hi = vocab_size

    B = batch_size
    prompts = torch.tensor(
        [[rng.randrange(lo, hi) for _ in range(half)] for _ in range(B)],
        dtype=torch.long, device=device,
    )
    sep = torch.full((B, 1), SEP_TOKEN, dtype=torch.long, device=device)
    full = torch.cat([prompts, sep, prompts], dim=1)  # (B, seq_len)

    # Targets: shift-by-1 next-token.  We want the model to predict the copy
    # half. Mark those positions with the real next token; mark everything
    # BEFORE the copy half with -100 so CE ignores it.
    targets = torch.full_like(full, -100)
    # Position i in `full` produces logits used to predict `full[:, i+1]`.
    # The copy half starts at index `half + 1` (right after SEP). So logits
    # at positions (half, half+1, ..., seq_len-2) should predict the copy
    # tokens (half+1 ... seq_len-1).
    # We assign targets at positions (half, ..., seq_len-2) = copy-half ids.
    targets[:, half:seq_len - 1] = full[:, half + 1:seq_len]
    return full, targets


# ---------------------------------------------------------------------------
# Monotonic CE loss (standalone copy — same formula as the smoke test prototype)
# ---------------------------------------------------------------------------

def compute_monotonic_ce_loss(step_losses: List[torch.Tensor]) -> torch.Tensor:
    """Zero when losses are non-increasing; quadratic penalty on regressions.
    Graph-preserving in the single-step / no-violation cases so .backward()
    is always safe to call on the result."""
    if len(step_losses) < 2:
        return step_losses[0] * 0.0
    total = step_losses[0] * 0.0
    for i in range(1, len(step_losses)):
        violation = F.relu(step_losses[i] - step_losses[i - 1])
        total = total + violation ** 2
    return total / (len(step_losses) - 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_run(
    steps: int = 200,
    batch_size: int = 4,
    seq_len: int = 31,
    lr: float = 5e-4,
    use_contractive: bool = True,
    use_dynamic_k: bool = True,
    use_monotonic: bool = True,
    mono_weight: float = 0.3,
    log_every: int = 10,
    seed: int = 0,
    device: str = "cpu",
    label: str = "full",
) -> dict:
    """Run a training loop. Returns {'losses': [...], 'final_ce': float, ...}."""

    torch.manual_seed(seed)
    random.seed(seed)
    rng = random.Random(seed)
    dev = torch.device(device)

    cfg = fant3_1m()
    cfg.mor_isrm_contractive = use_contractive
    # vocab / seq check
    assert seq_len <= cfg.max_seq_len
    assert cfg.vocab_size >= 2048

    model = FANT3Model(cfg).to(dev)
    n_params = model.n_params()
    print(f"[{label}] model built: {n_params/1e6:.3f}M stored params")
    print(f"[{label}] cfg: contractive={use_contractive} dynamic_k={use_dynamic_k} "
          f"monotonic={use_monotonic} (w={mono_weight})")

    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))

    losses: List[float] = []
    monos:  List[float] = []
    ks:     List[int]   = []

    model.train()
    t0 = time.time()

    for step in range(1, steps + 1):
        ids, targets = sample_batch(batch_size, seq_len, cfg.vocab_size, rng, dev)

        # Sample K for this batch.  Dynamic K = uniform over [1, n_depths].
        if use_dynamic_k:
            k_sampled = rng.randint(1, cfg.n_recursion_depths)
        else:
            k_sampled = cfg.n_recursion_depths  # fixed at max — the pre-ISRM behaviour

        if use_monotonic and k_sampled >= 2:
            # Run forward at K=1..k_sampled, collect per-pass CE.
            step_losses: List[torch.Tensor] = []
            for pass_k in range(1, k_sampled + 1):
                model.mor.inference_k_override = pass_k
                out = model(ids, targets=targets)
                step_losses.append(out["loss"])
            model.mor.inference_k_override = None

            ce_loss   = step_losses[-1]                        # use deepest pass as the CE
            mono_loss = compute_monotonic_ce_loss(step_losses)
            total = ce_loss + mono_weight * mono_loss
        else:
            # Single forward at k_sampled.
            model.mor.inference_k_override = k_sampled
            out = model(ids, targets=targets)
            model.mor.inference_k_override = None
            ce_loss = out["loss"]
            mono_loss = ce_loss * 0.0    # zero but graph-preserved
            total = ce_loss

        optim.zero_grad()
        total.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0).item()
        optim.step()

        losses.append(ce_loss.item())
        monos.append(mono_loss.item())
        ks.append(k_sampled)

        if step == 1 or step % log_every == 0 or step == steps:
            dt = time.time() - t0
            print(f"[{label}] step {step:>4d}  ce={ce_loss.item():6.3f}  "
                  f"mono={mono_loss.item():7.4f}  k={k_sampled}  gnorm={grad_norm:5.2f}  "
                  f"elapsed={dt:5.1f}s")

    return {
        "label": label,
        "n_params": n_params,
        "losses": losses,
        "monos":  monos,
        "ks":     ks,
        "final_ce":     losses[-1],
        "best_ce":      min(losses),
        "first_ce":     losses[0],
        "model":        model,
        "cfg":          cfg,
    }


# ---------------------------------------------------------------------------
# K-extrapolation eval after training
# ---------------------------------------------------------------------------

def eval_k_extrapolation(model: FANT3Model, cfg, seq_len: int = 31,
                         batch_size: int = 8, seed: int = 99,
                         device: str = "cpu") -> dict:
    """Copy-accuracy at K=1, K=max_depth, K=2*max_depth (extrapolation)."""
    dev = torch.device(device)
    rng = random.Random(seed)
    model.eval()

    ids, targets = sample_batch(batch_size, seq_len, cfg.vocab_size, rng, dev)
    half = (seq_len - 1) // 2

    def _acc_at_k(k: int) -> Tuple[float, float]:
        model.mor.inference_k_override = k
        with torch.no_grad():
            out = model(ids, targets=targets)
        logits = out["logits"]                                  # (B, T, V)
        preds  = logits.argmax(dim=-1)                          # (B, T)
        # Accuracy on the copy half only (positions half..seq_len-2 predict copy).
        valid = targets != -100
        correct = (preds == targets) & valid
        acc = correct.sum().item() / max(1, valid.sum().item())
        return acc, out["loss"].item()

    res = {}
    for k in (1, cfg.n_recursion_depths, cfg.n_recursion_depths * 2):
        acc, loss = _acc_at_k(k)
        res[k] = {"acc": acc, "loss": loss}

    model.mor.inference_k_override = None
    return res


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps",      type=int,  default=200)
    ap.add_argument("--batch-size", type=int,  default=4)
    ap.add_argument("--seq-len",    type=int,  default=31)   # 15 + SEP + 15
    ap.add_argument("--lr",         type=float, default=5e-4)
    ap.add_argument("--mono-weight", type=float, default=0.3)
    ap.add_argument("--seed",       type=int,  default=0)
    ap.add_argument("--device",     type=str,  default="cpu")
    ap.add_argument("--ablate",     action="store_true",
                    help="Also run a CE-only baseline for comparison.")
    args = ap.parse_args()

    print("=" * 72)
    print("FANT 3 ~1M local ISRM training smoke")
    print("=" * 72)

    # Full run — all ISRM features on
    full = train_run(
        steps=args.steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        use_contractive=True,
        use_dynamic_k=True,
        use_monotonic=True,
        mono_weight=args.mono_weight,
        seed=args.seed,
        device=args.device,
        label="ISRM-on",
    )

    print()
    print(f"[ISRM-on] first CE:  {full['first_ce']:.3f}")
    print(f"[ISRM-on] final CE:  {full['final_ce']:.3f}")
    print(f"[ISRM-on] best CE:   {full['best_ce']:.3f}")
    print(f"[ISRM-on] uniform-random baseline for vocab=2048: {math.log(2048):.3f}")

    # K-extrapolation eval
    print()
    print("K-extrapolation eval (copy-half accuracy):")
    eval_res = eval_k_extrapolation(full["model"], full["cfg"],
                                    seq_len=args.seq_len, device=args.device)
    for k, r in eval_res.items():
        print(f"  K={k:>2d}: acc={r['acc']*100:5.2f}%  loss={r['loss']:.3f}")

    if args.ablate:
        print()
        print("=" * 72)
        print("Ablation: CE-only baseline (all ISRM features off)")
        print("=" * 72)
        base = train_run(
            steps=args.steps,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            lr=args.lr,
            use_contractive=False,
            use_dynamic_k=False,
            use_monotonic=False,
            mono_weight=0.0,
            seed=args.seed,
            device=args.device,
            label="ISRM-off",
        )
        print()
        print(f"[ISRM-off] first CE: {base['first_ce']:.3f}")
        print(f"[ISRM-off] final CE: {base['final_ce']:.3f}")
        print(f"[ISRM-off] best CE:  {base['best_ce']:.3f}")

        print()
        print("Summary:")
        print(f"  ISRM-on   first→final:  {full['first_ce']:.3f} → {full['final_ce']:.3f}"
              f"  (drop {full['first_ce']-full['final_ce']:+.3f})")
        print(f"  ISRM-off  first→final:  {base['first_ce']:.3f} → {base['final_ce']:.3f}"
              f"  (drop {base['first_ce']-base['final_ce']:+.3f})")


if __name__ == "__main__":
    main()
