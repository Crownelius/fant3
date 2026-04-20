"""
Option K — Heavy procedural-math ramp + held-out procedural eval.

Loads the Option I checkpoint (real BPE on FineWeb-Edu), continues with
N_STEPS of Phase 2 cross-entropy training on procedurally-generated math
problems formatted with the FANT 2 <think>/<answer> structured prompt.

Then evaluates on a SEPARATE held-out seed of the same procedural stream
(no public benchmarks touched).

The point: demonstrate that the model CAN learn the procedural-math
distribution, and quantify the accuracy on the in-distribution procedural
eval — this is the "FANT 2's native task" measurement that the no-benchmark
constraint allows.

Run:
    PYTHONPATH=. python scripts/option_k_procedural_ramp.py
"""

from __future__ import annotations

import os
import re
import math
import time
import json
from typing import Iterator, List

import torch

from fant2.config import fant2_tiny
from fant2.data import TokenizedBatchStream
from fant2.inference import FANT2Generator
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training import TrainConfig, FANT2Trainer
from fant2.training.phase5_rollout import ProceduralMathStream, format_prompt


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

OPTION_I_CKPT = "output/option_i/pretrain/final.pt"
OPTION_I_TOK  = "output/option_i/tokenizer.json"

OUT_BASE = "output/option_k"
OUT_RAMP = os.path.join(OUT_BASE, "math_ramp")
RESULTS_JSON = os.path.join(OUT_BASE, "results.json")

N_STEPS    = 2500          # heavy procedural-math ramp
SEQ_LEN    = 128
BATCH_SIZE = 8
TRAIN_SEED = 11            # procedural math training seed
EVAL_SEED  = 9999          # held-out procedural eval seed (different from training)
EVAL_MAX_VALUE = 12        # in-distribution: small numbers same as training
EVAL_N_PROBLEMS = 200
EVAL_MAX_NEW_TOKENS = 64


# -----------------------------------------------------------------------------
# Procedural math text stream — wraps ProceduralMathStream into a string stream
# that TokenizedBatchStream can consume.
# -----------------------------------------------------------------------------

class ProceduralMathTextStream:
    """
    Yields formatted training strings from a ProceduralMathStream.

    Each yielded string is:
        Question: <q>
        Solve step by step. Wrap reasoning in <think>...</think> and the final
        answer in <answer>...</answer>.
        <think>
        Let me work it out. The answer is <gold>.
        </think>
        <answer><gold></answer>

    This teaches the model BOTH the reasoning format AND the answer.
    """

    def __init__(self, seed: int, max_value: int = 12):
        self.stream = ProceduralMathStream(seed=seed, max_value=max_value)

    def __iter__(self) -> Iterator[str]:
        for ex in self.stream:
            prompt = format_prompt(ex.question)
            answer_block = (
                f" Let me work it out. The answer is {ex.gold_answer}.\n"
                f"</think>\n"
                f"<answer>{ex.gold_answer}</answer>"
            )
            yield prompt + answer_block


def make_train_stream(tokenizer: FANT2Tokenizer) -> TokenizedBatchStream:
    text = ProceduralMathTextStream(seed=TRAIN_SEED, max_value=EVAL_MAX_VALUE)
    return TokenizedBatchStream(
        text_stream=text, tokenizer=tokenizer,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN, device="cpu",
    )


# -----------------------------------------------------------------------------
# Procedural math eval — generate, extract, compare
# -----------------------------------------------------------------------------

_ANSWER_TAG = re.compile(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>")
_ANY_NUM    = re.compile(r"-?\d+(?:\.\d+)?")


def _extract_answer(text: str) -> str | None:
    m = _ANSWER_TAG.search(text)
    if m:
        return m.group(1)
    nums = _ANY_NUM.findall(text)
    if nums:
        return nums[-1]
    return None


def evaluate_procedural_math(
    model, tokenizer, *, seed: int, max_value: int, n_problems: int,
    max_new_tokens: int, device: str = "cpu", verbose: bool = True,
) -> dict:
    """
    Generate continuations for `n_problems` procedural math prompts, extract
    the numeric answer, compare to gold. Returns {correct, total, accuracy}.

    Uses a held-out seed (different from training), so the actual problems
    are unseen even though the distribution matches.
    """
    gen = FANT2Generator(model, tokenizer, device=device)
    stream = ProceduralMathStream(seed=seed, max_value=max_value)
    it = iter(stream)
    correct = 0
    total = 0
    extracted_count = 0
    examples_log: List[dict] = []
    t0 = time.time()
    for i in range(n_problems):
        try:
            ex = next(it)
        except StopIteration:
            break
        prompt = format_prompt(ex.question)
        completion = gen.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            greedy=True,
            return_full_text=False,
        )
        pred = _extract_answer(completion)
        if pred is not None:
            extracted_count += 1
        is_correct = (pred is not None) and (pred.lstrip("0") == ex.gold_answer.lstrip("0")
                                              or pred == ex.gold_answer
                                              or (pred == "0" and ex.gold_answer == "0"))
        # Looser numeric compare
        if pred is not None and not is_correct:
            try:
                is_correct = float(pred) == float(ex.gold_answer)
            except ValueError:
                pass
        total += 1
        if is_correct:
            correct += 1
        if i < 5:
            examples_log.append({
                "q": ex.question,
                "gold": ex.gold_answer,
                "completion": completion[:300],
                "pred": pred,
                "correct": is_correct,
            })
        if verbose and (i + 1) % 25 == 0:
            print(f"    [proc-math {i+1}/{n_problems}] acc={correct/max(total,1):.3f} "
                  f"({correct}/{total}) extracted={extracted_count}")
    dt = time.time() - t0
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / max(total, 1),
        "extraction_rate": extracted_count / max(total, 1),
        "wall_seconds": dt,
        "first_examples": examples_log,
    }


# -----------------------------------------------------------------------------
# Trainer construction (resume from Option I)
# -----------------------------------------------------------------------------

def build_trainer(tokenizer, n_steps: int, resume_from: str | None) -> FANT2Trainer:
    cfg = fant2_tiny()
    assert tokenizer.vocab_size <= cfg.vocab_size, (
        f"tokenizer vocab_size {tokenizer.vocab_size} exceeds preset {cfg.vocab_size}"
    )
    model = FANT2Model(cfg)
    train_stream = make_train_stream(tokenizer)
    train_cfg = TrainConfig(
        phase=2, n_steps=n_steps,
        batch_size=BATCH_SIZE, seq_len=SEQ_LEN,
        muon_lr=8e-4, adam_lr=2e-4,    # slightly lower than Option I — finetuning
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.05,
        fep_kl_beta_max=0.2,
        fep_kl_anneal_steps=max(n_steps, 1),
        telemetry_every=2000, tikkun_every=2000, fana_every=10000,
        log_every=max(1, n_steps // 25),
        save_every=500,                # was 10000; want intermediate ckpts
        out_dir=OUT_RAMP,
        resume_from=resume_from,
        device="cpu",
        bf16=False, grad_checkpoint=False, use_8bit_adam=False,
    )
    return FANT2Trainer(model, train_cfg, train_stream)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    print("=" * 64)
    print(" FANT 2 — Option K: heavy procedural-math ramp + procedural eval")
    print(" (procedural data only — no public benchmarks touched)")
    print("=" * 64)

    if not os.path.exists(OPTION_I_CKPT):
        print(f"  ✗ Option I checkpoint not found at {OPTION_I_CKPT}")
        print(f"    run `python scripts/option_i_real_pretrain.py` first")
        return 1
    if not os.path.exists(OPTION_I_TOK):
        print(f"  ✗ Option I tokenizer not found at {OPTION_I_TOK}")
        return 1

    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(OUT_RAMP, exist_ok=True)

    print()
    print("  loading Option I tokenizer + checkpoint")
    tokenizer = FANT2Tokenizer.load(OPTION_I_TOK)
    print(f"    vocab_size = {tokenizer.vocab_size}")

    # ---------- Phase A: bench Option I as-is on procedural math ----------
    print()
    print("  ===== Phase A: pre-ramp procedural-math eval =====")
    pre_trainer = build_trainer(tokenizer, n_steps=1, resume_from=OPTION_I_CKPT)
    print(f"  loaded at step {pre_trainer.step}")
    pre_res = evaluate_procedural_math(
        pre_trainer.model, tokenizer,
        seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=EVAL_N_PROBLEMS, max_new_tokens=EVAL_MAX_NEW_TOKENS,
        verbose=True,
    )
    print(f"  pre-ramp: {pre_res['correct']}/{pre_res['total']} = {pre_res['accuracy']:.1%} "
          f"(extraction rate {pre_res['extraction_rate']:.1%})")

    # ---------- Phase B: ramp procedural math ----------
    print()
    print(f"  ===== Phase B: procedural-math ramp ({N_STEPS} steps) =====")
    trainer = build_trainer(tokenizer, n_steps=N_STEPS, resume_from=OPTION_I_CKPT)
    t0 = time.time()
    train_exc = None
    try:
        trainer.train()
    except (KeyboardInterrupt, Exception) as exc:
        train_exc = exc
        print(f"  ! training interrupted: {type(exc).__name__}: {exc}")
    dt = time.time() - t0
    steps_done = max(trainer.step, 1)
    print(f"  ramp done/halted in {dt / 60:.1f} min ({dt / steps_done * 1000:.0f} ms/step)")

    # Crash-safe save: only write our own final.pt if training crashed early.
    # If trainer.train() completed normally, it already wrote a richer final.pt
    # via save_checkpoint() (with `opt` and `cfg` keys for downstream resume).
    final_ckpt = os.path.join(OUT_RAMP, "final.pt")
    if train_exc is not None:
        torch.save({
            "model": trainer.model.state_dict(),
            "opt":   trainer.opt.state_dict(),
            "cfg":   trainer.cfg,
            "step":  trainer.step,
            "halted_early": True,
        }, final_ckpt)
        print(f"  saved partial (crash-safe) checkpoint to {final_ckpt}")
    else:
        print(f"  trainer wrote final checkpoint to {final_ckpt}")

    # ---------- Phase C: post-ramp eval ----------
    print()
    print("  ===== Phase C: post-ramp procedural-math eval =====")
    post_res = evaluate_procedural_math(
        trainer.model, tokenizer,
        seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=EVAL_N_PROBLEMS, max_new_tokens=EVAL_MAX_NEW_TOKENS,
        verbose=True,
    )
    print(f"  post-ramp: {post_res['correct']}/{post_res['total']} = {post_res['accuracy']:.1%} "
          f"(extraction rate {post_res['extraction_rate']:.1%})")

    # ---------- Report ----------
    print()
    print("=" * 64)
    print(" RESULTS")
    print("=" * 64)
    print(f"  pre-ramp  ({pre_trainer.step:>5d} steps): "
          f"{pre_res['correct']:>3d}/{pre_res['total']:<3d} = {pre_res['accuracy']*100:>5.1f}%")
    print(f"  post-ramp ({trainer.step:>5d} steps): "
          f"{post_res['correct']:>3d}/{post_res['total']:<3d} = {post_res['accuracy']*100:>5.1f}%")
    delta = post_res["accuracy"] - pre_res["accuracy"]
    print(f"  delta: {delta * 100:+.1f}pp")
    print()

    results = {
        "config": {
            "n_steps": N_STEPS, "seq_len": SEQ_LEN, "batch_size": BATCH_SIZE,
            "train_seed": TRAIN_SEED, "eval_seed": EVAL_SEED,
            "eval_max_value": EVAL_MAX_VALUE, "eval_n_problems": EVAL_N_PROBLEMS,
        },
        "pre_ramp": pre_res,
        "post_ramp": post_res,
        "ckpt": final_ckpt,
        "tokenizer": OPTION_I_TOK,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  results JSON: {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
