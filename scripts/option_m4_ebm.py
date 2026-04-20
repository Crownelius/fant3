"""
Option M4-EBM — M4 synthesis + energy-based scoring head.

Same as M4 (L1.5 base + HELM upstream + ce_surprise classifier) but replaces
the success_estimator's BCE loss with a contrastive energy-based objective.

The energy head E(h, y) takes the hidden state h and a token prediction y,
and learns to assign:
  - LOW energy to correct (h, y_correct) pairs
  - HIGH energy to incorrect (h, y_wrong) pairs

This is motivated by EORM (arXiv:2505.14999) which showed a 55M-param
energy-based verifier outperforms much larger reward models for math
verification. The key insight: energy-based scoring gives compositional
constraint satisfaction that per-token BCE lacks.

Changes from M4:
  1. Adds an EnergyHead module to the model (replaces success_estimator usage)
  2. Trains with contrastive energy loss (correct vs negative samples)
  3. Uses energy score for pass-2 refinement gating

Everything else identical to M4: HELM upstream, ce_surprise, no gate,
no filter, no Coconut, no SpiralThinker.

Run:
    PYTHONPATH=. python scripts/option_m4_ebm.py
"""

from __future__ import annotations

import os
import re
import time
import json
from typing import Iterator, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from fant2.config import fant2_tiny
from fant2.data import TokenizedBatchStream
from fant2.inference import FANT2Generator
from fant2.model import FANT2Model
from fant2.tokenizer import FANT2Tokenizer
from fant2.training import TrainConfig, FANT2Trainer
from fant2.training.phase5_rollout import ProceduralMathStream, format_prompt


# Configuration

OPTION_I_CKPT = "output/option_i/pretrain/final.pt"
OPTION_I_TOK  = "output/option_i/tokenizer.json"

OUT_BASE = "output/option_m4_ebm"
OUT_RAMP = os.path.join(OUT_BASE, "math_ramp")
RESULTS_JSON = os.path.join(OUT_BASE, "results.json")

N_STEPS    = 2500
SEQ_LEN    = 128
BATCH_SIZE = 8
TRAIN_SEED = 11
EVAL_SEED  = 9999
EVAL_MAX_VALUE = 12
EVAL_N_PROBLEMS = 1000   # 1K eval for statistical power
EVAL_MAX_NEW_TOKENS = 64

OUTPUT_GATE_INIT     = 0.1
CURVATURE_THRESHOLD  = 1.0

# M4 classifier flags
PHASE4_CLASS_UPSTREAM = True
PHASE4_CLASS_MODE     = "ce_surprise"

# EBM config
ENERGY_HIDDEN = 64
ENERGY_N_NEGATIVES = 4       # negative samples per token for contrastive loss
ENERGY_MARGIN = 1.0          # margin for contrastive energy loss
ENERGY_WEIGHT = 0.5          # weight of energy loss in total refine loss


# ====== Energy-Based Scoring Head ======

class EnergyHead(nn.Module):
    """
    Contrastive energy head: E(h, y) -> scalar energy.

    Takes a hidden state h (dim,) and a token embedding y (dim,) and computes
    a scalar energy. Low energy = good (h, y) pair, high = bad.

    Architecture: bilinear interaction + MLP projection.
    This is a Joint Energy Model (JEM) inspired design — the energy function
    captures the compatibility between the representation and the prediction.
    """

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        # Bilinear interaction: captures compatibility
        self.W_interact = nn.Linear(dim, hidden, bias=False)
        # MLP energy projection
        self.energy_mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden, bias=False),
            nn.GELU(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, h: torch.Tensor, y_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h:     (*, dim) hidden states
            y_emb: (*, dim) token embeddings for predictions
        Returns:
            energy: (*,) scalar energy per position
        """
        # Project both through the interaction layer
        h_proj = self.W_interact(h)      # (*, hidden)
        y_proj = self.W_interact(y_emb)  # (*, hidden)
        # Concatenate projections
        combined = torch.cat([h_proj, y_proj * h_proj], dim=-1)  # (*, hidden*2)
        # hadamard product captures element-wise compatibility
        energy = self.energy_mlp(combined).squeeze(-1)  # (*,)
        return energy


def contrastive_energy_loss(
    energy_head: EnergyHead,
    hidden: torch.Tensor,        # (B, T, dim)
    target_ids: torch.Tensor,    # (B, T)
    tok_emb: nn.Embedding,       # token embedding layer
    n_negatives: int = 4,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Contrastive energy loss: E(h, y_correct) should be lower than
    E(h, y_wrong) by at least `margin`.

    For each position, we compute:
      E_pos = E(h_t, embed(target_t))
      E_neg = E(h_t, embed(random_token))  [n_negatives samples]
      loss = mean(relu(E_pos - E_neg + margin))

    This trains the energy head to assign low energy to correct completions
    and high energy to incorrect ones.
    """
    B, T, dim = hidden.shape
    device = hidden.device

    # Positive energy: correct token embeddings
    with torch.no_grad():
        y_pos = tok_emb(target_ids)  # (B, T, dim)
    e_pos = energy_head(hidden, y_pos)  # (B, T)

    # Negative energies: random token embeddings
    vocab_size = tok_emb.num_embeddings
    neg_losses = []
    for _ in range(n_negatives):
        neg_ids = torch.randint(0, vocab_size, (B, T), device=device)
        # Avoid using the correct token as a negative
        neg_ids = torch.where(neg_ids == target_ids, (neg_ids + 1) % vocab_size, neg_ids)
        with torch.no_grad():
            y_neg = tok_emb(neg_ids)  # (B, T, dim)
        e_neg = energy_head(hidden, y_neg)  # (B, T)
        # Margin loss: want E_pos < E_neg - margin
        neg_losses.append(F.relu(e_pos - e_neg + margin))

    # Average over negatives and positions
    loss = torch.stack(neg_losses).mean()

    # Also add a regularizer to keep energies bounded
    energy_reg = 0.01 * (e_pos ** 2).mean()

    return loss + energy_reg


# ====== Data streams (same as M4 — unfiltered) ======

class ProceduralMathTextStream:
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


# ====== Eval (same extract logic) ======

_ANSWER_TAG = re.compile(r"<answer>\s*(-?\d+(?:\.\d+)?)\s*</answer>")
_ANY_NUM = re.compile(r"-?\d+(?:\.\d+)?")

import math

def _extract_answer(text: str) -> str | None:
    m = _ANSWER_TAG.search(text)
    if m:
        return m.group(1)
    nums = _ANY_NUM.findall(text)
    if nums:
        return nums[-1]
    return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0, centre - spread), min(1, centre + spread))


def evaluate_procedural_math(
    model, tokenizer, *, seed, max_value, n_problems, max_new_tokens,
    device="cpu", verbose=True,
) -> dict:
    gen = FANT2Generator(model, tokenizer, device=device)
    stream = ProceduralMathStream(seed=seed, max_value=max_value)
    it = iter(stream)
    correct = total = extracted = 0
    examples_log: List[dict] = []
    t0 = time.time()
    for i in range(n_problems):
        try:
            ex = next(it)
        except StopIteration:
            break
        prompt = format_prompt(ex.question)
        completion = gen.generate(
            prompt, max_new_tokens=max_new_tokens, greedy=True,
            return_full_text=False,
        )
        pred = _extract_answer(completion)
        if pred is not None:
            extracted += 1
        is_correct = (pred is not None) and (
            pred.lstrip("0") == ex.gold_answer.lstrip("0")
            or pred == ex.gold_answer
            or (pred == "0" and ex.gold_answer == "0")
        )
        if pred is not None and not is_correct:
            try:
                is_correct = float(pred) == float(ex.gold_answer)
            except ValueError:
                pass
        total += 1
        if is_correct:
            correct += 1
        if i < 10:
            examples_log.append({
                "q": ex.question, "gold": ex.gold_answer,
                "completion": completion[:300], "pred": pred,
                "correct": is_correct,
            })
        if verbose and (i + 1) % 100 == 0:
            ci = wilson_ci(correct, total)
            print(f"    [proc-math {i+1}/{n_problems}] acc={correct/total:.3f} "
                  f"({correct}/{total}) CI=[{ci[0]:.3f}, {ci[1]:.3f}]")
    dt = time.time() - t0
    ci = wilson_ci(correct, total)
    return {
        "correct": correct, "total": total,
        "accuracy": correct / max(total, 1),
        "wilson_ci_95": list(ci),
        "extraction_rate": extracted / max(total, 1),
        "wall_seconds": dt, "first_examples": examples_log,
    }


# ====== Memory helpers ======

def bump_output_gates(model, value):
    n = 0
    for m in model.modules():
        if hasattr(m, "output_gate") and isinstance(m.output_gate, nn.Parameter):
            with torch.no_grad():
                m.output_gate.fill_(value)
            n += 1
    return n


def bump_curvature_threshold(model, value):
    prev = float(model.memory.curvature_threshold)
    model.memory.curvature_threshold = float(value)
    return prev


def memory_diagnostics(model):
    mem = model.memory
    fills = mem.fill_rates()
    curvs = mem.curvature_statistics()
    alpha_pl = mem.estimate_power_law_exponent("alpha")
    beta_pl = mem.estimate_power_law_exponent("beta")
    gates = []
    for m in model.modules():
        if hasattr(m, "output_gate") and isinstance(m.output_gate, nn.Parameter):
            gates.append(float(m.output_gate.item()))
    return {
        "fills": fills, "curvature": curvs,
        "alpha_power_law_exp": alpha_pl, "beta_power_law_exp": beta_pl,
        "output_gates": gates, "curvature_threshold": float(mem.curvature_threshold),
    }


# ====== Custom Phase 4 training loop with EBM head ======

def train_phase4_ebm(model, tokenizer, n_steps, resume_from):
    """
    Custom Phase 4 training with energy-based scoring.

    This is a manual training loop (not using FANT2Trainer) because we need
    to inject the EnergyHead into the two-pass flow and use contrastive
    energy loss instead of success_estimator BCE.
    """
    from fant2.training.losses import fep_unified_loss

    cfg = fant2_tiny()
    cfg.phase4_classifier_upstream = PHASE4_CLASS_UPSTREAM
    cfg.phase4_classifier_mode = PHASE4_CLASS_MODE

    # Load checkpoint weights
    ckpt = torch.load(resume_from, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    bump_output_gates(model, OUTPUT_GATE_INIT)
    bump_curvature_threshold(model, CURVATURE_THRESHOLD)

    # Create energy head
    energy_head = EnergyHead(dim=cfg.dim, hidden=ENERGY_HIDDEN)

    # Optimizer: model params + energy head params
    all_params = list(model.parameters()) + list(energy_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=2e-4, weight_decay=0.01)

    # Data stream
    train_stream = make_train_stream(tokenizer)
    train_iter = iter(train_stream)

    model.train()
    energy_head.train()

    step_offset = ckpt.get("step", 3000) if isinstance(ckpt, dict) else 3000
    log_every = max(1, n_steps // 25)
    save_every = 500

    t0 = time.time()
    fep_kl_beta = 0.05
    fep_kl_beta_max = 0.2

    for step_i in range(n_steps):
        step = step_offset + step_i + 1

        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_stream)
            batch = next(train_iter)

        input_ids, target_ids = batch

        # Anneal FEP KL beta
        progress = min(step_i / max(n_steps, 1), 1.0)
        fep_kl_beta = 0.05 + progress * (fep_kl_beta_max - 0.05)

        # ===== Pass 1: free run =====
        out1 = model(input_ids)

        # Feedback: legacy pooled mean (no Coconut)
        feedback = out1["final_hidden"].mean(dim=1).detach()

        # ce_surprise classifier scores
        with torch.no_grad():
            ce_per_tok = F.cross_entropy(
                out1["logits"].reshape(-1, cfg.vocab_size),
                target_ids.reshape(-1),
                reduction="none", ignore_index=-100,
            ).detach()
            mean_ce = ce_per_tok.mean().clamp(min=1e-6)
            external_scores = ce_per_tok / mean_ce

        # ===== Pass 2: refinement with memory fill =====
        out2 = model(
            input_ids, targets=target_ids,
            store_to_memory=True,
            prepend_vec=feedback,
            external_classifier_scores=external_scores,
        )

        # ===== FEP loss on pass-2 =====
        base = fep_unified_loss(
            logits=out2["logits"], targets=target_ids,
            router_outputs=out2["router_outputs"],
            z_loss_alpha=1e-3, fep_kl_beta=fep_kl_beta,
        )

        # ===== Energy-based contrastive loss (REPLACES success_estimator BCE) =====
        energy_loss = contrastive_energy_loss(
            energy_head,
            hidden=out2["final_hidden"],
            target_ids=target_ids,
            tok_emb=model.tok_emb,
            n_negatives=ENERGY_N_NEGATIVES,
            margin=ENERGY_MARGIN,
        )

        # ===== Soft hidden-state consistency (keep from M4) =====
        consistency = F.mse_loss(
            out2["final_hidden"], out1["final_hidden"].detach(),
        )

        # ===== Total loss =====
        # Replace succ_bce + succ_gap with energy_loss
        refine_total = ENERGY_WEIGHT * energy_loss + 0.5 * consistency
        total = base["total"] + 0.5 * refine_total

        optimizer.zero_grad()
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        optimizer.step()

        if (step_i + 1) % log_every == 0:
            print(f"  [step {step:5d}]  ce={base['ce'].item():.4f}  "
                  f"fep_kl={base['fep_kl'].item():.4f}  "
                  f"energy={energy_loss.item():.4f}  "
                  f"total={total.item():.4f}")

        if (step_i + 1) % save_every == 0:
            ckpt_path = os.path.join(OUT_RAMP, f"step_{step}.pt")
            torch.save({
                "model": model.state_dict(),
                "energy_head": energy_head.state_dict(),
                "step": step,
            }, ckpt_path)
            print(f"  saved checkpoint to {ckpt_path}")

    dt = time.time() - t0
    print(f"  training done in {dt / 60:.1f} min ({dt / n_steps * 1000:.0f} ms/step)")

    # Save final
    final_ckpt = os.path.join(OUT_RAMP, "final.pt")
    torch.save({
        "model": model.state_dict(),
        "energy_head": energy_head.state_dict(),
        "step": step,
    }, final_ckpt)

    return model, energy_head, dt


# ====== Main ======

def main() -> int:
    print("=" * 64)
    print(" FANT 2 — Option M4-EBM: M4 + energy-based scoring head")
    print(" Contrastive energy verification (inspired by EORM)")
    print("=" * 64)
    print()
    print(f"  phase4_classifier_upstream = {PHASE4_CLASS_UPSTREAM}")
    print(f"  phase4_classifier_mode     = {PHASE4_CLASS_MODE}")
    print(f"  energy_hidden              = {ENERGY_HIDDEN}")
    print(f"  energy_n_negatives         = {ENERGY_N_NEGATIVES}")
    print(f"  energy_margin              = {ENERGY_MARGIN}")
    print(f"  energy_weight              = {ENERGY_WEIGHT}")
    print(f"  eval_n_problems            = {EVAL_N_PROBLEMS}")
    print()

    if not os.path.exists(OPTION_I_CKPT):
        print(f"  x checkpoint not found: {OPTION_I_CKPT}")
        return 1

    os.makedirs(OUT_BASE, exist_ok=True)
    os.makedirs(OUT_RAMP, exist_ok=True)

    tokenizer = FANT2Tokenizer.load(OPTION_I_TOK)
    cfg = fant2_tiny()
    cfg.phase4_classifier_upstream = PHASE4_CLASS_UPSTREAM
    cfg.phase4_classifier_mode = PHASE4_CLASS_MODE
    model = FANT2Model(cfg)

    # ---------- Phase A: pre-ramp eval ----------
    print("  ===== Phase A: pre-ramp eval =====")
    ckpt = torch.load(OPTION_I_CKPT, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    model.eval()
    bump_output_gates(model, OUTPUT_GATE_INIT)
    bump_curvature_threshold(model, CURVATURE_THRESHOLD)

    pre_res = evaluate_procedural_math(
        model, tokenizer, seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=200, max_new_tokens=EVAL_MAX_NEW_TOKENS,
    )
    print(f"  pre-ramp: {pre_res['correct']}/{pre_res['total']} = {pre_res['accuracy']:.1%}")

    # ---------- Phase B: Phase 4 ramp with EBM head ----------
    print()
    print(f"  ===== Phase B: Phase 4 ramp ({N_STEPS} steps, EBM scoring) =====")
    # Re-create fresh model for training
    model = FANT2Model(cfg)
    model, energy_head, train_time = train_phase4_ebm(
        model, tokenizer, n_steps=N_STEPS, resume_from=OPTION_I_CKPT,
    )

    # ---------- Phase C: post-ramp diagnostics + 1K eval ----------
    print()
    print(f"  ===== Phase C: post-ramp diagnostics + {EVAL_N_PROBLEMS}-sample eval =====")
    model.eval()
    post_diag = memory_diagnostics(model)
    print(f"  fills:     {post_diag['fills']}")
    print(f"  curvature: {post_diag['curvature']}")
    print(f"  alpha PLExp: {post_diag['alpha_power_law_exp']:.3f} (target ~1.305)")
    print(f"  beta  PLExp: {post_diag['beta_power_law_exp']:.3f}")

    post_res = evaluate_procedural_math(
        model, tokenizer, seed=EVAL_SEED, max_value=EVAL_MAX_VALUE,
        n_problems=EVAL_N_PROBLEMS, max_new_tokens=EVAL_MAX_NEW_TOKENS,
    )
    ci = post_res["wilson_ci_95"]
    print(f"  post-ramp: {post_res['correct']}/{post_res['total']} = "
          f"{post_res['accuracy']*100:.1f}%  CI=[{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")

    # ---------- Report ----------
    print()
    print("=" * 64)
    print(" RESULTS — Option M4-EBM (contrastive energy scoring)")
    print("=" * 64)
    print(f"  pre-ramp:  {pre_res['correct']}/{pre_res['total']} = {pre_res['accuracy']*100:.1f}%")
    print(f"  post-ramp: {post_res['correct']}/{post_res['total']} = {post_res['accuracy']*100:.1f}%")
    print(f"  Wilson 95% CI: [{ci[0]*100:.1f}%, {ci[1]*100:.1f}%]")
    print()
    print("  Comparison (200-sample historical):")
    print(f"    L1.5 (LLM baseline):      123/200 = 61.5%")
    print(f"    M4   (LLM + classifier):  116/200 = 58.0%")
    print(f"    M4-EBM (energy scoring):  {post_res['correct']}/{post_res['total']} = "
          f"{post_res['accuracy']*100:.1f}%")
    print()

    # Energy head parameter count
    ebm_params = sum(p.numel() for p in energy_head.parameters())
    model_params = sum(p.numel() for p in model.parameters())
    print(f"  Energy head params: {ebm_params:,} ({ebm_params/model_params*100:.1f}% of model)")
    print(f"  Training time: {train_time/60:.1f} min")

    results = {
        "config": {
            "phase": 4, "variant": "ebm",
            "energy_hidden": ENERGY_HIDDEN,
            "energy_n_negatives": ENERGY_N_NEGATIVES,
            "energy_margin": ENERGY_MARGIN,
            "energy_weight": ENERGY_WEIGHT,
            "phase4_classifier_upstream": PHASE4_CLASS_UPSTREAM,
            "phase4_classifier_mode": PHASE4_CLASS_MODE,
            "n_steps": N_STEPS, "eval_n_problems": EVAL_N_PROBLEMS,
        },
        "pre_ramp": pre_res,
        "post_ramp": post_res,
        "post_memory": post_diag,
        "energy_head_params": ebm_params,
        "model_params": model_params,
        "train_time_seconds": train_time,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  results: {RESULTS_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
