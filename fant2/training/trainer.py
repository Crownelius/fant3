"""
FANT2Trainer — the unified 7-phase training driver.

A single class that runs any of the 7 training phases:

    Phase 0:  BPE tokenizer training (no model required)
    Phase 1:  LLM-JEPA self-supervised pretraining (+ SIGReg)
    Phase 2:  MoE expert specialization (the FEP unified loss)
    Phase 3:  Active-layer calibration (rank/condition repair)
    Phase 4:  Self-refinement + STaR + Apollonian fill
    Phase 5:  Dr.GRPO RL on math/code tasks
    Phase 6:  SimPO + KTO preference optimization

The trainer is the SINGLE place where:
  - the optimizer step happens
  - the router bias is updated (DeepSeek aux-loss-free)
  - Tikkun + fanā are triggered (every N steps)
  - telemetry is collected and monitors are run
  - checkpoints are saved

Phase-specific logic is delegated to small "phase callbacks" that customize the
loss function for that phase. Everything else is shared.
"""

import math
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict, Any, Iterable

import torch

from ..config import FANT2Config
from ..model import FANT2Model
from ..constants import (
    TELEMETRY_EVERY_N_STEPS,
    TIKKUN_CHECK_EVERY_N_STEPS,
)
from .optimizer import HybridOptimizer
from .losses import (
    fep_unified_loss,
    llm_jepa_loss,
    success_estimator_loss,
    calibration_loss,
    progressive_alignment_loss,
)
from .telemetry import collect_telemetry, TelemetrySnapshot
from .monitors import default_monitors, run_monitors


@dataclass
class TrainConfig:
    phase: int = 2                 # which of the 7 phases (0..6)
    n_steps: int = 1000            # total optimizer steps
    batch_size: int = 8
    seq_len: int = 1024
    grad_accum: int = 4

    # Optimizer
    muon_lr: float = 1e-3
    adam_lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    use_8bit_adam: bool = True

    # FEP loss
    z_loss_alpha: float = 1e-3
    fep_kl_beta_init: float = 0.1
    fep_kl_beta_max: float = 1.0
    fep_kl_anneal_steps: int = 5000

    # Phase 3: calibration
    calib_weight: float = 0.1
    calib_n_samples: int = 4
    calib_rank_target_frac: float = 0.9
    calib_max_condition: float = 100.0

    # Campaign N1: expert orthogonality + router variance (arXiv:2505.22323)
    ortho_alpha: float = 0.0   # expert orthogonality loss weight (0 = off)
    var_alpha: float = 0.0     # router variance loss weight (0 = off)

    # Campaign N3: SleepGate memory consolidation
    sleep_consolidate_every: int = 0       # 0 = off; e.g. 100 = every 100 steps
    sleep_merge_threshold: float = 0.92    # cosine sim above which entries merge
    sleep_staleness_horizon: int = 200     # entries older than this get evicted

    # 2026-04-16 optimization: optionally populate the Apollonian α/β packs
    # during Phase 2 (bulk pretraining). Default False preserves historic
    # behavior where Phase 2 doesn't write to memory — Phase 4 does. Setting
    # True lets the packs accumulate a warm start so Phase 4's STaR refinement
    # can query a non-empty memory from its first step. No gradient cost; the
    # store happens inside no_grad().
    populate_apollonian_in_phase2: bool = False

    # Phase 4: self-refinement
    refine_weight: float = 0.5

    # Phase 5: Dr.GRPO RL
    grpo_n_rollouts:    int   = 8       # G in spec §8 (full default 16; use 4-8 for tiny)
    grpo_max_new_tokens: int  = 96      # cap response length for rollouts
    grpo_temperature:   float = 0.9
    grpo_top_p:         float = 0.95
    grpo_clip_eps:      float = 0.20    # ε_lo
    grpo_clip_eps_hi:   float = 0.28    # ε_hi (DAPO clip-higher, spec §8 Phase 5)

    # Phase 6: SimPO + KTO preference optimization
    simpo_beta:    float = 2.0    # SimPO temperature
    simpo_gamma:   float = 1.6    # SimPO target margin
    kto_beta:      float = 0.1    # KTO temperature
    kto_weight:    float = 0.5    # weight on KTO term in composite loss

    # Telemetry / repair cadence
    telemetry_every: int = TELEMETRY_EVERY_N_STEPS
    tikkun_every: int = TIKKUN_CHECK_EVERY_N_STEPS
    fana_every: int = 1000
    log_every: int = 50
    save_every: int = 1000

    # Checkpoint
    out_dir: str = "out/fant2"
    resume_from: Optional[str] = None

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    bf16: bool = True
    grad_checkpoint: bool = True


class FANT2Trainer:
    """The single training driver for all 7 FANT 2 phases."""

    def __init__(
        self,
        model: FANT2Model,
        train_cfg: TrainConfig,
        data_stream: Iterable,
    ):
        self.model = model
        self.cfg = train_cfg
        self.data_stream = data_stream
        self.step = 0

        os.makedirs(train_cfg.out_dir, exist_ok=True)

        # Move model to device
        self.model = self.model.to(train_cfg.device)
        if train_cfg.bf16 and train_cfg.device == "cuda":
            self.model = self.model.to(torch.bfloat16)

        # Build optimizer
        self.opt = HybridOptimizer.from_model(
            self.model,
            muon_lr=train_cfg.muon_lr,
            adam_lr=train_cfg.adam_lr,
            weight_decay=train_cfg.weight_decay,
            use_8bit_adam=train_cfg.use_8bit_adam,
        )

        # Monitors
        self.monitors = default_monitors()

        # Telemetry log
        self.telemetry_log: list[TelemetrySnapshot] = []

        # Phase 5 only: frozen reference model for Dr.GRPO old_logps.
        # The Phase 5 entry-point script sets this before calling .train().
        self.ref_model: Optional[FANT2Model] = None

        # Resume?
        if train_cfg.resume_from:
            self.load_checkpoint(train_cfg.resume_from)

    # -------------------------------------------------------------------------
    # FEP β annealing schedule (linear from init to max over fep_kl_anneal_steps)
    # -------------------------------------------------------------------------

    def current_fep_beta(self) -> float:
        c = self.cfg
        if c.fep_kl_anneal_steps <= 0:
            return c.fep_kl_beta_max
        frac = min(1.0, self.step / c.fep_kl_anneal_steps)
        return c.fep_kl_beta_init + frac * (c.fep_kl_beta_max - c.fep_kl_beta_init)

    # -------------------------------------------------------------------------
    # Single training step
    # -------------------------------------------------------------------------

    def train_step(self, batch: tuple) -> Dict[str, float]:
        """
        One full optimizer step.

        Dispatches forward + loss computation to a phase-specific helper:
            Phase 1 → _phase1_jepa_forward
            Phase 2 → _phase2_moe_forward  (the canonical FEP unified loss)
            Phase 3 → _phase3_calibrate_forward
            Phase 4 → _phase4_refine_forward
            Phase 5 → _phase5_grpo_forward  (stub: falls back to Phase 2 loss)
            Phase 6 → _phase6_simpo_kto_forward  (stub: falls back to Phase 2 loss)

        The backward / grad-clip / router-bias-update / opt.step cadence is shared.

        Returns:
            dict of named scalar losses for logging
        """
        c = self.cfg
        self.model.train()
        self.opt.zero_grad()

        input_ids, target_ids = batch
        input_ids = input_ids.to(c.device)
        target_ids = target_ids.to(c.device)

        # ===== Phase-specific forward + loss =====
        dispatch = {
            1: self._phase1_jepa_forward,
            2: self._phase2_moe_forward,
            3: self._phase3_calibrate_forward,
            4: self._phase4_refine_forward,
            5: self._phase5_grpo_forward,
            6: self._phase6_simpo_kto_forward,
        }
        handler = dispatch.get(c.phase, self._phase2_moe_forward)
        out, losses = handler(input_ids, target_ids)

        # ===== Backward =====
        losses["total"].backward()

        # Grad clip
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.grad_clip)

        # ===== DeepSeek aux-loss-free bias update =====
        # MUST be after backward() and before optimizer.step()
        if out.get("router_outputs"):
            self.model.update_router_biases(out["router_outputs"])

        # ===== Optimizer step =====
        self.opt.step()

        return {k: float(v.item()) for k, v in losses.items()}

    # -------------------------------------------------------------------------
    # Phase-specific forward handlers
    # -------------------------------------------------------------------------

    def _phase1_jepa_forward(self, input_ids, target_ids):
        """Phase 1: LLM-JEPA + SIGReg pretraining (no FEP KL prior)."""
        out = self.model(input_ids, targets=target_ids, store_to_memory=False)
        # JEPA: first-half mean context → predictor → second-half mean target (detached)
        B, T, D = out["final_hidden"].shape
        half = max(1, T // 2)
        ctx = out["final_hidden"][:, :half, :].mean(dim=1)
        tgt = out["final_hidden"][:, half:, :].mean(dim=1)
        pred = self.model.jepa_predictor(ctx)
        jepa_dict = llm_jepa_loss(pred.unsqueeze(1), tgt.unsqueeze(1).detach())
        losses = {
            "ce":     out["loss"],
            "jepa":   jepa_dict["jepa"],
            "sigreg": jepa_dict["sigreg"],
            "total":  out["loss"] + jepa_dict["total"],
        }
        return out, losses

    def _phase2_moe_forward(self, input_ids, target_ids):
        """Phase 2: MoE expert specialization via FEP unified loss."""
        c = self.cfg
        out = self.model(
            input_ids, targets=target_ids,
            store_to_memory=c.populate_apollonian_in_phase2,
        )
        losses = fep_unified_loss(
            logits=out["logits"],
            targets=target_ids,
            router_outputs=out["router_outputs"],
            z_loss_alpha=c.z_loss_alpha,
            fep_kl_beta=self.current_fep_beta(),
            # Campaign N1: expert orthogonality + router variance
            ortho_alpha=c.ortho_alpha,
            var_alpha=c.var_alpha,
            moe_layers=self.model.moe_layers if c.ortho_alpha > 0 else None,
        )
        return out, losses

    def _phase3_calibrate_forward(self, input_ids, target_ids):
        """
        Phase 3: Active-layer calibration (rank / condition number repair).

        Adds `calibration_loss` on a few randomly-materialized expert weights
        to the base FEP loss. Gradients flow back to A_expert and B_layer
        through the kron3 op.
        """
        c = self.cfg
        out = self.model(input_ids, targets=target_ids, store_to_memory=False)

        base = fep_unified_loss(
            logits=out["logits"],
            targets=target_ids,
            router_outputs=out["router_outputs"],
            z_loss_alpha=c.z_loss_alpha,
            fep_kl_beta=self.current_fep_beta(),
        )

        sampled = self.model.sample_materialized_expert_weights(
            n_samples=c.calib_n_samples, seed=self.step
        )
        calib = calibration_loss(
            sampled,
            rank_target_frac=c.calib_rank_target_frac,
            max_condition=c.calib_max_condition,
        )
        losses = {
            "ce":         base["ce"],
            "z_loss":     base["z_loss"],
            "fep_kl":     base["fep_kl"],
            "calib_rank": calib["rank"],
            "calib_cond": calib["cond"],
            "total":      base["total"] + c.calib_weight * calib["total"],
        }
        return out, losses

    def _phase4_refine_forward(self, input_ids, target_ids):
        """
        Phase 4: Self-refinement + STaR + Apollonian fill (true two-pass).

        Pass 1: forward(input_ids) → out1
        Pass 2: forward(input_ids, prepend=<feedback>, store_to_memory=True)
                → out2 (sees pass-1's state as virtual prepend position(s))

        Base loss:
            L_FEP(out2)                                       # main LM signal on the refined pass
            + refine_weight * BCE(out2.success, correctness)  # STaR self-confidence supervision
            + refine_weight * relu(out1.succ - out2.succ)     # pass 2 must be ≥ pass 1's confidence
            + 0.5 * MSE(out2.final_hidden, out1.final_hidden.detach())  # soft consistency

        Option M additions (all gated by `self.model.config.phase4_*` flags;
        default values preserve the legacy L1.5 behavior):

            #1 Think-at-Hard per-token gate (arxiv:2511.08577)
               — if pass-1 confidence on a token exceeds `phase4_gate_threshold`,
                 that token's pass-2 CE is masked out so confident tokens don't
                 get corrupted by refinement. Recomputes CE with per-token mean.

            #2 Coconut full-tensor feedback (arxiv:2412.06769)
               — if `phase4_prepend_k > 0`, feed the last K positions of pass-1's
                 final_hidden back as a [B,K,D] prepend instead of the pooled
                 [B,D] mean. Breaks the single-vector bottleneck.

            #4 Titans-style surprise classifier (arxiv:2501.00663, proxy version)
               — if `phase4_classifier_mode == "ce_surprise"`, compute per-token
                 CE from pass 2 and pass it to memory.store() as the α/β
                 assignment signal instead of the L2-norm proxy.

            #6 SpiralThinker progressive alignment (arxiv:2511.08983)
               — if `phase4_alignment_weight > 0`, add a cosine-alignment
                 penalty between pass-1 and pass-2 final_hidden in addition
                 to the legacy MSE consistency.

        Apollonian α/β memory is filled on pass 2 (the gradient-flowing one).
        Apollonian retrieval cross-attention activates automatically at the
        config-listed layers (default last 2) once memory has content.
        """
        c = self.cfg
        mcfg = self.model.config

        # ===== Pass 1: free run (full grad — model parameters get the
        # original forward signal too) =====
        out1 = self.model(input_ids)

        # ===== Feedback selection (Option M #2 Coconut vs legacy pool) =====
        prepend_k = int(getattr(mcfg, "phase4_prepend_k", 0))
        if prepend_k > 0:
            # Coconut full-tensor feedback: last K positions as [B, K, D]
            h1_full = out1["final_hidden"].detach()  # (B, T, D)
            K = min(prepend_k, h1_full.size(1))
            feedback = h1_full[:, -K:, :].contiguous()  # (B, K, D)
        else:
            # Legacy pooled-mean [B, D]
            feedback = out1["final_hidden"].mean(dim=1).detach()  # (B, dim)

        # ===== Compute external classifier scores if Titans mode is on =====
        # NOTE: this currently peeks at pass-1 predictions to estimate per-token
        # difficulty (pass 2 hasn't run yet, so we can't use its CE here without
        # double-forwarding). Pass-1 CE is a valid proxy: tokens that pass 1
        # already finds hard are the same tokens that refinement should target.
        classifier_mode = getattr(mcfg, "phase4_classifier_mode", "curvature")
        external_scores = None
        if classifier_mode == "ce_surprise":
            with torch.no_grad():
                ce_per_tok = torch.nn.functional.cross_entropy(
                    out1["logits"].reshape(-1, mcfg.vocab_size),
                    target_ids.reshape(-1),
                    reduction="none",
                    ignore_index=-100,
                ).detach()  # (B*T,)
                # Normalize to the same scale as the legacy curvature (~1.0)
                # so the classifier_threshold stays meaningful.
                mean_ce = ce_per_tok.mean().clamp(min=1e-6)
                external_scores = ce_per_tok / mean_ce

        # ===== Pass 2: refinement with virtual prepend + memory fill =====
        out2 = self.model(
            input_ids,
            targets=target_ids,
            store_to_memory=True,
            prepend_vec=feedback,
            external_classifier_scores=external_scores,
        )

        # ===== Main FEP loss on the refined pass =====
        base = fep_unified_loss(
            logits=out2["logits"],
            targets=target_ids,
            router_outputs=out2["router_outputs"],
            z_loss_alpha=c.z_loss_alpha,
            fep_kl_beta=self.current_fep_beta(),
            # Campaign N1: expert orthogonality + router variance
            ortho_alpha=c.ortho_alpha,
            var_alpha=c.var_alpha,
            moe_layers=self.model.moe_layers if c.ortho_alpha > 0 else None,
        )

        # ===== Option M #1: Think-at-Hard per-token pass-2 gate =====
        gate_enabled = bool(getattr(mcfg, "phase4_gate_enabled", False))
        gate_fraction_hard = 1.0
        if gate_enabled:
            gate_threshold = float(getattr(mcfg, "phase4_gate_threshold", 0.7))
            with torch.no_grad():
                # Pass 1 softmax confidence per token
                pass1_probs = torch.nn.functional.softmax(out1["logits"], dim=-1)
                pass1_conf = pass1_probs.max(dim=-1).values  # (B, T)
                # "Hard" tokens are the ones Think-at-Hard says pass 2 should refine
                hard_mask = (pass1_conf < gate_threshold).float()  # (B, T)
                gate_fraction_hard = float(hard_mask.mean().item())

            # Recompute pass-2 CE weighted by hard_mask so confident tokens
            # contribute zero gradient on the CE term.
            per_tok_ce2 = torch.nn.functional.cross_entropy(
                out2["logits"].reshape(-1, mcfg.vocab_size),
                target_ids.reshape(-1),
                reduction="none",
                ignore_index=-100,
            ).reshape(input_ids.shape)
            denom = hard_mask.sum().clamp(min=1.0)
            gated_ce = (per_tok_ce2 * hard_mask).sum() / denom

            # Override the CE component (and total) in the base dict.
            old_ce = base["ce"]
            base["ce"] = gated_ce
            base["total"] = base["total"] - old_ce + gated_ce

        # ===== STaR-style success-estimator supervision on pass 2 =====
        with torch.no_grad():
            pred_ids = out2["logits"].argmax(dim=-1)
            correct = (pred_ids == target_ids).float().unsqueeze(-1)
        succ_bce = torch.nn.functional.binary_cross_entropy(
            out2["success_pred"], correct, reduction="mean"
        )

        # ===== Refinement gap: pass 2 must be at least as confident as pass 1 =====
        # If out1.succ > out2.succ at any token, that's a regression — penalize.
        succ_gap = torch.relu(
            out1["success_pred"].detach() - out2["success_pred"]
        ).mean()

        # ===== Soft hidden-state consistency (legacy MSE) =====
        consistency = torch.nn.functional.mse_loss(
            out2["final_hidden"],
            out1["final_hidden"].detach(),
        )

        # ===== Option M #6: SpiralThinker progressive alignment =====
        alignment_weight = float(getattr(mcfg, "phase4_alignment_weight", 0.0))
        if alignment_weight > 0:
            alignment = progressive_alignment_loss(
                out1["final_hidden"],
                out2["final_hidden"],
                weight=alignment_weight,
            )
        else:
            alignment = torch.tensor(0.0, device=input_ids.device)

        refine_total = succ_bce + succ_gap + 0.5 * consistency + alignment

        losses = {
            "ce":          base["ce"],
            "z_loss":      base["z_loss"],
            "fep_kl":      base["fep_kl"],
            "succ":        succ_bce,
            "succ_gap":    succ_gap,
            "consistency": consistency,
            "alignment":   alignment,
            "gate_frac":   torch.tensor(gate_fraction_hard, device=input_ids.device),
            "total":       base["total"] + c.refine_weight * refine_total,
        }
        return out2, losses

    def _phase5_grpo_forward(self, input_ids, target_ids):
        """
        Phase 5: Dr.GRPO RL training (true rollout loop).

        Reads `MathExample`s from `self.data_stream.last_examples` (set by
        `Phase5BatchStream` on each `__next__`), generates G rollouts per
        example with `phase5_rollout.grpo_step`, and accumulates the
        Dr.GRPO loss across the batch. The frozen reference model is read
        from `self.ref_model` (set by the Phase 5 entry-point script before
        calling `.train()`).

        The `(input_ids, target_ids)` arguments are dummy padding tensors
        emitted by `Phase5BatchStream` to satisfy the trainer's per-step
        contract — they are intentionally ignored here.

        Training data is procedurally generated (see `phase5_rollout`); no
        public benchmark is touched, per the locked feedback constraint.
        """
        from .phase5_rollout import grpo_step, Phase5BatchStream

        c = self.cfg
        if not isinstance(self.data_stream, Phase5BatchStream):
            raise RuntimeError(
                "Phase 5 requires a Phase5BatchStream as the data stream. "
                "Wrap your problem source via "
                "`Phase5BatchStream(tokenizer=..., problems=ProceduralMathStream(), ...)`."
            )
        if self.ref_model is None:
            raise RuntimeError(
                "Phase 5 requires `trainer.ref_model` to be set to a frozen "
                "reference policy before calling .train(). The entry-point "
                "script in `fant2/training/phase5_grpo.py` does this for you."
            )

        examples = list(self.data_stream.last_examples)
        tokenizer = self.data_stream.tokenizer
        if not examples:
            raise RuntimeError("Phase5BatchStream produced an empty last_examples list.")

        # One grpo_step per prompt; sum the losses (mean across the batch).
        results = []
        loss_sum = None
        for ex in examples:
            res = grpo_step(
                model=self.model,
                ref_model=self.ref_model,
                tokenizer=tokenizer,
                example=ex,
                n_rollouts=c.grpo_n_rollouts,
                max_new_tokens=c.grpo_max_new_tokens,
                temperature=c.grpo_temperature,
                top_p=c.grpo_top_p,
                clip_eps=c.grpo_clip_eps,
                clip_eps_hi=c.grpo_clip_eps_hi,
                device=c.device,
            )
            results.append(res)
            loss_sum = res.loss if loss_sum is None else loss_sum + res.loss
        loss = loss_sum / max(len(results), 1)

        # Aggregate per-batch metrics for logging.
        all_rewards = [r for res in results for r in res.rewards]
        all_lens = [L for res in results for L in res.response_lens]
        n_correct = sum(res.n_correct for res in results)
        n_total = len(all_rewards)
        mean_reward = sum(all_rewards) / max(n_total, 1)
        mean_len = sum(all_lens) / max(n_total, 1)

        losses = {
            "grpo_loss":   loss,
            "mean_reward": torch.tensor(mean_reward),
            "mean_resp_len": torch.tensor(mean_len),
            "frac_correct": torch.tensor(n_correct / max(n_total, 1)),
            "total":       loss,
        }
        # Empty router_outputs so the post-backward router-bias update is skipped
        # (Phase 5 does not perform aux-loss-free router updates).
        out: Dict[str, Any] = {"router_outputs": []}
        return out, losses

    def _phase6_simpo_kto_forward(self, input_ids, target_ids):
        """
        Phase 6: SimPO + KTO preference optimization (real loop).

        Reads `PrefExample`s from `self.data_stream.last_examples` (set by
        `Phase6BatchStream` on each `__next__`), runs `simpo_kto_step` for
        each example, and accumulates the composite SimPO+KTO loss across
        the batch. The frozen reference model is read from `self.ref_model`
        (set by the Phase 6 entry-point script before calling `.train()`).

        The `(input_ids, target_ids)` arguments are dummy padding tensors
        emitted by `Phase6BatchStream` to satisfy the trainer's per-step
        contract — they are intentionally ignored here.

        Training data is procedurally generated (see `phase6_pref`); no
        public benchmark is touched, per the locked feedback constraint.
        """
        from .phase6_pref import simpo_kto_step, Phase6BatchStream

        c = self.cfg
        if not isinstance(self.data_stream, Phase6BatchStream):
            raise RuntimeError(
                "Phase 6 requires a Phase6BatchStream as the data stream. "
                "Wrap your preference source via "
                "`Phase6BatchStream(tokenizer=..., pairs=SyntheticPreferenceStream(), ...)`."
            )
        if self.ref_model is None:
            raise RuntimeError(
                "Phase 6 requires `trainer.ref_model` to be set to a frozen "
                "reference policy before calling .train(). The entry-point "
                "script in `fant2/training/phase6_simpo_kto.py` does this for you."
            )

        examples = list(self.data_stream.last_examples)
        tokenizer = self.data_stream.tokenizer
        if not examples:
            raise RuntimeError("Phase6BatchStream produced an empty last_examples list.")

        # One simpo_kto_step per preference triple; mean across the batch.
        results = []
        loss_sum: Optional[torch.Tensor] = None
        simpo_sum: Optional[torch.Tensor] = None
        kto_sum: Optional[torch.Tensor] = None
        for ex in examples:
            res = simpo_kto_step(
                model=self.model,
                ref_model=self.ref_model,
                tokenizer=tokenizer,
                example=ex,
                simpo_beta=c.simpo_beta,
                simpo_gamma=c.simpo_gamma,
                kto_beta=c.kto_beta,
                kto_weight=c.kto_weight,
                device=c.device,
            )
            results.append(res)
            loss_sum = res.loss if loss_sum is None else loss_sum + res.loss
            simpo_sum = res.simpo if simpo_sum is None else simpo_sum + res.simpo
            kto_sum = res.kto if kto_sum is None else kto_sum + res.kto
        n = max(len(results), 1)
        loss = loss_sum / n
        simpo_mean = simpo_sum / n
        kto_mean = kto_sum / n

        # Aggregate per-batch metrics for logging.
        margins = [r.margin for r in results]
        chosen_lps = [r.chosen_lp for r in results]
        rejected_lps = [r.rejected_lp for r in results]
        # "Preference accuracy" = fraction of triples where the live policy
        # already prefers chosen (per-token logp) over rejected. With a frozen
        # ref this is a sanity signal that the chosen response really is more
        # likely under the model.
        n_correct = sum(1 for m in margins if m > 0)

        losses = {
            "pref_loss":     loss,
            "simpo":         simpo_mean,
            "kto":           kto_mean,
            "mean_margin":   torch.tensor(sum(margins) / n),
            "mean_chosen":   torch.tensor(sum(chosen_lps) / n),
            "mean_rejected": torch.tensor(sum(rejected_lps) / n),
            "pref_acc":      torch.tensor(n_correct / n),
            "total":         loss,
        }
        # Skip the post-backward router-bias update — Phase 6 does no router
        # statistics, the routing is "as-is" from Phase 5.
        out: Dict[str, Any] = {"router_outputs": []}
        return out, losses

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------

    def train(self):
        """Run the training loop for cfg.n_steps."""
        c = self.cfg
        print(f"=== FANT 2 Phase {c.phase} training, {c.n_steps} steps ===")
        print(f"  out_dir: {c.out_dir}")
        print(f"  device:  {c.device}, bf16={c.bf16}, ckpt={c.grad_checkpoint}")
        print(f"  optimizer: HybridMuon+AdamW (lr={c.muon_lr}/{c.adam_lr})")

        data_iter = iter(self.data_stream)
        t0 = time.time()
        # Running averages over any key the phase loss dict produces
        running: Dict[str, float] = {}
        n_running = 0

        for step in range(self.step + 1, self.step + c.n_steps + 1):
            self.step = step

            try:
                batch = next(data_iter)
            except StopIteration:
                # Restart the stream
                data_iter = iter(self.data_stream)
                batch = next(data_iter)

            losses = self.train_step(batch)

            # Running averages (accumulate every key produced by the phase handler)
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v
            n_running += 1

            # ----- Periodic Tikkun + fanā -----
            if step % c.tikkun_every == 0:
                n_repaired = self.model.tikkun_repair_all()
                if n_repaired > 0:
                    print(f"  [step {step}] Tikkun repaired {n_repaired} layers")
            if step % c.fana_every == 0:
                self.model.fana_dropout_all(p=0.5)
                print(f"  [step {step}] Fanā dropout (50% prob)")

            # ----- Campaign N3: SleepGate memory consolidation -----
            if (c.sleep_consolidate_every > 0
                    and step % c.sleep_consolidate_every == 0
                    and hasattr(self.model, "memory")):
                from .campaign_n import run_sleep_consolidation
                stats = run_sleep_consolidation(
                    self.model,
                    merge_threshold=c.sleep_merge_threshold,
                    staleness_horizon=c.sleep_staleness_horizon,
                )

            # ----- Periodic telemetry -----
            if step % c.telemetry_every == 0:
                with torch.no_grad():
                    # Use the most recent forward's final_hidden as the probe sample
                    # (not stored across steps to save memory; do a fresh forward)
                    probe = self.model(batch[0][:1].to(c.device))
                    # Concat all MoE-layer mega-pool decisions into one sequence
                    # (T*n_moe_layers,) for box-counting / MFDFA / avalanche probes.
                    router_seq = None
                    ros = probe.get("router_outputs") or []
                    if ros:
                        router_seq = torch.cat([
                            ro.megapool_idx.flatten() for ro in ros
                        ], dim=0)
                    snap = collect_telemetry(
                        self.model, step,
                        sample_activations=probe["final_hidden"][0],
                        sample_router_seq=router_seq,
                    )
                self.telemetry_log.append(snap)
                # Run monitors
                reports = run_monitors(self.monitors, step, self.model, snap)
                for r in reports:
                    print(f"  [step {step}] [{r.severity}] {r.name}: {r.message}")

                # ----- Avalanche-τ homeostat (Phase 4 criticality drive) -----
                # Spec §8 Phase 4: drive avalanche exponent τ → 1.5. If two
                # consecutive snapshots show |τ - 1.5| > 0.2, kick the router
                # with a small fanā perturbation. This is the criticality
                # analog of Tikkun (perturb-then-let-recover).
                if c.phase >= 4 and len(self.telemetry_log) >= 2:
                    tau_now = self.telemetry_log[-1].avalanche_tau
                    tau_prev = self.telemetry_log[-2].avalanche_tau
                    if (tau_now is not None and tau_prev is not None
                            and not (math.isnan(tau_now) or math.isnan(tau_prev))
                            and abs(tau_now - 1.5) > 0.2
                            and abs(tau_prev - 1.5) > 0.2):
                        self.model.fana_dropout_all(p=0.2)
                        print(f"  [step {step}] [homeostat] avalanche τ drift "
                              f"({tau_prev:.2f} → {tau_now:.2f}), kicked router")

            # ----- Periodic logging -----
            if step % c.log_every == 0:
                avg = {k: v / max(n_running, 1) for k, v in running.items()}
                dt = time.time() - t0
                tps = (step * c.batch_size * c.seq_len) / max(dt, 1e-6)
                # Build log line from whatever keys this phase emits.
                # Put "total" last; skip "total" if it's the only key.
                parts = [f"[step {step:6d}]"]
                priority = ["ce", "jepa", "sigreg", "fep_kl", "z_loss",
                            "calib_rank", "calib_cond", "succ",
                            "ortho", "rvar"]
                for k in priority:
                    if k in avg:
                        parts.append(f"{k}={avg[k]:.4f}")
                if "total" in avg:
                    parts.append(f"total={avg['total']:.4f}")
                parts.append(f"({tps:.0f} tok/s)")
                print("  " + "  ".join(parts))
                running = {}
                n_running = 0

            # ----- Periodic checkpoint -----
            if step % c.save_every == 0:
                self.save_checkpoint(os.path.join(c.out_dir, f"step_{step}.pt"))

        # Final save
        self.save_checkpoint(os.path.join(c.out_dir, "final.pt"))
        print(f"=== Training done. Total time: {(time.time() - t0):.1f}s ===")

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------

    def save_checkpoint(self, path: str) -> None:
        torch.save({
            "step": self.step,
            "model": self.model.state_dict(),
            "opt": self.opt.state_dict(),
            "cfg": self.cfg,
        }, path)
        print(f"  saved checkpoint to {path}")

    def load_checkpoint(self, path: str) -> None:
        # weights_only=False because the checkpoint contains a TrainConfig
        # dataclass alongside the state_dict. We trust our own checkpoints.
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        # Optimizer state is optional — partial/crash-safe checkpoints from
        # outer scripts may only have model weights. Cross-phase resumes
        # (Option I → Option K) also legitimately want a fresh optimizer.
        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
            opt_msg = "opt restored"
        else:
            opt_msg = "opt skipped (not in ckpt)"
        self.step = ckpt.get("step", 0)
        print(f"  resumed from {path} at step {self.step} ({opt_msg})")
