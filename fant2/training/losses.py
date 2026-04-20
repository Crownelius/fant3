"""
Unified loss functions for FANT 2 training.

This file implements the SINGLE Free Energy Principle (FEP) loss that replaces
the FANT 350M's stack of 4 separate auxiliary losses. The unified form is:

    L_total  =  L_CE                          (cross-entropy on next token)
              + α_z   * L_z_loss              (OLMoE router z-loss)
              + β_kl  * L_FEP_KL              (Free Energy Principle KL prior)
              + α_jepa * L_JEPA               (LLM-JEPA self-supervised, Phase 1)
              + α_sigreg * L_SIGReg           (signal regularization, Phase 1)
              + α_succ * L_succ               (success estimator, Phase 4)

The FEP form replaces the FANT 350M losses (load balance + diversity entropy +
calibration + STaR) with a single information-theoretic objective derived from
the Free Energy Principle (Friston 2010, Buckley et al. 2017): minimize the KL
divergence between the routing posterior and the uniform prior. This is
mathematically equivalent to:

    L_FEP_KL = KL(routing_dist || uniform_prior)
             = sum_e p(e | x) * log(p(e | x) / (1/N))
             = -H(routing_dist) + log(N)

So minimizing L_FEP_KL is the same as MAXIMIZING the routing entropy. This is
also equivalent to minimizing the load-balance loss + the diversity loss
together. One coefficient β_kl now controls what previously needed three.

References:
    - Friston (2010) "The free-energy principle: a unified brain theory?"
    - Buckley et al. (2017) "The free energy principle for action and perception"
    - DeepSeek-V3 (2024) §3.2 (z-loss)
    - OLMoE (2024) §4.1 (z-loss)
    - LLM-JEPA (2024) Garrido et al.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.router import RouterOutput


# -----------------------------------------------------------------------------
# 1. Cross-entropy with optional label smoothing
# -----------------------------------------------------------------------------

def cross_entropy_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Standard next-token cross-entropy."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
    )


# -----------------------------------------------------------------------------
# 2. OLMoE router z-loss
# -----------------------------------------------------------------------------

def router_z_loss(router_outputs: List[RouterOutput]) -> torch.Tensor:
    """
    Average z-loss across all MoE layers.

        L_z = mean over layers of (logsumexp(router_logits))^2

    Penalizes the magnitude of the router logits to prevent runaway scaling.
    """
    if not router_outputs:
        return torch.tensor(0.0)
    parts = []
    for ro in router_outputs:
        z = (torch.logsumexp(ro.expert_logits, dim=-1) ** 2).mean()
        parts.append(z)
    return torch.stack(parts).mean()


# -----------------------------------------------------------------------------
# 3. Free Energy Principle KL prior
# -----------------------------------------------------------------------------

def fep_kl_prior(router_outputs: List[RouterOutput]) -> torch.Tensor:
    """
    Average FEP KL prior across all MoE layers.

    For each layer, the routing distribution is the (megapool_load, expert_load)
    tuple. We compute KL(routing_dist || uniform) for each tier and sum them.
    """
    if not router_outputs:
        return torch.tensor(0.0)
    eps = 1e-8
    parts = []
    for ro in router_outputs:
        n_mp = ro.megapool_load.numel()
        n_ex = ro.expert_load.numel()
        target_mp = 1.0 / n_mp
        target_ex = 1.0 / n_ex
        # KL(p || target_uniform) = sum p * log(p / target)
        mp = ro.megapool_load + eps
        ex = ro.expert_load.flatten() + eps
        kl_mp = (mp * (mp.log() - torch.log(torch.full_like(mp, target_mp)))).sum()
        kl_ex = (ex * (ex.log() - torch.log(torch.full_like(ex, target_ex)))).sum()
        parts.append(kl_mp + kl_ex)
    return torch.stack(parts).mean()


# -----------------------------------------------------------------------------
# 4. LLM-JEPA loss (Phase 1 self-supervised pretraining)
# -----------------------------------------------------------------------------

def llm_jepa_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    sigreg_coef: float = 0.5,
) -> Dict[str, torch.Tensor]:
    """
    LLM-JEPA = Joint Embedding Predictive Architecture loss for language models.

    Args:
        pred:        (B, T, dim) JEPA-predictor output
        target:      (B, T, dim) the masked-target embeddings (DETACHED from grad)
        sigreg_coef: weight on the SIGReg variance regularizer

    Returns:
        dict with "jepa", "sigreg", "total"
    """
    # Predictor MSE loss (target is detached so no EMA needed)
    target_detached = target.detach()
    jepa = F.mse_loss(pred, target_detached)

    # SIGReg: penalize feature collapse (require unit-ish std along feature dim)
    # std-deviation of the prediction across batch+sequence positions
    flat = pred.reshape(-1, pred.size(-1))
    std = flat.std(dim=0)  # (dim,)
    sigreg = torch.relu(1.0 - std).mean()

    return {
        "jepa":   jepa,
        "sigreg": sigreg_coef * sigreg,
        "total":  jepa + sigreg_coef * sigreg,
    }


# -----------------------------------------------------------------------------
# 5. Success estimator loss (Phase 4 self-refinement)
# -----------------------------------------------------------------------------

def success_estimator_loss(
    pred: torch.Tensor,
    target_correct: torch.Tensor,
) -> torch.Tensor:
    """
    BCE between the success_estimator's per-token output and the actual
    correctness label (1 if the model's argmax == ground truth, 0 else).

    Args:
        pred:           (B, T, 1) sigmoid'd success estimator output
        target_correct: (B, T) bool tensor of correctness labels
    """
    return F.binary_cross_entropy(
        pred.squeeze(-1),
        target_correct.float(),
        reduction="mean",
    )


# -----------------------------------------------------------------------------
# 5b. Progressive alignment loss (Phase 4, Option M: SpiralThinker 2511.08983)
# -----------------------------------------------------------------------------

def progressive_alignment_loss(
    h1: torch.Tensor,
    h2: torch.Tensor,
    weight: float = 1.0,
) -> torch.Tensor:
    """
    SpiralThinker-style progressive alignment between consecutive refinement
    passes. Penalizes cosine drift between pass-1 and pass-2 final hidden
    states so pass 2 can refine but not wander off the pass-1 manifold.

    Reference: arxiv:2511.08983 — "Bounded Progressive Alignment for Latent
    Reasoning" (reports the drift-as-failure-mode for iterative latent stacks
    and demonstrates the alignment anchor as the stabilizer).

    Args:
        h1:     (B, T, dim) pass-1 final hidden state (treated as anchor; detached)
        h2:     (B, T, dim) pass-2 final hidden state (the one receiving the loss)
        weight: scalar multiplier applied to the `1 - cos(h1, h2)` term

    Returns:
        scalar alignment penalty suitable for direct addition to a loss sum.
    """
    h1_flat = h1.detach().reshape(-1, h1.size(-1))
    h2_flat = h2.reshape(-1, h2.size(-1))
    cos_sim = F.cosine_similarity(h1_flat, h2_flat, dim=-1, eps=1e-8)
    # We want pass 2 to stay aligned with pass 1 → minimize (1 - cos)
    return weight * (1.0 - cos_sim).mean()


# -----------------------------------------------------------------------------
# 6. Unified FEP loss (the single combined objective)
# -----------------------------------------------------------------------------

def fep_unified_loss(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    router_outputs: List[RouterOutput],
    z_loss_alpha: float = 1e-3,
    fep_kl_beta: float = 0.1,
    label_smoothing: float = 0.0,
    ignore_index: int = -100,
    # Campaign N1 coefficients (default 0.0 = off for backward compat)
    ortho_alpha: float = 0.0,
    var_alpha: float = 0.0,
    moe_layers: Optional[nn.ModuleList] = None,
) -> Dict[str, torch.Tensor]:
    """
    The single FEP-derived loss that subsumes:
        - cross-entropy
        - load balance     (subsumed by KL prior)
        - diversity entropy (subsumed by KL prior)
        - z-loss
        - [N1] expert orthogonality (if ortho_alpha > 0)
        - [N1] router variance      (if var_alpha > 0)

    Returns a dict of named scalars + the "total" key for backward().
    """
    ce = cross_entropy_loss(logits, targets, ignore_index=ignore_index, label_smoothing=label_smoothing)
    z = router_z_loss(router_outputs)
    kl = fep_kl_prior(router_outputs)
    total = ce + z_loss_alpha * z + fep_kl_beta * kl

    result: Dict[str, torch.Tensor] = {
        "ce":      ce,
        "z_loss":  z_loss_alpha * z,
        "fep_kl":  fep_kl_beta * kl,
    }

    # Campaign N1: expert orthogonality
    if ortho_alpha > 0 and moe_layers is not None:
        ortho = expert_orthogonality_loss(moe_layers)
        total = total + ortho_alpha * ortho
        result["ortho"] = ortho_alpha * ortho

    # Campaign N1: router variance (negative = maximize variance)
    if var_alpha > 0:
        rvar = router_variance_loss(router_outputs)
        total = total + var_alpha * rvar
        result["rvar"] = var_alpha * rvar

    result["total"] = total
    return result


# -----------------------------------------------------------------------------
# 6a. Campaign N1: Expert orthogonality + router variance losses
#     arXiv:2505.22323 — Advancing Expert Specialization in MoE
# -----------------------------------------------------------------------------

def expert_orthogonality_loss(moe_layers: nn.ModuleList) -> torch.Tensor:
    """
    Push expert A-factor matrices toward mutual orthogonality.

    For each MoE layer, computes pairwise ||A_i^T A_j||_F across experts
    within the same megapool (9 experts per pool, 8 pools).  Using A factors
    (8×40) instead of materialized weights (256×1280) is:
      - 10× cheaper  (320 vs 327K elements per expert)
      - More meaningful (A is the unique per-expert component; B is shared)

    L_ortho = mean_over_layers mean_over_pools mean_over_pairs ||A_i^T A_j||_F
    """
    if not moe_layers:
        return torch.tensor(0.0)

    layer_losses = []
    for layer in moe_layers:
        experts = layer.fractal_experts  # nn.ModuleList of 72 FractalSeedExperts
        n_per_pool = getattr(layer, "n_per_pool", 9)
        n_pools = len(experts) // n_per_pool

        pool_losses = []
        for p in range(n_pools):
            start = p * n_per_pool
            # Stack A_gate for all experts in this pool: (n_per_pool, 8, 40)
            A_stack = torch.stack([
                experts[start + i].A_gate for i in range(n_per_pool)
            ], dim=0)
            # Flatten to (n_per_pool, 320)
            A_flat = A_stack.reshape(n_per_pool, -1)
            # Gram matrix: (n_per_pool, n_per_pool)
            gram = A_flat @ A_flat.T
            # Zero the diagonal (self-similarity is fine)
            mask = 1.0 - torch.eye(n_per_pool, device=gram.device)
            # Mean of off-diagonal squared entries
            ortho = (gram * mask).pow(2).sum() / max(mask.sum(), 1.0)
            pool_losses.append(ortho)

        layer_losses.append(torch.stack(pool_losses).mean())
    return torch.stack(layer_losses).mean()


def router_variance_loss(router_outputs: List[RouterOutput]) -> torch.Tensor:
    """
    Encourage more decisive routing by maximizing logit variance.

    L_var = -mean(Var(expert_logits)) across layers and tokens.
    Negative because we MAXIMIZE variance (minimize -variance).

    Higher variance → more confident routing → less expert overlap.
    """
    if not router_outputs:
        return torch.tensor(0.0)

    parts = []
    for ro in router_outputs:
        # Expert logits: (B*T, n_per_pool)
        var_expert = ro.expert_logits.var(dim=-1).mean()
        # Megapool logits: (B*T, n_megapools)
        var_mega = ro.megapool_logits.var(dim=-1).mean()
        # Negative: we want to MAXIMIZE variance
        parts.append(-(var_expert + var_mega))
    return torch.stack(parts).mean()


# -----------------------------------------------------------------------------
# 6b. Active-layer calibration loss (Phase 3)
# -----------------------------------------------------------------------------

def effective_rank(matrix: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Entropy-based effective rank of a matrix.

        EffRank(W) = exp( - sum_i p_i log p_i )
    where p_i are the normalized singular values.

    For a matrix with k equal singular values and the rest zero, EffRank = k.
    For a full-rank orthogonal matrix of size (n, n), EffRank = n.
    """
    s = torch.linalg.svdvals(matrix.float())
    p = s / (s.sum() + eps)
    h = -(p * (p + eps).log()).sum()
    return h.exp()


def calibration_loss(
    materialized_weights: List[torch.Tensor],
    rank_target_frac: float = 0.9,
    max_condition: float = 100.0,
) -> Dict[str, torch.Tensor]:
    """
    Phase 3 active-layer calibration penalty.

    For each randomly materialized expert weight matrix, penalize:
      * effective rank < rank_target_frac * full_rank (rank collapse)
      * condition number > max_condition (ill-conditioning)

    Args:
        materialized_weights: list of 2D tensors (the kron-generated W_gate/up/down
                              for a few sampled experts, NOT detached — gradients
                              flow back into A, B, C via the kron3 op)
        rank_target_frac:     fraction of full rank we want to maintain (0.9 = 90%)
        max_condition:        cap on condition number σ_max / σ_min

    Returns:
        dict with "rank", "cond", "total"
    """
    if not materialized_weights:
        z = torch.tensor(0.0)
        return {"rank": z, "cond": z, "total": z}

    rank_parts = []
    cond_parts = []
    for W in materialized_weights:
        if W.dim() != 2:
            W = W.view(W.size(0), -1)
        full_rank = float(min(W.shape))
        s = torch.linalg.svdvals(W.float())
        # Effective rank penalty: we want eff_rank / full_rank >= rank_target_frac
        p = s / (s.sum() + 1e-8)
        eff_rank = torch.exp(-(p * (p + 1e-8).log()).sum())
        rank_ratio = eff_rank / full_rank
        rank_parts.append(torch.relu(rank_target_frac - rank_ratio))

        # Condition number penalty: sigma_max / sigma_min <= max_condition
        # (If min singular value is nearly zero, this blows up — that's the signal)
        cond = s[0] / (s[-1] + 1e-6)
        cond_parts.append(torch.relu(cond - max_condition) / max_condition)

    rank_loss = torch.stack(rank_parts).mean()
    cond_loss = torch.stack(cond_parts).mean()
    return {
        "rank":  rank_loss,
        "cond":  cond_loss,
        "total": rank_loss + cond_loss,
    }


# -----------------------------------------------------------------------------
# 7. KTO loss (Kahneman-Tversky Optimization, Phase 6)
# -----------------------------------------------------------------------------

def kto_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    ref_chosen_logps: torch.Tensor,
    ref_rejected_logps: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """
    Kahneman-Tversky Optimization loss (Ethayarajh et al. 2024).

    Unlike DPO, KTO does not require pairs — only individual desirable
    (chosen) and undesirable (rejected) examples with separate weights.
    Loss prefers actions with high "human prospect-theoretic value":
        v(chosen)   = sigmoid(beta * (chosen_logps - ref_chosen_logps - z_chosen))
        v(rejected) = sigmoid(beta * (z_rejected - (rejected_logps - ref_rejected_logps)))
    where z_chosen, z_rejected are running KL estimates.
    """
    chosen_kl = (chosen_logps - ref_chosen_logps).clamp(min=0).detach()
    rejected_kl = (rejected_logps - ref_rejected_logps).clamp(min=0).detach()
    chosen_value = torch.sigmoid(beta * (chosen_logps - ref_chosen_logps - rejected_kl.mean()))
    rejected_value = torch.sigmoid(beta * (chosen_kl.mean() - (rejected_logps - ref_rejected_logps)))
    return -(chosen_value.mean() + rejected_value.mean())


# -----------------------------------------------------------------------------
# 8. SimPO loss (Phase 6)
# -----------------------------------------------------------------------------

def simpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    chosen_lengths: torch.Tensor,
    rejected_lengths: torch.Tensor,
    beta: float = 2.0,
    gamma: float = 1.6,
) -> torch.Tensor:
    """
    Simple Preference Optimization (Meng et al. 2024).

    Length-normalized preference optimization that does NOT require a
    reference model — just the policy log-probs and the chosen/rejected
    sequence lengths. This is much cheaper than DPO.

        L_SimPO = -log sigmoid(beta * (logp_chosen / |chosen|) -
                                beta * (logp_rejected / |rejected|) - gamma)
    """
    chosen_norm = chosen_logps / chosen_lengths
    rejected_norm = rejected_logps / rejected_lengths
    margin = beta * (chosen_norm - rejected_norm) - gamma
    return -F.logsigmoid(margin).mean()


# -----------------------------------------------------------------------------
# 9. Dr.GRPO loss (Phase 5 RL training)
# -----------------------------------------------------------------------------

def dr_grpo_loss(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2,
    clip_eps_hi: Optional[float] = None,
) -> torch.Tensor:
    """
    Dr.GRPO = "Done right" Group Relative Policy Optimization (Liu et al. 2025).

    Differs from vanilla GRPO by removing the bias-introducing length
    normalization (which causes the model to prefer shorter rollouts).

        ratio = exp(new_logps - old_logps)
        clipped_ratio = clip(ratio, 1 - eps_lo, 1 + eps_hi)
        L = -mean(min(ratio * advantage, clipped_ratio * advantage))

    The asymmetric upper clip (`clip_eps_hi`, default = `clip_eps`) is the
    DAPO / Dr.GRPO "clip-higher" trick: it lets the policy tolerate larger
    upward ratio swings on positive-advantage tokens, which empirically
    improves exploration on math/code tasks. Spec §8 Phase 5 sets ε_hi=0.28.

    Note: there is no length normalization. Sum is over all tokens, mean is
    over examples in the batch.
    """
    if clip_eps_hi is None:
        clip_eps_hi = clip_eps
    ratio = torch.exp(new_logps - old_logps)
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps_hi)
    return -torch.min(ratio * advantages, clipped * advantages).mean()
