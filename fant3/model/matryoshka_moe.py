"""
Matryoshka MoE routing — nested coarse-to-fine expert activation.

Reference: arxiv:2509.26520 (Wang et al. 2025).

Standard MoE picks top-k experts per token. Matryoshka MoE instead trains the
model with VARYING numbers of active experts per token, such that:

  Level 0:  only expert[0] active         — learns COARSE behavior
  Level 1:  experts[0..1]  active         — adds first-order detail
  Level 2:  experts[0..3]  active         — adds second-order detail
  Level L:  experts[0..2^L-1]  active     — adds Lth-order detail

This gives a MONOTONE nested hierarchy: at inference time you can decide how
many experts to activate based on compute budget, and the model still works
(elastic inference). Lower levels generalize because they're trained against
more inputs; higher levels specialize because they see harder residuals.

FANT 3 integration:
  - Nested within each megapool. Each megapool has n_per_megapool (16) experts,
    arranged in `n_matryoshka_levels=4` nested bands of 1, 2, 4, 8 experts.
  - Router picks the matryoshka LEVEL per token (softmax over levels), then
    activates the corresponding nested band.
  - Preserves FANT 2's fractal expert factorization (Kronecker 3-level):
    each expert is still materialized from shared A ⊗ B ⊗ C at call time.
  - Parisi RSB interpretation: the megapool tier becomes the "temperature"
    of replica-symmetry breaking, the matryoshka level within megapool is
    the "ultrametric depth."

This module implements only the routing + aggregation logic. The Kronecker
materialization of each expert weight comes from a separate expert module
(reusing FANT 2's pattern).
"""

from __future__ import annotations
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Router: picks megapool + matryoshka level per token
# ─────────────────────────────────────────────────────────────────────────────

class MatryoshkaRouter(nn.Module):
    """
    Two-stage router:
        1. Megapool logits  —  (dim → n_megapools)     — picks WHICH megapool
        2. Level logits     —  (dim → n_matryoshka_levels) — picks how many experts
           within that megapool to activate (nested band: 1, 2, 4, 8, ...)

    The final routing weights are a product of the two distributions (per-token),
    then summed with a learned shared-expert gate.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.n_megapools = cfg.n_megapools
        self.n_levels    = cfg.n_matryoshka_levels
        self.n_per_mp    = cfg.n_per_megapool

        # Nested band sizes: 1, 2, 4, 8, ... (MERA-style scale-invariant tree,
        # Belin-Myers-Ruan-Sárosi-Speranza CDS 2837843). For full MERA coverage
        # set cfg.n_matryoshka_levels = floor(log2(n_per_megapool)) + 1.
        # Last level is capped at n_per_megapool.
        self.band_sizes = []
        for lv in range(self.n_levels):
            sz = min(2 ** lv, self.n_per_mp)
            self.band_sizes.append(sz)
        assert self.band_sizes[-1] <= self.n_per_mp, \
            f"Matryoshka levels ({self.band_sizes}) exceed n_per_megapool ({self.n_per_mp})"

        self.megapool_proj = nn.Linear(self.dim, self.n_megapools, bias=False)
        self.level_proj    = nn.Linear(self.dim, self.n_levels,    bias=False)

        # Bias correction buffers (DeepSeek-V3 style aux-loss-free balance)
        self.register_buffer("megapool_bias", torch.zeros(self.n_megapools))
        self.register_buffer("level_bias",    torch.zeros(self.n_levels))

        # EMA of load per megapool / level, for Tikkun repair
        self.register_buffer("megapool_load_ema", torch.full((self.n_megapools,), 1.0 / self.n_megapools))
        self.register_buffer("level_load_ema",    torch.full((self.n_levels,),    1.0 / self.n_levels))

        self.ema_decay = 0.99

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, T, dim)
        Returns a dict with:
            mp_logits:    (B*T, n_megapools)
            mp_probs:     (B*T, n_megapools)
            lv_logits:    (B*T, n_levels)
            lv_probs:     (B*T, n_levels)
            band_sizes:   list[int]      — static nested band sizes
            mp_idx:       (B*T,)         — argmax megapool (for dispatch)
            lv_idx:       (B*T,)         — argmax level
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)  # (B*T, D)

        mp_logits = self.megapool_proj(flat) + self.megapool_bias  # (N, n_mp)
        lv_logits = self.level_proj(flat)    + self.level_bias     # (N, n_lv)

        mp_probs = F.softmax(mp_logits, dim=-1)
        lv_probs = F.softmax(lv_logits, dim=-1)

        mp_idx = mp_logits.argmax(dim=-1)   # (N,)
        lv_idx = lv_logits.argmax(dim=-1)   # (N,)

        # Update load EMA (detached; used for Tikkun, not gradients)
        if self.training:
            with torch.no_grad():
                mp_load = F.one_hot(mp_idx, self.n_megapools).float().mean(dim=0)
                lv_load = F.one_hot(lv_idx, self.n_levels).float().mean(dim=0)
                self.megapool_load_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * mp_load)
                self.level_load_ema.mul_(self.ema_decay).add_((1 - self.ema_decay) * lv_load)

        # OLMoE-style z-loss of BOTH logit projections. Sum to total loss with
        # a small coefficient (~1e-4) to keep router logits bounded. Without
        # this the router drifts unbounded and MoE collapses (+NaN CE).
        z = self.z_loss(mp_logits, lv_logits)

        return {
            "mp_logits":  mp_logits,
            "mp_probs":   mp_probs,
            "lv_logits":  lv_logits,
            "lv_probs":   lv_probs,
            "band_sizes": self.band_sizes,
            "mp_idx":     mp_idx,
            "lv_idx":     lv_idx,
            "mp_replicon": self.compute_replicon(mp_logits),
            "lv_replicon": self.compute_replicon(lv_logits),
            "z_loss":     z,
        }

    @staticmethod
    def compute_replicon(logits: torch.Tensor, temperature: float = 1.0,
                         capacity_factor: float = 1.0) -> torch.Tensor:
        # Ritort AT-line replicon eigenvalue (CERN CDS 263665). Positive value
        # = routing on the unstable side of replica-symmetry breaking (expert
        # collapse imminent). Logged as scalar diagnostic; no gradient path.
        with torch.no_grad():
            if logits.shape[-1] < 2:
                return logits.new_zeros(())
            top2 = torch.topk(logits, k=2, dim=-1).values
            gap = top2[..., 0] - top2[..., 1]
            return gap.var() - (temperature * capacity_factor) ** 2

    # -------------------------------------------------------------------------
    #  Auxiliary losses
    # -------------------------------------------------------------------------

    def z_loss(self, mp_logits: torch.Tensor, lv_logits: torch.Tensor) -> torch.Tensor:
        """OLMoE-style z-loss on both levels of routing."""
        mp_z = torch.logsumexp(mp_logits, dim=-1).square().mean()
        lv_z = torch.logsumexp(lv_logits, dim=-1).square().mean()
        return 0.5 * (mp_z + lv_z)

    def fep_kl_prior(self) -> torch.Tensor:
        """
        FEP KL between the empirical megapool/level load EMA and the uniform
        prior (1/n). This is the same role as FANT 2's FEP term.
        """
        mp_uniform = torch.full_like(self.megapool_load_ema, 1.0 / self.n_megapools)
        lv_uniform = torch.full_like(self.level_load_ema,    1.0 / self.n_levels)
        mp_kl = (self.megapool_load_ema * (self.megapool_load_ema.clamp(min=1e-12).log() - mp_uniform.log())).sum()
        lv_kl = (self.level_load_ema    * (self.level_load_ema.clamp(min=1e-12).log()    - lv_uniform.log())).sum()
        return 0.5 * (mp_kl + lv_kl)


# ─────────────────────────────────────────────────────────────────────────────
#  Matryoshka MoE FFN (the full block: router + experts + aggregation)
# ─────────────────────────────────────────────────────────────────────────────

class MatryoshkaMoEFFN(nn.Module):
    """
    Full Matryoshka MoE feed-forward block.

    For this initial version, each expert is a standard SwiGLU FFN (like
    FANT 2's shared_expert pattern). Kronecker 3-level factorization is a
    later upgrade (`_materialize_kron_expert`).

    Forward computation:
        1. Router picks a megapool and matryoshka level per token.
        2. Activate the nested band of experts[0..band_size-1] in that megapool.
        3. Normalized expert weights: softmax over ONLY the active band.
        4. Aggregate: sum_i (w_i * expert_i(x))  +  shared_expert(x) * shared_gate.

    Complexity: O(N_tokens * avg_band_size) expert-calls, vs classic top-k MoE
    which is O(N * k). The Matryoshka average band size can be smaller OR
    larger depending on the level distribution.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = cfg.dim
        self.n_mp = cfg.n_megapools
        self.n_per_mp = cfg.n_per_megapool
        self.hidden = cfg.moe_hidden
        self.shared_hidden = cfg.shared_expert_hidden

        self.router = MatryoshkaRouter(cfg)

        # Experts: for now, dense SwiGLU. Shape (n_mp * n_per_mp, dim, 2*hidden)
        # for gate+value projection, and (n_mp * n_per_mp, hidden, dim) for output.
        # Kron factorization comes in a later version.
        n_experts = self.n_mp * self.n_per_mp
        # Weight shapes follow SwiGLU: W_up: (dim, 2*hidden); W_down: (hidden, dim)
        self.W_up   = nn.Parameter(torch.randn(n_experts, self.dim, 2 * self.hidden) * (1.0 / math.sqrt(self.dim)))
        self.W_down = nn.Parameter(torch.randn(n_experts, self.hidden, self.dim)     * (1.0 / math.sqrt(self.hidden)))

        # Always-on shared expert (narrower)
        self.shared_up   = nn.Linear(self.dim, 2 * self.shared_hidden, bias=False)
        self.shared_down = nn.Linear(self.shared_hidden, self.dim, bias=False)
        self.shared_gate = nn.Parameter(torch.zeros(1))

    def _swiglu(self, x: torch.Tensor, W_up: torch.Tensor, W_down: torch.Tensor) -> torch.Tensor:
        """
        x: (..., dim). W_up: (dim, 2*hidden). W_down: (hidden, dim).
        """
        up = x @ W_up                        # (..., 2*hidden)
        gate, val = up.chunk(2, dim=-1)      # (..., hidden) each
        h = F.silu(gate) * val               # (..., hidden)
        return h @ W_down                    # (..., dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        x: (B, T, dim)
        Returns (out, router_info) where out has the same shape as x.
        """
        B, T, D = x.shape
        N = B * T
        flat = x.reshape(N, D)

        r = self.router(x)
        mp_idx    = r["mp_idx"]        # (N,)
        lv_idx    = r["lv_idx"]        # (N,)
        band_sizes = r["band_sizes"]   # list[int] len n_levels

        # Per-token active band size
        band_per_token = torch.tensor(band_sizes, device=x.device)[lv_idx]  # (N,)

        # Iterate over the levels since each level has a different band size.
        # (This is a simple first implementation; a production version would
        # use scatter_reduce + flash-style batching.)
        out = torch.zeros_like(flat)
        for lv, band_size in enumerate(band_sizes):
            mask = (lv_idx == lv)  # (N,)
            if not mask.any():
                continue
            x_sel = flat[mask]                # (M, D)
            mp_sel = mp_idx[mask]             # (M,)
            M = x_sel.shape[0]

            # For each selected token, activate experts in its megapool:
            # expert_id = mp_sel * n_per_mp + local_expert_idx, local_expert_idx in [0, band_size)
            # Expert weights: we run each of the `band_size` experts and do a
            # softmax-weighted sum.

            # Collect the W_up/W_down tensors for the active band.
            # Indices: (M, band_size) = mp_sel[:, None] * n_per_mp + arange(band_size)
            local = torch.arange(band_size, device=x.device)        # (band_size,)
            idx = mp_sel.unsqueeze(-1) * self.n_per_mp + local      # (M, band_size)

            W_up_sel = self.W_up[idx]     # (M, band_size, D, 2*hidden)
            W_down_sel = self.W_down[idx] # (M, band_size, hidden, D)

            # Batched matmul via einsum (cleaner for variable band_size)
            # up[m, b, :] = x_sel[m, :] @ W_up_sel[m, b, :, :]
            up = torch.einsum('md,mbdh->mbh', x_sel, W_up_sel)       # (M, band_size, 2*hidden)
            gate, val = up.chunk(2, dim=-1)
            h = F.silu(gate) * val                                    # (M, band_size, hidden)
            # e_out[m, b, :] = h[m, b, :] @ W_down_sel[m, b, :, :]
            e_out = torch.einsum('mbh,mbhd->mbd', h, W_down_sel)      # (M, band_size, D)

            # Uniform weighting within the band (all band members contribute equally).
            w = torch.full((M, band_size), 1.0 / band_size, device=x.device, dtype=e_out.dtype)
            e_sum = (w.unsqueeze(-1) * e_out).sum(dim=1)              # (M, D)
            out[mask] = e_sum

        # Shared expert contribution
        shared = self._swiglu_linear(flat, self.shared_up, self.shared_down)
        out = out + torch.sigmoid(self.shared_gate) * shared

        return out.reshape(B, T, D), r

    def _swiglu_linear(self, x: torch.Tensor, up: nn.Linear, down: nn.Linear) -> torch.Tensor:
        u = up(x)
        gate, val = u.chunk(2, dim=-1)
        return down(F.silu(gate) * val)
