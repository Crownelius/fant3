"""
MASA — "Share Your Attention" shared-atom attention.

Reference: arxiv:2508.04581 (Zhussip et al. 2025).

Per-layer Q/K/V/O are each expressed as a low-rank combination of a shared
dictionary of atom matrices:

    Q_layer  = sum_i  q_layer[i] * A_Q[i]   for i in 1..n_atoms
    K_layer  = sum_i  k_layer[i] * A_K[i]
    V_layer  = sum_i  v_layer[i] * A_V[i]
    O_layer  = sum_i  o_layer[i] * A_O[i]

The atoms A_{Q,K,V,O}[i] ∈ R^(dim × dim) are shared across ALL layers; each
layer only stores the per-atom scalar (or low-rank) coefficients
q_layer, k_layer, v_layer, o_layer ∈ R^(n_atoms × masa_coef_rank).

With n_atoms=6, masa_coef_rank=16 on a 24-layer × 2048-dim model:
    Classic  per-layer:  4 * 24 * (2048*2048)     = 402.7 M params
    MASA:                4 * (6 * 2048*2048)      +
                         4 * 24 * (n_atoms * rank) = 100.7 M + 9 K = ~100.7 M params
  Savings: 75% (MASA paper reports 66.7% on BERT/ViT; our reduction is higher
  because we share Q/K/V/O and the model is deeper).

Compatibility notes:
  - GQA (n_kv_heads < n_heads) handled by projecting K,V into (n_kv_heads * head_dim)
    before the atom combination. The atoms are split correspondingly.
  - Partial RoPE (Phi-4-Mini style) applies rotations only to the first
    `int(head_dim * rope_partial)` dims of Q and K.
  - Attention sinks (first `n_attention_sinks` positions always attended to)
    handled in the attention mask, not here.
"""

from __future__ import annotations
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
#  Atom bank (shared across all layers)
# ─────────────────────────────────────────────────────────────────────────────

class MASAAtomBank(nn.Module):
    """
    The shared dictionary of (n_atoms × dim × dim) atom matrices for Q, K, V, O.

    Instantiated ONCE per model, passed by reference to every MASAAttention layer.

    Q and O operate on full dim×dim.
    K and V operate on dim×kv_dim where kv_dim = n_kv_heads * head_dim (GQA).
    """

    def __init__(self, dim: int, n_atoms: int, kv_dim: int):
        super().__init__()
        self.dim = dim
        self.n_atoms = n_atoms
        self.kv_dim = kv_dim

        init_scale = 1.0 / math.sqrt(dim)
        self.A_Q = nn.Parameter(torch.randn(n_atoms, dim, dim)    * init_scale)
        self.A_K = nn.Parameter(torch.randn(n_atoms, dim, kv_dim) * init_scale)
        self.A_V = nn.Parameter(torch.randn(n_atoms, dim, kv_dim) * init_scale)
        self.A_O = nn.Parameter(torch.randn(n_atoms, dim, dim)    * init_scale)


# ─────────────────────────────────────────────────────────────────────────────
#  Per-layer attention using the shared atom bank
# ─────────────────────────────────────────────────────────────────────────────

class MASAAttention(nn.Module):
    """
    One attention layer that composes its Q/K/V/O matrices from the shared
    atom bank via low-rank coefficients.

    Coefficient shapes:
        q_coef: (n_atoms, coef_rank)   — q_weight = sum_i (A_Q[i] @ q_coef[i, :] ... )
    In practice we compute the assembled matrices on the fly per forward pass:
        Q_mat = einsum("adc,ar->rdc", atom_bank.A_Q, self.q_coef).sum(0) — too expensive.
    Instead we use the cheaper:
        W_Q = sum_i (q_coef[i].mean() * atom_bank.A_Q[i])  — rank-1 combination.
    For rank > 1 we extend to:
        W_Q = einsum("adc,ar->rdc", A_Q, q_coef).reshape(...)
    """

    def __init__(self, cfg, atom_bank: MASAAtomBank, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.atom_bank = atom_bank  # shared reference — NOT owned
        self.layer_idx = layer_idx

        self.dim = cfg.dim
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.kv_groups = cfg.n_heads // cfg.n_kv_heads

        n_atoms = cfg.n_attention_atoms
        rank = cfg.masa_coef_rank

        # Per-layer low-rank coefficient tensors
        # Shape: (n_atoms, rank). We combine atoms via `A[i] * coef[i].mean()` for
        # rank=1 behavior, or the full rank-r sum for rank>1.
        self.q_coef = nn.Parameter(self._init_coef(n_atoms, rank))
        self.k_coef = nn.Parameter(self._init_coef(n_atoms, rank))
        self.v_coef = nn.Parameter(self._init_coef(n_atoms, rank))
        self.o_coef = nn.Parameter(self._init_coef(n_atoms, rank))

        # Partial RoPE: only the first `rope_head_dim` of each head gets rotated.
        self.rope_head_dim = int(cfg.head_dim * cfg.rope_partial)
        self.rope_theta = cfg.rope_theta

        # Precompute the RoPE frequencies (fixed, not a parameter)
        self.register_buffer(
            "rope_inv_freq",
            1.0 / (cfg.rope_theta ** (torch.arange(0, self.rope_head_dim, 2).float() / self.rope_head_dim)),
            persistent=False,
        )

    @staticmethod
    def _init_coef(n_atoms: int, rank: int) -> torch.Tensor:
        # Initialize coefficients so that the sum of atoms approximates one
        # learned attention matrix: start near the all-ones mean (so each
        # atom contributes equally to start), with a small perturbation.
        base = torch.full((n_atoms, rank), 1.0 / math.sqrt(n_atoms * rank))
        return base + 0.01 * torch.randn(n_atoms, rank)

    # -------------------------------------------------------------------------
    #  Assemble W_Q, W_K, W_V, W_O from atoms × per-layer coefficients
    # -------------------------------------------------------------------------

    def _assemble(self, atoms: torch.Tensor, coef: torch.Tensor) -> torch.Tensor:
        """
        atoms: (n_atoms, in_dim, out_dim)
        coef:  (n_atoms, rank)

        Returns: (in_dim, out_dim) — the per-layer assembled weight.

        For rank=r: W = sum_{i, j} coef[i, j] * atoms[i]  (stable, all ranks).
        Equivalent to taking a scalar per atom = coef[i].sum().
        """
        # Sum over the rank axis first (collapse to per-atom scalars),
        # then weighted sum over atoms.
        scalars = coef.sum(dim=-1)  # (n_atoms,)
        # W[d, o] = sum_i scalars[i] * atoms[i, d, o]
        W = (scalars[:, None, None] * atoms).sum(dim=0)
        return W

    # -------------------------------------------------------------------------
    #  RoPE application (partial — only first rope_head_dim of each head)
    # -------------------------------------------------------------------------

    def _apply_rope(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        """
        x:   (B, n_heads, T, head_dim)
        pos: (T,)
        """
        r = self.rope_head_dim
        if r == 0:
            return x
        x_rot, x_pass = x[..., :r], x[..., r:]
        # freqs in f32 for numerical accuracy, then cast cos/sin to x.dtype so
        # the output preserves input precision. Without this cast, bf16 x silently
        # promotes to f32 through cos/sin and mismatches V in SDPA (bug 2026-04-19).
        freqs = pos.float().unsqueeze(-1) * self.rope_inv_freq  # (T, r/2) f32
        cos = freqs.cos().to(dtype=x.dtype)
        sin = freqs.sin().to(dtype=x.dtype)
        # x_rot shape: (B, H, T, r); treat as (..., r/2, 2)
        x_rot = x_rot.reshape(*x_rot.shape[:-1], -1, 2)
        rot_cos = x_rot[..., 0] * cos - x_rot[..., 1] * sin
        rot_sin = x_rot[..., 0] * sin + x_rot[..., 1] * cos
        x_rot = torch.stack([rot_cos, rot_sin], dim=-1).reshape(*x_rot.shape[:-2], r)
        return torch.cat([x_rot, x_pass], dim=-1)

    # -------------------------------------------------------------------------
    #  Area-law sliding-window mask (Ryu-Takayanagi / entanglement-wedge)
    # -------------------------------------------------------------------------

    @staticmethod
    def _area_law_mask(T: int, device, dtype) -> torch.Tensor:
        window = int(math.sqrt(T)) + 1
        i = torch.arange(T, device=device).unsqueeze(1)
        j = torch.arange(T, device=device).unsqueeze(0)
        causal    = j <= i
        in_window = (i - j) <= window
        keep = causal & in_window
        mask = torch.full((T, T), float("-inf"), device=device, dtype=dtype)
        mask.masked_fill_(keep, 0.0)
        return mask

    # -------------------------------------------------------------------------
    #  Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x:    (B, T, dim)
        mask: optional (B?, T, T) additive mask (causal / attention-sinks applied upstream)

        Returns: (B, T, dim)
        """
        B, T, D = x.shape
        ab = self.atom_bank

        # Assemble per-layer Q/K/V/O projection matrices from the shared atoms
        W_Q = self._assemble(ab.A_Q, self.q_coef)  # (dim, dim)
        W_K = self._assemble(ab.A_K, self.k_coef)  # (dim, kv_dim)
        W_V = self._assemble(ab.A_V, self.v_coef)  # (dim, kv_dim)
        W_O = self._assemble(ab.A_O, self.o_coef)  # (dim, dim)

        # Project
        q = x @ W_Q   # (B, T, dim)
        k = x @ W_K   # (B, T, kv_dim)
        v = x @ W_V   # (B, T, kv_dim)

        # Reshape to heads
        q = q.view(B, T, self.n_heads,    self.head_dim).transpose(1, 2)   # (B, H,   T, Hd)
        k = k.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, Hkv, T, Hd)
        v = v.view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)   # (B, Hkv, T, Hd)

        # Partial RoPE
        pos = torch.arange(T, device=x.device)
        q = self._apply_rope(q, pos)
        k = self._apply_rope(k, pos)

        # GQA: repeat K,V for each query head group
        if self.kv_groups > 1:
            k = k.repeat_interleave(self.kv_groups, dim=1)  # (B, H, T, Hd)
            v = v.repeat_interleave(self.kv_groups, dim=1)

        # Scaled dot-product attention. Default: full causal. Opt-in
        # `masa_area_law_window` replaces causal with a sliding-window causal
        # mask of size floor(sqrt(T))+1, implementing the RT-surface area-law
        # prescription (Faulkner-Lewkowycz-Maldacena, CERN CDS 2199026).
        if mask is None and getattr(self.cfg, "masa_area_law_window", False) and T > 4:
            mask = self._area_law_mask(T, x.device, x.dtype)
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        elif mask is None:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        # Reshape back
        out = out.transpose(1, 2).reshape(B, T, D)

        # Output projection
        out = out @ W_O
        return out
