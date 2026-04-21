"""
Mixture-of-Recursions (MoR) — per-token dynamic recursion depth over a shared
layer stack.

Reference: arxiv:2507.10524 (Bae et al. 2025, NeurIPS 2025).

Standard transformers apply N distinct layers in sequence. MoR instead applies
the SAME layer repeatedly — for a DIFFERENT number of recursions per token —
with a lightweight router selecting the recursion depth.

    Token α (easy/recent):    layer(x)                    — 1 pass
    Token β (hard/schema):    layer(layer(layer(x)))      — 3 passes

FANT 3 integration:
  - The MoR wrapper sits AROUND a sub-stack of shared transformer layers.
    For example, of 24 total layers, we might make 18 of them "shared" and
    the other 6 (3 dense prefix + 3 postfix) fixed-depth.
  - The depth router is a small (dim → n_recursion_depths) linear head.
  - Depth bias: use Apollonian curvature classification as a prior
    (α tokens → shallow, β tokens → deep). Configured by `mor_depth_bias`.

Current implementation (v1): uniform-depth routing. Apollonian-biased routing
is a follow-up once Phase 4 memory population is active.
"""

from __future__ import annotations
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoRDepthRouter(nn.Module):
    """
    Lightweight per-token recursion-depth router.
    Outputs a softmax over `n_depths` options; argmax selects the depth.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_depths = cfg.n_recursion_depths
        # Bottleneck through mor_router_dim to keep the head tiny
        self.fc1 = nn.Linear(cfg.dim, cfg.mor_router_dim, bias=False)
        self.fc2 = nn.Linear(cfg.mor_router_dim, self.n_depths, bias=False)

    def forward(self, x: torch.Tensor) -> dict:
        """
        x: (B, T, dim)
        Returns: {
            "logits":  (B*T, n_depths),
            "probs":   (B*T, n_depths),
            "idx":     (B*T,)    — argmax depth per token (1..n_depths)
        }
        """
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        h = F.silu(self.fc1(flat))
        logits = self.fc2(h)
        probs = F.softmax(logits, dim=-1)
        # Depth is 1-indexed: level 0 → 1 recursion, level 1 → 2, ...
        depth_idx = logits.argmax(dim=-1)   # (N,)
        depth     = depth_idx + 1
        return {"logits": logits, "probs": probs, "idx": depth_idx, "depth": depth}


class MoRShared(nn.Module):
    """
    Wraps a single shared `TransformerBlock` module and applies it 1..max_depth
    times per-token based on the depth router.

    Token grouping by depth:
        For each depth d in 1..max_depth:
            subset = tokens whose router chose depth d
            apply shared_block(subset) d times, writing results back

    Because each recursion can re-visit the SAME block with different x,
    there's no increase in parameters — just compute. The Pareto frontier
    improvement the paper reports comes from hard tokens getting more compute
    without easy tokens wasting any.
    """

    def __init__(self, cfg, shared_block: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.block = shared_block
        self.router = MoRDepthRouter(cfg)
        self.max_depth = cfg.n_recursion_depths
        # Gradient checkpointing of inner recursion passes. MoR recurses up to
        # n_recursion_depths times through the shared block; without ckpt we
        # store activations for EVERY pass = 2-3× normal. With ckpt, activations
        # for earlier passes are recomputed on backward. Massive VRAM saver.
        self.use_gc = getattr(cfg, "use_gradient_checkpointing", False)

        # --- Mythos / Recurrent-Depth Transformer (RDT) augmentations -------
        # See cfg.mor_lti_injection_enabled docstring for the update rule.
        # All three are opt-in; defaults keep bit-compatibility with v1 MoR.
        self.lti_enabled      = getattr(cfg, "mor_lti_injection_enabled", False)
        self.spectral_enabled = getattr(cfg, "mor_spectral_constraint", False)
        self.loop_idx_enabled = getattr(cfg, "mor_loop_index_enabled", False)
        self.lti_apollonian   = getattr(cfg, "mor_lti_apollonian_channel", True)

        if self.lti_enabled:
            # Diagonal A matrix — parameterized as a_diag (free tensor); the
            # actual A = -softplus(a_diag) when mor_spectral_constraint is True.
            # That guarantees each diagonal entry a_ii in (-inf, 0) so
            # |1 + a_ii| < 1 for small |a_ii|, keeping the LTI recurrence stable.
            # We initialize near zero so injection starts as a near-no-op
            # (h_{t+1} ≈ 0*h_t + 0*x_orig + block(h_t) ≈ block(h_t), recovering v1).
            self.a_diag = nn.Parameter(torch.zeros(cfg.dim))
            # B: x_original → injected contribution.
            self.b_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
            nn.init.zeros_(self.b_proj.weight)  # start as no-op
            # C: Apollonian-retrieved context → injected contribution.
            # Only materialized if cfg asks for it; stays None otherwise so the
            # parameter count at lti_apollonian=False matches v1 + a_diag + B.
            if self.lti_apollonian:
                self.c_proj = nn.Linear(cfg.dim, cfg.dim, bias=False)
                nn.init.zeros_(self.c_proj.weight)
            else:
                self.c_proj = None

        if self.loop_idx_enabled:
            # Learned per-pass positional signal. Shape (max_depth, dim).
            # Initialized small so pass 0 ≈ pass 1 early in training.
            self.loop_emb = nn.Parameter(
                torch.randn(self.max_depth, cfg.dim) * 0.02
            )

    # -------------------------------------------------------------------------
    #  Helpers for the LTI update
    # -------------------------------------------------------------------------

    def _effective_A(self) -> torch.Tensor:
        """Return the diagonal A used in the LTI update.
        With spectral constraint: A = -softplus(a_diag), so each entry is in
        (-inf, 0); the magnitude |1 + a_ii| < 1 for |a_ii| small enough, giving
        rho(A) < 1 on the update operator (I + A).  Without the constraint,
        A = a_diag (free real-valued)."""
        if not self.lti_enabled:
            raise RuntimeError("_effective_A called when LTI injection disabled")
        if self.spectral_enabled:
            return -F.softplus(self.a_diag)
        return self.a_diag

    def _lti_injection(
        self,
        current: torch.Tensor,
        x_original: torch.Tensor,
        retrieved: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute A*current + B*x_original [+ C*retrieved].  Shapes all (B,T,D)."""
        A = self._effective_A()                      # (D,)
        inj = A * current + self.b_proj(x_original)  # (B, T, D)
        if self.c_proj is not None and retrieved is not None:
            inj = inj + self.c_proj(retrieved)
        return inj

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        curvatures: Optional[torch.Tensor] = None,
        retrieved: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        x:          (B, T, dim) — input to the MoR loop
        mask:       optional attention mask passed to the shared block
        curvatures: optional (B, T) per-token Apollonian curvature. If given
                    AND cfg.mor_depth_bias == "alpha", we bias the depth
                    router toward shallow for high-curvature (instance) tokens.
        retrieved:  optional (B, T, dim) Apollonian-retrieved context. Only
                    used when mor_lti_injection_enabled AND lti_apollonian AND
                    cfg.spinor_apollonian_enabled. This is the knob that keeps
                    FANT-style memory central in the Mythos-style update:
                    the recurrent injection reads from our Apollonian packs.
        Returns: (out, router_info)
        """
        B, T, D = x.shape

        # Depth selection
        r = self.router(x)
        depth_idx = r["idx"]                # (B*T,) — 0..n_depths-1

        # Optional curvature-informed bias: α (high curv) → shallow; β → deep
        if curvatures is not None and self.cfg.mor_depth_bias == "alpha":
            flat_curv = curvatures.reshape(-1)   # (B*T,)
            high_curv = flat_curv > flat_curv.median()
            depth_idx = torch.where(high_curv, (depth_idx - 1).clamp(min=0), depth_idx)

        depth = (depth_idx + 1).reshape(B, T)  # (B, T), values in {1..max_depth}

        # x_original is captured here for the LTI B*x_orig injection channel —
        # prevents hidden-state drift over deep recursions (Mythos trick).
        x_original = x
        current = x

        # Gradient checkpointing setup (unchanged from v1)
        if self.use_gc and self.training:
            from torch.utils.checkpoint import checkpoint as _ckpt
            def _block_call(c, m):
                return self.block(c, mask=m) if m is not None else self.block(c)
        else:
            _ckpt = None

        for pass_idx in range(1, self.max_depth + 1):
            # Loop-index positional signal — gives each pass a distinct identity
            # so the same shared block can behave differently at pass 0 vs pass k.
            if self.loop_idx_enabled:
                # Expand (dim,) -> (1, 1, dim) for broadcast over (B, T, dim)
                k_emb = self.loop_emb[pass_idx - 1].view(1, 1, D)
                block_input = current + k_emb
            else:
                block_input = current

            # Run the shared block
            if _ckpt is not None:
                next_state = _ckpt(_block_call, block_input, mask, use_reentrant=False)
            else:
                next_state = self.block(block_input, mask=mask) if mask is not None else self.block(block_input)

            # Mythos-style LTI injection (opt-in).  When enabled, adds
            #    A*current + B*x_original [+ C*retrieved]
            # to the block output before it becomes the next state.  This
            # stabilizes the recurrence AND re-injects fresh context each pass,
            # preventing the hidden state from drifting away from the input
            # manifold during deep recursion.
            if self.lti_enabled:
                injection = self._lti_injection(current, x_original, retrieved)
                next_state = next_state + injection

            # Active mask: tokens whose chosen depth >= current pass continue
            # to use the NEW state; others stay frozen at their depth.
            active = (depth >= pass_idx).unsqueeze(-1)  # (B, T, 1)
            current = torch.where(active, next_state, current)

        return current, r
