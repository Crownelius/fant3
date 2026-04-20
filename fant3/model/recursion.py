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

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        curvatures: Optional[torch.Tensor] = None,
    ) -> tuple:
        """
        x:          (B, T, dim)
        mask:       optional attention mask passed to the shared block
        curvatures: optional (B, T) per-token Apollonian curvature. If given
                    AND cfg.mor_depth_bias == "alpha", we bias the depth
                    router toward shallow for high-curvature (instance) tokens.
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
            # Shift high-curv tokens down one depth level (bias towards 0)
            depth_idx = torch.where(high_curv, (depth_idx - 1).clamp(min=0), depth_idx)

        # Apply shared block, grouping tokens by their chosen depth
        #
        # Simple implementation (v1): apply the block max_depth times on the
        # FULL batch, but on each pass only WRITE BACK the result for tokens
        # whose depth ≥ pass_idx. Tokens with shallow depth are "frozen" after
        # their last pass.
        #
        # This wastes some compute (we compute the block for all tokens on each
        # pass, even ones that don't need it), but is simple and GPU-friendly.
        # A production version would gather/scatter by depth group.

        depth = (depth_idx + 1).reshape(B, T)  # (B, T), values in {1..max_depth}
        current = x

        # Each pass through the shared block can be gradient-checkpointed
        # independently — this is where MoR's activation memory really costs.
        if self.use_gc and self.training:
            from torch.utils.checkpoint import checkpoint as _ckpt
            def _block_call(c, m):
                return self.block(c, mask=m) if m is not None else self.block(c)
        else:
            _ckpt = None

        for pass_idx in range(1, self.max_depth + 1):
            if _ckpt is not None:
                next_state = _ckpt(_block_call, current, mask, use_reentrant=False)
            else:
                next_state = self.block(current, mask=mask) if mask is not None else self.block(current)
            # For tokens with depth >= pass_idx, we continue to use `next_state`.
            # For tokens with depth < pass_idx, we keep `current` (they're done).
            active = (depth >= pass_idx).unsqueeze(-1)  # (B, T, 1)
            current = torch.where(active, next_state, current)

        return current, r
