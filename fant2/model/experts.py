"""
FANT 2 expert classes — the four types of computation a token can route through.

1. FractalSeedExpert  — one of 72 unique fractal seeds. Each stores its own (kron_A_p,kron_A_q)
                        A_gate, A_up, A_down matrices. The full SwiGLU FFN weight is built on
                        demand by 2-level Kronecker product with the per-layer B template
                        (and optionally a global C correction at the layer-output level).

2. ZeroExpert         — outputs all zeros. Routable from anywhere. Acts as "skip this token".
                        From FANT 350M, kept verbatim because it served as a useful safety
                        valve and got ~2% of routing.

3. CopyExpert         — passthrough identity. Routable from anywhere. The other ~3% of
                        FANT 350M routing went here. Kept verbatim.

4. SharedNarrowExpert — always-on dense SwiGLU at narrow hidden=256 (vs 1280 for fractal).
                        Per DeepSeek V3 / Llama 4 / Kimi K2: every layer has one always-on
                        shared expert that handles the "common substrate" of computation.
                        We narrow it to 256 to fit the 60M param budget.

5. DenseSwiGLU        — full-size dense SwiGLU baseline. Used in:
                          - the first n_dense_layers transformer layers (per DeepSeek V3
                            first_k_dense_replace=3 pattern)
                          - ablation comparison runs vs FractalSeedExpert
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .kron3 import kron2, heavy_tail_t_init


# -----------------------------------------------------------------------------
# 1. FractalSeedExpert
# -----------------------------------------------------------------------------

class FractalSeedExpert(nn.Module):
    """
    One of N_FRACTAL=72 fractal-seed experts.

    Stored per-expert: A_gate, A_up, A_down each of shape (kron_A_q, kron_A_p) for
    gate/up and (kron_A_p, kron_A_q) for down. With kron_A_p=40, kron_A_q=8 these are
    each 8*40 = 320 scalars, so the entire expert is 3 * 320 = 960 stored scalars.

    The materialized SwiGLU FFN weights have shape (dim, moe_hidden) = (768, 1280)
    after Kronecker product with the per-layer B template (which is shared across
    all 72 experts in the same layer).

    materialize() builds the full effective weights and is called only for the
    top-k selected experts per forward pass, keeping memory bounded.
    """

    def __init__(
        self,
        kron_A_p: int = 40,
        kron_A_q: int = 8,
        kron_B_p: int = 32,
        kron_B_q: int = 32,
        dim: int = 768,
        moe_hidden: int = 1280,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.kron_A_p = kron_A_p
        self.kron_A_q = kron_A_q
        self.kron_B_p = kron_B_p
        self.kron_B_q = kron_B_q
        self.dim = dim
        self.moe_hidden = moe_hidden

        # Sanity: A ⊗ B effective shape must match (dim, moe_hidden)
        eff_p = kron_A_p * kron_B_p   # rows
        eff_q = kron_A_q * kron_B_q   # cols
        # We design so that eff_p == moe_hidden and eff_q maps to dim via reshape.
        # FANT default: kron_A_p=40, kron_B_p=32 → eff_p=1280=moe_hidden ✓
        #               kron_A_q=8,  kron_B_q=32 → eff_q=256
        # The remaining dim/eff_q = 768/256 = 3 factor is absorbed by reshaping
        # the SwiGLU input via a small (256→768) projection per-layer (not per-expert).
        # See moe.py for the per-layer reshape projection.

        # A_gate, A_up: (kron_A_q, kron_A_p) = (8, 40) — used to build (256, 1280)
        self.A_gate = nn.Parameter(heavy_tail_t_init(kron_A_q, kron_A_p, df=3.0, scale=init_std))
        self.A_up   = nn.Parameter(heavy_tail_t_init(kron_A_q, kron_A_p, df=3.0, scale=init_std))
        # A_down: (kron_A_p, kron_A_q) = (40, 8) — used to build (1280, 256)
        self.A_down = nn.Parameter(heavy_tail_t_init(kron_A_p, kron_A_q, df=3.0, scale=init_std))

    def materialize(
        self,
        B_gate: torch.Tensor,  # (kron_B_q, kron_B_p) e.g. (32, 32)
        B_up:   torch.Tensor,  # (kron_B_q, kron_B_p)
        B_down: torch.Tensor,  # (kron_B_p, kron_B_q)
    ):
        """
        Build the full effective Kronecker SwiGLU weights for this expert.

        Returns:
            W_gate, W_up: (kron_A_q*kron_B_q, kron_A_p*kron_B_p) = (256, 1280)
            W_down:        (kron_A_p*kron_B_p, kron_A_q*kron_B_q) = (1280, 256)

        These small (256, 1280) and (1280, 256) effective shapes are then mapped
        to the full (dim, moe_hidden)=(768, 1280) by the per-layer reshape
        projection in FractalMoELayer.
        """
        W_gate = kron2(self.A_gate, B_gate)
        W_up   = kron2(self.A_up,   B_up)
        W_down = kron2(self.A_down, B_down)
        return W_gate, W_up, W_down

    def forward_with_B(
        self,
        x_proj: torch.Tensor,    # (n_tokens, kron_A_q*kron_B_q) = (n, 256)
        B_gate: torch.Tensor,
        B_up:   torch.Tensor,
        B_down: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the expert computation on already-down-projected input tokens.

        Args:
            x_proj: (n_tokens, 256) input AFTER the per-layer 768→256 down projection
            B_gate, B_up, B_down: per-layer shared B templates

        Returns:
            (n_tokens, 256) output BEFORE the per-layer 256→768 up projection
        """
        W_gate, W_up, W_down = self.materialize(B_gate, B_up, B_down)
        # SwiGLU: F.silu(x @ W_gate.T) * (x @ W_up.T) @ W_down.T
        gate = F.linear(x_proj, W_gate.T)  # (n, 1280)
        up   = F.linear(x_proj, W_up.T)    # (n, 1280)
        h    = F.silu(gate) * up           # (n, 1280)
        out  = F.linear(h, W_down.T)       # (n, 256)
        return out

    def stored_param_count(self) -> int:
        return self.A_gate.numel() + self.A_up.numel() + self.A_down.numel()


# -----------------------------------------------------------------------------
# 2. ZeroExpert
# -----------------------------------------------------------------------------

class ZeroExpert(nn.Module):
    """Outputs zeros. The 'skip' route. Routable from anywhere."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x)

    @staticmethod
    def stored_param_count() -> int:
        return 0


# -----------------------------------------------------------------------------
# 3. CopyExpert
# -----------------------------------------------------------------------------

class CopyExpert(nn.Module):
    """Identity passthrough. The 'skip residual' route. Routable from anywhere."""

    def __init__(self, dim: int = 768):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    @staticmethod
    def stored_param_count() -> int:
        return 0


# -----------------------------------------------------------------------------
# 4. SharedNarrowExpert
# -----------------------------------------------------------------------------

class SharedNarrowExpert(nn.Module):
    """
    Always-on dense SwiGLU at NARROW hidden dim.

    Every MoE layer has exactly one of these. Every token goes through it on
    every forward pass, in addition to the top-k fractal experts.

    From DeepSeek V3 §3.1, Llama 4 Maverick, Kimi K2: the always-on shared
    expert handles the "common substrate" computation that all tokens need
    (e.g. de-tokenization noise removal, basic syntactic features), freeing
    the fractal experts to specialize on domain-specific computation.

    We use hidden=256 instead of the full 1280 to fit the 60M parameter budget.
    Cost per layer: 3 * 768 * 256 = 590k params -> x12 layers = 7.08M.
    """

    def __init__(self, dim: int = 768, hidden: int = 256, init_std: float = 0.02):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up   = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        # Std init
        for m in [self.w_gate, self.w_up, self.w_down]:
            nn.init.normal_(m.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    def stored_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# 5. DenseSwiGLU (baseline / first n_dense_layers)
# -----------------------------------------------------------------------------

class DenseSwiGLU(nn.Module):
    """
    Full-size dense SwiGLU FFN.

    Used in:
    - the first n_dense_layers transformer layers (DeepSeek V3 first_k_dense_replace
      pattern: dense layers near the embedding handle low-level tokenization,
      MoE layers further up handle high-level semantics)
    - ablation comparison runs against FractalSeedExpert
    """

    def __init__(self, dim: int = 768, hidden: int = 1280, init_std: float = 0.02):
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.w_gate = nn.Linear(dim, hidden, bias=False)
        self.w_up   = nn.Linear(dim, hidden, bias=False)
        self.w_down = nn.Linear(hidden, dim, bias=False)
        for m in [self.w_gate, self.w_up, self.w_down]:
            nn.init.normal_(m.weight, std=init_std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

    def stored_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
