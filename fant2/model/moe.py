"""
FractalMoELayer — the per-layer Mixture of Experts module of FANT 2.

This is the module that *combines* everything routing-related:
  - HierarchicalApollonianRouter (8 mega-pools × 9 fractal seeds = 72 experts)
  - 72 FractalSeedExpert instances (each one fractal seed)
  - 1 SharedNarrowExpert (always-on, "common substrate")
  - 1 ZeroExpert + 1 CopyExpert (kept as safety-valve modules; reserved for
    re-introduction in Phase 4 self-refinement)
  - per-layer B template (Daubechies-4 wavelet init, shared across 72 experts)
  - per-layer 768↔256 bridge projections
  - optional global C correction (passed in by the model, shared across layers)

Forward pass:
    x : (B, T, dim)
    --> router decides top-k of 72 experts per token (2-stage: pool then expert)
    --> tokens dispatched to those experts via index_add scatter
    --> each expert materializes its W_gate, W_up, W_down via kron(A_expert, B_layer)
    --> weighted sum of expert outputs (in 256-d effective space)
    --> projected back to dim
    --> add always-on shared expert output
    --> add optional global C correction (residual SwiGLU bottleneck)
    --> return (output, router_output)

The trainer is responsible for:
    - calling router.update_biases(out.megapool_load, out.expert_load)
      after backward() and before optimizer.step() (DeepSeek aux-loss-free)
    - calling router.tikkun_repair() ~every 200 steps
    - calling router.fana_dropout() ~every 1000 steps
    - including router.z_loss(out.expert_logits) in the total loss
    - including router.fep_kl_prior(out.megapool_load, out.expert_load) in the
      total loss with the annealed FEP β coefficient

Param count for default config (per MoE layer, dim=768, 72 fractal experts):
    fractal experts:    72 × 3 × 320 = 69,120  (tiny — the whole point of fractal)
    B template:         3 × 32 × 32  =  3,072
    in_proj:            768 × 256    = 196,608
    out_proj:           256 × 768    = 196,608
    shared expert:      3 × 768 × 256 = 589,824
    router:             ~13,000
    ----------------------------------------
    total per MoE layer ≈ 1.07M
    × 10 MoE layers     ≈ 10.7M  (within the 60M stored budget)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .experts import (
    FractalSeedExpert,
    ZeroExpert,
    CopyExpert,
    SharedNarrowExpert,
)
from .router import HierarchicalApollonianRouter, RouterOutput
from .kron3 import daubechies4_init, orthogonal_init


# -----------------------------------------------------------------------------
# Optional global C correction (3-level Kronecker, layer-output residual form)
# -----------------------------------------------------------------------------

class GlobalCCorrection(nn.Module):
    """
    The "C" tier of the 3-level Kronecker hierarchy A ⊗ B ⊗ C.

    Per the kron3.py docstring, the full 3-level kron(kron(A,B), C) effective
    weight would be 40960 × 10240 — too big to materialize. Instead, FANT 2 v2.0
    applies C as a SEPARATE multiplicative correction at the layer-output level,
    realized as a global SwiGLU bottleneck (dim → kron_C_q → dim).

    Construction:
        - C_gate, C_up : (kron_C_q, dim) ← initialized via orthogonal init
        - C_down       : (dim, kron_C_q)

    A single instance is shared across ALL FractalMoELayers (passed in by FANT2Model).
    The bottleneck (kron_C_q = 40) is intentionally narrow so the correction is
    coarse-grained — that's the entire point of the "global" tier of the hierarchy.

    Param count: 3 × 768 × 40 = 92,160 (one-time, shared across all layers).
    """

    def __init__(self, dim: int = 768, kron_C_q: int = 40, kron_C_p: int = 32, gain: float = 1.0):
        super().__init__()
        self.dim = dim
        self.kron_C_q = kron_C_q
        self.kron_C_p = kron_C_p
        # Bottleneck SwiGLU: dim → kron_C_q → dim
        self.C_gate = nn.Parameter(orthogonal_init(kron_C_q, dim, gain=gain))
        self.C_up   = nn.Parameter(orthogonal_init(kron_C_q, dim, gain=gain))
        self.C_down = nn.Parameter(orthogonal_init(dim, kron_C_q, gain=gain))
        # Learnable scale (small init so the C correction starts as a near-identity residual)
        self.alpha = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        gate = F.linear(x, self.C_gate)        # (..., kron_C_q)
        up   = F.linear(x, self.C_up)          # (..., kron_C_q)
        h    = F.silu(gate) * up               # (..., kron_C_q)
        out  = F.linear(h, self.C_down)        # (..., dim)
        return self.alpha * out


# -----------------------------------------------------------------------------
# FractalMoELayer
# -----------------------------------------------------------------------------

class FractalMoELayer(nn.Module):
    """
    One MoE layer of FANT 2: router + 72 fractal experts + shared expert + C correction.

    See module docstring for the full data-flow.
    """

    def __init__(
        self,
        dim: int = 768,
        n_megapools: int = 8,
        n_per_megapool: int = 9,
        top_k: int = 4,
        moe_hidden: int = 1280,
        shared_expert_hidden: int = 256,
        kron_A_p: int = 40,
        kron_A_q: int = 8,
        kron_B_p: int = 32,
        kron_B_q: int = 32,
        init_std: float = 0.02,
        c_global: Optional[GlobalCCorrection] = None,
        # Router knobs
        router_gamma: float = 1e-3,
        router_ema_decay: float = 0.99,
        router_tikkun_threshold: float = 0.30,
        router_cayley_every_n_steps: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.n_megapools = n_megapools
        self.n_per_megapool = n_per_megapool
        self.n_fractal = n_megapools * n_per_megapool          # 72
        self.top_k = top_k
        self.moe_hidden = moe_hidden
        self.kron_A_p = kron_A_p
        self.kron_A_q = kron_A_q
        self.kron_B_p = kron_B_p
        self.kron_B_q = kron_B_q

        # Effective expert input dim after the in_proj down-projection.
        # = kron_A_q * kron_B_q (e.g. 8 * 32 = 256)
        self.expert_in_dim = kron_A_q * kron_B_q

        # ----- Per-layer SHARED B templates (Daubechies-4 wavelet init) -----
        # Shared across all 72 fractal experts in THIS layer.
        # Different layers have different B templates (mid-grain of the 3-level kron).
        # B_gate, B_up : (kron_B_q, kron_B_p) = (32, 32)
        # B_down       : (kron_B_p, kron_B_q) = (32, 32)
        self.B_gate = nn.Parameter(daubechies4_init(kron_B_q, kron_B_p))
        self.B_up   = nn.Parameter(daubechies4_init(kron_B_q, kron_B_p))
        self.B_down = nn.Parameter(daubechies4_init(kron_B_p, kron_B_q))

        # ----- Per-layer 768↔256 bridge projections -----
        # in_proj  : dim (768) → expert_in_dim (256)
        # out_proj : expert_in_dim (256) → dim (768)
        # These absorb the dim/kron_in mismatch; see kron3.py for the math.
        self.in_proj  = nn.Linear(dim, self.expert_in_dim, bias=False)
        self.out_proj = nn.Linear(self.expert_in_dim, dim,  bias=False)
        nn.init.normal_(self.in_proj.weight,  std=init_std)
        nn.init.normal_(self.out_proj.weight, std=init_std)

        # ----- 72 FractalSeedExpert instances -----
        self.fractal_experts = nn.ModuleList([
            FractalSeedExpert(
                kron_A_p=kron_A_p,
                kron_A_q=kron_A_q,
                kron_B_p=kron_B_p,
                kron_B_q=kron_B_q,
                dim=dim,
                moe_hidden=moe_hidden,
                init_std=init_std,
            )
            for _ in range(self.n_fractal)
        ])

        # ----- Always-on shared (narrow) expert -----
        self.shared_expert = SharedNarrowExpert(
            dim=dim,
            hidden=shared_expert_hidden,
            init_std=init_std,
        )

        # ----- ZeroExpert / CopyExpert: kept as side modules -----
        # Not in the main routing path for v2.0 — reserved for Phase 4
        # self-refinement, where the trainer can reintroduce them as residual
        # gates ("safety valves") if needed. They contribute zero to forward().
        self.zero_expert = ZeroExpert(dim=dim)
        self.copy_expert = CopyExpert(dim=dim)

        # ----- Hierarchical Apollonian router -----
        self.router = HierarchicalApollonianRouter(
            dim=dim,
            n_megapools=n_megapools,
            n_per_pool=n_per_megapool,
            top_k=top_k,
            gamma=router_gamma,
            ema_decay=router_ema_decay,
            tikkun_threshold=router_tikkun_threshold,
            cayley_every_n_steps=router_cayley_every_n_steps,
        )

        # ----- Optional global C correction (shared module passed in) -----
        # If None, the layer skips the C tier; the model can still train.
        self.c_global = c_global  # Module reference, NOT a child (so it's not double-counted)

        # NOTE: we deliberately do NOT register c_global as a submodule, to keep
        # parameter sharing across layers clean. The model is responsible for
        # adding c_global to its own parameter list exactly once.

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, RouterOutput]:
        """
        Args:
            x: (B, T, dim) input activations

        Returns:
            (output, router_out) where
                output      : (B, T, dim)
                router_out  : RouterOutput (see router.py)
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        # ===== 1. Always-on shared expert (operates in dim directly) =====
        shared_out = self.shared_expert(x)  # (B, T, dim)

        # ===== 2. Run the hierarchical router =====
        router_out = self.router(x)
        # router_out.expert_idx_global : (N, top_k) long, in [0, 72)
        # router_out.expert_weights    : (N, top_k) float, sums to 1 per token

        # ===== 3. Project tokens into expert effective space (256) =====
        x_proj = self.in_proj(x_flat)  # (N, expert_in_dim=256)

        # ===== 4. Dispatch to fractal experts via index_add scatter =====
        out_proj_acc = torch.zeros(
            N, self.expert_in_dim, device=x.device, dtype=x.dtype
        )

        # Standard MoE dispatch: for each expert, gather → run → scatter
        # The .any() guard handles the (common) case of an expert receiving 0 tokens
        for e in range(self.n_fractal):
            # Find all (token, slot) pairs that picked expert e
            expert_mask = router_out.expert_idx_global == e   # (N, top_k) bool
            if not expert_mask.any():
                continue
            token_idx, slot_idx = expert_mask.nonzero(as_tuple=True)  # (n_e,), (n_e,)

            expert_input = x_proj[token_idx]                         # (n_e, 256)
            slot_weights = router_out.expert_weights[token_idx, slot_idx]  # (n_e,)

            expert_out = self.fractal_experts[e].forward_with_B(
                expert_input,
                self.B_gate,
                self.B_up,
                self.B_down,
            )  # (n_e, 256)

            weighted = expert_out * slot_weights.unsqueeze(-1).to(expert_out.dtype)
            out_proj_acc.index_add_(0, token_idx, weighted)

        # ===== 5. Project back to model dim =====
        moe_out = self.out_proj(out_proj_acc).reshape(B, T, D)  # (B, T, dim)

        # ===== 6. Combine fractal MoE + shared expert =====
        out = moe_out + shared_out

        # ===== 7. Optional global C correction (3-level Kron, layer-output form) =====
        if self.c_global is not None:
            out = out + self.c_global(out)

        return out, router_out

    # -------------------------------------------------------------------------
    # Diagnostic / introspection helpers
    # -------------------------------------------------------------------------

    def materialize_all_experts(self) -> torch.Tensor:
        """
        Materialize the W_gate of every fractal expert in this layer.
        Used by telemetry to compute the per-layer effective rank, condition
        number, Martin-Mahoney spectral α, etc.

        Returns: (n_fractal, kron_A_q*kron_B_q, kron_A_p*kron_B_p) = (72, 256, 1280)
        """
        with torch.no_grad():
            stack = []
            for e in self.fractal_experts:
                W_gate, _, _ = e.materialize(self.B_gate, self.B_up, self.B_down)
                stack.append(W_gate)
            return torch.stack(stack, dim=0)

    def stored_param_count(self) -> int:
        """
        Sum of stored parameters in this layer (excluding the c_global which is
        shared and counted at the model level, and excluding the router buffers).
        """
        n = 0
        # B template
        n += self.B_gate.numel() + self.B_up.numel() + self.B_down.numel()
        # Bridge projections
        n += self.in_proj.weight.numel() + self.out_proj.weight.numel()
        # Fractal experts
        for e in self.fractal_experts:
            n += e.stored_param_count()
        # Shared expert
        n += self.shared_expert.stored_param_count()
        # Router has only buffers (no Parameters), so 0 stored params
        return n
