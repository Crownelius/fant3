"""
CerebellumModule — echo-state reservoir + Purkinje linear readout.

Design = (FANT 2 neurology agent §6 + brain-inspired-fractal agent §3):

The biological cerebellum is the brain's high-bandwidth motor / error-correction
coprocessor. It contains ~80% of the brain's neurons in ~10% of its volume, all
arranged into a stereotyped 3-layer feedforward circuit:

    Mossy fibers          (input)         ~10⁹ axons from cortex/brainstem
        ↓
    Granule cells         (10× expansion)  ~10¹¹ tiny neurons, ~4 dendrites each
        ↓
    Parallel fibers        (long, thin)    each granule's axon spans many Purkinjes
        ↓
    Purkinje cells        (output)        ~10⁷, large dendritic trees, single axon

Key properties:
  1. MASSIVE FAN-OUT (10× expansion at the granule layer):
       Mossy → granule projection turns N inputs into ~10N high-dimensional
       sparse codes. Each granule sees only ~4 mossy fibers, so the code is
       extremely sparse (the "expansion encoding" of Marr-Albus 1969).

  2. EDGE-OF-CHAOS RECURRENCE:
       Parallel fibers and the molecular/nuclear loops give the granule layer
       an effective recurrent dynamic. Bertschinger-Natschläger (2004) showed
       reservoirs at the edge of chaos (spectral radius ≈ 1) maximize their
       information-processing capacity. We initialize at spectral radius 0.95
       — slightly stable but right at the boundary.

  3. PURKINJE LINEAR READOUT:
       The Purkinje cell is a *linear* combiner of parallel fiber inputs (with
       inhibitory tuning via the climbing fiber, but the basic operation is
       linear). This is the same as the "linear readout" in echo-state networks
       (Jaeger 2001): the reservoir does the nonlinear lifting, and only the
       readout is learned.

In FANT 2 we use the cerebellum module as an EVERY-LAYER side path that lifts
the model's hidden state into a high-D sparse code, runs a few steps of
edge-of-chaos recurrence, and projects back. This gives the model:
  - Multi-scale temporal processing (the recurrence does the multi-scale)
  - Heavy-tailed statistics (sparse + edge-of-chaos = power-law spectrum)
  - A cheap-to-compute "second cortex" that handles the iterative refinement
    that pure feedforward layers cannot do (Coconut, MoR, Titans).

Architecture:
    x  : (B, T, in_dim)
    --> mossy_proj    : (in_dim → expand_dim)        [LEARNED]
    --> tanh
    --> for k in range(n_layers):                    [FROZEN reservoir]
            h ← (1-leak) * h + leak * tanh(W_res @ h + mossy)
    --> purkinje      : (expand_dim → out_dim)       [LEARNED]
    --> RMSNorm

The reservoir matrix W_res is FROZEN at initialization (random sparse with
spectral radius rescaled to 0.95). It is NOT a Parameter, so it does not
contribute to the optimizer state or the gradient computation.

Param count for default config (in=out=768, expand=7680, sparsity=0.001):
    mossy_proj  :   768 × 7680   = 5.9 M (LEARNED)
    purkinje    :   7680 × 768   = 5.9 M (LEARNED)
    leak_rate   :   1            (LEARNED, scalar)
    out_norm    :   768          (LEARNED, RMSNorm weight)
    reservoir   :   ~59K nonzero (FROZEN sparse buffer, ~0.7 MB)
    total LEARNED ≈ 11.8M per cerebellum

Note: 11.8M is too much to put one in every layer. The model uses ONE shared
cerebellum applied to the residual stream after the dense (early) layers.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .norm import RMSNorm


# -----------------------------------------------------------------------------
# Sparse reservoir helper
# -----------------------------------------------------------------------------

def _build_sparse_reservoir(
    dim: int,
    sparsity: float,
    target_radius: float,
    n_power_iters: int = 80,
    init_scale: float = 0.5,
    device=None,
) -> tuple:
    """
    Build a sparse random reservoir with rescaled spectral radius.

    Returns (rows, cols, vals) representing W_res ∈ R^(dim, dim).
    The matrix W_res is implicitly:
        W_res[rows[k], cols[k]] = vals[k]   for each k

    Spectral radius is estimated by power iteration on the sparse representation
    and the values are rescaled so that the largest eigenvalue magnitude is
    `target_radius`.

    Args:
        dim:           reservoir size
        sparsity:      fraction of (i,j) entries that are non-zero
        target_radius: desired spectral radius (≤ 1 for stability, ~0.95 for edge of chaos)
        n_power_iters: number of power iterations to estimate the largest eigenvalue
        init_scale:    initial std of the non-zero values (rescaled afterwards)

    Returns:
        rows, cols, vals — three 1D tensors of equal length
    """
    # Minimum connections per row: a sparse random matrix needs at least
    # ~6-10 nonzeros per row to have a non-degenerate spectrum at small dims.
    # Without this floor, low-dim presets (e.g. tiny @ 512) get a reservoir
    # whose power-iteration spectral radius collapses to ~0.
    MIN_CONN_PER_ROW = 6
    n_connections = max(MIN_CONN_PER_ROW * dim, int(dim * dim * sparsity))

    g = torch.Generator(device="cpu")
    g.manual_seed(42)  # deterministic reservoir for reproducibility

    rows = torch.randint(0, dim, (n_connections,), generator=g)
    cols = torch.randint(0, dim, (n_connections,), generator=g)
    vals = torch.randn(n_connections, generator=g) * init_scale

    # ---- Power iteration to estimate spectral radius ----
    def sparse_matvec(v: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(v)
        out.index_add_(0, rows, v[cols] * vals)
        return out

    v = torch.randn(dim, generator=g)
    v = v / (v.norm() + 1e-12)
    for _ in range(n_power_iters):
        v_new = sparse_matvec(v)
        norm = v_new.norm() + 1e-12
        v = v_new / norm
    estimated_radius = sparse_matvec(v).norm().item()

    if estimated_radius > 1e-6:
        vals = vals * (target_radius / estimated_radius)

    if device is not None:
        rows = rows.to(device)
        cols = cols.to(device)
        vals = vals.to(device)
    return rows, cols, vals


# -----------------------------------------------------------------------------
# CerebellumModule
# -----------------------------------------------------------------------------

class CerebellumModule(nn.Module):
    """
    Echo-state reservoir + Purkinje linear readout, applied per token.

    See module docstring for the design rationale and parameter accounting.
    """

    def __init__(
        self,
        in_dim: int = 768,
        expand_dim: int = 7680,
        out_dim: int = 768,
        n_layers: int = 4,
        spectral_radius: float = 0.95,
        sparsity: float = 0.001,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.expand_dim = expand_dim
        self.out_dim = out_dim
        self.n_layers = n_layers

        # ----- LEARNED: mossy fiber projection (in_dim → expand_dim) -----
        self.mossy_proj = nn.Linear(in_dim, expand_dim, bias=False)
        nn.init.normal_(self.mossy_proj.weight, std=init_std)

        # ----- LEARNED: leak rate (parameterized via sigmoid for [0,1]) -----
        # Initialize so sigmoid ≈ 0.5 (moderate leak)
        self.leak_rate_raw = nn.Parameter(torch.tensor(0.0))

        # ----- FROZEN: sparse reservoir buffers -----
        rows, cols, vals = _build_sparse_reservoir(
            dim=expand_dim,
            sparsity=sparsity,
            target_radius=spectral_radius,
        )
        self.register_buffer("res_rows", rows)
        self.register_buffer("res_cols", cols)
        self.register_buffer("res_vals", vals)

        # ----- LEARNED: Purkinje linear readout (expand_dim → out_dim) -----
        self.purkinje = nn.Linear(expand_dim, out_dim, bias=False)
        nn.init.normal_(self.purkinje.weight, std=init_std)

        # ----- LEARNED: output normalization -----
        self.out_norm = RMSNorm(out_dim)

    # -------------------------------------------------------------------------
    # Sparse reservoir apply
    # -------------------------------------------------------------------------

    def _reservoir_apply(self, h: torch.Tensor) -> torch.Tensor:
        """
        Apply the sparse frozen reservoir matrix to a (..., expand_dim) tensor.

        Mathematically: out[..., r] = sum over connections (r, c) of h[..., c] * w
        Implemented via gather + scatter-add along the last dim.
        """
        # Gather: get h[..., cols] of shape (..., n_conn)
        gathered = h.index_select(dim=-1, index=self.res_cols)  # (..., n_conn)
        # Multiply by vals (broadcasts over leading dims)
        contributions = gathered * self.res_vals
        # Scatter-add into the output along the last dim, indexed by rows
        out = torch.zeros_like(h)
        # index_add_ on the last dim — tensor must be contiguous
        out.index_add_(-1, self.res_rows, contributions)
        return out

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_dim) input activations

        Returns:
            (B, T, out_dim) cerebellum output (to be added as a residual)
        """
        # ===== 1. Mossy fiber projection + initial nonlinearity =====
        mossy = self.mossy_proj(x)              # (B, T, expand_dim)
        h = torch.tanh(mossy)

        # ===== 2. Edge-of-chaos reservoir iteration =====
        leak = torch.sigmoid(self.leak_rate_raw)
        for _ in range(self.n_layers):
            res_out = self._reservoir_apply(h)              # (B, T, expand_dim)
            new_h = torch.tanh(res_out + mossy)
            h = (1.0 - leak) * h + leak * new_h

        # ===== 3. Purkinje linear readout =====
        out = self.purkinje(h)                  # (B, T, out_dim)

        # ===== 4. Output norm =====
        out = self.out_norm(out)
        return out

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def stored_param_count(self) -> int:
        """Stored learnable parameters only (does NOT count the frozen reservoir)."""
        return sum(p.numel() for p in self.parameters())

    def reservoir_size(self) -> int:
        """Number of non-zero entries in the frozen reservoir."""
        return self.res_vals.numel()

    @torch.no_grad()
    def estimate_spectral_radius(self, n_iters: int = 50) -> float:
        """Re-estimate the reservoir spectral radius (for telemetry)."""
        v = torch.randn(self.expand_dim, device=self.res_vals.device)
        v = v / (v.norm() + 1e-12)
        for _ in range(n_iters):
            out = torch.zeros_like(v)
            out.index_add_(0, self.res_rows, v[self.res_cols] * self.res_vals)
            norm = out.norm() + 1e-12
            v = out / norm
        out = torch.zeros_like(v)
        out.index_add_(0, self.res_rows, v[self.res_cols] * self.res_vals)
        return float(out.norm().item())
