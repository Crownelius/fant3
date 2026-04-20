"""
3-level Kronecker hierarchy: W = kron(kron(A_expert, B_layer), C_global)

This is the FANT 2 generalization of FANT 350M's 2-level kron(A, B).
The hierarchy gives three independent grain levels:

    A_expert  : (kron_A_p, kron_A_q)  -- per-expert,  fine grain
    B_layer   : (kron_B_p, kron_B_q)  -- per-layer,   mid grain  (shared across experts in a layer)
    C_global  : (kron_C_p, kron_C_q)  -- global,      coarse grain (shared across all layers and experts)

The effective weight matrix has shape:
    W : (kron_A_p * kron_B_p * kron_C_p,  kron_A_q * kron_B_q * kron_C_q)

For FANT 2 default:
    A : (40, 8)
    B : (32, 32)
    C : (32, 40)
    -> W : (40*32*32, 8*32*40) = (40960, 10240)  # WAY too big

The trick: we re-use the same outer Kronecker product but the OUTPUT shape we
need is (dim, moe_hidden) = (768, 1280). So we use a 2-level Kron of A⊗B then
reshape, NOT the full 3-level. For FANT 2 v2.0 we use:

    A : (40, 8)   --> kron(A, B) shape = (40*32, 8*32) = (1280, 256)  (W_down for SwiGLU)
    B : (32, 32)
    -> kron(A, B) is the per-expert effective weight after merging A and B.

    W_gate, W_up : kron(A_gate (8, 40), B (32, 32)) -> (256, 1280)  reshape -> (768, 1280)
    W_down       : kron(A_down (40, 8), B (32, 32)) -> (1280, 256)  reshape -> (1280, 768)

Then C_global is applied as a SEPARATE multiplicative correction at the LAYER
output level, not folded into the per-expert W. That keeps the per-expert
materialization cost manageable while preserving the 3-level structure.

This file provides:
    - kron2(A, B)              : standard 2D Kronecker (torch.kron is fine)
    - kron3(A, B, C)           : 3-level Kronecker (used at materialization time)
    - heavy_tail_t_init(...)   : Student-t init for A (Martin-Mahoney)
    - daubechies4_init(...)    : wavelet init for B (Mallat scattering)
    - orthogonal_init(...)     : orthogonal init for C (Pennington dynamical isometry)
    - validate_kron3_shapes(...) : sanity check at config-load time
"""

import math

import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Kronecker products
# -----------------------------------------------------------------------------

def kron2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Standard 2D Kronecker product. Wraps torch.kron for consistency."""
    return torch.kron(A, B)


def kron3(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """
    3-level Kronecker product: kron(kron(A, B), C).

    Note that torch.kron is associative, so kron(kron(A,B),C) == kron(A,kron(B,C)).
    We compute left-associative for cache friendliness.
    """
    return torch.kron(torch.kron(A, B), C)


# -----------------------------------------------------------------------------
# Initialization recipes (from hyperdim agent #7 + spec §5.1)
# -----------------------------------------------------------------------------

def heavy_tail_t_init(
    *shape: int,
    df: float = 3.0,
    scale: float = 0.02,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Student-t initialization with heavy tails (Martin-Mahoney 2018).

    Args:
        *shape: tensor shape
        df: degrees of freedom (alpha ≈ 3 → heavy-tailed self-regularization regime)
        scale: standard deviation scale
    """
    # Sample from t-distribution: T = Z / sqrt(chi2/df)
    z = torch.randn(*shape, device=device, dtype=dtype)
    chi2 = torch.distributions.Chi2(df).sample(torch.Size(shape)).to(device or 'cpu')
    if dtype is not None:
        chi2 = chi2.to(dtype)
    t = z / torch.sqrt(chi2 / df)
    return t * scale


def daubechies4_init(
    p: int,
    q: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Discrete Daubechies-4 wavelet initialization for the B-layer template.

    Builds a (p, q) matrix whose rows are tiled, scaled Daubechies-4 wavelet
    coefficients. This places the B-layer init on the dyadic wavelet basis,
    which is the canonical sparsity-promoting init for multi-scale signals
    (Mallat scattering tradition).
    """
    # Daubechies-4 wavelet coefficients (filter length 4)
    # h0, h1, h2, h3 are the standard D4 lowpass coefficients
    sqrt3 = math.sqrt(3.0)
    denom = 4.0 * math.sqrt(2.0)
    h = torch.tensor(
        [
            (1 + sqrt3) / denom,
            (3 + sqrt3) / denom,
            (3 - sqrt3) / denom,
            (1 - sqrt3) / denom,
        ],
        device=device,
        dtype=dtype or torch.float32,
    )

    M = torch.zeros(p, q, device=device, dtype=dtype or torch.float32)
    for i in range(p):
        for j in range(min(4, q)):
            M[i, (i + j) % q] = h[j]
    # Add small noise so different rows are not perfectly correlated
    M = M + 0.01 * torch.randn_like(M)
    return M


def orthogonal_init(
    p: int,
    q: int,
    gain: float = 1.0,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Orthogonal initialization for the C-global template (Pennington-Schoenholz-Ganguli
    dynamical isometry). Ensures the spectrum is concentrated near 1.
    """
    M = torch.empty(p, q, device=device, dtype=dtype or torch.float32)
    torch.nn.init.orthogonal_(M, gain=gain)
    return M


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------

def validate_kron3_shapes(
    kron_A_p: int, kron_A_q: int,
    kron_B_p: int, kron_B_q: int,
    kron_C_p: int, kron_C_q: int,
    target_p: int, target_q: int,
) -> None:
    """
    Sanity-check that the 3-level Kronecker shapes multiply to the target.

    NOTE: For FANT 2 v2.0 we use a 2-level kron(A, B) for the per-expert
    weight, with C applied as a separate layer-output correction. So the
    relevant constraint is:
        kron_A_p * kron_B_p == target_p  AND
        kron_A_q * kron_B_q == target_q
    """
    eff_p = kron_A_p * kron_B_p
    eff_q = kron_A_q * kron_B_q
    if eff_p != target_p:
        raise ValueError(
            f"kron_A_p ({kron_A_p}) * kron_B_p ({kron_B_p}) = {eff_p} != target_p ({target_p})"
        )
    if eff_q != target_q:
        raise ValueError(
            f"kron_A_q ({kron_A_q}) * kron_B_q ({kron_B_q}) = {eff_q} != target_q ({target_q})"
        )
    # C global must be applicable as a (kron_C_p, kron_C_q) correction afterward
    if kron_C_p <= 0 or kron_C_q <= 0:
        raise ValueError(f"kron_C dimensions must be positive: ({kron_C_p}, {kron_C_q})")
