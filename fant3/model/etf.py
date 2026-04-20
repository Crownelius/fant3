"""
Simplex Equiangular Tight Frame (ETF) initialization for router heads.

Implements the "Intermediate Neural Collapse + ETF freezing" trick from
  arxiv:2412.00884 — Leveraging Intermediate Neural Collapse with Simplex ETFs

Background:
  A simplex ETF in d dimensions with k ≤ d+1 classes is a set of unit vectors
  {m_c} with ‖m_c‖ = 1 and  m_c · m_c' = -1/(k-1) for c ≠ c'. This is the
  maximally-separated configuration for k points on the unit sphere in R^d.

  The paper shows that modern classifiers *converge* to this geometry at the
  final layer during training (the "neural collapse" phenomenon). Intermediate
  layers past an "effective depth" also converge toward it. FREEZING the
  router weights to an ETF early in training gives free compression with no
  accuracy loss — the model doesn't have to learn what it was going to learn
  anyway.

FANT 3 use:
  Matryoshka MoE routers project dim→k (k = n_megapools or n_per_megapool).
  After `etf_freeze_after_step` training steps, we freeze the router weights
  to the simplex ETF of the appropriate dimension. This zeroes out ~k·dim
  parameters of router gradient and saves a small amount of compute per step.
"""

from __future__ import annotations
import math
import torch


def simplex_etf(k: int, dim: int, *, dtype=torch.float32, device=None, rotate: bool = True) -> torch.Tensor:
    """
    Return a (k, dim) matrix whose rows form a simplex ETF (if k ≤ dim+1).

    Algorithm (centered-identity, well-conditioned):
        1. M_k = I_k - (1/k) J_k    where J_k is the all-ones k×k matrix.
           Each row v_i = e_i - (1/k) 1 has ‖v_i‖² = (k-1)/k and pairwise
           inner product -1/k, giving cos-angle -1/(k-1) ✓ (simplex ETF).
        2. Normalize rows to unit norm.
        3. The rank is k-1 (all rows sum to zero). Project onto the (k-1)-dim
           principal subspace via SVD and pad with zeros out to `dim`.
        4. Optionally rotate by a random orthogonal matrix (so the ETF isn't
           axis-aligned — slightly better init).

    For k > dim+1 we can't embed a proper simplex in `dim` coordinates;
    fall back to a random orthonormal set of size `dim` (repeats rows).
    """
    if k < 2:
        return torch.zeros((k, dim), dtype=dtype, device=device)

    # Step 1-2: centered identity, unit-normalized rows
    I = torch.eye(k, dtype=torch.float64)
    J = torch.ones(k, k, dtype=torch.float64) / k
    M = I - J                                               # (k, k), rank k-1
    M = M / M.norm(dim=-1, keepdim=True).clamp(min=1e-12)   # unit rows

    if k <= dim + 1:
        # Step 3: SVD to find the (k-1)-dim basis of M
        U, S, Vh = torch.linalg.svd(M, full_matrices=False)
        # M = U diag(S) Vh   — the first k-1 right-singular-vectors are the
        # principal basis (last singular value is ~0 because rows sum to 0).
        rank = min(k - 1, dim)
        principal = M @ Vh[:rank].T                          # (k, rank)
        out = torch.zeros((k, dim), dtype=torch.float64)
        out[:, :rank] = principal
        # Re-normalize after projection (SVD preserves row norms already but
        # round-off makes this safer)
        out = out / out.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        if rotate:
            Q = _random_orthogonal(dim, dtype=torch.float64, device="cpu")
            out = out @ Q
        return out.to(dtype=dtype, device=device)

    # k > dim+1: truly impossible — use random orthonormal rows
    Q = _random_orthogonal(dim, dtype=dtype, device=device)
    # Repeat Q rows cyclically if k > dim (approximate ETF; pairwise cosine
    # is NOT exactly -1/(k-1), but rows are unit and maximally spread in dim).
    idx = torch.arange(k, device=device) % dim
    return Q[idx]


def _random_orthogonal(n: int, dtype=torch.float32, device=None) -> torch.Tensor:
    A = torch.randn(n, n, dtype=dtype, device=device)
    Q, _ = torch.linalg.qr(A)
    return Q


@torch.no_grad()
def freeze_linear_to_etf(linear: torch.nn.Linear) -> None:
    """
    Overwrite `linear.weight` with the simplex ETF of shape (k, dim) where
    k = linear.out_features, dim = linear.in_features, and set
    requires_grad=False. The bias (if any) is zeroed and also frozen.
    """
    k, dim = linear.out_features, linear.in_features
    W = simplex_etf(k, dim, dtype=linear.weight.dtype, device=linear.weight.device)
    linear.weight.data.copy_(W)
    linear.weight.requires_grad_(False)
    if linear.bias is not None:
        linear.bias.data.zero_()
        linear.bias.requires_grad_(False)


def etf_quality(linear: torch.nn.Linear) -> dict:
    """
    Diagnostic: how close is this linear's weight to a perfect ETF?

    Returns the mean and max deviation of off-diagonal cosines from -1/(k-1).
    """
    W = linear.weight.data.detach().float()
    k, dim = W.shape
    W_norm = W / W.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    cos = W_norm @ W_norm.T  # (k, k)
    target_offdiag = -1.0 / max(k - 1, 1)
    mask = ~torch.eye(k, dtype=torch.bool)
    offdiag = cos[mask]
    return {
        "k": k,
        "dim": dim,
        "offdiag_mean": float(offdiag.mean()),
        "offdiag_target": target_offdiag,
        "offdiag_max_dev": float((offdiag - target_offdiag).abs().max()),
        "row_norm_mean": float(W.norm(dim=-1).mean()),
        "row_norm_std":  float(W.norm(dim=-1).std()),
    }
