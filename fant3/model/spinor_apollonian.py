"""
SpinorApollonianMemory — drop-in replacement for ApollonianMemory that
classifies memory into α/β packs via a 2D tangency spinor rather than a
scalar curvature threshold.

THEORETICAL MOTIVATION
======================
The original ApollonianMemory uses L2-norm as a proxy for Apollonian
curvature and then applies a fixed threshold to split items into α (high
curvature, instance) vs β (low curvature, schema) packs.  The documented
starvation bug (all norms cluster in [0.9916, 1.0127] → everything classified
as α) shows that a 1D scalar is too information-poor to separate two
structurally distinct populations.

Kocik (arXiv:2001.05866, "Spinors and Descartes") shows that every tangent
pair of circles in an Apollonian packing corresponds to a Minkowski spinor
s = (s₀, s₁) ∈ ℝ² satisfying the Clifford quadratic form

    Q_Cl(s) = s₀² − s₁²        (signature (1,−1), i.e. Cl(2,1) subalgebra)

and that the sign of s₁ determines which of the two complementary sub-
packings (left-chiral vs right-chiral) the circle belongs to.  The Descartes
theorem

    (Σ bᵢ)² = 2 Σ bᵢ²          (bᵢ = Apollonian curvatures)

is exactly the (1,3) Minkowski quadratic form evaluated on the 4-vector of
curvatures, and SO(3,1) (the Apollonian group) acts on this vector.

We map each hidden-state vector h ∈ ℝ^dim to a 2D spinor via a small learned
linear projection:

    s = proj_spinor(h)   ∈ ℝ²

then use:

    chirality = sign(s[1])     →  α if > 0,  β if ≤ 0

and monitor the Clifford norm ‖s‖²_Cl = s₀² + s₁² as the "curvature"
observable (Euclidean norm, not pseudo-Riemannian, for dashboard stability).

Retrieval combines cosine similarity on embeddings with the Clifford bilinear
form on spinors:

    score(q, m) = 0.7 · cos(q_emb, m_emb)  +  0.3 · bilinear_Cl(q_s, m_s)

where bilinear_Cl(a, b) = a[0]·b[0] − a[1]·b[1]  (Minkowski (1,−1)).

References
----------
- Kocik, J. (2001). "Clifford algebras and Euclid's parametrisation of
  Pythagorean triples." arXiv:2001.05866 — tangency spinors §3.
- Boyd, Lagarias, Mallows, Wilks (2003) — Apollonian circle packings.
- Project memory: research_spinor_apollonian_2026_04_16.md
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Helper: Clifford bilinear kernel (Minkowski signature (1, −1))
# ---------------------------------------------------------------------------

def clifford_bilinear(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Evaluate the (1,−1) Clifford bilinear form between two sets of spinors.

    Args:
        a: (..., 2)  spinors
        b: (..., 2)  spinors  (broadcast-compatible with a)

    Returns:
        (...,)  scalar  a[...,0]*b[...,0] − a[...,1]*b[...,1]
    """
    return a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]


def clifford_norm(s: torch.Tensor) -> torch.Tensor:
    """
    Euclidean‖s‖² (not pseudo-Riemannian) used as the dashboard 'curvature'.

    Args:
        s: (..., 2) spinors

    Returns:
        (...,) curvature scalars  s[0]² + s[1]²
    """
    return (s ** 2).sum(dim=-1)


# ---------------------------------------------------------------------------
# SpinorApollonianMemory
# ---------------------------------------------------------------------------

class SpinorApollonianMemory(nn.Module):
    """
    Dual α/β memory pack whose pack assignment is governed by the chirality
    of a learned 2D tangency spinor rather than a scalar curvature threshold.

    API-compatible with fant2.model.apollonian.ApollonianMemory.

    Parameters
    ----------
    dim        : int   — hidden-state dimension
    alpha_cap  : int   — FIFO capacity of the α (positive chirality) pack
    beta_cap   : int   — FIFO capacity of the β (negative chirality) pack
    """

    def __init__(
        self,
        dim: int = 768,
        alpha_cap: int = 5000,
        beta_cap: int = 5000,
    ) -> None:
        super().__init__()

        self.dim = dim
        self.alpha_cap = alpha_cap
        self.beta_cap = beta_cap

        # ── Spinor projection (the only trainable param in this module) ──────
        # Projects PRE-RMSNorm hidden state h ∈ ℝ^dim → spinor s ∈ ℝ²
        # Bias=False keeps the spinor origin at zero when h=0.
        self.proj_spinor = nn.Linear(dim, 2, bias=False)
        # Initialise to a small scale so the chirality split starts near 50/50.
        nn.init.normal_(self.proj_spinor.weight, std=0.01)

        # ── α pack buffers ───────────────────────────────────────────────────
        self.register_buffer("alpha_emb",       torch.zeros(alpha_cap, dim))
        self.register_buffer("alpha_spinor",    torch.zeros(alpha_cap, 2))
        self.register_buffer("alpha_curvature", torch.zeros(alpha_cap))
        self.register_buffer("alpha_age",       torch.zeros(alpha_cap, dtype=torch.long))
        self.register_buffer("alpha_count",     torch.tensor(0, dtype=torch.long))
        self.register_buffer("alpha_head",      torch.tensor(0, dtype=torch.long))

        # ── β pack buffers ───────────────────────────────────────────────────
        self.register_buffer("beta_emb",        torch.zeros(beta_cap, dim))
        self.register_buffer("beta_spinor",     torch.zeros(beta_cap, 2))
        self.register_buffer("beta_curvature",  torch.zeros(beta_cap))
        self.register_buffer("beta_age",        torch.zeros(beta_cap, dtype=torch.long))
        self.register_buffer("beta_count",      torch.tensor(0, dtype=torch.long))
        self.register_buffer("beta_head",       torch.tensor(0, dtype=torch.long))

        # ── Global step (for age tracking / SleepGate compatibility) ─────────
        self.register_buffer("global_step",     torch.tensor(0, dtype=torch.long))

    # -----------------------------------------------------------------------
    # Internal utilities
    # -----------------------------------------------------------------------

    def _compute_spinors(
        self,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project a batch of hidden states to spinors.

        Args:
            hidden: (N, dim) or (B, T, dim)

        Returns:
            spinors: same leading dims ... 2
        """
        # We want gradients through proj_spinor during training, but callers
        # that are inside no_grad contexts (store) will detach afterwards.
        return self.proj_spinor(hidden)  # (..., 2)

    @staticmethod
    @torch.no_grad()
    def _fifo_write(
        emb_buf: torch.Tensor,
        sp_buf: torch.Tensor,
        cur_buf: torch.Tensor,
        age_buf: torch.Tensor,
        count_buf: torch.Tensor,
        head_buf: torch.Tensor,
        cap: int,
        new_emb: torch.Tensor,
        new_sp: torch.Tensor,
        new_cur: torch.Tensor,
        step: int,
    ) -> int:
        """In-place FIFO write.  Returns number of items written."""
        n_new = new_emb.shape[0]
        if n_new == 0:
            return 0
        head = int(head_buf.item())
        count = int(count_buf.item())

        for i in range(n_new):
            slot = (head + i) % cap
            emb_buf[slot] = new_emb[i]
            sp_buf[slot]  = new_sp[i]
            cur_buf[slot] = new_cur[i]
            age_buf[slot] = step

        new_head = (head + n_new) % cap
        new_count = min(count + n_new, cap)
        head_buf.fill_(new_head)
        count_buf.fill_(new_count)
        return n_new

    # -----------------------------------------------------------------------
    # Store
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def store(
        self,
        embeddings: torch.Tensor,
        hidden_preRMSnorm: Optional[torch.Tensor] = None,
    ) -> Dict[str, int]:
        """
        Classify and store a batch of embeddings.

        Parameters
        ----------
        embeddings       : [B, T, dim] or [N, dim]
            The vectors stored as retrieval keys in the memory banks.
        hidden_preRMSnorm: same shape, optional.
            Used to compute the tangency spinor.  If None, the embeddings
            themselves are used as a fallback (still better than L2-norm).

        Returns
        -------
        {'alpha_stored': N_a, 'beta_stored': N_b}
        """
        # ── Flatten to (N, dim) ──────────────────────────────────────────────
        emb = embeddings.detach()
        if emb.dim() == 3:
            B, T, D = emb.shape
            emb = emb.reshape(B * T, D)
        elif emb.dim() == 1:
            emb = emb.unsqueeze(0)

        N, D = emb.shape
        assert D == self.dim, f"embedding dim {D} != memory dim {self.dim}"

        # ── Compute spinors ──────────────────────────────────────────────────
        if hidden_preRMSnorm is not None:
            hid = hidden_preRMSnorm.detach()
            if hid.dim() == 3:
                hid = hid.reshape(-1, self.dim)
        else:
            hid = emb  # fallback

        # Temporarily enable grad context for proj_spinor (it's a Parameter)
        # but we immediately detach the output.
        with torch.enable_grad():
            spinors = self._compute_spinors(hid).detach()   # (N, 2)

        curvature = clifford_norm(spinors)  # (N,) — for monitoring

        # ── Chirality split ──────────────────────────────────────────────────
        # sign(s[1]) > 0  →  α (positive chirality, instance memory)
        # sign(s[1]) ≤ 0  →  β (negative chirality, schema memory)
        is_alpha = spinors[:, 1] > 0

        # ── Increment global step ────────────────────────────────────────────
        self.global_step.add_(1)
        step = int(self.global_step.item())

        # ── Write to packs ───────────────────────────────────────────────────
        n_a = self._fifo_write(
            self.alpha_emb, self.alpha_spinor, self.alpha_curvature, self.alpha_age,
            self.alpha_count, self.alpha_head, self.alpha_cap,
            emb[is_alpha], spinors[is_alpha], curvature[is_alpha], step,
        )
        n_b = self._fifo_write(
            self.beta_emb, self.beta_spinor, self.beta_curvature, self.beta_age,
            self.beta_count, self.beta_head, self.beta_cap,
            emb[~is_alpha], spinors[~is_alpha], curvature[~is_alpha], step,
        )

        return {"alpha_stored": n_a, "beta_stored": n_b}

    # -----------------------------------------------------------------------
    # Retrieve
    # -----------------------------------------------------------------------

    def retrieve(
        self,
        query: torch.Tensor,
        top_k: int = 8,
        pool: str = "both",
        hidden_preRMSnorm: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Retrieve top-k items from the α pack, β pack, or both.

        Parameters
        ----------
        query  : [B, T, dim]
        top_k  : number of neighbours per query token
        pool   : 'alpha' | 'beta' | 'both'
        hidden_preRMSnorm : [B, T, dim] optional — used to compute query
            spinors.  Falls back to query if not given.

        Returns
        -------
        {
          'values': [B, T, top_k, dim],
          'scores': [B, T, top_k]
        }
        """
        B, T, D = query.shape
        N = B * T
        q_flat = query.reshape(N, D)

        # ── Query spinors ────────────────────────────────────────────────────
        hid_q = hidden_preRMSnorm.reshape(N, D) if hidden_preRMSnorm is not None else q_flat
        q_spinors = self.proj_spinor(hid_q)  # (N, 2)   — differentiable

        # ── Gather live pool embeddings and spinors ──────────────────────────
        alpha_n = int(self.alpha_count.item())
        beta_n  = int(self.beta_count.item())

        pool_embs: list[torch.Tensor] = []
        pool_spinors: list[torch.Tensor] = []

        if pool in ("alpha", "both") and alpha_n > 0:
            pool_embs.append(self.alpha_emb[:alpha_n])
            pool_spinors.append(self.alpha_spinor[:alpha_n])
        if pool in ("beta", "both") and beta_n > 0:
            pool_embs.append(self.beta_emb[:beta_n])
            pool_spinors.append(self.beta_spinor[:beta_n])

        if not pool_embs:
            # Both packs empty — return zeros
            values = torch.zeros(B, T, top_k, D, device=query.device, dtype=query.dtype)
            scores = torch.zeros(B, T, top_k,    device=query.device, dtype=query.dtype)
            return {"values": values, "scores": scores}

        pool_emb  = torch.cat(pool_embs,   dim=0)   # (M, dim)
        pool_sp   = torch.cat(pool_spinors, dim=0)   # (M, 2)
        M = pool_emb.shape[0]

        # ── Cosine similarity term ───────────────────────────────────────────
        q_norm = F.normalize(q_flat.float(),     dim=-1)   # (N, dim)
        m_norm = F.normalize(pool_emb.float(),   dim=-1)   # (M, dim)
        cos_sim = q_norm @ m_norm.T                         # (N, M)

        # ── Clifford bilinear term ────────────────────────────────────────────
        # a[...,0]*b[...,0] − a[...,1]*b[...,1]
        # q_spinors: (N, 2), pool_sp: (M, 2)
        # Expand to (N, M) via broadcasting:
        #   q_spinors[:, None, :] * pool_sp[None, :, :]
        q_sp = q_spinors.float()          # (N, 2)
        m_sp = pool_sp.float()            # (M, 2)
        cl_sim = (q_sp[:, None, 0] * m_sp[None, :, 0]
                 - q_sp[:, None, 1] * m_sp[None, :, 1])  # (N, M)

        # Normalise Clifford term to [-1, 1] range to keep scores comparable
        cl_scale = (cl_sim.abs().max() + 1e-8)
        cl_sim_normed = cl_sim / cl_scale

        # ── Combined score ────────────────────────────────────────────────────
        scores_flat = 0.7 * cos_sim + 0.3 * cl_sim_normed   # (N, M)

        # ── Top-k ────────────────────────────────────────────────────────────
        k_eff = min(top_k, M)
        topk_scores, topk_idx = scores_flat.topk(k_eff, dim=-1)   # (N, k_eff)

        # Gather value embeddings
        topk_vals = pool_emb[topk_idx]   # (N, k_eff, dim)

        # Pad if needed
        if k_eff < top_k:
            pad = top_k - k_eff
            topk_vals   = torch.cat([topk_vals,
                                     torch.zeros(N, pad, D, device=query.device,
                                                 dtype=topk_vals.dtype)], dim=1)
            topk_scores = torch.cat([topk_scores,
                                     torch.zeros(N, pad, device=query.device,
                                                 dtype=topk_scores.dtype)], dim=1)

        # Cast back to original dtype and reshape to (B, T, top_k, ...)
        values = topk_vals.to(query.dtype).reshape(B, T, top_k, D)
        ret_scores = topk_scores.to(query.dtype).reshape(B, T, top_k)

        return {"values": values, "scores": ret_scores}

    # -----------------------------------------------------------------------
    # Descartes regularizer
    # -----------------------------------------------------------------------

    def descartes_loss(
        self,
        query_spinors: torch.Tensor,
        lmbda: float = 1e-4,
        top_n: int = 4,
    ) -> torch.Tensor:
        """
        Descartes-budget regularizer.

        For each query spinor find its top-4 nearest memory spinors (by
        Euclidean spinor distance), then compute the Descartes violation:

            L_desc = mean_over_queries( ((Σ bᵢ)² − 2 Σ bᵢ²)² )

        where bᵢ = ‖sᵢ‖²_Cl  (Clifford Euclidean norm) of the 4 retrieved
        neighbours.

        This measures how far the local 4-spinor neighbourhood is from
        satisfying the Descartes circle theorem.  Return value is a scalar
        that can be added to the training loss weighted by lmbda.

        Parameters
        ----------
        query_spinors : [Q, 2]  — e.g. the spinors from a batch of queries
        lmbda         : float   — default weight when caller does lmbda * loss
        top_n         : int     — number of neighbours (default 4, as in Descartes)

        Returns
        -------
        Scalar tensor (can be 0.0 if all packs are empty).
        """
        alpha_n = int(self.alpha_count.item())
        beta_n  = int(self.beta_count.item())

        # Gather all stored spinors
        parts: list[torch.Tensor] = []
        if alpha_n > 0:
            parts.append(self.alpha_spinor[:alpha_n])
        if beta_n > 0:
            parts.append(self.beta_spinor[:beta_n])

        if not parts:
            return query_spinors.new_zeros(())

        mem_sp = torch.cat(parts, dim=0).float()  # (M, 2)
        q_sp   = query_spinors.float()             # (Q, 2)
        Q = q_sp.shape[0]
        M = mem_sp.shape[0]
        n = min(top_n, M)

        # Euclidean distance in spinor space: (Q, M)
        diff = q_sp[:, None, :] - mem_sp[None, :, :]   # (Q, M, 2)
        dist2 = (diff ** 2).sum(dim=-1)                  # (Q, M)

        # Top-n nearest  (smallest distance)
        _, topn_idx = dist2.topk(n, dim=-1, largest=False)   # (Q, n)

        # Clifford norms of the n neighbours: bᵢ = sᵢ[0]² + sᵢ[1]²
        nbr_sp = mem_sp[topn_idx]                            # (Q, n, 2)
        b = clifford_norm(nbr_sp)                            # (Q, n)

        # Descartes violation: ((Σ bᵢ)² − 2 Σ bᵢ²)²
        sum_b  = b.sum(dim=-1)                               # (Q,)
        sum_b2 = (b ** 2).sum(dim=-1)                        # (Q,)
        violation = (sum_b ** 2 - 2.0 * sum_b2) ** 2        # (Q,)

        return violation.mean()

    # -----------------------------------------------------------------------
    # Introspection
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def get_stats(self) -> Dict[str, Any]:
        """
        Return diagnostic statistics compatible with the monitoring dashboard.

        Keys
        ----
        alpha_fill           : int   — items currently in α pack
        beta_fill            : int   — items currently in β pack
        alpha_curvature_mean : float — mean Clifford norm of α spinors
        beta_curvature_mean  : float — mean Clifford norm of β spinors
        chirality_balance    : float — α_fill / (α_fill + β_fill)  ∈ [0, 1]
                                       0.5 = perfect balance; 0 or 1 = starvation
        """
        alpha_n = int(self.alpha_count.item())
        beta_n  = int(self.beta_count.item())
        total = alpha_n + beta_n

        alpha_curv_mean = (
            float(self.alpha_curvature[:alpha_n].mean().item()) if alpha_n > 0 else 0.0
        )
        beta_curv_mean = (
            float(self.beta_curvature[:beta_n].mean().item()) if beta_n > 0 else 0.0
        )
        chirality_balance = float(alpha_n) / float(total) if total > 0 else 0.5

        pq = self._sample_pq_overlap(n_samples=512)

        return {
            "alpha_fill":           alpha_n,
            "beta_fill":            beta_n,
            "alpha_curvature_mean": alpha_curv_mean,
            "beta_curvature_mean":  beta_curv_mean,
            "chirality_balance":    chirality_balance,
            "pq_overlap_mean":      pq["mean"],
            "pq_overlap_std":       pq["std"],
            "pq_bimodality":        pq["bimodality"],
            "chsh_S":               self.chsh_correlator(n_samples=512),
        }

    @torch.no_grad()
    def chsh_correlator(self, n_samples: int = 512) -> float:
        # Bell CHSH audit between alpha/beta chirality packs (Bell CERN CDS
        # 111654). Picks four unit axes (a, a', b, b') in the 2D spinor plane
        # and returns S = |E(a,b) + E(a,b') + E(a',b) - E(a',b')|. Classical
        # hidden-variable bound is S <= 2; quantum bound is 2*sqrt(2) ≈ 2.828.
        # When packs behave as entangled subsystems, S pushes past 2; when the
        # chirality split has collapsed into a single local cluster, S falls
        # toward 0. Dashboard metric, no gradient.
        alpha_n = int(self.alpha_count.item())
        beta_n  = int(self.beta_count.item())
        if alpha_n == 0 or beta_n == 0:
            return 0.0
        n = min(n_samples, alpha_n, beta_n)
        dev = self.alpha_spinor.device
        ai = torch.randint(0, alpha_n, (n,), device=dev)
        bi = torch.randint(0, beta_n,  (n,), device=dev)
        sa = F.normalize(self.alpha_spinor[ai].float(), dim=-1)  # (n, 2)
        sb = F.normalize(self.beta_spinor[bi].float(),  dim=-1)  # (n, 2)
        a  = torch.tensor([1.0, 0.0], device=dev)
        ap = torch.tensor([0.0, 1.0], device=dev)
        b  = torch.tensor([math.cos(math.pi / 4), math.sin(math.pi / 4)], device=dev)
        bp = torch.tensor([math.cos(math.pi / 4), -math.sin(math.pi / 4)], device=dev)
        E = lambda u, v: ((sa @ u) * (sb @ v)).mean().item()
        return abs(E(a, b) + E(a, bp) + E(ap, b) - E(ap, bp))

    @torch.no_grad()
    def _sample_pq_overlap(self, n_samples: int = 512) -> Dict[str, float]:
        # Berg-Billoire-Janke (CERN CDS 782816): the two-replica overlap
        # distribution P(q) of a spin glass is bimodal in the healthy
        # two-cluster regime, delta-like on collapse, broad when over-mixed.
        # Here "replicas" = alpha pack and beta pack. Overlap = cosine
        # similarity between one sampled element from each. Bimodality index
        # is |skew| * kurtosis_excess, larger = more bimodal structure.
        alpha_n = int(self.alpha_count.item())
        beta_n  = int(self.beta_count.item())
        if alpha_n == 0 or beta_n == 0:
            return {"mean": 0.0, "std": 0.0, "bimodality": 0.0}
        n = min(n_samples, alpha_n, beta_n)
        dev = self.alpha_emb.device
        ai = torch.randint(0, alpha_n, (n,), device=dev)
        bi = torch.randint(0, beta_n,  (n,), device=dev)
        a = F.normalize(self.alpha_emb[ai].float(), dim=-1)
        b = F.normalize(self.beta_emb[bi].float(),  dim=-1)
        q = (a * b).sum(dim=-1)  # (n,) cosine overlaps
        m  = q.mean()
        sd = q.std()
        if sd.item() < 1e-8:
            return {"mean": float(m), "std": float(sd), "bimodality": 0.0}
        z = (q - m) / sd
        skew = (z ** 3).mean().abs()
        kurt = (z ** 4).mean() - 3.0
        return {
            "mean":       float(m),
            "std":        float(sd),
            "bimodality": float(skew * kurt.clamp(min=0.0)),
        }

    # -----------------------------------------------------------------------
    # SleepGate compatibility stub
    # (full implementation deferred — mirrors apollonian.py.sleep_consolidate)
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def sleep_consolidate(
        self,
        merge_threshold: float = 0.92,
        staleness_horizon: int = 200,
    ) -> Dict[str, int]:
        """
        SleepGate consolidation on the α pack (same API as ApollonianMemory).

        Evicts stale entries and greedily merges near-duplicate embeddings
        (cosine > merge_threshold).  Spinors and curvatures of merged pairs
        are averaged.
        """
        n_alpha = int(self.alpha_count.item())
        if n_alpha < 4:
            return {"n_merged": 0, "n_evicted": 0, "n_before": n_alpha, "n_after": n_alpha}

        step = int(self.global_step.item())
        embs  = self.alpha_emb[:n_alpha].clone()
        sps   = self.alpha_spinor[:n_alpha].clone()
        curvs = self.alpha_curvature[:n_alpha].clone()
        ages  = self.alpha_age[:n_alpha].clone()

        # Phase 1: evict stale
        age_cutoff = max(step - staleness_horizon, 0)
        fresh = ages >= age_cutoff
        n_evicted = int((~fresh).sum().item())
        embs, sps, curvs, ages = embs[fresh], sps[fresh], curvs[fresh], ages[fresh]

        # Phase 2: greedy cosine merge
        n_merged = 0
        if embs.shape[0] >= 2:
            norms = F.normalize(embs, dim=-1)
            sim = norms @ norms.T
            merged_into = list(range(embs.shape[0]))
            for i in range(embs.shape[0]):
                if merged_into[i] != i:
                    continue
                for j in range(i + 1, embs.shape[0]):
                    if merged_into[j] != j:
                        continue
                    if float(sim[i, j].item()) > merge_threshold:
                        embs[i]  = (embs[i] + embs[j]) / 2.0
                        sps[i]   = (sps[i]  + sps[j])  / 2.0
                        curvs[i] = max(curvs[i].item(), curvs[j].item())
                        ages[i]  = max(ages[i].item(),  ages[j].item())
                        merged_into[j] = i
                        n_merged += 1
            keep = torch.tensor(
                [merged_into[i] == i for i in range(len(merged_into))],
                dtype=torch.bool, device=embs.device,
            )
            embs, sps, curvs, ages = embs[keep], sps[keep], curvs[keep], ages[keep]

        # Phase 3: write back
        n_after = embs.shape[0]
        self.alpha_emb[:n_after]       = embs
        self.alpha_spinor[:n_after]    = sps
        self.alpha_curvature[:n_after] = curvs
        self.alpha_age[:n_after]       = ages
        if n_after < self.alpha_cap:
            self.alpha_emb[n_after:].zero_()
            self.alpha_spinor[n_after:].zero_()
            self.alpha_curvature[n_after:].zero_()
            self.alpha_age[n_after:].zero_()
        self.alpha_count.fill_(n_after)
        self.alpha_head.fill_(n_after % self.alpha_cap)

        return {
            "n_merged":  n_merged,
            "n_evicted": n_evicted,
            "n_before":  n_alpha,
            "n_after":   n_after,
        }

    @torch.no_grad()
    def reset(self) -> None:
        """Wipe both packs (for tests / between runs)."""
        for buf_name in (
            "alpha_emb", "alpha_spinor", "alpha_curvature", "alpha_age",
            "alpha_count", "alpha_head",
            "beta_emb",  "beta_spinor",  "beta_curvature",  "beta_age",
            "beta_count", "beta_head",
            "global_step",
        ):
            getattr(self, buf_name).zero_()
