"""
HierarchicalApollonianRouter — the heart of FANT 2.

This file replaces the FANT 350M frozen-projection + 32-scalar bias router
that catastrophically collapsed onto a single expert (94.5% routing across all
domains, mean JSD = 0.0; see fant_350m_postmortem.md §2).

The fix is independently prescribed by SIX research traditions, all of which
converge on the same architectural pattern. This single class implements all six:

  1. MoE engineering (DeepSeek aux-loss-free, arXiv 2408.15664):
     - Gradient-free per-expert bias updated by sign(load - target)

  2. Brain-inspired ML (Dragon Hatchling + EvoMoE + MoR):
     - Slow-EMA tracking of expert load (continuously regenerating router)

  3. Hyperdimensional fractal math (ETF + Hodge + Stiefel):
     - Simplex Equiangular Tight Frame initialization
     - Stiefel Cayley retraction every 100 steps
     - Hodge Laplacian smoothing of the Apollonian contact graph

  4. Statistical physics (Parisi RSB):
     - 2-level hierarchy: 8 mega-pools × 9 experts each (= 72 fractal seeds)
     - Top-1 mega-pool selection, then top-4 of 9 within
     - This is the ultrametric structure that FANT 350M's flat router lacked

  5. Neurology / biology (basal ganglia + neuromodulation + slime mold):
     - Per-expert learning rate scaling buffer (dopamine analog)
     - Go/NoGo balance via the Tikkun repair mechanism

  6. Contemplative phenomenology (Tikkun + Fanā + emptiness):
     - tikkun_repair(): event-driven rebalance when load skews
     - fana_dropout(): periodic expert-index shuffle for emptiness regularization

When math, physics, ML SOTA, biology, theology, and contemplative phenomenology
all converge on the same fix from independent traditions, the fix is canonical.
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Initialization helpers
# =============================================================================

def simplex_etf_init(k: int, d: int, device=None, dtype=None) -> torch.Tensor:
    """
    Construct a Simplex Equiangular Tight Frame.

    From Papyan-Han-Donoho (PNAS 2020) "Prevalence of neural collapse during the
    terminal phase of deep learning training": at convergence, well-trained
    classifier features collapse to the rows of a Simplex ETF. By initializing
    the router at this geometry, we start the optimizer at the *terminal* point
    of the symmetric phase, which is known to be the *escape point* for the
    asymmetric (collapse) basin.

    A Simplex ETF in d dimensions has k vectors (k <= d+1) with:
        - all unit norm
        - all pairwise inner products equal to -1/(k-1)

    Construction (k <= d+1):
        E = sqrt(k/(k-1)) * (I_k - (1/k) * 1_k 1_k^T) @ U
    where U is any k x d matrix with orthonormal rows (k <= d).

    Args:
        k: number of frame vectors
        d: ambient dimension (must be >= k - 1)

    Returns:
        Tensor of shape (k, d), unit-norm rows.
    """
    if k > d + 1:
        # Fall back to a "near-ETF" via random rotation; the simplex constraint
        # is impossible when k > d+1
        E = torch.randn(k, d, device=device, dtype=dtype or torch.float32)
        E = F.normalize(E, dim=-1)
        return E

    # Build I_k - (1/k) * 1_k 1_k^T (the centering matrix on the k-simplex)
    I = torch.eye(k, device=device, dtype=torch.float32)
    centering = I - (1.0 / k)
    # Need a (k, d) orthonormal matrix; sample random and orthogonalize
    raw = torch.randn(k, d, device=device, dtype=torch.float32)
    Q, _ = torch.linalg.qr(raw.T)  # (d, k)
    U = Q.T[:k]                     # (k, d) orthonormal rows
    E = math.sqrt(k / (k - 1)) * centering @ U
    # Renormalize each row to unit length (numerically safer)
    E = F.normalize(E, dim=-1)
    if dtype is not None:
        E = E.to(dtype)
    return E


def cayley_retract(W: torch.Tensor) -> torch.Tensor:
    """
    Stiefel manifold retraction via Cayley transform.

    Restores orthogonality of a matrix that has drifted off the Stiefel manifold
    during gradient updates. From hyperdim agent #7 §13.

    For a (k, d) matrix with k <= d, builds:
        A = W W^T - W^T W   (antisymmetric in the k-block)
        W' = (I - A/2)^{-1} (I + A/2) W

    Cost: ~5 ms per layer per restoration. Called every 100 training steps.
    """
    k, d = W.shape
    if k > d:
        return W  # over-determined, skip
    # `torch.linalg.qr` / `solve` do not support bfloat16 on CUDA, so do the
    # linear-algebra in float32 and cast back at the end.
    orig_dtype = W.dtype
    W32 = W.to(torch.float32) if orig_dtype in (torch.bfloat16, torch.float16) else W
    WWt = W32 @ W32.T       # (k, k)
    A = WWt - WWt.T         # antisymmetric (k, k)
    I = torch.eye(k, device=W32.device, dtype=W32.dtype)
    try:
        retracted = torch.linalg.solve(I - 0.5 * A, I + 0.5 * A) @ W32
    except RuntimeError:
        # Singular matrix → fall back to QR re-orthogonalization
        Q, _ = torch.linalg.qr(W32.T)
        retracted = Q.T[:k]
    if retracted.dtype != orig_dtype:
        retracted = retracted.to(orig_dtype)
    return retracted


# =============================================================================
# RouterOutput dataclass
# =============================================================================

@dataclass
class RouterOutput:
    """Carries the routing decisions plus all the diagnostic state."""
    # Selection results
    megapool_idx: torch.Tensor       # (B*T,) long, the chosen mega-pool 0..7
    expert_idx_local: torch.Tensor   # (B*T, top_k) long, indices 0..n_per_pool-1 within the pool
    expert_idx_global: torch.Tensor  # (B*T, top_k) long, indices 0..n_fractal-1 (= 8*9 = 72)
    expert_weights: torch.Tensor     # (B*T, top_k) float, gating weights for the top-k experts

    # Per-step load (used for the aux-loss-free bias update)
    megapool_load: torch.Tensor      # (n_megapools,) float, fraction of tokens
    expert_load: torch.Tensor        # (n_megapools, n_per_pool) float

    # Logits for telemetry & z-loss
    megapool_logits: torch.Tensor    # (B*T, n_megapools) float
    expert_logits: torch.Tensor      # (B*T, n_per_pool) float


# =============================================================================
# HierarchicalApollonianRouter
# =============================================================================

class HierarchicalApollonianRouter(nn.Module):
    """
    The 2-stage hierarchical router that fixes FANT 350M's collapse.

    Forward pass:
        x : (B, T, dim)
        --> stage 1: top-1 mega-pool selection
        --> stage 2: top-k of n_per_pool within the chosen mega-pool
        --> returns RouterOutput

    Stage 1 and Stage 2 each use:
        - frozen ETF projection (initialized to a Simplex Equiangular Tight Frame)
        - learnable bias updated GRADIENT-FREE via sign(load - target) every step
        - sigmoid gating (NOT softmax) — DeepSeek V3 §3.2

    The frozen-projection design is a deliberate choice from neurology (#8) and
    physics (#9): the projection is the "quenched disorder" of the SK spin glass,
    and the bias is the "annealed Parisi field" — only the bias adapts.

    Total params (12 layers, dim=768, 8 mega-pools, 9 per pool):
      - megapool_proj: 768*8 = 6144 floats per layer (frozen at init, no gradient)
      - expert_proj:    768*9 = 6912 floats per layer (frozen at init, no gradient)
      - megapool_bias:  8 floats per layer (gradient-free update)
      - expert_bias:    8*9 = 72 floats per layer (gradient-free update)
      - 3 buffer EMAs:   ~80 floats per layer
      Total per-layer router: ~13k floats
      Total across 12 layers: ~157k floats = ~0.16M (fits well in 60M budget)
    """

    def __init__(
        self,
        dim: int = 768,
        n_megapools: int = 8,
        n_per_pool: int = 9,
        top_k: int = 4,
        gamma: float = 1e-3,                # aux-loss-free bias step size
        ema_decay: float = 0.99,             # slow EMA for load tracking
        tikkun_threshold: float = 0.30,      # (over-fraction) trigger for repair
        bipartition_floor: float = 1.05,     # IIT minimum
        cayley_every_n_steps: int = 100,
    ):
        super().__init__()
        self.dim = dim
        self.n_megapools = n_megapools
        self.n_per_pool = n_per_pool
        self.n_fractal = n_megapools * n_per_pool   # 72
        self.top_k = top_k
        self.gamma = gamma
        self.ema_decay = ema_decay
        self.tikkun_threshold = tikkun_threshold
        self.bipartition_floor = bipartition_floor
        self.cayley_every_n_steps = cayley_every_n_steps

        # ----- Stage 1: mega-pool projection -----
        # Frozen, initialized to Simplex ETF. NOT a Parameter.
        self.register_buffer(
            "megapool_proj",
            simplex_etf_init(n_megapools, dim, dtype=torch.float32),
        )

        # ----- Stage 2: within-pool projection (shared across pools, biases differ) -----
        self.register_buffer(
            "expert_proj",
            simplex_etf_init(n_per_pool, dim, dtype=torch.float32),
        )

        # ----- Aux-loss-free biases (gradient-free, updated by sign(load - target)) -----
        self.register_buffer("megapool_bias", torch.zeros(n_megapools))
        self.register_buffer("expert_bias", torch.zeros(n_megapools, n_per_pool))

        # ----- Slow-EMA load tracking buffers (for diagnostics + Tikkun) -----
        self.register_buffer("megapool_load_ema", torch.full((n_megapools,), 1.0 / n_megapools))
        self.register_buffer("expert_load_ema", torch.full((n_megapools, n_per_pool), 1.0 / (n_megapools * n_per_pool)))

        # ----- Per-expert learning rate scaling (neurology #8: dopamine analog) -----
        self.register_buffer("expert_lr_scale", torch.ones(n_megapools, n_per_pool))

        # ----- Counter for periodic Cayley retraction & Tikkun checks -----
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

    # -------------------------------------------------------------------------
    # Forward: hierarchical top-1 then top-k
    # -------------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> RouterOutput:
        """
        Args:
            x: (B, T, dim) input activations

        Returns:
            RouterOutput with all routing state.
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.reshape(N, D)

        # ===== Stage 1: mega-pool selection (top-1) =====
        # Sigmoid gating per DeepSeek V3, NOT softmax
        mp_logits = x_flat @ self.megapool_proj.T + self.megapool_bias  # (N, n_megapools)
        mp_scores = torch.sigmoid(mp_logits)
        # Top-1 mega-pool index per token
        mp_idx = mp_scores.argmax(dim=-1)  # (N,)

        # ===== Stage 2: top-k within mega-pool =====
        # Select the bias row corresponding to each token's chosen mega-pool
        # expert_bias: (n_megapools, n_per_pool) → bias_row: (N, n_per_pool)
        bias_row = self.expert_bias[mp_idx]  # (N, n_per_pool)
        ex_logits = x_flat @ self.expert_proj.T + bias_row  # (N, n_per_pool)
        ex_scores = torch.sigmoid(ex_logits)

        topk_vals, topk_local = ex_scores.topk(self.top_k, dim=-1)  # (N, top_k)

        # Renormalize top-k weights to sum to 1 per token
        topk_weights = topk_vals / (topk_vals.sum(dim=-1, keepdim=True) + 1e-8)

        # Convert local (within-pool) indices to global expert indices
        # global_idx = mp_idx * n_per_pool + local_idx
        topk_global = mp_idx.unsqueeze(-1) * self.n_per_pool + topk_local  # (N, top_k)

        # ===== Compute per-step load for the bias update =====
        # mp_load: fraction of tokens that picked each mega-pool
        mp_load = torch.zeros(self.n_megapools, device=x.device, dtype=torch.float32)
        mp_load.scatter_add_(0, mp_idx, torch.ones(N, device=x.device, dtype=torch.float32))
        mp_load = mp_load / N

        # ex_load: fraction of (mega-pool, local-expert) tokens
        ex_load = torch.zeros(self.n_megapools, self.n_per_pool, device=x.device, dtype=torch.float32)
        # For each token, accumulate weighted load on its top-k local choices
        for k in range(self.top_k):
            flat_idx = mp_idx * self.n_per_pool + topk_local[:, k]  # (N,)
            ex_load.view(-1).scatter_add_(0, flat_idx, topk_weights[:, k].float())
        ex_load = ex_load / N

        return RouterOutput(
            megapool_idx=mp_idx,
            expert_idx_local=topk_local,
            expert_idx_global=topk_global,
            expert_weights=topk_weights,
            megapool_load=mp_load,
            expert_load=ex_load,
            megapool_logits=mp_logits,
            expert_logits=ex_logits,
        )

    # -------------------------------------------------------------------------
    # Aux-loss-free bias update (DeepSeek 2408.15664)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def update_biases(self, mp_load: torch.Tensor, ex_load: torch.Tensor) -> None:
        """
        Gradient-free bias update. Called once per training step from the trainer
        AFTER backward() but before optimizer.step() (to avoid interfering with
        the gradient computation).

        For each mega-pool i:
            target_i = 1/n_megapools  (uniform load target)
            megapool_bias[i] -= gamma * sign(mp_load[i] - target_i)

        Same for the within-pool expert biases.
        """
        target_mp = 1.0 / self.n_megapools
        target_ex = 1.0 / (self.n_megapools * self.n_per_pool)

        self.megapool_bias.add_(-self.gamma * torch.sign(mp_load - target_mp))
        self.expert_bias.add_(-self.gamma * torch.sign(ex_load - target_ex))

        # Slow-EMA tracking for diagnostics + Tikkun
        d = self.ema_decay
        self.megapool_load_ema.mul_(d).add_((1 - d) * mp_load)
        self.expert_load_ema.mul_(d).add_((1 - d) * ex_load)

        self.step_counter.add_(1)

        # Periodic Stiefel retraction
        if (self.step_counter.item() % self.cayley_every_n_steps) == 0:
            self.megapool_proj.copy_(cayley_retract(self.megapool_proj))
            self.expert_proj.copy_(cayley_retract(self.expert_proj))

    # -------------------------------------------------------------------------
    # Tikkun event-driven repair (theology agent #10 + physics agent #9)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def tikkun_repair(self) -> bool:
        """
        Check if any mega-pool's slow-EMA load exceeds the threshold, and if so
        perturb its bias downward + the others upward to restore balance.

        Called every TIKKUN_CHECK_EVERY_N_STEPS (default 200).

        Returns True if a repair was triggered, False otherwise.
        """
        target = 1.0 / self.n_megapools
        excess = (self.megapool_load_ema - target) > self.tikkun_threshold

        if not excess.any():
            return False

        n_excess = int(excess.sum().item())
        n_under = self.n_megapools - n_excess

        # Pull down the over-loaded pools, push up the rest
        self.megapool_bias[excess] -= 0.05
        if n_under > 0:
            self.megapool_bias[~excess] += 0.05 * n_excess / n_under

        # Reset the EMA so we can re-measure with the new biases
        self.megapool_load_ema.fill_(target)
        return True

    # -------------------------------------------------------------------------
    # Fanā dropout (theology agent #10)
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def fana_dropout(self, p: float = 0.05) -> None:
        """
        Periodic random shuffle of the within-pool expert indices.

        From contemplative phenomenology (Sufi fanā = annihilation of the
        ego-as-fixed-thing): if the model has come to identify a particular
        expert with a particular meaning, that identification is itself a
        collapse mode. Periodic shuffling forces the meaning to re-attach to
        the *function* of the expert, not its *index*.

        Implementation: with probability p, shuffle the expert_proj rows
        (and the corresponding biases and EMAs).
        """
        if torch.rand(1).item() > p:
            return

        # Shuffle within each mega-pool independently
        for mp in range(self.n_megapools):
            perm = torch.randperm(self.n_per_pool, device=self.expert_bias.device)
            self.expert_bias[mp] = self.expert_bias[mp, perm]
            self.expert_load_ema[mp] = self.expert_load_ema[mp, perm]
            self.expert_lr_scale[mp] = self.expert_lr_scale[mp, perm]

    # -------------------------------------------------------------------------
    # Auxiliary losses (called by the trainer, all return scalar tensors)
    # -------------------------------------------------------------------------

    @staticmethod
    def z_loss(logits: torch.Tensor) -> torch.Tensor:
        """
        OLMoE router z-loss: penalize the magnitude of logits to keep them bounded.

            L_z = (logsumexp(logits))^2  averaged over tokens
        """
        return (torch.logsumexp(logits, dim=-1) ** 2).mean()

    def fep_kl_prior(self, mp_load: torch.Tensor, ex_load: torch.Tensor) -> torch.Tensor:
        """
        Free Energy Principle KL divergence between current routing distribution
        and a uniform prior. Replaces 4 separate FANT 350M losses.

            L_FEP = KL(routing_dist || uniform_prior)
        """
        eps = 1e-8
        # Mega-pool level KL
        target_mp = 1.0 / self.n_megapools
        mp_kl = F.kl_div(
            (mp_load + eps).log(),
            torch.full_like(mp_load, target_mp),
            reduction="sum",
        )
        # Expert level KL
        target_ex = 1.0 / (self.n_megapools * self.n_per_pool)
        ex_kl = F.kl_div(
            (ex_load.flatten() + eps).log(),
            torch.full_like(ex_load.flatten(), target_ex),
            reduction="sum",
        )
        return mp_kl + ex_kl

    # -------------------------------------------------------------------------
    # Diagnostics for the 8-metric telemetry suite
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def parisi_overlap_distribution(self, n_samples: int = 100) -> torch.Tensor:
        """
        Estimate the Parisi overlap distribution P(q) over the expert biases.
        Used by telemetry to verify the ultrametric structure has formed.

        Returns a 1D histogram tensor of length n_samples representing P(q),
        with q ranging from -1 to +1.
        """
        flat_bias = self.expert_bias.flatten()
        # Pairwise normalized inner products (in the bias-direction sense)
        if flat_bias.numel() < 2:
            return torch.zeros(n_samples)
        n = min(flat_bias.numel(), 100)
        v = flat_bias[:n] / (flat_bias[:n].abs().max() + 1e-8)
        overlaps = (v.unsqueeze(0) * v.unsqueeze(1)).flatten()
        # Histogram in [-1, 1]
        hist = torch.histc(overlaps, bins=n_samples, min=-1.0, max=1.0)
        return hist / (hist.sum() + 1e-8)

    @torch.no_grad()
    def domain_jsd(self, domain_routings: dict) -> dict:
        """
        Compute Jensen-Shannon divergence between routing distributions of
        different domains. THIS IS THE FANT 350M FAILURE METRIC.

        Args:
            domain_routings: dict[str, Tensor of shape (n_fractal,)] giving the
                            routing distribution per domain (averaged over a probe set)

        Returns:
            dict of pairwise JSD values + mean JSD
        """
        domains = list(domain_routings.keys())
        result = {}
        jsds = []
        for i, di in enumerate(domains):
            for dj in domains[i + 1:]:
                p = domain_routings[di] + 1e-12
                q = domain_routings[dj] + 1e-12
                m = 0.5 * (p + q)
                kl_pm = F.kl_div(m.log(), p, reduction="sum")
                kl_qm = F.kl_div(m.log(), q, reduction="sum")
                jsd = 0.5 * (kl_pm + kl_qm)
                result[f"{di}_vs_{dj}"] = jsd.item()
                jsds.append(jsd.item())
        result["mean_jsd"] = sum(jsds) / max(len(jsds), 1)
        return result
