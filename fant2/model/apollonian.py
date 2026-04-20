"""
ApollonianMemory — the dual α/β memory bank at the heart of FANT 2.

THIS MODULE IS THE CENTRAL THEORETICAL COMMITMENT OF FANT 2.

The brain-as-Apollonian-packing hypothesis:
    Mature cognition stores knowledge as a 3D Apollonian sphere packing in
    representation space. Two complementary stores fill the gaps of the packing:

      α (alpha)  packing  =  high-curvature spheres,  RECENT, INSTANCE memory
      β (beta)   packing  =  low-curvature  spheres, OLD,    SCHEMA  memory

    Curvature ≈ 1/radius. A sphere with high curvature is *small* and *specific*
    (a particular event, a particular face). A sphere with low curvature is
    *large* and *general* (the abstract concept of "face", the schema of
    "celebration"). The Apollonian gasket is the unique structure that fills any
    region with non-overlapping circles/spheres of all scales — the only known
    space-filling, scale-free, parameter-free packing.

    Predictions made by the hypothesis:
      1. The two memory pools should have *power-law* size distributions
         (Apollonian curvatures follow a power law with exponent ≈ 1.305)
      2. Retrieval should be O(log n) by descending the packing hierarchy
      3. The two pools should be *complementary* — α fills the gaps β cannot
      4. Forgetting should preserve the packing structure (eviction is by age,
         not by importance)

References:
    - Edelman & Mountcastle  (1978) — cortical column hypothesis
    - Mandelbrot             (1982) — fractal geometry of nature, ch. 18
    - Boyd, Lagarias, Mallows, Wilks (2003) — Apollonian circle packings
    - Hassabis et al.        (2007) — episodic memory's role in imagination
    - Tulving                (1985) — episodic / semantic memory distinction
    - Marr                   (1971) — simple memory: hippocampus → cortex consolidation

This module is NOT a Module of trainable parameters. It is a non-differentiable
memory bank that the model READS via the ApollonianRetrievalAttention module
in memory_retrieval.py, and that the trainer WRITES via the .store() method
during the Phase 4 self-refinement loop.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApollonianMemory(nn.Module):
    """
    Dual α/β memory pack for the FANT 2 model.

    Both packs are stored as fixed-size circular buffers (FIFO eviction).
    The model code WRITES to them via .store() (no_grad), and READS from
    them via .retrieve() which returns top-k cosine-similarity matches.

    The buffers are stored as nn.Module buffers (not Parameters) so they:
      - move with .to(device)
      - get saved with state_dict()
      - do NOT contribute to the optimizer state
      - do NOT generate gradients
    """

    def __init__(
        self,
        dim: int = 768,
        alpha_cap: int = 5000,
        beta_cap: int = 5000,
        curvature_threshold: float = 0.5,
    ):
        super().__init__()
        self.dim = dim
        self.alpha_cap = alpha_cap
        self.beta_cap = beta_cap
        self.curvature_threshold = curvature_threshold

        # ----- α pack: high-curvature, recent, instance memory -----
        self.register_buffer("alpha_emb",       torch.zeros(alpha_cap, dim))
        self.register_buffer("alpha_curvature", torch.zeros(alpha_cap))
        self.register_buffer("alpha_age",       torch.zeros(alpha_cap, dtype=torch.long))
        # Number of slots actually used and the next write head (FIFO)
        self.register_buffer("alpha_count",     torch.tensor(0, dtype=torch.long))
        self.register_buffer("alpha_head",      torch.tensor(0, dtype=torch.long))

        # ----- β pack: low-curvature, old, schema memory -----
        self.register_buffer("beta_emb",        torch.zeros(beta_cap, dim))
        self.register_buffer("beta_curvature",  torch.zeros(beta_cap))
        self.register_buffer("beta_age",        torch.zeros(beta_cap, dtype=torch.long))
        self.register_buffer("beta_count",      torch.tensor(0, dtype=torch.long))
        self.register_buffer("beta_head",       torch.tensor(0, dtype=torch.long))

        # ----- Global step counter (used for age tracking) -----
        self.register_buffer("global_step",     torch.tensor(0, dtype=torch.long))

    # -------------------------------------------------------------------------
    # Curvature estimation (the proxy for sphere radius in the packing analogy)
    # -------------------------------------------------------------------------

    @staticmethod
    @torch.no_grad()
    def estimate_curvature(emb: torch.Tensor, ref_norm: float = 1.0) -> torch.Tensor:
        """
        Estimate the "curvature" of each embedding in the Apollonian sense.

        We use the L2 norm of the embedding as the proxy:
            curvature(e) = ||e|| / ref_norm

        High curvature = high norm = a sphere with a small radius (specific).
        Low  curvature = low  norm = a sphere with a large radius (general).

        This is one of three valid choices the spec allows:
            (a) L2 norm proxy (this method) — fast, memory-free
            (b) Local density via k-NN to existing buffer — more accurate but O(n)
            (c) Effective rank of a local Hessian — most principled but O(d²)

        We use (a) for v2.0; the others are TODO refinements.

        Args:
            emb:      (..., dim) embeddings
            ref_norm: normalization constant (typically the mean norm of recent embeddings)

        Returns:
            (...,) curvature scores
        """
        return emb.norm(dim=-1) / max(ref_norm, 1e-8)

    # -------------------------------------------------------------------------
    # Store
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def store(
        self,
        embeddings: torch.Tensor,
        curvatures: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store a batch of embeddings into either the α or β pack based on curvature.

        Args:
            embeddings: (N, dim) batch of embeddings to maybe-store
            curvatures: (N,) per-embedding curvature scores (or None to compute)
        """
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0)
        embeddings = embeddings.detach()

        N, D = embeddings.shape
        assert D == self.dim, f"embedding dim {D} != memory dim {self.dim}"

        if curvatures is None:
            curvatures = self.estimate_curvature(embeddings)
        curvatures = curvatures.detach()

        # Increment the global step
        self.global_step.add_(1)
        step = self.global_step.item()

        # Decide pack assignment by curvature threshold
        is_alpha = curvatures > self.curvature_threshold
        alpha_emb_in = embeddings[is_alpha]
        alpha_cur_in = curvatures[is_alpha]
        beta_emb_in  = embeddings[~is_alpha]
        beta_cur_in  = curvatures[~is_alpha]

        # FIFO write to alpha
        self._fifo_write(
            self.alpha_emb, self.alpha_curvature, self.alpha_age,
            self.alpha_count, self.alpha_head, self.alpha_cap,
            alpha_emb_in, alpha_cur_in, step,
        )

        # FIFO write to beta
        self._fifo_write(
            self.beta_emb, self.beta_curvature, self.beta_age,
            self.beta_count, self.beta_head, self.beta_cap,
            beta_emb_in, beta_cur_in, step,
        )

    @staticmethod
    @torch.no_grad()
    def _fifo_write(
        emb_buf: torch.Tensor,
        cur_buf: torch.Tensor,
        age_buf: torch.Tensor,
        count_buf: torch.Tensor,
        head_buf: torch.Tensor,
        cap: int,
        new_emb: torch.Tensor,
        new_cur: torch.Tensor,
        step: int,
    ) -> None:
        """In-place FIFO write of new entries into a circular buffer."""
        n_new = new_emb.shape[0]
        if n_new == 0:
            return
        head = int(head_buf.item())
        count = int(count_buf.item())

        # Write each new entry, wrapping around at the head
        for i in range(n_new):
            slot = (head + i) % cap
            emb_buf[slot] = new_emb[i]
            cur_buf[slot] = new_cur[i]
            age_buf[slot] = step
        # Advance head and count
        new_head = (head + n_new) % cap
        new_count = min(count + n_new, cap)
        head_buf.fill_(new_head)
        count_buf.fill_(new_count)

    # -------------------------------------------------------------------------
    # Retrieve
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def retrieve(
        self,
        query: torch.Tensor,
        pack: str = "alpha",
        k: int = 8,
    ) -> tuple:
        """
        Top-k cosine similarity retrieval from a pack.

        Args:
            query: (B*T, dim) or (N, dim) query embeddings
            pack:  "alpha" or "beta"
            k:     number of nearest neighbors to retrieve per query

        Returns:
            (mem_emb, sim) where
                mem_emb: (N, k, dim) the retrieved memory embeddings
                sim:     (N, k)      the cosine similarities (used as attention weights)
        """
        if pack == "alpha":
            buf_emb, count = self.alpha_emb, int(self.alpha_count.item())
        elif pack == "beta":
            buf_emb, count = self.beta_emb,  int(self.beta_count.item())
        else:
            raise ValueError(f"pack must be 'alpha' or 'beta', got {pack}")

        N, D = query.shape
        if count == 0:
            # Empty pack — return zeros (the retrieval attention will mask them out)
            mem = torch.zeros(N, k, D, device=query.device, dtype=query.dtype)
            sim = torch.zeros(N, k, device=query.device, dtype=query.dtype)
            return mem, sim

        # Restrict to filled slots
        live_emb = buf_emb[:count]                                 # (count, dim)
        # Cosine similarity (normalize both)
        q_norm = F.normalize(query.float(),    dim=-1)             # (N, dim)
        m_norm = F.normalize(live_emb.float(), dim=-1)             # (count, dim)
        sims = q_norm @ m_norm.T                                   # (N, count)

        # Top-k
        k_eff = min(k, count)
        topk_sim, topk_idx = sims.topk(k_eff, dim=-1)              # (N, k_eff)

        # Gather embeddings
        topk_mem = live_emb[topk_idx]                              # (N, k_eff, dim)

        # Pad to k if k_eff < k (with zeros)
        if k_eff < k:
            pad_count = k - k_eff
            mem_pad = torch.zeros(N, pad_count, D, device=query.device, dtype=topk_mem.dtype)
            sim_pad = torch.zeros(N, pad_count,    device=query.device, dtype=topk_sim.dtype)
            topk_mem = torch.cat([topk_mem, mem_pad], dim=1)
            topk_sim = torch.cat([topk_sim, sim_pad], dim=1)

        return topk_mem.to(query.dtype), topk_sim.to(query.dtype)

    # -------------------------------------------------------------------------
    # Diagnostics — these answer the 14 testable predictions of the hypothesis
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def fill_rates(self) -> dict:
        """Return how full each pack is, for telemetry."""
        return {
            "alpha_count": int(self.alpha_count.item()),
            "alpha_cap":   self.alpha_cap,
            "alpha_fill":  float(self.alpha_count.item()) / self.alpha_cap,
            "beta_count":  int(self.beta_count.item()),
            "beta_cap":    self.beta_cap,
            "beta_fill":   float(self.beta_count.item()) / self.beta_cap,
        }

    @torch.no_grad()
    def curvature_statistics(self) -> dict:
        """
        Compute curvature distribution stats. The Apollonian prediction is that
        the curvatures should follow a power law with exponent ≈ 1.305.
        """
        alpha_n = int(self.alpha_count.item())
        beta_n  = int(self.beta_count.item())
        stats = {}
        if alpha_n > 0:
            ac = self.alpha_curvature[:alpha_n]
            stats["alpha_curv_mean"] = float(ac.mean().item())
            stats["alpha_curv_std"]  = float(ac.std().item())
            stats["alpha_curv_min"]  = float(ac.min().item())
            stats["alpha_curv_max"]  = float(ac.max().item())
        if beta_n > 0:
            bc = self.beta_curvature[:beta_n]
            stats["beta_curv_mean"] = float(bc.mean().item())
            stats["beta_curv_std"]  = float(bc.std().item())
            stats["beta_curv_min"]  = float(bc.min().item())
            stats["beta_curv_max"]  = float(bc.max().item())
        return stats

    @torch.no_grad()
    def estimate_power_law_exponent(self, pack: str = "alpha") -> float:
        """
        MLE estimate of the curvature distribution's power-law exponent.

        The Apollonian prediction is that the exponent ≈ 1.305 ± 0.05 once
        the buffer is full and the model has converged. Significant deviation
        indicates the brain-as-Apollonian-packing hypothesis is FALSIFIED for
        this dataset / training regime.

        Returns NaN if the pack has fewer than 50 entries.
        """
        if pack == "alpha":
            n = int(self.alpha_count.item())
            curvs = self.alpha_curvature[:n].float() if n > 0 else None
        else:
            n = int(self.beta_count.item())
            curvs = self.beta_curvature[:n].float() if n > 0 else None
        if curvs is None or n < 50:
            return float("nan")

        # MLE estimator for a continuous power-law (Clauset-Shalizi-Newman 2009)
        x_min = float(curvs.min().item()) + 1e-8
        x = curvs[curvs >= x_min]
        if x.numel() < 10:
            return float("nan")
        alpha_hat = 1.0 + x.numel() / (torch.log(x / x_min).sum().item() + 1e-8)
        return float(alpha_hat)

    # -------------------------------------------------------------------------
    # N3 — SleepGate memory consolidation (arXiv:2603.14517)
    #
    # Periodic "micro-sleep" that:
    #   1. Clusters similar α entries (cosine > merge_threshold)
    #   2. Merges clusters → averaged embeddings + max curvature
    #   3. Evicts stale entries (age > staleness_horizon)
    #   4. Compacts the buffer to remove gaps
    #
    # No extra loss — purely structural change to memory quality.
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def sleep_consolidate(
        self,
        merge_threshold: float = 0.92,
        staleness_horizon: int = 200,
    ) -> dict:
        """
        Run one SleepGate consolidation cycle on the α pack.

        Args:
            merge_threshold:   cosine similarity above which two α entries
                               are merged into one (average embedding, max curvature).
            staleness_horizon: entries older than (global_step - horizon) are evicted.

        Returns:
            dict with stats: n_merged, n_evicted, n_before, n_after
        """
        n_alpha = int(self.alpha_count.item())
        if n_alpha < 4:
            return {"n_merged": 0, "n_evicted": 0, "n_before": n_alpha, "n_after": n_alpha}

        step = int(self.global_step.item())
        embs = self.alpha_emb[:n_alpha].clone()       # (n, dim)
        curvs = self.alpha_curvature[:n_alpha].clone() # (n,)
        ages = self.alpha_age[:n_alpha].clone()         # (n,)

        # --- Phase 1: Evict stale entries ---
        age_cutoff = max(step - staleness_horizon, 0)
        fresh_mask = ages >= age_cutoff
        n_evicted = int((~fresh_mask).sum().item())
        embs = embs[fresh_mask]
        curvs = curvs[fresh_mask]
        ages = ages[fresh_mask]

        # --- Phase 2: Merge similar entries ---
        n_merged = 0
        if embs.shape[0] >= 2:
            norms = F.normalize(embs, dim=-1)
            # Compute pairwise cosine similarity
            sim = norms @ norms.T  # (n, n)
            # Greedy merge: find pairs above threshold, merge into first
            merged_into = list(range(embs.shape[0]))  # union-find root
            for i in range(embs.shape[0]):
                if merged_into[i] != i:
                    continue  # already merged
                for j in range(i + 1, embs.shape[0]):
                    if merged_into[j] != j:
                        continue
                    if sim[i, j] > merge_threshold:
                        # Merge j into i: average embeddings, take max curvature
                        embs[i] = (embs[i] + embs[j]) / 2.0
                        curvs[i] = max(curvs[i].item(), curvs[j].item())
                        ages[i] = max(ages[i].item(), ages[j].item())
                        merged_into[j] = i
                        n_merged += 1

            # Compact: keep only roots
            keep = torch.tensor([merged_into[i] == i for i in range(len(merged_into))],
                                dtype=torch.bool, device=embs.device)
            embs = embs[keep]
            curvs = curvs[keep]
            ages = ages[keep]

        # --- Phase 3: Write back to buffer ---
        n_after = embs.shape[0]
        self.alpha_emb[:n_after] = embs
        self.alpha_curvature[:n_after] = curvs
        self.alpha_age[:n_after] = ages
        # Zero out vacated slots
        if n_after < self.alpha_cap:
            self.alpha_emb[n_after:].zero_()
            self.alpha_curvature[n_after:].zero_()
            self.alpha_age[n_after:].zero_()
        self.alpha_count.fill_(n_after)
        self.alpha_head.fill_(n_after % self.alpha_cap)

        return {
            "n_merged": n_merged,
            "n_evicted": n_evicted,
            "n_before": n_alpha,
            "n_after": n_after,
        }

    @torch.no_grad()
    def reset(self) -> None:
        """Wipe both packs (used between runs / for tests)."""
        self.alpha_emb.zero_()
        self.alpha_curvature.zero_()
        self.alpha_age.zero_()
        self.alpha_count.zero_()
        self.alpha_head.zero_()
        self.beta_emb.zero_()
        self.beta_curvature.zero_()
        self.beta_age.zero_()
        self.beta_count.zero_()
        self.beta_head.zero_()
        self.global_step.zero_()
