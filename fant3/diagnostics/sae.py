"""
ApollonianSAE — TopK sparse autoencoder for introspecting FANT's Apollonian
memory and hidden states.

Grounded in:
  - Anthropic "Scaling Monosemanticity" (Templeton et al. 2024):
      monosemanticity through dictionary learning on residual stream activations.
  - OpenAI "Scaling and evaluating sparse autoencoders" (Gao et al. 2024):
      TopK activation function + auxiliary "dead-feature" loss.

Design choices:
  - TopK(ReLU(·)) encoder (Gao et al.) rather than ReLU + L1 penalty:
      avoids the shrinkage bias of L1, gives exact sparsity control.
  - Auxiliary loss targets dead latents by penalizing their *pre-topk*
      activations, pushing them toward the reconstruction residual.
  - Decoder columns are unit-norm (normalize_decoder=True) so the norm of
      feature activations has a consistent geometric interpretation — the
      "feature amplitude" is purely in the scalar, not folded into the direction.
  - No bias in the encoder output (b_enc shifts the pre-activation;
      b_dec is the input centring offset, following Anthropic's implementation).

This module is NOT imported during training.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ─────────────────────────────────────────────────────────────────────────────
#  Core SAE
# ─────────────────────────────────────────────────────────────────────────────

class ApollonianSAE(nn.Module):
    """
    TopK sparse autoencoder for hidden-state introspection.

    Training objective:
        L = ||x - decode(encode(x))||^2 + lam * L_aux

    where:
        encode(x)  = TopK(ReLU(W_enc @ (x - b_dec) + b_enc), k)
        decode(z)  = W_dec @ z + b_dec

    L_aux is the dead-feature auxiliary loss from Gao et al. 2024:
        Let r = x - decode(z) be the reconstruction residual.
        Let z_pre = ReLU(W_enc @ (x - b_dec) + b_enc) be the pre-topk activations.
        L_aux = ||r - decode(z_dead_topk)||^2
    where z_dead_topk is a TopK mask applied to z_pre restricted to dead latents.
    This encourages dead latents to reconstruct the current residual, reviving them.

    Args:
        d_in:              dimension of the input hidden states.
        n_features:        dictionary size (number of latent features).
                           Typically 4–16× d_in.
        k:                 TopK sparsity: at most k features are nonzero per token.
        lam:               auxiliary-loss coefficient (weight of L_aux).
        dead_threshold:    fraction of batches a feature must miss before
                           it is counted as dead for the L_aux mask.
        normalize_decoder: if True, decoder column norms are constrained to 1
                           after each gradient step (call .normalize_decoder_weights()).
    """

    def __init__(
        self,
        d_in: int,
        n_features: int,
        k: int = 32,
        lam: float = 1e-2,
        dead_threshold: float = 0.01,
        normalize_decoder: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.n_features = n_features
        self.k = k
        self.lam = lam
        self.dead_threshold = dead_threshold
        self.normalize_decoder = normalize_decoder

        # Encoder: W_enc (n_features × d_in), bias b_enc (n_features,)
        self.W_enc = nn.Parameter(torch.empty(n_features, d_in))
        self.b_enc = nn.Parameter(torch.zeros(n_features))

        # Decoder: W_dec (d_in × n_features), bias b_dec (d_in,)
        # b_dec also serves as the input centring offset.
        self.W_dec = nn.Parameter(torch.empty(d_in, n_features))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        # Dead-feature tracking (not a Parameter — just state)
        # fires_since_last_check[i] counts how many batches feature i was nonzero.
        self.register_buffer(
            "_fires_since_reset", torch.zeros(n_features, dtype=torch.float32)
        )
        self.register_buffer(
            "_batches_since_reset", torch.tensor(0, dtype=torch.long)
        )

        self._init_weights()

    # -------------------------------------------------------------------------
    #  Initialisation
    # -------------------------------------------------------------------------

    def _init_weights(self) -> None:
        """
        Encoder initialised with Kaiming uniform (fan-out = n_features, fan-in = d_in).
        Decoder initialised as the transpose of the encoder then column-normalised.
        This gives a warm start where the encoder and decoder are near-inverses.
        """
        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        # Initialise W_dec as the transpose of W_enc (d_in × n_features)
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T.clone())
            if self.normalize_decoder:
                self._normalize_decoder_inplace()

    @torch.no_grad()
    def _normalize_decoder_inplace(self) -> None:
        """Renormalise each column of W_dec to unit L2 norm."""
        col_norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.div_(col_norms)

    def normalize_decoder_weights(self) -> None:
        """
        Public API — call after each optimiser step to enforce unit-norm columns.

        Usage::

            loss.backward()
            optimizer.step()
            sae.normalize_decoder_weights()
        """
        if self.normalize_decoder:
            self._normalize_decoder_inplace()

    # -------------------------------------------------------------------------
    #  Dead-feature tracking
    # -------------------------------------------------------------------------

    @torch.no_grad()
    def _update_dead_tracking(self, z: Tensor) -> None:
        """
        Update firing statistics.

        z: [..., n_features] sparse feature activations (after TopK mask).
        """
        fired = (z.detach().abs() > 0).float().reshape(-1, self.n_features).any(dim=0)
        self._fires_since_reset.add_(fired)
        self._batches_since_reset.add_(1)

    @torch.no_grad()
    def dead_feature_mask(self) -> Tensor:
        """
        Return a boolean mask (n_features,) — True for features considered dead.

        A feature is dead if it fired in fewer than `dead_threshold` fraction
        of batches since the last reset.
        """
        n_batches = max(int(self._batches_since_reset.item()), 1)
        fire_rate = self._fires_since_reset / n_batches
        return fire_rate < self.dead_threshold

    @torch.no_grad()
    def dead_feature_fraction(self) -> float:
        """Fraction of features currently classified as dead."""
        return float(self.dead_feature_mask().float().mean().item())

    def reset_dead_tracking(self) -> None:
        """Reset firing counters (call at the start of each training pass)."""
        self._fires_since_reset.zero_()
        self._batches_since_reset.zero_()

    # -------------------------------------------------------------------------
    #  Encode / Decode
    # -------------------------------------------------------------------------

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode hidden states to sparse feature activations.

        Args:
            x: [..., d_in]

        Returns:
            z: [..., n_features] with at most k nonzero entries per token.
        """
        # Centre the input (b_dec plays the role of the input mean offset)
        x_cent = x - self.b_dec                         # [..., d_in]
        # Pre-activation: linear + bias + ReLU
        pre = F.relu(x_cent @ self.W_enc.T + self.b_enc)  # [..., n_features]
        # TopK masking — zero out all but the top-k activations
        z = self._topk_mask(pre, self.k)                # [..., n_features]
        return z

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode sparse feature activations back to hidden-state space.

        Args:
            z: [..., n_features]

        Returns:
            xhat: [..., d_in]
        """
        return z @ self.W_dec.T + self.b_dec            # [..., d_in]

    @staticmethod
    def _topk_mask(pre: Tensor, k: int) -> Tensor:
        """
        Sparse TopK: keep the top-k values per last dimension, zero the rest.

        Shape: [..., n_features] → [..., n_features]
        """
        if k >= pre.shape[-1]:
            return pre  # all features active — no masking needed
        flat = pre.reshape(-1, pre.shape[-1])            # (N, n_features)
        topk_vals, topk_idx = flat.topk(k, dim=-1, sorted=False)
        out = torch.zeros_like(flat)
        out.scatter_(-1, topk_idx, topk_vals)
        return out.reshape(pre.shape)

    # -------------------------------------------------------------------------
    #  Forward (compute losses)
    # -------------------------------------------------------------------------

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Full forward pass with loss computation.

        Args:
            x: [..., d_in] hidden states.

        Returns a dict with:
            'reconstruction': xhat [..., d_in]
            'features':       z    [..., n_features]  (sparse)
            'loss':           scalar total loss  L = L_rec + lam * L_aux
            'l0':             scalar mean L0 sparsity (actual nonzero count per token)
            'l_rec':          scalar reconstruction loss (MSE)
            'l_aux':          scalar auxiliary dead-feature loss
        """
        x_cent = x - self.b_dec                                     # [..., d_in]
        pre = F.relu(x_cent @ self.W_enc.T + self.b_enc)            # [..., n_features]

        # TopK sparse features
        z = self._topk_mask(pre, self.k)                            # [..., n_features]

        # Reconstruction
        xhat = z @ self.W_dec.T + self.b_dec                        # [..., d_in]

        # L2 reconstruction loss (mean over tokens and dims)
        l_rec = F.mse_loss(xhat, x)

        # Dead-feature auxiliary loss (Gao et al. 2024 §3.2)
        l_aux = self._auxiliary_loss(x, xhat, pre)

        loss = l_rec + self.lam * l_aux

        # L0 sparsity: mean number of nonzero features per token
        with torch.no_grad():
            l0 = (z.abs() > 0).float().reshape(-1, self.n_features).sum(dim=-1).mean()
            self._update_dead_tracking(z)

        return {
            "reconstruction": xhat,
            "features":       z,
            "loss":           loss,
            "l0":             l0,
            "l_rec":          l_rec,
            "l_aux":          l_aux,
        }

    def _auxiliary_loss(self, x: Tensor, xhat: Tensor, pre: Tensor) -> Tensor:
        """
        Auxiliary dead-feature loss.

        Apply a TopK mask restricted to dead latents and penalise its failure
        to reconstruct the current residual:
            r = x - xhat  (stop-gradient)
            z_dead = TopK(pre * dead_mask, k_aux)
            L_aux = ||r - decode(z_dead)||^2

        k_aux = k so the dead features get the same budget as the live ones.
        If there are no dead features, L_aux = 0.
        """
        with torch.no_grad():
            dead = self.dead_feature_mask()                          # (n_features,) bool
        if not dead.any():
            return torch.tensor(0.0, device=x.device, dtype=x.dtype)

        r = (x - xhat).detach()                                      # residual, no grad

        # Mask pre-activations to dead features only
        pre_dead = pre * dead.float()                                # [..., n_features]

        # TopK over dead features (budget = k, capped at n_dead)
        n_dead = int(dead.sum().item())
        k_aux = min(self.k, n_dead)
        z_dead = self._topk_mask(pre_dead, k_aux)                   # [..., n_features]

        # Decode only via the dead-feature columns
        xhat_dead = z_dead @ self.W_dec.T                           # [..., d_in] (no b_dec — residual target has it removed)
        l_aux = F.mse_loss(xhat_dead, r)
        return l_aux


# ─────────────────────────────────────────────────────────────────────────────
#  Training helper
# ─────────────────────────────────────────────────────────────────────────────

def train_on_hidden_states(
    sae: ApollonianSAE,
    hidden_states: Tensor,
    n_epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
) -> Dict[str, List[float]]:
    """
    Train the SAE on a collected tensor of hidden states (offline, no FANT model needed).

    Args:
        sae:           an ApollonianSAE instance (will be set to train mode).
        hidden_states: (N, d_in) tensor of hidden states from FANT forward passes.
        n_epochs:      number of full passes over the data.
        batch_size:    mini-batch size.
        lr:            learning rate for AdamW.

    Returns:
        dict with keys:
            'losses':           per-batch total loss values (float).
            'l0_sparsity':      per-batch mean L0 sparsity.
            'dead_feature_frac': per-epoch dead-feature fraction.
    """
    assert hidden_states.ndim == 2, "hidden_states must be (N, d_in)"
    N, d_in = hidden_states.shape
    assert d_in == sae.d_in, f"hidden_states d_in={d_in} != sae.d_in={sae.d_in}"

    device = next(sae.parameters()).device
    hidden_states = hidden_states.to(device)

    optimizer = torch.optim.AdamW(sae.parameters(), lr=lr, weight_decay=0.0)

    history: Dict[str, List[float]] = {
        "losses":            [],
        "l0_sparsity":       [],
        "dead_feature_frac": [],
    }

    sae.train()
    sae.reset_dead_tracking()

    for epoch in range(n_epochs):
        # Shuffle
        perm = torch.randperm(N, device=device)
        shuffled = hidden_states[perm]

        for start in range(0, N, batch_size):
            batch = shuffled[start : start + batch_size]

            optimizer.zero_grad(set_to_none=True)
            out = sae(batch)
            out["loss"].backward()
            optimizer.step()

            # Enforce unit-norm decoder columns after each step
            sae.normalize_decoder_weights()

            history["losses"].append(float(out["loss"].item()))
            history["l0_sparsity"].append(float(out["l0"].item()))

        dead_frac = sae.dead_feature_fraction()
        history["dead_feature_frac"].append(dead_frac)

    sae.eval()
    return history


# ─────────────────────────────────────────────────────────────────────────────
#  Apollonian-memory introspection
# ─────────────────────────────────────────────────────────────────────────────

def analyze_apollonian_memory(
    sae: ApollonianSAE,
    memory: Any,
    top_n_features: int = 20,
) -> Dict[str, Any]:
    """
    Apply the trained SAE to every stored embedding in both α and β packs.

    Accepts either:
      - fant2.model.apollonian.ApollonianMemory  (buffers: alpha_emb/alpha_count/beta_emb/beta_count)
      - SpinorApollonianMemory                   (same buffer names + chirality buffer)
      - Any nn.Module with tensor attributes     alpha_bank and beta_bank  (for mocks / tests)

    Computes:
      feature_activation_histograms:
          For each pack, a list of n_features mean activation values —
          i.e. how strongly each SAE feature fires on average across that pack.
      chirality_correlation (if SpinorApollonianMemory):
          Pearson correlation between each feature's mean activation and
          a binary chirality label (+1 / -1).  Returns None if memory has
          no chirality buffer.
      top_discriminating_features:
          The top_n_features feature indices that MOST distinguish α from β,
          ranked by absolute difference in mean activations (a chi-square proxy).
      ghost_features:
          Indices of features that never fired on EITHER pack (dead for this memory).
      pack_sizes:
          {'alpha': int, 'beta': int}  — how many embeddings were analysed.

    Returns a dict with Python primitives (no tensors).
    """
    sae.eval()
    device = next(sae.parameters()).device

    # ------------------------------------------------------------------
    # Resolve memory layout: supports two naming conventions
    # ------------------------------------------------------------------
    alpha_emb, beta_emb = _extract_memory_banks(memory, device, sae.d_in)

    n_alpha = alpha_emb.shape[0]
    n_beta  = beta_emb.shape[0]

    # ------------------------------------------------------------------
    # Encode both packs
    # ------------------------------------------------------------------
    with torch.no_grad():
        z_alpha = sae.encode(alpha_emb) if n_alpha > 0 else torch.zeros(0, sae.n_features, device=device)
        z_beta  = sae.encode(beta_emb)  if n_beta  > 0 else torch.zeros(0, sae.n_features, device=device)

    # ------------------------------------------------------------------
    # Mean activations per feature (histogram proxy)
    # ------------------------------------------------------------------
    mean_alpha = z_alpha.mean(dim=0).tolist() if n_alpha > 0 else [0.0] * sae.n_features
    mean_beta  = z_beta.mean(dim=0).tolist()  if n_beta  > 0 else [0.0] * sae.n_features

    # ------------------------------------------------------------------
    # Ghost features: never fire on either pack
    # ------------------------------------------------------------------
    fired_alpha = (z_alpha.abs() > 0).any(dim=0) if n_alpha > 0 else torch.zeros(sae.n_features, dtype=torch.bool, device=device)
    fired_beta  = (z_beta.abs()  > 0).any(dim=0) if n_beta  > 0 else torch.zeros(sae.n_features, dtype=torch.bool, device=device)
    ghost_mask  = ~(fired_alpha | fired_beta)
    ghost_features: List[int] = ghost_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    if isinstance(ghost_features, int):
        ghost_features = [ghost_features]

    # ------------------------------------------------------------------
    # Top discriminating features (by absolute difference in mean activation)
    # ------------------------------------------------------------------
    diff = torch.tensor(mean_alpha, device=device) - torch.tensor(mean_beta, device=device)
    abs_diff = diff.abs()
    n_candidates = min(top_n_features, sae.n_features)
    top_idx = abs_diff.topk(n_candidates, sorted=True).indices.tolist()

    top_discriminating: List[Dict[str, Any]] = []
    for idx in top_idx:
        top_discriminating.append({
            "feature_index":  idx,
            "mean_alpha":     float(mean_alpha[idx]),
            "mean_beta":      float(mean_beta[idx]),
            "abs_difference": float(abs_diff[idx].item()),
            "prefers":        "alpha" if diff[idx].item() > 0 else "beta",
        })

    # ------------------------------------------------------------------
    # Chirality correlation (SpinorApollonianMemory only)
    # ------------------------------------------------------------------
    chirality_corr: Optional[List[float]] = None
    chirality_buf = getattr(memory, "chirality", None)
    if chirality_buf is not None and isinstance(chirality_buf, Tensor):
        # chirality buffer should be (cap,) with +1/-1 values
        n_alpha_live = min(n_alpha, chirality_buf.shape[0])
        if n_alpha_live > 0:
            chir = chirality_buf[:n_alpha_live].float().to(device)    # (n,)
            # Pearson correlation between chirality and each feature column
            z_a = z_alpha[:n_alpha_live].float()                      # (n, F)
            chir_c = chir - chir.mean()
            z_c    = z_a - z_a.mean(dim=0, keepdim=True)
            denom  = (chir_c.norm() * z_c.norm(dim=0)).clamp(min=1e-8)
            corr   = (chir_c @ z_c) / denom                           # (F,)
            chirality_corr = corr.tolist()

    # ------------------------------------------------------------------
    # Assemble result
    # ------------------------------------------------------------------
    return {
        "pack_sizes": {
            "alpha": n_alpha,
            "beta":  n_beta,
        },
        "feature_activation_histograms": {
            "alpha": mean_alpha,
            "beta":  mean_beta,
        },
        "top_discriminating_features": top_discriminating,
        "ghost_features":              ghost_features,
        "ghost_feature_count":         len(ghost_features),
        "ghost_feature_fraction":      len(ghost_features) / sae.n_features,
        "chirality_correlation":       chirality_corr,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_memory_banks(
    memory: Any,
    device: torch.device,
    d_in: int,
) -> tuple:
    """
    Normalise different memory object layouts into two tensors:
        (alpha_embeddings: [N_a, d_in], beta_embeddings: [N_b, d_in])

    Supported layouts:
      1. ApollonianMemory style:
            .alpha_emb     (cap, dim) buffer
            .alpha_count   scalar long buffer
            .beta_emb      (cap, dim) buffer
            .beta_count    scalar long buffer
      2. Mock / SpinorApollonianMemory with bank tensors:
            .alpha_bank    (N, dim) tensor  (already dense, no count needed)
            .beta_bank     (N, dim) tensor
    """
    # Layout 1: FANT buffers
    if hasattr(memory, "alpha_emb") and hasattr(memory, "alpha_count"):
        n_alpha = int(memory.alpha_count.item())
        n_beta  = int(memory.beta_count.item())
        alpha_emb = memory.alpha_emb[:n_alpha].float().to(device) if n_alpha > 0 else torch.zeros(0, d_in, device=device)
        beta_emb  = memory.beta_emb[:n_beta].float().to(device)   if n_beta  > 0 else torch.zeros(0, d_in, device=device)
        return alpha_emb, beta_emb

    # Layout 2: dense bank tensors (mock / SpinorApollonianMemory)
    if hasattr(memory, "alpha_bank") and hasattr(memory, "beta_bank"):
        alpha_emb = memory.alpha_bank.float().to(device)
        beta_emb  = memory.beta_bank.float().to(device)
        return alpha_emb, beta_emb

    raise AttributeError(
        "memory object must have either (alpha_emb, alpha_count, beta_emb, beta_count) "
        "or (alpha_bank, beta_bank) attributes. "
        f"Got: {[a for a in dir(memory) if not a.startswith('__')]}"
    )
