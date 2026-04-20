"""
ArtificialHippocampusNetwork (AHN) — sliding short-term window + compressed
long-term memory gate for FANT 3.

Grounded in ByteDance "Artificial Hippocampus Networks" 2025, which demonstrates
that a sliding short-term window combined with compressed long-term memory
outperforms either mechanism alone.

Architecture
------------
1. Short-term sliding window
   Fixed-size FIFO of recent K/V states (size `short_window`). Full-fidelity
   attention over this window at every forward call.

2. Long-term compressed memory
   When the short-term window fills and a new token arrives, the oldest short-term
   K/V pair is compressed (via a linear compressor) and pushed into the long-term
   bank (size `long_capacity`). The long-term bank stores compact latent K/V pairs.
   When the long-term bank is also full, oldest latents are evicted (FIFO).

3. Gated read
   At each position the query attends to both memory stores; a 2-way softmax gate
   (conditioned on the query) controls the blend:

       out = alpha_short * attn(q, K_short, V_short)
           + alpha_long  * attn(q, K_long_dec, V_long_dec)

   where K_long_dec / V_long_dec are the latents projected back to full dim via
   the learned decompressor.

Buffer updates happen inside torch.no_grad() and are stored as nn.Module buffers
(not Parameters), so they:
  - Move with .to(device)
  - Save with state_dict()
  - Do NOT contribute to optimizer state
  - Do NOT generate gradients

References
----------
- ByteDance AHN 2025 (committed to MemPalace as lab_bytedance_paper_*)
- ApollonianMemory (fant2/model/apollonian.py) — circular buffer patterns
- MASAAttention (fant3/model/attention.py) — style / import conventions
"""

from __future__ import annotations
from typing import Dict

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ArtificialHippocampusNetwork(nn.Module):
    """
    Hippocampal memory module with sliding short-term window and compressed
    long-term memory, gated into the residual stream.

    Parameters
    ----------
    dim : int
        Model hidden dimension.
    n_heads : int
        Number of attention heads for both short- and long-term attention.
    short_window : int
        Maximum number of recent token K/V pairs kept at full fidelity.
    long_capacity : int
        Maximum number of compressed latent K/V pairs kept in long-term memory.
    compress_ratio : float
        Compression ratio; latent dim = int(dim * compress_ratio).
    """

    def __init__(
        self,
        dim: int,
        n_heads: int = 4,
        short_window: int = 256,
        long_capacity: int = 512,
        compress_ratio: float = 0.25,
    ):
        super().__init__()

        assert dim % n_heads == 0, f"dim ({dim}) must be divisible by n_heads ({n_heads})"
        assert 0.0 < compress_ratio <= 1.0, "compress_ratio must be in (0, 1]"

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.short_window = short_window
        self.long_capacity = long_capacity
        self.latent_dim = int(dim * compress_ratio)
        self.scale = math.sqrt(self.head_dim)

        # ── Learned projections ────────────────────────────────────────────────
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)

        # Compressor: full K or V → latent  (used when evicting from short term)
        self.compressor = nn.Linear(dim, self.latent_dim, bias=False)
        # Decompressor: latent → full dim  (used at attention time)
        self.decompressor = nn.Linear(self.latent_dim, dim, bias=False)

        # Output projection: dim → dim (like a standard attention out proj)
        self.out_proj = nn.Linear(dim, dim, bias=False)

        # Gate: given mean-pooled query → 2-way softmax over [short, long]
        self.gate_proj = nn.Linear(dim, 2, bias=True)

        # ── Non-parameter buffers (circular, per-batch) ────────────────────────
        # Short-term stores full-dim K and V
        self.register_buffer("short_K",   torch.zeros(short_window, dim),       persistent=False)
        self.register_buffer("short_V",   torch.zeros(short_window, dim),       persistent=False)
        # Long-term stores latent-dim K and V
        self.register_buffer("long_K",    torch.zeros(long_capacity, self.latent_dim), persistent=False)
        self.register_buffer("long_V",    torch.zeros(long_capacity, self.latent_dim), persistent=False)
        # Running write pointers and fill counts
        self.register_buffer("short_ptr",  torch.tensor(0, dtype=torch.long),   persistent=False)
        self.register_buffer("short_fill", torch.tensor(0, dtype=torch.long),   persistent=False)
        self.register_buffer("long_ptr",   torch.tensor(0, dtype=torch.long),   persistent=False)
        self.register_buffer("long_fill",  torch.tensor(0, dtype=torch.long),   persistent=False)

        # Weight initialisation
        self._reset_parameters()

    # ─────────────────────────────────────────────────────────────────────────
    #  Init
    # ─────────────────────────────────────────────────────────────────────────

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        # Compressor / decompressor: orthogonal init preserves norm
        nn.init.orthogonal_(self.compressor.weight)
        nn.init.orthogonal_(self.decompressor.weight)
        # Gate: start neutral (equal weight between short and long)
        nn.init.zeros_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

    # ─────────────────────────────────────────────────────────────────────────
    #  Buffer helpers (all no_grad)
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _push_to_short(self, k_tok: Tensor, v_tok: Tensor) -> None:
        """
        Push one token's K/V into the short-term window.
        k_tok, v_tok: (dim,)

        If the window is full, the oldest entry is first compressed into the
        long-term bank before being overwritten.
        """
        ptr  = int(self.short_ptr.item())
        fill = int(self.short_fill.item())
        cap  = self.short_window

        if fill == cap:
            # Evict the oldest slot (the one we are about to overwrite) into long-term.
            # oldest slot == ptr (FIFO circular buffer: ptr always points to the oldest)
            old_k = self.short_K[ptr]  # (dim,)
            old_v = self.short_V[ptr]  # (dim,)
            self._push_to_long(old_k, old_v)

        # Write new token into the current slot
        self.short_K[ptr] = k_tok
        self.short_V[ptr] = v_tok

        # Advance pointer
        new_ptr = (ptr + 1) % cap
        self.short_ptr.fill_(new_ptr)
        if fill < cap:
            self.short_fill.fill_(fill + 1)
        # (fill stays at cap when window is full — it just wraps)

    @torch.no_grad()
    def _push_to_long(self, k_tok: Tensor, v_tok: Tensor) -> None:
        """
        Compress one full-dim K/V pair and write it into the long-term bank.
        k_tok, v_tok: (dim,)
        """
        lk = self.compressor(k_tok)   # (latent_dim,)
        lv = self.compressor(v_tok)   # (latent_dim,)

        ptr  = int(self.long_ptr.item())
        fill = int(self.long_fill.item())
        cap  = self.long_capacity

        self.long_K[ptr] = lk
        self.long_V[ptr] = lv

        new_ptr = (ptr + 1) % cap
        self.long_ptr.fill_(new_ptr)
        new_fill = min(fill + 1, cap)
        self.long_fill.fill_(new_fill)

    # ─────────────────────────────────────────────────────────────────────────
    #  Attention helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _multihead_attn(
        self,
        q: Tensor,   # (B, T, dim)
        K: Tensor,   # (S, dim)
        V: Tensor,   # (S, dim)
    ) -> Tensor:
        """
        Multi-head attention of queries against a fixed memory bank.

        q: (B, T, dim)
        K: (S, dim) — S = number of filled slots
        V: (S, dim)
        Returns: (B, T, dim)
        """
        B, T, D = q.shape
        S = K.shape[0]
        H = self.n_heads
        Hd = self.head_dim

        # Reshape Q into heads: (B, H, T, Hd)
        q_h = q.view(B, T, H, Hd).transpose(1, 2)

        # Reshape K, V into heads: (H, S, Hd)
        K_h = K.view(S, H, Hd).transpose(0, 1)  # (H, S, Hd)
        V_h = V.view(S, H, Hd).transpose(0, 1)  # (H, S, Hd)

        # Broadcast K, V across batch
        K_h = K_h.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, S, Hd)
        V_h = V_h.unsqueeze(0).expand(B, -1, -1, -1)

        # Scaled dot-product attention (no causal mask — memory is fully visible)
        out = F.scaled_dot_product_attention(q_h, K_h, V_h)  # (B, H, T, Hd)

        # Merge heads
        out = out.transpose(1, 2).reshape(B, T, D)  # (B, T, dim)
        return out

    # ─────────────────────────────────────────────────────────────────────────
    #  Forward
    # ─────────────────────────────────────────────────────────────────────────

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, dim)
        Returns: (B, T, dim)

        Side effects: updates short_K/V and long_K/V buffers (no_grad).

        Gradient design
        ---------------
        k_proj / v_proj:  current-token K/V are used directly for short-term
            self-attention (the current T tokens attend to each other in the
            short-term bank PLUS the buffered history).  This keeps gradients
            flowing through k_proj / v_proj on every forward pass.

        compressor / decompressor: current-token latents (compress(k), compress(v))
            are concatenated onto the buffered long-term bank before attention.
            This ensures compressor always has a gradient path, and decompressor
            always gets a gradient through the long-term attention output.

        Buffer writes (to the FIFO circular buffers) happen in no_grad and use
        detached tensors, so they do not pollute the autograd graph.
        """
        B, T, D = x.shape
        assert D == self.dim, f"input dim {D} != AHN dim {self.dim}"

        # ── Project Q, K, V from current input ─────────────────────────────
        q = self.q_proj(x)   # (B, T, dim)  — used for attention + gate
        k = self.k_proj(x)   # (B, T, dim)  — used for attn AND buffer writes
        v = self.v_proj(x)   # (B, T, dim)

        # ── Compute gate weights from query (mean over T, per batch) ────────
        gate_logits = self.gate_proj(q.mean(dim=1))  # (B, 2)
        gate = torch.softmax(gate_logits, dim=-1)    # (B, 2)
        alpha_short = gate[:, 0].view(B, 1, 1)       # (B, 1, 1)
        alpha_long  = gate[:, 1].view(B, 1, 1)

        # ── Differentiable compress path for current tokens ──────────────────
        # compress k/v for the current batch element so compressor always has
        # a gradient, even before the long-term buffer has any history.
        k_lat = self.compressor(k)   # (B, T, latent_dim)
        v_lat = self.compressor(v)   # (B, T, latent_dim)

        # ── Update buffers with current tokens (no_grad) ────────────────────
        # Use batch index 0 for the shared (stateful) buffers.
        # API compromise: multi-batch calls share one buffer — designed for
        # online / single-sequence use (same as ApollonianMemory).
        with torch.no_grad():
            k_det = k[0].detach()  # (T, dim)
            v_det = v[0].detach()  # (T, dim)
            for t in range(T):
                self._push_to_short(k_det[t], v_det[t])

        # ── Short-term attention ─────────────────────────────────────────────
        # Combine buffered short-term history with the CURRENT token projections
        # so that k_proj / v_proj always carry a gradient.
        sf  = int(self.short_fill.item())
        ptr = int(self.short_ptr.item())
        cap = self.short_window

        # Current tokens as additional short-term keys/values (batch 0)
        cur_K = k[0]  # (T, dim)  — differentiable
        cur_V = v[0]  # (T, dim)

        if sf > 0:
            if sf < cap:
                hist_K = self.short_K[:sf]   # (sf, dim) — buffer (no grad)
                hist_V = self.short_V[:sf]
            else:
                idx = (torch.arange(cap, device=x.device) + ptr) % cap
                hist_K = self.short_K[idx]
                hist_V = self.short_V[idx]
            # Concat buffered history with current tokens for full short-term bank
            K_short = torch.cat([hist_K, cur_K], dim=0)   # (sf + T, dim)
            V_short = torch.cat([hist_V, cur_V], dim=0)
        else:
            K_short = cur_K   # first forward — only current tokens
            V_short = cur_V

        out_short = self._multihead_attn(q, K_short, V_short)  # (B, T, dim)

        # ── Long-term attention ──────────────────────────────────────────────
        # Concat buffered latents with current-token latents (differentiable)
        # so that compressor/decompressor always carry a gradient.
        lf   = int(self.long_fill.item())
        lptr = int(self.long_ptr.item())
        lcap = self.long_capacity

        # Current token latents (differentiable path for compressor grad)
        cur_lK = k_lat[0]   # (T, latent_dim)
        cur_lV = v_lat[0]

        if lf > 0:
            if lf < lcap:
                hist_lK = self.long_K[:lf]   # (lf, latent_dim) — no grad
                hist_lV = self.long_V[:lf]
            else:
                lidx = (torch.arange(lcap, device=x.device) + lptr) % lcap
                hist_lK = self.long_K[lidx]
                hist_lV = self.long_V[lidx]
            all_lK = torch.cat([hist_lK, cur_lK], dim=0)   # (lf + T, latent_dim)
            all_lV = torch.cat([hist_lV, cur_lV], dim=0)
        else:
            all_lK = cur_lK   # first forward — only current tokens compressed
            all_lV = cur_lV

        # Decompress latents → full dim (differentiable through decompressor)
        lK = self.decompressor(all_lK)   # (lf + T, dim)
        lV = self.decompressor(all_lV)
        out_long = self._multihead_attn(q, lK, lV)   # (B, T, dim)

        # ── Gated combination ────────────────────────────────────────────────
        combined = alpha_short * out_short + alpha_long * out_long  # (B, T, dim)

        # ── Output projection ────────────────────────────────────────────────
        out = self.out_proj(combined)   # (B, T, dim)
        return out

    # ─────────────────────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def reset_memory(self) -> None:
        """Clear both short-term and long-term buffers (call between sequences or at eval)."""
        self.short_K.zero_()
        self.short_V.zero_()
        self.long_K.zero_()
        self.long_V.zero_()
        self.short_ptr.zero_()
        self.short_fill.zero_()
        self.long_ptr.zero_()
        self.long_fill.zero_()

    @torch.no_grad()
    def get_stats(self) -> Dict[str, float]:
        """
        Diagnostic stats about memory fill and current gate weights.

        Returns
        -------
        dict with keys:
            short_fill   : fraction of short window occupied  [0, 1]
            long_fill    : fraction of long bank occupied     [0, 1]
            gate_short   : expected gate weight for short-term (averaged over dummy query)
            gate_long    : expected gate weight for long-term
        """
        sf = float(self.short_fill.item()) / self.short_window
        lf = float(self.long_fill.item()) / self.long_capacity

        # Estimate gate with a zero query (reflects gate_proj bias).
        # Match dtype of gate_proj so this helper works under bf16 too.
        dummy_q = torch.zeros(
            1, self.dim,
            device=self.gate_proj.weight.device,
            dtype=self.gate_proj.weight.dtype,
        )
        gate_logits = self.gate_proj(dummy_q)           # (1, 2)
        gate = torch.softmax(gate_logits, dim=-1)[0]    # (2,)
        gs = float(gate[0].item())
        gl = float(gate[1].item())

        return {
            "short_fill":  sf,
            "long_fill":   lf,
            "gate_short":  gs,
            "gate_long":   gl,
        }
