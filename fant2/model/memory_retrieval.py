"""
ApollonianRetrievalAttention — cross-attention over the α/β Apollonian memory.

This module is inserted at the apollonian_retrieval_layers (default: layers
10 and 11, the last two layers of a 12-layer FANT 2 model). It lets the model
*query* the Apollonian memory for relevant past instances and schemas, and
fold them back into the residual stream.

Architecture:
    x : (B, T, dim)              — current residual stream (the query source)
    memory : ApollonianMemory     — the dual α/β pack (the key/value source)

    1. Project x to a query (per-token, per-head)
    2. Retrieve top-k from α pack and top-k from β pack via cosine similarity
       (this is done by ApollonianMemory.retrieve())
    3. Concatenate the α and β retrievals into a (N, 2k, dim) memory tensor
    4. Project memory to keys and values (per-head)
    5. Cross-attention: out = softmax(Q K^T / √d) V
    6. Residual: returned as a delta to be added to x

The cross-attention is masked so that:
    - α retrievals contribute only when the query is in the "instance lookup" mode
    - β retrievals contribute only when the query is in the "schema lookup" mode
    - The split is decided by a small per-token gate (sigmoid)

Param count (default config, dim=768, n_heads=8, head_dim=96):
    Q proj      : 768 × 768   = 589 K
    K proj      : 768 × 768   = 589 K
    V proj      : 768 × 768   = 589 K
    O proj      : 768 × 768   = 589 K
    pack_gate   : 768 × 2     =  1.5K
    Total       : ~2.36 M per retrieval layer
    × 2 layers  : ~4.72 M

This sits inside the 60M parameter budget (the budget reserves ~5M for memory
retrieval per the spec §1).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .apollonian import ApollonianMemory


class ApollonianRetrievalAttention(nn.Module):
    """
    Cross-attention over the Apollonian α/β memory pack.

    Forward signature:
        forward(x, memory) -> (B, T, dim)
    """

    def __init__(
        self,
        dim: int = 768,
        n_heads: int = 8,
        head_dim: int = 96,
        n_retrieve_per_pack: int = 8,
        attention_dropout: float = 0.0,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.n_retrieve_per_pack = n_retrieve_per_pack
        self.attention_dropout = attention_dropout

        # ----- Q / K / V / O projections -----
        self.W_q = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.W_k = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.W_v = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.W_o = nn.Linear(n_heads * head_dim, dim, bias=False)

        for m in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.normal_(m.weight, std=init_std)

        # ----- α/β pack gate (per-token sigmoid) -----
        # Outputs a (B, T, 2) tensor with the (alpha_weight, beta_weight) gating
        # scalars. We multiply each retrieval by its gate before attention.
        self.pack_gate = nn.Linear(dim, 2, bias=True)
        nn.init.normal_(self.pack_gate.weight, std=init_std)
        nn.init.zeros_(self.pack_gate.bias)

        # ----- Output gate (per-token, learned residual scale) -----
        # The Apollonian retrieval is OFF by default (output_gate ≈ 0) and
        # turns on as the model learns to use it. This avoids destabilizing
        # the early training when the memory is empty.
        self.output_gate = nn.Parameter(torch.tensor(0.01))

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        memory: ApollonianMemory,
    ) -> torch.Tensor:
        """
        Args:
            x:      (B, T, dim) current residual stream
            memory: ApollonianMemory instance to query

        Returns:
            (B, T, dim) the cross-attention output (a residual delta — the caller
            is responsible for adding it to the input residual stream).
        """
        B, T, D = x.shape
        H, hd = self.n_heads, self.head_dim
        N = B * T
        x_flat = x.reshape(N, D)

        # ===== 1. Retrieve top-k from α and β packs =====
        # Each retrieval gives (N, k, dim) embeddings + (N, k) similarities.
        alpha_mem, alpha_sim = memory.retrieve(x_flat, pack="alpha", k=self.n_retrieve_per_pack)
        beta_mem,  beta_sim  = memory.retrieve(x_flat, pack="beta",  k=self.n_retrieve_per_pack)

        # ===== 2. Concatenate the two packs into (N, 2k, dim) =====
        mem = torch.cat([alpha_mem, beta_mem], dim=1)        # (N, 2k, dim)
        K_total = mem.shape[1]

        # ===== 3. Pack gating: per-token (alpha_weight, beta_weight) =====
        gate_logits = self.pack_gate(x_flat)                 # (N, 2)
        gate = torch.sigmoid(gate_logits)                    # (N, 2)
        alpha_w = gate[:, 0:1]                               # (N, 1)
        beta_w  = gate[:, 1:2]                               # (N, 1)
        # Per-key weight: alpha keys get alpha_w, beta keys get beta_w
        key_weights = torch.cat(
            [alpha_w.expand(-1, self.n_retrieve_per_pack),
             beta_w .expand(-1, self.n_retrieve_per_pack)],
            dim=1,
        )  # (N, 2k)

        # ===== 4. Project Q, K, V =====
        q = self.W_q(x_flat).view(N, H, hd)                  # (N, H, hd)
        k = self.W_k(mem).view(N, K_total, H, hd)            # (N, 2k, H, hd)
        v = self.W_v(mem).view(N, K_total, H, hd)            # (N, 2k, H, hd)

        # ===== 5. Cross-attention =====
        # Reshape for batched matmul: q -> (N, H, 1, hd), k -> (N, H, 2k, hd), v -> (N, H, 2k, hd)
        q = q.unsqueeze(2)                                   # (N, H, 1, hd)
        k = k.permute(0, 2, 1, 3)                            # (N, H, 2k, hd)
        v = v.permute(0, 2, 1, 3)                            # (N, H, 2k, hd)

        # Build attention mask: empty memory slots have similarity 0 (exact zero)
        # which we can detect and mask out.
        empty_mask = (alpha_sim.abs() + 1e-12 < 1e-8)        # (N, k) bool
        empty_mask_b = (beta_sim.abs() + 1e-12 < 1e-8)
        empty_full = torch.cat([empty_mask, empty_mask_b], dim=1)  # (N, 2k) — True = empty

        # Build the additive bias mask (-inf at empty slots, 0 elsewhere)
        attn_bias = torch.zeros(N, K_total, device=x.device, dtype=q.dtype)
        attn_bias.masked_fill_(empty_full, float("-inf"))
        # Add the log of the per-key weight (so the gate is multiplicative on the attention)
        attn_bias = attn_bias + torch.log(key_weights.clamp_min(1e-8))
        # Reshape for broadcasting: (N, 1, 1, 2k)
        attn_bias = attn_bias.unsqueeze(1).unsqueeze(2)

        # SDPA expects float-additive masks via attn_mask
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_bias,
            dropout_p=self.attention_dropout if self.training else 0.0,
        )  # (N, H, 1, hd)

        # ===== 6. Reshape and project back =====
        attn_out = attn_out.squeeze(2).reshape(N, H * hd)
        out = self.W_o(attn_out)                             # (N, dim)

        # ===== 7. Output gate (slow learning of "should we use memory at all") =====
        out = out * self.output_gate

        return out.view(B, T, D)

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def stored_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
