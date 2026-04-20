"""
TransformerBlock — one layer of FANT 2.

The block has the standard pre-norm structure:

    h := x + Attention(RMSNorm(x))
    h := h + FFN(RMSNorm(h))
    [optional] h := h + ApollonianRetrievalAttention(RMSNorm(h), memory)

The FFN is one of:
    - DenseSwiGLU            for the first n_dense_layers (DeepSeek V3 first_k_dense_replace)
    - FractalMoELayer        for all subsequent layers

The Apollonian retrieval is only inserted in the layers listed in
config.apollonian_retrieval_layers (default: the last two).

Gradient checkpointing is supported via the use_checkpoint flag — when on, the
attention and FFN sub-blocks are recomputed on the backward pass to save VRAM.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.utils.checkpoint as ckpt

from .norm import RMSNorm
from .hub_attention import HubAttention
from .moe import FractalMoELayer, GlobalCCorrection
from .experts import DenseSwiGLU
from .memory_retrieval import ApollonianRetrievalAttention
from .apollonian import ApollonianMemory
from .router import RouterOutput


class TransformerBlock(nn.Module):
    """
    One transformer layer of FANT 2.

    Holds (in this order):
      - pre_attn_norm  : RMSNorm before attention
      - attn           : HubAttention (GQA-2 + hubs + sinks + window)
      - pre_ffn_norm   : RMSNorm before FFN
      - ffn            : DenseSwiGLU OR FractalMoELayer (depending on `is_dense`)
      - pre_mem_norm   : RMSNorm before Apollonian retrieval (only if `use_memory`)
      - mem_attn       : ApollonianRetrievalAttention (only if `use_memory`)
    """

    def __init__(
        self,
        layer_idx: int,
        dim: int = 768,
        n_heads: int = 8,
        n_kv_heads: int = 2,
        head_dim: int = 96,
        n_hub_tokens: int = 32,
        hub_dim_mult: float = 2.0,
        local_window: int = 128,
        n_attention_sinks: int = 4,
        rope_partial: float = 0.25,
        max_seq_len: int = 1024,
        # FFN config
        is_dense: bool = False,
        moe_hidden: int = 1280,
        n_megapools: int = 8,
        n_per_megapool: int = 9,
        top_k: int = 4,
        kron_A_p: int = 40,
        kron_A_q: int = 8,
        kron_B_p: int = 32,
        kron_B_q: int = 32,
        shared_expert_hidden: int = 256,
        c_global: Optional[GlobalCCorrection] = None,
        # Memory config
        use_memory: bool = False,
        n_retrieve_per_pack: int = 8,
        # Misc
        rms_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        init_std: float = 0.02,
        use_checkpoint: bool = False,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.is_dense = is_dense
        self.use_memory = use_memory
        self.use_checkpoint = use_checkpoint

        # ----- Attention -----
        self.pre_attn_norm = RMSNorm(dim, eps=rms_eps)
        self.attn = HubAttention(
            dim=dim,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            head_dim=head_dim,
            n_hub_tokens=n_hub_tokens,
            hub_dim_mult=hub_dim_mult,
            local_window=local_window,
            n_attention_sinks=n_attention_sinks,
            rope_partial=rope_partial,
            max_seq_len=max_seq_len,
            attention_dropout=attention_dropout,
            init_std=init_std,
        )

        # ----- FFN (dense or MoE) -----
        self.pre_ffn_norm = RMSNorm(dim, eps=rms_eps)
        if is_dense:
            self.ffn = DenseSwiGLU(dim=dim, hidden=moe_hidden, init_std=init_std)
        else:
            self.ffn = FractalMoELayer(
                dim=dim,
                n_megapools=n_megapools,
                n_per_megapool=n_per_megapool,
                top_k=top_k,
                moe_hidden=moe_hidden,
                shared_expert_hidden=shared_expert_hidden,
                kron_A_p=kron_A_p,
                kron_A_q=kron_A_q,
                kron_B_p=kron_B_p,
                kron_B_q=kron_B_q,
                init_std=init_std,
                c_global=c_global,
            )

        # ----- Optional Apollonian memory retrieval -----
        if use_memory:
            self.pre_mem_norm = RMSNorm(dim, eps=rms_eps)
            self.mem_attn = ApollonianRetrievalAttention(
                dim=dim,
                n_heads=n_heads,
                head_dim=head_dim,
                n_retrieve_per_pack=n_retrieve_per_pack,
                attention_dropout=attention_dropout,
                init_std=init_std,
            )
        else:
            self.pre_mem_norm = None
            self.mem_attn = None

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        memory: Optional[ApollonianMemory] = None,
    ) -> Tuple[torch.Tensor, Optional[RouterOutput]]:
        """
        Args:
            x         : (B, T, dim) input
            freqs_cis : (max_seq_len, rope_dim/2) precomputed RoPE
            memory    : ApollonianMemory (only used if self.use_memory)

        Returns:
            (out, router_out) where router_out is None for dense layers.
        """
        if self.use_checkpoint and self.training:
            return self._forward_checkpointed(x, freqs_cis, memory)
        return self._forward_uncheckpointed(x, freqs_cis, memory)

    def _forward_uncheckpointed(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        memory: Optional[ApollonianMemory],
    ) -> Tuple[torch.Tensor, Optional[RouterOutput]]:
        # ----- Attention sub-block -----
        h = x + self.attn(self.pre_attn_norm(x), freqs_cis)

        # ----- FFN sub-block -----
        ffn_in = self.pre_ffn_norm(h)
        if self.is_dense:
            h = h + self.ffn(ffn_in)
            router_out: Optional[RouterOutput] = None
        else:
            ffn_out, router_out = self.ffn(ffn_in)
            h = h + ffn_out

        # ----- Optional memory retrieval sub-block -----
        if self.use_memory and memory is not None:
            h = h + self.mem_attn(self.pre_mem_norm(h), memory)

        return h, router_out

    def _forward_checkpointed(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        memory: Optional[ApollonianMemory],
    ) -> Tuple[torch.Tensor, Optional[RouterOutput]]:
        """
        Gradient-checkpointed variant.

        We checkpoint the attention sub-block separately from the FFN sub-block
        because the FFN MoE dispatch creates router output that we need access
        to AFTER the forward (for the bias update). Checkpointing the whole
        block would lose access to the router output.
        """
        # Attention sub-block (checkpointed)
        def attn_fn(x_in):
            return self.attn(self.pre_attn_norm(x_in), freqs_cis)

        h = x + ckpt.checkpoint(attn_fn, x, use_reentrant=False)

        # FFN sub-block (NOT checkpointed for MoE — we need router_out)
        ffn_in = self.pre_ffn_norm(h)
        if self.is_dense:
            def ffn_fn(ffn_in_):
                return self.ffn(ffn_in_)
            h = h + ckpt.checkpoint(ffn_fn, ffn_in, use_reentrant=False)
            router_out: Optional[RouterOutput] = None
        else:
            ffn_out, router_out = self.ffn(ffn_in)
            h = h + ffn_out

        # Memory sub-block
        if self.use_memory and memory is not None:
            def mem_fn(h_in):
                return self.mem_attn(self.pre_mem_norm(h_in), memory)
            h = h + ckpt.checkpoint(mem_fn, h, use_reentrant=False)

        return h, router_out

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def stored_param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
