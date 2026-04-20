"""
FANT3Model — top-level assembly of the FANT 3 architecture.

Layer plan (with `n_layers=24, n_dense_layers=3` defaults):
    layers 0..2     : DenseBlock (MASA attention + vanilla SwiGLU FFN)
    layers 3..20    : MoR-wrapped shared MoEBlock (MASA + Matryoshka MoE FFN),
                       each token recurses 1..n_recursion_depths times through
                       this single shared block — represents 18 logical layers
                       with the parameter cost of 1
    layers 21..23   : MoEBlock (MASA + Matryoshka MoE FFN), distinct each
    Apollonian retrieval mixed into final 2 layers' attention

Cerebellum: parallel-residual path, fixed dim (768) regardless of model dim.
Final norm + tied LM head (weight shared with token embedding).
"""

from __future__ import annotations
import math
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reused from fant2 — already validated
from fant2.model.norm import RMSNorm
from fant2.model.apollonian import ApollonianMemory
from fant2.model.cerebellum import CerebellumModule

from .attention import MASAAtomBank, MASAAttention
from .matryoshka_moe import MatryoshkaMoEFFN
from .recursion import MoRShared
from .etf import freeze_linear_to_etf
# Fix 2 (2026-04-19) — spinor-based α/β classifier (drop-in, opt-in via cfg)
from .spinor_apollonian import SpinorApollonianMemory
# Fix 4 (2026-04-19) — sliding + compressed long-term residual (opt-in via cfg)
from .ahn import ArtificialHippocampusNetwork


# ─────────────────────────────────────────────────────────────────────────────
#  Block primitives
# ─────────────────────────────────────────────────────────────────────────────

class DenseSwiGLU(nn.Module):
    """Vanilla SwiGLU FFN for the dense prefix."""
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.up = nn.Linear(dim, 2 * hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        gate, val = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(gate) * val)


class DenseBlock(nn.Module):
    """MASA attention + dense SwiGLU + RMSNorm pre-norm + residual."""
    def __init__(self, cfg, atom_bank: MASAAtomBank, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(cfg.dim, eps=cfg.rms_eps)
        self.attn  = MASAAttention(cfg, atom_bank, layer_idx)
        self.norm2 = RMSNorm(cfg.dim, eps=cfg.rms_eps)
        self.ffn   = DenseSwiGLU(cfg.dim, cfg.moe_hidden)
        self.use_gc = getattr(cfg, "use_gradient_checkpointing", False)

    def _forward_inner(self, x, mask):
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ffn(self.norm2(x))
        return x

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if self.use_gc and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_inner, x, mask, use_reentrant=False)
        return self._forward_inner(x, mask)


class MoEBlock(nn.Module):
    """MASA attention + Matryoshka MoE + RMSNorm pre-norm + residual."""
    def __init__(self, cfg, atom_bank: MASAAtomBank, layer_idx: int):
        super().__init__()
        self.cfg = cfg
        self.layer_idx = layer_idx
        self.norm1 = RMSNorm(cfg.dim, eps=cfg.rms_eps)
        self.attn  = MASAAttention(cfg, atom_bank, layer_idx)
        self.norm2 = RMSNorm(cfg.dim, eps=cfg.rms_eps)
        self.moe   = MatryoshkaMoEFFN(cfg)
        self.last_router_info: Optional[Dict[str, Any]] = None
        self.use_gc = getattr(cfg, "use_gradient_checkpointing", False)

    def _forward_inner(self, x, mask):
        x = x + self.attn(self.norm1(x), mask=mask)
        moe_out, router_info = self.moe(self.norm2(x))
        # Side effect: record router info for aux-loss telemetry. Gets called
        # twice under gradient checkpointing (once on forward save, once on
        # backward recompute) — same value written, idempotent.
        self.last_router_info = router_info
        x = x + moe_out
        return x

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        if self.use_gc and self.training:
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_inner, x, mask, use_reentrant=False)
        return self._forward_inner(x, mask)


# ─────────────────────────────────────────────────────────────────────────────
#  Top-level model
# ─────────────────────────────────────────────────────────────────────────────

class FANT3Model(nn.Module):
    """
    Full FANT 3 model. Returns logits + auxiliary signals.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Token embedding
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
        nn.init.normal_(self.tok_emb.weight, std=0.02)

        # Shared MASA atom bank (one per model)
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.atom_bank = MASAAtomBank(cfg.dim, cfg.n_attention_atoms, kv_dim)

        # Layer plan
        n_dense  = cfg.n_dense_layers
        n_total  = cfg.n_layers
        # Suffix: distinct MoE blocks for the LAST 3 layers (or fewer if model is small)
        n_suffix = min(3, max(1, n_total - n_dense - 1))
        # Middle: ONE shared block, MoR applied (the wrapper handles 1..max_depth recursions)
        # represents (n_total - n_dense - n_suffix) logical layers with the param cost of 1
        self.n_dense  = n_dense
        self.n_suffix = n_suffix
        self.n_middle_logical = n_total - n_dense - n_suffix

        # Dense prefix
        self.dense_blocks = nn.ModuleList([
            DenseBlock(cfg, self.atom_bank, layer_idx=i) for i in range(n_dense)
        ])

        # MoR-wrapped shared middle block
        if cfg.mor_enabled and self.n_middle_logical > 0:
            shared_middle = MoEBlock(cfg, self.atom_bank, layer_idx=n_dense)
            self.shared_middle = shared_middle
            self.mor = MoRShared(cfg, shared_middle)
        else:
            self.shared_middle = None
            self.mor = None
            # Fallback: distinct MoE blocks for the middle range
            self.middle_blocks = nn.ModuleList([
                MoEBlock(cfg, self.atom_bank, layer_idx=n_dense + i)
                for i in range(self.n_middle_logical)
            ])

        # MoE suffix: distinct blocks
        self.suffix_blocks = nn.ModuleList([
            MoEBlock(cfg, self.atom_bank, layer_idx=n_total - n_suffix + i)
            for i in range(n_suffix)
        ])

        # Final norm + LM head (tied to token embedding)
        self.final_norm = RMSNorm(cfg.dim, eps=cfg.rms_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)
        # Weight tying — LM head shares parameters with token embedding (saves vocab*dim)
        self.lm_head.weight = self.tok_emb.weight

        # Apollonian dual α/β memory. Two implementations:
        #   - SpinorApollonianMemory (Fix 2, 2026-04-19): Kocik tangency spinor
        #     chirality split; fixes scalar-curvature degeneracy.
        #   - ApollonianMemory: original scalar curvature, kept for A/B.
        if getattr(cfg, "spinor_apollonian_enabled", False):
            self.memory = SpinorApollonianMemory(
                dim=cfg.dim,
                alpha_cap=cfg.apollonian_alpha_cap,
                beta_cap=cfg.apollonian_beta_cap,
            )
            self._memory_is_spinor = True
        else:
            self.memory = ApollonianMemory(
                dim=cfg.dim,
                alpha_cap=cfg.apollonian_alpha_cap,
                beta_cap=cfg.apollonian_beta_cap,
                curvature_threshold=cfg.apollonian_curvature_threshold,
            )
            self._memory_is_spinor = False

        # AHN (Fix 4): sliding-window + compressed long-term memory as a
        # gated residual applied before the final norm. Opt-in via cfg.
        if getattr(cfg, "ahn_enabled", False):
            self.ahn = ArtificialHippocampusNetwork(
                dim=cfg.dim,
                n_heads=cfg.ahn_n_heads,
                short_window=cfg.ahn_short_window,
                long_capacity=cfg.ahn_long_capacity,
                compress_ratio=cfg.ahn_compress_ratio,
            )
            # Zero-init gate — AHN contributes nothing until the model learns to use it
            self.ahn_gate = nn.Parameter(torch.zeros(1))
        else:
            self.ahn = None

        # Cerebellum (FIXED dim, NOT scaled with cfg.dim)
        if cfg.cerebellum_enabled:
            self.cerebellum = CerebellumModule(
                in_dim=cfg.cerebellum_in_dim,
                expand_dim=cfg.cerebellum_expand_dim,
                out_dim=cfg.cerebellum_out_dim,
                n_layers=cfg.cerebellum_layers,
                spectral_radius=cfg.cerebellum_spectral_radius,
                sparsity=cfg.cerebellum_sparsity,
            )
            # Project model_dim → cerebellum_in_dim and back
            self.cereb_in_proj  = nn.Linear(cfg.dim, cfg.cerebellum_in_dim,  bias=False)
            self.cereb_out_proj = nn.Linear(cfg.cerebellum_out_dim, cfg.dim, bias=False)
            # Gated residual — starts at zero so cerebellum doesn't disrupt training
            self.cereb_gate = nn.Parameter(torch.zeros(1))
        else:
            self.cerebellum = None

    # -------------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        store_to_memory: bool = False,
    ) -> Dict[str, Any]:
        """
        input_ids: (B, T) int64
        targets:   (B, T) int64 or None  — if given, computes CE loss
        store_to_memory: whether to push final_hidden into Apollonian (Phase 4+)

        Returns dict with keys: logits, loss (if targets given), final_hidden,
                                router_infos, mor_info
        """
        B, T = input_ids.shape

        # Token embed
        x = self.tok_emb(input_ids)  # (B, T, dim)

        # Dense prefix
        for blk in self.dense_blocks:
            x = blk(x)

        # MoR middle (or fallback to distinct middle blocks)
        mor_info = None
        if self.mor is not None:
            x, mor_info = self.mor(x)
        elif hasattr(self, "middle_blocks"):
            for blk in self.middle_blocks:
                x = blk(x)

        # Suffix MoE blocks
        router_infos = []
        for blk in self.suffix_blocks:
            x = blk(x)
            if blk.last_router_info is not None:
                router_infos.append(blk.last_router_info)

        # Optional Cerebellum residual
        if self.cerebellum is not None:
            cer_in  = self.cereb_in_proj(x)
            cer_out = self.cerebellum(cer_in)
            cer_back = self.cereb_out_proj(cer_out)
            x = x + torch.sigmoid(self.cereb_gate) * cer_back

        # Optional AHN residual (Fix 4, 2026-04-19)
        # Zero-init gate → starts as no-op, lets model learn whether to use it.
        if self.ahn is not None:
            ahn_out = self.ahn(x)
            x = x + torch.sigmoid(self.ahn_gate) * ahn_out

        # Final norm
        pre_norm_hidden = x
        x = self.final_norm(x)
        final_hidden = x

        # LM head
        logits = self.lm_head(final_hidden)  # (B, T, vocab_size)

        # CE loss if targets given
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )

        # Apollonian memory store (only if requested — typically Phase 4+)
        if store_to_memory:
            with torch.no_grad():
                emb_flat = final_hidden.reshape(-1, self.cfg.dim).detach()
                if self._memory_is_spinor:
                    # Spinor API: pass pre-RMSnorm hiddens; classifier uses spinor chirality.
                    pre_flat = pre_norm_hidden.reshape(-1, self.cfg.dim).detach()
                    self.memory.store(emb_flat, hidden_preRMSnorm=pre_flat)
                else:
                    # Legacy scalar-curvature API
                    use_upstream = self.cfg.phase4_classifier_upstream
                    classifier_flat = (pre_norm_hidden if use_upstream else final_hidden).reshape(-1, self.cfg.dim).detach()
                    ref = classifier_flat.norm(dim=-1).mean().item()
                    curvs = self.memory.estimate_curvature(classifier_flat, ref_norm=ref)
                    self.memory.store(emb_flat, curvs)

        return {
            "logits":          logits,
            "loss":            loss,
            "final_hidden":    final_hidden,
            "pre_norm_hidden": pre_norm_hidden,
            "router_infos":    router_infos,
            "mor_info":        mor_info,
        }

    # -------------------------------------------------------------------------

    def freeze_intermediate_routers_to_etf(self):
        """
        Apply the ETF-freezing trick (arxiv:2412.00884) to the routers of all
        layers in `cfg.etf_freeze_layers`. Called from the trainer once the
        configured `etf_freeze_after_step` is reached.
        """
        if not self.cfg.etf_freeze_enabled:
            return 0
        n_frozen = 0
        for layer_idx in self.cfg.etf_freeze_layers:
            block = self._get_block_at(layer_idx)
            if block is None or not isinstance(block, MoEBlock):
                continue
            # Freeze megapool projection AND level projection
            freeze_linear_to_etf(block.moe.router.megapool_proj)
            freeze_linear_to_etf(block.moe.router.level_proj)
            n_frozen += 1
        return n_frozen

    def _get_block_at(self, layer_idx: int):
        """Return the block module corresponding to a logical layer index."""
        if layer_idx < self.n_dense:
            return self.dense_blocks[layer_idx]
        elif layer_idx >= self.cfg.n_layers - self.n_suffix:
            suffix_idx = layer_idx - (self.cfg.n_layers - self.n_suffix)
            return self.suffix_blocks[suffix_idx]
        else:
            # Middle range — MoR-shared block (same for all middle layers)
            return self.shared_middle if self.mor is not None else None

    # -------------------------------------------------------------------------

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        n = self.n_params()
        cfg = self.cfg
        return (
            f"FANT 3 Model — stored {n/1e6:.1f}M\n"
            f"  dim={cfg.dim} layers={cfg.n_layers} (dense={self.n_dense}, "
            f"middle-shared={self.n_middle_logical} via MoR x{cfg.n_recursion_depths}, "
            f"suffix={self.n_suffix})\n"
            f"  heads={cfg.n_heads}q/{cfg.n_kv_heads}kv head_dim={cfg.head_dim}\n"
            f"  Matryoshka MoE: {cfg.n_megapools}x{cfg.n_per_megapool} = {cfg.n_megapools * cfg.n_per_megapool} experts, "
            f"{cfg.n_matryoshka_levels} levels (bands {[min(2**i, cfg.n_per_megapool) for i in range(cfg.n_matryoshka_levels)]}), "
            f"top_k={cfg.top_k}\n"
            f"  MASA: {cfg.n_attention_atoms} atoms, rank-{cfg.masa_coef_rank} per-layer coefs\n"
            f"  Cerebellum: {cfg.cerebellum_enabled} ({cfg.cerebellum_in_dim}->{cfg.cerebellum_expand_dim}->{cfg.cerebellum_out_dim})\n"
            f"  Apollonian: alpha={cfg.apollonian_alpha_cap} beta={cfg.apollonian_beta_cap} thr={cfg.apollonian_curvature_threshold}\n"
            f"  ETF freeze: {cfg.etf_freeze_enabled} after step {cfg.etf_freeze_after_step}, layers {cfg.etf_freeze_layers}\n"
        )
