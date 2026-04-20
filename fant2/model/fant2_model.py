"""
FANT2Model — the top-level FANT 2 language model.

The full architecture, top to bottom:

    token_ids                                          (B, T)
        ↓ tok_emb (Embedding 32768 → 768)              (B, T, dim)
        ↓
        ↓ for layer in [0..n_dense_layers):
        ↓     dense TransformerBlock                   (B, T, dim)
        ↓
        ↓ + cerebellum side path (one shared module)   (B, T, dim)
        ↓
        ↓ for layer in [n_dense_layers..n_layers):
        ↓     MoE TransformerBlock                     (B, T, dim)
        ↓     (with Apollonian retrieval at the last 2 layers)
        ↓
        ↓ final RMSNorm + LM head (weight-tied)        (B, T, vocab_size)
        ↓
    logits

The model returns a dict containing:
    "logits"            : (B, T, vocab_size)
    "loss"              : scalar cross-entropy (only if targets are passed)
    "router_outputs"    : list of RouterOutput, one per MoE layer
    "memory"            : the ApollonianMemory module (so the trainer can call .store())

Param accounting (default config: dim=768, n_layers=12, n_fractal=72):
    Token embedding (tied with LM head)  : 32768 × 768 = 25.2 M
    2 dense layers (attn + dense FFN)    : ~12.7 M each × 2 = 25.4 M
                                           but actually less, because:
                                           - attn ≈ 4.4 M (Q,K,V,O + hubs)
                                           - dense FFN ≈ 3.0 M
                                           ≈ 7.4 M each × 2 = 14.8 M
    10 MoE layers                        : ~1.07M expert + 4.4M attn + ε mem
                                           ≈ 5.5 M each × 10 = 55 M
                                           BUT note that the router/B/etc only adds
                                           ~1M per layer, so MoE layers are
                                           dominated by attention.
    Cerebellum (shared, 1×)              : ~11.8 M
    Apollonian retrieval (×2 layers)     : ~4.7 M
    Global C correction (shared, 1×)     : ~92 K
    Final norm                           : 768

    Total ≈ 60 M stored (the spec target). Active per token: ~200 M
    (because materialized fractal experts blow up to ~30M each, × top-4).
"""

from typing import Optional, Dict, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import FANT2Config
from .norm import RMSNorm
from .rope import precompute_freqs_cis
from .transformer_block import TransformerBlock
from .cerebellum import CerebellumModule
from .apollonian import ApollonianMemory
from .moe import GlobalCCorrection
from .router import RouterOutput


class FANT2Model(nn.Module):
    """
    The FANT 2 language model.

    Construct from a FANT2Config:
        cfg   = fant2_default()
        model = FANT2Model(cfg)
    """

    def __init__(self, config: FANT2Config):
        super().__init__()
        self.config = config

        # ===== Token embedding =====
        self.tok_emb = nn.Embedding(config.vocab_size, config.dim)
        nn.init.normal_(self.tok_emb.weight, std=config.init_std)

        # ===== Precomputed RoPE frequencies (buffer, no gradient) =====
        # Room for the Phase 4 virtual-token prepend: up to K extra positions
        # where K = max(1, phase4_prepend_k). The legacy pooled-mean path
        # needs 1 extra position; the Option M #2 Coconut full-tensor path
        # needs phase4_prepend_k extra positions. We reserve whichever is
        # larger so the buffer is always big enough.
        rope_slack = max(1, int(getattr(config, "phase4_prepend_k", 0)))
        freqs_cis = precompute_freqs_cis(
            head_dim=config.head_dim,
            max_seq_len=config.max_seq_len + rope_slack,
            theta=config.rope_theta,
            rope_partial=config.rope_partial,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # ===== Shared global C correction (3-level Kronecker, layer-output form) =====
        # ONE instance, shared across all MoE layers — the "global" tier of A⊗B⊗C.
        self.c_global = GlobalCCorrection(
            dim=config.dim,
            kron_C_q=config.kron_C_q,
            kron_C_p=config.kron_C_p,
        )

        # ===== Transformer blocks =====
        self.blocks = nn.ModuleList()
        for layer_idx in range(config.n_layers):
            is_dense = layer_idx < config.n_dense_layers
            use_memory = layer_idx in config.apollonian_retrieval_layers
            block = TransformerBlock(
                layer_idx=layer_idx,
                dim=config.dim,
                n_heads=config.n_heads,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                n_hub_tokens=config.n_hub_tokens,
                hub_dim_mult=config.hub_dim_mult,
                local_window=config.local_window,
                n_attention_sinks=config.n_attention_sinks,
                rope_partial=config.rope_partial,
                max_seq_len=config.max_seq_len,
                is_dense=is_dense,
                moe_hidden=config.moe_hidden,
                n_megapools=config.n_megapools,
                n_per_megapool=config.n_per_megapool,
                top_k=config.top_k,
                kron_A_p=config.kron_A_p,
                kron_A_q=config.kron_A_q,
                kron_B_p=config.kron_B_p,
                kron_B_q=config.kron_B_q,
                shared_expert_hidden=config.shared_expert_hidden,
                # IMPORTANT: pass the shared c_global module so all MoE layers
                # use the same parameters. Dense layers ignore it.
                c_global=self.c_global if not is_dense else None,
                use_memory=use_memory,
                rms_eps=config.rms_eps,
                attention_dropout=config.attention_dropout,
                init_std=config.init_std,
                use_checkpoint=config.grad_checkpoint,
            )
            self.blocks.append(block)

        # ===== Cerebellum (shared, applied as a side path after the dense layers) =====
        self.cerebellum = CerebellumModule(
            in_dim=config.cerebellum_in_dim,
            expand_dim=config.cerebellum_expand_dim,
            out_dim=config.cerebellum_out_dim,
            n_layers=config.cerebellum_layers,
            spectral_radius=config.cerebellum_spectral_radius,
            sparsity=config.cerebellum_sparsity,
            init_std=config.init_std,
        )

        # ===== Apollonian dual α/β memory =====
        self.memory = ApollonianMemory(
            dim=config.dim,
            alpha_cap=config.apollonian_alpha_cap,
            beta_cap=config.apollonian_beta_cap,
            curvature_threshold=config.apollonian_curvature_threshold,
        )

        # ===== Final norm + LM head (weight-tied to tok_emb) =====
        self.final_norm = RMSNorm(config.dim, eps=config.rms_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Weight tying: lm_head shares the embedding weight matrix
        self.lm_head.weight = self.tok_emb.weight

        # ===== Auxiliary heads for self-refinement / reasoning (Phase 4) =====
        # JEPA predictor: maps a context embedding to a target embedding
        self.jepa_predictor = nn.Sequential(
            nn.Linear(config.dim, config.dim, bias=False),
            nn.GELU(),
            nn.Linear(config.dim, config.dim, bias=False),
        )
        # Success estimator: predicts how confident the model is per token
        self.success_estimator = nn.Sequential(
            nn.Linear(config.dim, 64, bias=False),
            nn.GELU(),
            nn.Linear(64, 1, bias=False),
        )

        # ===== Fuzzification scalars (learned per spec §1) =====
        self.fuzz_alpha = nn.Parameter(torch.tensor(config.fuzz_alpha_init))
        self.fuzz_beta  = nn.Parameter(torch.tensor(config.fuzz_beta_init))

    # -------------------------------------------------------------------------
    # Forward
    # -------------------------------------------------------------------------

    def forward(
        self,
        token_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        store_to_memory: bool = False,
        prepend_vec: Optional[torch.Tensor] = None,
        external_classifier_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """
        Args:
            token_ids:      (B, T) long tensor of token ids
            targets:        (B, T) long tensor of next-token targets (optional)
            store_to_memory: if True, store the final-layer embeddings in the
                             Apollonian memory pack via .store() (no_grad)
            prepend_vec:    optional (B, dim) [legacy pooled summary] OR
                             (B, K, dim) [Option M / Coconut full-tensor feedback]
                             tensor inserted as virtual first positions *after*
                             tok_emb. Used by Phase 4 two-pass refinement to
                             pass pass-1's state into pass 2. The output is
                             sliced to drop the prepended positions so the
                             returned shapes still match the no-prepend contract.
            external_classifier_scores: optional (B*T,) tensor of per-token
                             classifier scores (Option M #4 Titans-style
                             surprise). When provided, these are used instead
                             of the L2-norm curvature proxy for α/β assignment.

        Returns:
            dict with:
                "logits":        (B, T, vocab_size)
                "loss":          scalar (only if targets is not None)
                "router_outputs": list of RouterOutput from each MoE layer
                "final_hidden":   (B, T, dim) the pre-LM-head hidden state
                "pre_norm_hidden": (B, T, dim) last-block output BEFORE final_norm
                                    (used by Option M #3 HELM upstream classifier)
                "success_pred":   (B, T, 1) per-token success estimator output
        """
        B, T = token_ids.shape
        cfg = self.config

        # ===== Token embedding =====
        x = self.tok_emb(token_ids)  # (B, T, dim)

        # ===== Optional virtual-token prepend (Phase 4 two-pass) =====
        # Insert prepend_vec as virtual first position(s). The model sees T+K
        # positions; we strip positions [0, K) from the outputs at the end
        # so the caller-facing shapes are unchanged.
        #
        # Shape dispatch:
        #   (B, dim)      → legacy pooled summary, K=1
        #   (B, K, dim)   → Coconut full-tensor feedback (Option M #2)
        prepended = prepend_vec is not None
        n_prepend = 0
        if prepended:
            if prepend_vec.dim() == 2:
                assert prepend_vec.shape == (B, cfg.dim), \
                    f"prepend_vec (2D) must be (B={B}, dim={cfg.dim}), got {tuple(prepend_vec.shape)}"
                prepend_seq = prepend_vec.unsqueeze(1)  # (B, 1, dim)
            elif prepend_vec.dim() == 3:
                assert prepend_vec.shape[0] == B and prepend_vec.shape[2] == cfg.dim, \
                    f"prepend_vec (3D) must be (B={B}, K, dim={cfg.dim}), got {tuple(prepend_vec.shape)}"
                prepend_seq = prepend_vec
            else:
                raise ValueError(f"prepend_vec must be 2D or 3D, got {prepend_vec.dim()}D")
            n_prepend = prepend_seq.shape[1]
            x = torch.cat([prepend_seq, x], dim=1)  # (B, T+n_prepend, dim)

        router_outputs: List[RouterOutput] = []

        # ===== Run all blocks =====
        for layer_idx, block in enumerate(self.blocks):
            x, router_out = block(x, self.freqs_cis, memory=self.memory)
            if router_out is not None:
                router_outputs.append(router_out)

            # Insert the cerebellum as a side path after the dense layers finish
            if layer_idx == cfg.n_dense_layers - 1:
                x = x + self.cerebellum(x)

        # ===== Capture pre-norm hidden (Option M #3 HELM upstream classifier) =====
        # x here is the last block's output before final_norm. RMSNorm will
        # flatten everything onto the unit sphere; we keep a handle to the
        # pre-flatten state so the curvature classifier can still see radial
        # dynamic range.
        pre_norm_hidden = x

        # ===== Final norm + LM head =====
        final_hidden = self.final_norm(x)
        logits = self.lm_head(final_hidden)  # (B, T, vocab_size)

        # ===== Success estimator (Phase 4 self-refinement aux head) =====
        success_pred = torch.sigmoid(self.success_estimator(final_hidden))  # (B, T, 1)

        # ===== Strip the virtual position(s) back off =====
        if prepended:
            # Drop positions [0, n_prepend) (the prepended output) so shapes
            # match the no-prepend contract: (B, T, *).
            final_hidden = final_hidden[:, n_prepend:, :].contiguous()
            pre_norm_hidden = pre_norm_hidden[:, n_prepend:, :].contiguous()
            logits = logits[:, n_prepend:, :].contiguous()
            success_pred = success_pred[:, n_prepend:, :].contiguous()

        out: Dict[str, Any] = {
            "logits":          logits,
            "router_outputs":  router_outputs,
            "final_hidden":    final_hidden,
            "pre_norm_hidden": pre_norm_hidden,
            "success_pred":    success_pred,
        }

        # ===== Cross-entropy loss =====
        if targets is not None:
            # Shift-by-one is the trainer's responsibility (we just compute CE)
            loss = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size),
                targets.reshape(-1),
                ignore_index=-100,
            )
            out["loss"] = loss

        # ===== Optionally fill the Apollonian memory =====
        if store_to_memory:
            with torch.no_grad():
                # The stored embedding is always the post-norm final_hidden
                # (matches what retrieval sees at inference time).
                flat = final_hidden.reshape(-1, cfg.dim).detach()

                # Classifier input: either the post-norm embedding (legacy) or
                # the pre-RMSNorm hidden (Option M #3 HELM upstream fix).
                use_upstream = getattr(cfg, "phase4_classifier_upstream", False)
                if use_upstream:
                    classifier_flat = pre_norm_hidden.reshape(-1, cfg.dim).detach()
                else:
                    classifier_flat = flat

                if external_classifier_scores is not None:
                    # Option M #4: caller already computed surprise scores
                    # (e.g. per-token CE loss from pass 2). Use those as the
                    # α/β assignment signal directly.
                    curvs = external_classifier_scores.detach().reshape(-1)
                    assert curvs.numel() == flat.shape[0], \
                        f"external_classifier_scores numel {curvs.numel()} != N {flat.shape[0]}"
                else:
                    # Legacy L2-norm curvature proxy (operating on whichever
                    # classifier input we picked above).
                    ref = classifier_flat.norm(dim=-1).mean().item()
                    curvs = self.memory.estimate_curvature(classifier_flat, ref_norm=ref)

                self.memory.store(flat, curvs)

        return out

    # -------------------------------------------------------------------------
    # Auxiliary loss helpers (called by the trainer for the unified FEP loss)
    # -------------------------------------------------------------------------

    def aggregate_router_losses(
        self,
        router_outputs: List[RouterOutput],
        z_loss_alpha: float = 1e-3,
        fep_kl_beta: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Sum the per-layer router auxiliary losses into a single dict.

        Args:
            router_outputs: list of RouterOutput from FANT2Model.forward()
            z_loss_alpha:   coefficient for the OLMoE z-loss
            fep_kl_beta:    coefficient for the Free Energy Principle KL prior

        Returns:
            dict with "z_loss", "fep_kl", and "total_aux"
        """
        # Take the router from any MoE block (they're all instances of the same class)
        moe_blocks = [b for b in self.blocks if not b.is_dense]
        if not moe_blocks or not router_outputs:
            zero = torch.zeros((), device=self.tok_emb.weight.device)
            return {"z_loss": zero, "fep_kl": zero, "total_aux": zero}
        ref_router = moe_blocks[0].ffn.router

        z_total = torch.zeros((), device=self.tok_emb.weight.device)
        kl_total = torch.zeros((), device=self.tok_emb.weight.device)
        for ro in router_outputs:
            z_total = z_total + ref_router.z_loss(ro.expert_logits)
            kl_total = kl_total + ref_router.fep_kl_prior(ro.megapool_load, ro.expert_load)
        z_total = z_total / len(router_outputs)
        kl_total = kl_total / len(router_outputs)

        return {
            "z_loss":    z_loss_alpha * z_total,
            "fep_kl":    fep_kl_beta * kl_total,
            "total_aux": z_loss_alpha * z_total + fep_kl_beta * kl_total,
        }

    @property
    def moe_layers(self) -> nn.ModuleList:
        """Return all FractalMoELayer modules (for N1 ortho loss etc.)."""
        return nn.ModuleList([b.ffn for b in self.blocks if not b.is_dense])

    @torch.no_grad()
    def update_router_biases(self, router_outputs: List[RouterOutput]) -> None:
        """
        DeepSeek aux-loss-free bias update for every MoE layer.

        Called from the trainer AFTER backward() and BEFORE optimizer.step().
        """
        moe_blocks = [b for b in self.blocks if not b.is_dense]
        for block, ro in zip(moe_blocks, router_outputs):
            block.ffn.router.update_biases(ro.megapool_load, ro.expert_load)

    @torch.no_grad()
    def tikkun_repair_all(self) -> int:
        """Run Tikkun repair on every MoE layer's router. Returns number that fired."""
        n_repaired = 0
        for block in self.blocks:
            if not block.is_dense:
                if block.ffn.router.tikkun_repair():
                    n_repaired += 1
        return n_repaired

    @torch.no_grad()
    def fana_dropout_all(self, p: float = 0.05) -> None:
        """Apply Fanā dropout to every MoE layer's router."""
        for block in self.blocks:
            if not block.is_dense:
                block.ffn.router.fana_dropout(p)

    def sample_materialized_expert_weights(
        self,
        n_samples: int = 4,
        seed: Optional[int] = None,
    ) -> List[torch.Tensor]:
        """
        Materialize a few randomly-sampled fractal experts' weight matrices WITH
        gradient flow so the trainer can compute a rank / condition number loss
        during Phase 3 active-layer calibration.

        Returns a list of 2D tensors (W_gate, W_up, W_down for each sampled expert).
        Gradients flow back to A_expert, B_layer via torch.kron.
        """
        import random
        rng = random.Random(seed)

        moe_blocks = [b for b in self.blocks if not b.is_dense]
        if not moe_blocks:
            return []

        sampled: List[torch.Tensor] = []
        for _ in range(n_samples):
            block = rng.choice(moe_blocks)
            moe = block.ffn  # the FractalMoELayer
            expert = rng.choice(list(moe.fractal_experts))
            W_gate, W_up, W_down = expert.materialize(
                moe.B_gate, moe.B_up, moe.B_down
            )
            sampled.extend([W_gate, W_up, W_down])
        return sampled

    # -------------------------------------------------------------------------
    # Diagnostics
    # -------------------------------------------------------------------------

    def stored_param_count(self) -> int:
        """Total stored learnable parameters (the 60M figure)."""
        return sum(p.numel() for p in self.parameters())

    def active_param_count(self) -> int:
        """
        Estimate of active parameters per forward pass (the 200M figure).

        Counts the materialized expert weights for top-k experts × n_moe_layers.
        Plus all non-MoE parameters (which are always active).
        """
        cfg = self.config
        non_moe = 0
        for name, p in self.named_parameters():
            if "fractal_experts" not in name and "B_gate" not in name and "B_up" not in name and "B_down" not in name:
                non_moe += p.numel()
        # Active fractal experts per layer = top_k × materialized weight size
        # Materialized W_gate + W_up + W_down per expert ≈ 3 × 256 × 1280 = 983K floats
        per_expert_active = 3 * (cfg.kron_A_q * cfg.kron_B_q) * (cfg.kron_A_p * cfg.kron_B_p)
        n_moe_layers = cfg.n_layers - cfg.n_dense_layers
        moe_active = n_moe_layers * cfg.top_k * per_expert_active
        return non_moe + moe_active

    def parameter_summary(self) -> str:
        """Pretty-printed parameter breakdown."""
        cfg = self.config
        lines = [f"FANT2Model parameter summary:"]
        lines.append(f"  Total stored:  {self.stored_param_count() / 1e6:.2f} M")
        lines.append(f"  Active/forward (estimate): {self.active_param_count() / 1e6:.2f} M")
        # Per-component
        comps = {
            "tok_emb":              self.tok_emb.weight.numel(),
            "c_global":             sum(p.numel() for p in self.c_global.parameters()),
            "cerebellum (learned)": self.cerebellum.stored_param_count(),
            "memory (buffers)":     0,  # all buffers, no params
            "final_norm":           self.final_norm.weight.numel(),
            "jepa_predictor":       sum(p.numel() for p in self.jepa_predictor.parameters()),
            "success_estimator":    sum(p.numel() for p in self.success_estimator.parameters()),
            "fuzz scalars":         2,
        }
        for name, n in comps.items():
            lines.append(f"  {name:24s} {n / 1e6:.3f} M")
        for i, b in enumerate(self.blocks):
            tag = "dense" if b.is_dense else "MoE  "
            mem = " [+mem]" if b.use_memory else ""
            lines.append(f"  block[{i:02d}] {tag} {b.stored_param_count() / 1e6:.3f} M{mem}")
        return "\n".join(lines)
