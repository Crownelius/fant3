"""
FANT 3 model configuration.

Hardware budget (RTX 3060 12 GB, bf16 + 8-bit AdamW + grad ckpt, batch=2 seq=1024):
    weights       ~ 2.0 GB    (1.0B × bf16)
    gradients     ~ 2.0 GB
    8-bit AdamW   ~ 2.0 GB
    activations   ~ 2.5 GB    (100M active × seq × batch w/ ckpt)
    buffers       ~ 0.5 GB
    TOTAL         ~ 9.0 GB    (3 GB headroom)

Three presets:
    fant3_smoke — 40M tiny (for smoke tests; fits in 2 GB)
    fant3_742m  — 750M validation scale (VRAM-proven on 3060 via FANT 2)
    fant3_1b    — 1.05 B primary target
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FANT3Config:
    # --- Core dimensions -----------------------------------------------------
    # FIXED 2026-04-19: defaults now produce real ~987M params (not 6.6B).
    # Same preset-vs-reality bug as the old fant3_742m preset — full-rank
    # MoE experts ignore the kron_* fields, so param count is dominated by
    # n_experts * dim * moe_hidden * 2 (up+down).  Calibrated to land under
    # 1B stored so the '1b' TARGET_SCALE actually fits on A100 96 GB.
    dim:            int   = 1024       # was 2048
    n_layers:       int   = 20         # was 24
    n_dense_layers: int   = 3
    n_heads:        int   = 8          # was 16
    n_kv_heads:     int   = 2          # was 4 — still GQA-4 with n_heads=8
    head_dim:       int   = 128
    rope_partial:   float = 0.25        # Phi-4-Mini partial RoPE
    rope_theta:     float = 10000.0
    rms_eps:        float = 1e-6

    # --- Matryoshka MoE ------------------------------------------------------
    # Nested coarse-to-fine expert activation. Expert 0 learns coarse behavior,
    # +1 adds detail, etc. Elastic inference supported.
    n_megapools:      int = 4          # was 8
    n_per_megapool:   int = 8          # was 16 — now 32 experts total (was 128)
    n_matryoshka_levels: int = 2       # was 4
    top_k:            int = 2          # was 4
    shared_expert_hidden: int = 640    # was 768
    moe_hidden:       int = 2304       # was 2048 — bumped to reach 1B target
    n_special:        int = 2           # zero + copy experts

    # --- MASA shared-atom attention ------------------------------------------
    # All Q/K/V/O across all n_layers share a dictionary of `n_attention_atoms`
    # atom matrices, with per-layer rank-`masa_coef_rank` coefficients.
    masa_enabled:       bool = True
    n_attention_atoms:  int  = 5       # was 6
    masa_coef_rank:     int  = 8       # was 16

    # --- Mixture-of-Recursions (MoR) -----------------------------------------
    # Per-token recursion depth chosen by a lightweight router.
    # α (instance / recent) tokens → shallow; β (schema / stable) → deep.
    mor_enabled:        bool = True
    n_recursion_depths: int  = 2       # was 3 — halved activation cost at cost of max-depth reasoning
    mor_router_dim:     int  = 128
    mor_depth_bias:     str  = "alpha"  # "alpha" or "beta" or "uniform"

    # --- Kronecker 3-level expert factorization ------------------------------
    # Each fractal expert's weight matrix: W = kron(kron(A_expert, B_layer), C_global)
    kron_A_p: int = 80
    kron_A_q: int = 16
    kron_B_p: int = 32
    kron_B_q: int = 32
    kron_C_p: int = 48
    kron_C_q: int = 80

    # --- Vocabulary ----------------------------------------------------------
    vocab_size:  int = 32768
    max_seq_len: int = 1024            # was 2048 — longer sequences blow up activations

    # --- Hub attention (VEN analog) -----------------------------------------
    n_hub_tokens:      int   = 32
    hub_dim_mult:      float = 2.0
    local_window:      int   = 128
    n_attention_sinks: int   = 4

    # --- Cerebellum (FIXED SIZE — does NOT scale with dim) -------------------
    # Distinctive reservoir feature from FANT 2. Kept at ~25M params regardless
    # of overall model scale — parameter budget is better spent on MoE experts.
    cerebellum_enabled:        bool  = True
    cerebellum_in_dim:         int   = 768      # fixed (not dim)
    cerebellum_expand_dim:     int   = 7680     # 10× fanout
    cerebellum_out_dim:        int   = 768      # fixed (not dim)
    cerebellum_layers:         int   = 4
    cerebellum_spectral_radius: float = 0.95
    cerebellum_sparsity:       float = 0.001

    # --- Apollonian memory ---------------------------------------------------
    apollonian_alpha_cap:           int   = 10000
    apollonian_beta_cap:            int   = 10000
    apollonian_curvature_threshold: float = 0.5
    apollonian_retrieval_layers:    tuple = (18, 19)  # last 2 layers of 20 (was 22,23 for n_layers=24)
    # Fix 2 (2026-04-19): Replace scalar-curvature classifier with Kocik tangency
    # spinor chirality split (Cl(2,1) Minkowski, Descartes-theorem-aware).
    # When True, uses fant3.model.spinor_apollonian.SpinorApollonianMemory.
    # See research_spinor_apollonian_2026_04_16.md for the math.
    spinor_apollonian_enabled:      bool  = True

    # --- Artificial Hippocampus Network (AHN) --------------------------------
    # Fix 4 (2026-04-19): Sliding short-term + compressed long-term memory,
    # applied as a gated residual before final norm. Runs alongside (not
    # instead of) the Apollonian packs. Based on ByteDance AHN (lab_bytedance).
    ahn_enabled:                    bool  = True
    ahn_n_heads:                    int   = 4
    ahn_short_window:               int   = 256
    ahn_long_capacity:              int   = 512
    ahn_compress_ratio:             float = 0.25

    # --- Intermediate ETF freezing (arxiv:2412.00884) ------------------------
    # Freeze router weights to simplex ETFs after calibration for free compression.
    etf_freeze_enabled:       bool = True
    etf_freeze_after_step:    int  = 2000
    etf_freeze_layers:        tuple = tuple(range(3, 17))  # freeze middle of n_layers=20 (leave first 3 and last 3 learnable)

    # --- Phase 4 fixes inherited from FANT 2 (landed 2026-04-16) -------------
    phase4_classifier_upstream:     bool = True
    populate_apollonian_in_phase2:  bool = True
    phase4_classifier_mode:         str  = "curvature"  # or "ce_surprise"

    # --- Gradient checkpointing (landed 2026-04-19) -------------------------
    # When True, wrap each DenseBlock / MoEBlock / MoR inner-pass forward with
    # torch.utils.checkpoint.checkpoint(use_reentrant=False). Trades ~25-40%
    # compute for ~3-5x less activation memory. Required for 742m+ at T>=256
    # and essential for 1b on Colab A100 80 GB.
    use_gradient_checkpointing:     bool = False

    # --- MoR LTI-style injection (landed 2026-04-20) ------------------------
    # Adapted from Mythos / Recurrent-Depth Transformer (RDT) literature. Each
    # pass through the MoR shared block is updated with:
    #     current_{k+1} = A * current_k + B * x_original + C * retrieved + block(current_k + loop_emb[k])
    # where
    #   - A is a diagonal matrix with A = -softplus(a_diag), guaranteeing
    #     spectral radius rho(A) < 1 for recurrent stability.
    #   - B injects the ORIGINAL input to prevent hidden-state drift.
    #   - C injects the Apollonian-retrieved memory context (FANT-specific —
    #     this is what keeps FANT central in the synthesis).
    #   - loop_emb[k] is a learned per-pass positional signal so the same
    #     shared block behaves differently on different passes.
    # These are all OPTIONAL via the three flags below — default off so
    # existing checkpoints remain bit-compatible with the un-augmented MoR.
    mor_lti_injection_enabled:      bool = False
    mor_spectral_constraint:        bool = False
    mor_loop_index_enabled:         bool = False
    # When mor_lti_injection_enabled, the Apollonian-retrieved context C*retrieved
    # channel is added only when BOTH this flag and spinor_apollonian_enabled
    # are True AND the memory pack has >0 items.
    mor_lti_apollonian_channel:     bool = True

    # ------ CE stabilisation (Tier 1/2 landed 2026-04-22) --------------------
    # LM head logit soft-cap (Gemma-2 style): logits = cap * tanh(logits / cap).
    # Bounds softmax input; kills bf16 overflow + stabilises MoE router.
    # None = off. Recommended value at 1B scale: 30.0.
    lm_head_logit_cap:               Optional[float] = None
    # MoR C channel warmup — delay the Apollonian-retrieved memory injection
    # into MoR until step N, so freshly-filled packs don't inject noise before
    # their embeddings have meaningful gradients. 0 = never delay.
    apollonian_channel_warmup_steps: int = 500


# -----------------------------------------------------------------------------
#  Presets
# -----------------------------------------------------------------------------

def fant3_smoke() -> FANT3Config:
    """Tiny 40M config for smoke-testing. Fits in ~2 GB."""
    return FANT3Config(
        dim=512, n_layers=8, n_dense_layers=1,
        n_heads=8, n_kv_heads=2, head_dim=64,
        n_megapools=4, n_per_megapool=4, top_k=2,
        n_matryoshka_levels=2,
        shared_expert_hidden=256, moe_hidden=512,
        n_attention_atoms=3, masa_coef_rank=8,
        n_recursion_depths=2,
        kron_A_p=20, kron_A_q=8, kron_B_p=16, kron_B_q=16, kron_C_p=16, kron_C_q=20,
        max_seq_len=512,
        n_hub_tokens=8,
        cerebellum_enabled=False,  # skip for smoke
        apollonian_alpha_cap=1000, apollonian_beta_cap=1000,
        apollonian_retrieval_layers=(6, 7),
        etf_freeze_after_step=100,
        etf_freeze_layers=(2, 3, 4),
    )


def fant3_1m() -> FANT3Config:
    """~1M toy preset for local ISRM smoke (2026-04-24).

    Purpose: prove dynamic-K + monotonic-CE + contractive-alpha work end-to-end
    in a real training loop on CPU in minutes, not hours.

    Tight vocab (2048) so the tied emb doesn't dominate; dim=128 / 4 layers /
    2 experts / moe_hidden=192 brings stored params to ~0.99M. Cerebellum +
    AHN + ETF-freeze all disabled — we're measuring MoR/ISRM behaviour, not
    the full stack.

    Verified: 0.99M stored. Runs ~1 step/sec on a laptop CPU at B=4 T=64.
    """
    return FANT3Config(
        vocab_size=2048,
        dim=128, n_layers=4, n_dense_layers=1,
        n_heads=4, n_kv_heads=1, head_dim=32,
        n_megapools=1, n_per_megapool=2, top_k=1,    # 2 experts
        n_matryoshka_levels=1,
        shared_expert_hidden=96, moe_hidden=192,
        n_attention_atoms=2, masa_coef_rank=2,
        n_recursion_depths=3,                         # 3 so dynamic K has a real range
        kron_A_p=4, kron_A_q=4, kron_B_p=8, kron_B_q=8, kron_C_p=8, kron_C_q=8,
        max_seq_len=64,
        cerebellum_enabled=False,
        ahn_enabled=False,
        apollonian_alpha_cap=128, apollonian_beta_cap=128,
        apollonian_retrieval_layers=(2, 3),
        etf_freeze_enabled=False,
    )


def fant3_10m() -> FANT3Config:
    """~10M preset for budget Chinchilla-optimal runs (2026-04-23).

    Designed for the $15-30 RunPod budget range. At Chinchilla (~20 tok/param)
    = 200M tokens = ~3100 steps at T=16384 B=1 accum=4 = ~4 hours on H100.

    Architecture is minimal: the tied tok_emb + lm_head alone at vocab 32736
    consume 6.29M params (63% of total), so everything else must be aggressive.
    Cerebellum + AHN disabled (they carry fixed budget regardless of model
    size). MoR kept at depth 2 since it's cheap and load-bearing for the
    CERN-inspired physics path.

    Target: ~10M stored (measure to confirm).
    """
    return FANT3Config(
        dim=192, n_layers=6, n_dense_layers=1,
        n_heads=4, n_kv_heads=2, head_dim=48,
        n_megapools=2, n_per_megapool=2, top_k=1,   # 4 experts total
        n_matryoshka_levels=2,
        shared_expert_hidden=128, moe_hidden=256,
        n_attention_atoms=3, masa_coef_rank=4,
        n_recursion_depths=2,
        kron_A_p=4, kron_A_q=4, kron_B_p=8, kron_B_q=6, kron_C_p=6, kron_C_q=8,
        max_seq_len=1024,
        cerebellum_enabled=False,            # 25M budget killer at this scale
        ahn_enabled=False,                    # same
        apollonian_alpha_cap=500, apollonian_beta_cap=500,
        apollonian_retrieval_layers=(4, 5),  # last 2 of 6
        etf_freeze_after_step=1000,
        etf_freeze_layers=tuple(range(1, 5)),
    )


def fant3_15m() -> FANT3Config:
    """~15M preset (2026-04-23).

    Sweet spot for a small but capacity-respectable run on $30-40 budget.
    Chinchilla = 300M tokens = ~4600 steps at T=16384 B=1 accum=4 =
    ~6 hours on H100.

    Target: ~15M stored.
    """
    return FANT3Config(
        dim=256, n_layers=6, n_dense_layers=1,
        n_heads=4, n_kv_heads=2, head_dim=64,
        n_megapools=2, n_per_megapool=2, top_k=1,   # 4 experts total
        n_matryoshka_levels=2,
        shared_expert_hidden=192, moe_hidden=384,
        n_attention_atoms=3, masa_coef_rank=4,
        n_recursion_depths=2,
        kron_A_p=8, kron_A_q=4, kron_B_p=8, kron_B_q=8, kron_C_p=8, kron_C_q=8,
        max_seq_len=1024,
        cerebellum_enabled=False,
        ahn_enabled=False,
        apollonian_alpha_cap=500, apollonian_beta_cap=500,
        apollonian_retrieval_layers=(4, 5),
        etf_freeze_after_step=1000,
        etf_freeze_layers=tuple(range(1, 5)),
    )


def fant3_80m() -> FANT3Config:
    """~80M Chinchilla-optimal preset (2026-04-23).

    Designed for the 2x RTX PRO 6000 Blackwell budget regime on RunPod
    (~$4/hr combined). With $19 = 4.75h of compute, realistic tokens
    throughput is ~100K tok/sec DDP at this scale = 1.7B tokens =
    exactly 20 tokens/param (Chinchilla-optimal).

    Target: ~80M stored. Scales up from fant3_50m by widening dim
    320 -> 448 and moe_hidden 640 -> 896, keeping n_layers=10 and the
    rest of the architecture identical.
    """
    # Architecture calibrated empirically: FANT 3 caps n_suffix at min(3, ...)
    # in the model builder, so the only scaling levers at fixed n_layers are
    # dim, moe_hidden, and expert count.
    return FANT3Config(
        dim=512, n_layers=10, n_dense_layers=2,
        n_heads=8, n_kv_heads=2, head_dim=64,
        n_megapools=2, n_per_megapool=4, top_k=2,   # 8 experts
        n_matryoshka_levels=2,
        shared_expert_hidden=384, moe_hidden=1280,
        n_attention_atoms=4, masa_coef_rank=8,
        n_recursion_depths=2,
        kron_A_p=16, kron_A_q=8, kron_B_p=16, kron_B_q=16, kron_C_p=16, kron_C_q=16,
        max_seq_len=1024,
        cerebellum_enabled=False,             # 25M fixed budget dwarfs model at this scale
        ahn_enabled=False,
        apollonian_alpha_cap=2000, apollonian_beta_cap=2000,
        apollonian_retrieval_layers=(8, 9),
        etf_freeze_after_step=2000,
        etf_freeze_layers=tuple(range(2, 8)),
    )


def fant3_20m() -> FANT3Config:
    """20M chat-optimized preset (2026-04-19).

    Target use case: 12h training on A100 96 GB, ~1.4B training tokens,
    heavy distillation from Sonnet 4.6 + Opus 4.6. Following the TinyStories /
    MobileLLM / Phi philosophy of over-training a small model on high-quality
    distilled data. Honest target: fluent conversational English on short
    responses, basic arithmetic, simple code; NOT MMLU-level knowledge.

    Verified: 23.50M stored params.
    """
    return FANT3Config(
        dim=320, n_layers=10, n_dense_layers=2,
        n_heads=4, n_kv_heads=2, head_dim=80,
        n_megapools=2, n_per_megapool=2, top_k=1,   # 4 experts total
        n_matryoshka_levels=2,
        shared_expert_hidden=256, moe_hidden=640,
        n_attention_atoms=3, masa_coef_rank=4,
        n_recursion_depths=2,
        kron_A_p=8, kron_A_q=4, kron_B_p=10, kron_B_q=10, kron_C_p=8, kron_C_q=16,
        max_seq_len=1024,
        cerebellum_enabled=False,
        apollonian_alpha_cap=1000, apollonian_beta_cap=1000,
        apollonian_retrieval_layers=(8, 9),
        etf_freeze_after_step=3000,
        etf_freeze_layers=tuple(range(2, 8)),
    )


def fant3_50m() -> FANT3Config:
    """50M chat-optimized preset (2026-04-19).

    Target: 12h training on A100 96 GB, ~2B training tokens (~40× Chinchilla —
    heavily over-trained, which is what small chat models need). Heavy
    distillation from Sonnet 4.6 + Opus 4.6 + Cascade-2 chat/IF, following
    the TinyStories / MobileLLM / SmolLM2 / Phi playbook. Honest target:
    fluent conversational English on short exchanges, basic numeracy, simple
    code. NOT MMLU/GSM8K competitive — capacity floor.

    Verified: 50.79M stored params.
    """
    return FANT3Config(
        dim=384, n_layers=12, n_dense_layers=2,
        n_heads=6, n_kv_heads=2, head_dim=64,         # 6×64=384, GQA-3
        n_megapools=2, n_per_megapool=4, top_k=2,     # 8 experts
        n_matryoshka_levels=2,
        shared_expert_hidden=320, moe_hidden=896,
        n_attention_atoms=4, masa_coef_rank=6,
        n_recursion_depths=2,
        kron_A_p=12, kron_A_q=6, kron_B_p=12, kron_B_q=12, kron_C_p=12, kron_C_q=16,
        max_seq_len=1024,
        cerebellum_enabled=False,
        ahn_enabled=False,
        apollonian_alpha_cap=2000, apollonian_beta_cap=2000,
        apollonian_retrieval_layers=(10, 11),
        etf_freeze_after_step=6000,
        etf_freeze_layers=tuple(range(2, 10)),
    )


def fant3_742m() -> FANT3Config:
    """742M validation scale.
    FIXED 2026-04-19: original preset materialized 6.6B params because the
    MatryoshkaMoE implementation uses full-rank expert weights (ignores the
    kron_* config fields). The old preset had 128 experts × dim=2048 × moe_hidden=2048
    which is ~1.6B per MoE block × 4 blocks = 6.4B. Shrunk to 32 experts / dim=1024
    / moe_hidden=1792 which actually produces ~770M stored params."""
    return FANT3Config(
        dim=1024, n_layers=16, n_dense_layers=2,
        n_heads=8, n_kv_heads=2, head_dim=128,
        n_megapools=4, n_per_megapool=8, top_k=2,   # 32 experts total (was 128)
        n_matryoshka_levels=2,
        shared_expert_hidden=512, moe_hidden=1792,
        n_attention_atoms=4, masa_coef_rank=8,
        n_recursion_depths=2,
        kron_A_p=32, kron_A_q=8, kron_B_p=16, kron_B_q=16, kron_C_p=32, kron_C_q=32,
        max_seq_len=1024,
        apollonian_alpha_cap=5000, apollonian_beta_cap=5000,
        apollonian_retrieval_layers=(14, 15),
        etf_freeze_after_step=1000,
        etf_freeze_layers=tuple(range(2, 13)),
    )


def fant3_1b() -> FANT3Config:
    """Primary target: 1.05B stored / ~100M active, fits on RTX 3060 12GB."""
    return FANT3Config()  # uses dataclass defaults above
