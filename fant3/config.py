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
