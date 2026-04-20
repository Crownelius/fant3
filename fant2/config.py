"""
FANT 2 model configuration.

All hyperparameters from fant2_architecture_spec.md §1.
The defaults give the locked 60M-stored / 200M-active configuration.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FANT2Config:
    # ----- Core dimensions -----
    dim:            int = 768          # hidden dim
    n_layers:       int = 12           # 2 dense + 10 MoE
    n_dense_layers: int = 2            # first k layers use dense FFN (DeepSeek V3 first_k_dense_replace)
    n_heads:        int = 8            # query heads
    n_kv_heads:     int = 2            # GQA-2: 4x KV cache reduction
    head_dim:       int = 96           # dim // n_heads
    rope_partial:   float = 0.25       # Phi-4-Mini partial RoPE: only 25% of head_dim gets rotated
    rope_theta:     float = 10000.0
    rms_eps:        float = 1e-6

    # ----- MoE / fractal expert config -----
    n_megapools:    int = 8            # outer routing tier (Parisi RSB)
    n_per_megapool: int = 9            # inner routing tier
    n_fractal:      int = 72           # = n_megapools * n_per_megapool
    n_special:      int = 2            # zero + copy
    top_k:          int = 4            # top-k within selected mega-pool
    n_total_experts: int = 74          # 72 fractal + zero + copy (shared is separate)

    # Always-on shared expert (narrow, to fit budget)
    shared_expert_hidden: int = 256    # narrow shared expert (vs 1280 full SwiGLU)

    # MoE FFN (per fractal expert, when materialized)
    moe_hidden:     int = 1280         # SwiGLU expert FFN hidden size

    # ----- 3-level Kronecker hierarchy A ⊗ B ⊗ C -----
    # Each fractal expert effective weight matrix is built as:
    #   W = kron(kron(A_expert, B_layer), C_global)
    # so that effective shapes match (1024, 1280) ≈ (dim_A * dim_B * dim_C, ...)
    kron_A_p:       int = 40           # per-expert (fine grain)
    kron_A_q:       int = 8
    kron_B_p:       int = 32           # per-layer (mid grain)
    kron_B_q:       int = 32
    kron_C_p:       int = 32           # global (coarse grain) — kept small to be cheap
    kron_C_q:       int = 40

    # ----- Vocabulary -----
    vocab_size:     int = 32768
    max_seq_len:    int = 1024         # extendable to 4096 via YaRN at fine-tune

    # ----- Hub attention (VEN analog) -----
    n_hub_tokens:   int = 32
    hub_dim_mult:   float = 2.0        # hub tokens use 2x embedding dim
    local_window:   int = 128
    n_attention_sinks: int = 4

    # ----- Cerebellum module -----
    cerebellum_in_dim:     int = 768   # = dim
    cerebellum_expand_dim: int = 7680  # 10x expansion (mossy → granule fan-out)
    cerebellum_out_dim:    int = 768
    cerebellum_layers:     int = 4
    cerebellum_spectral_radius: float = 0.95  # echo state at edge of chaos
    cerebellum_sparsity:   float = 0.001       # ~10 inputs per neuron

    # ----- Apollonian memory -----
    apollonian_alpha_cap: int = 5000
    apollonian_beta_cap:  int = 5000
    apollonian_curvature_threshold: float = 0.5
    apollonian_retrieval_layers: tuple = (10, 11)  # last 2 layers of 12

    # ----- Phase 4 refinement controls (Option M campaign) -----
    # All fields default to the legacy L1.5 behavior; each new feature is opt-in.
    #
    #  #1 Think-at-Hard per-token pass-2 gate (arxiv:2511.08577)
    #     When True, pass-2 CE is masked on tokens where pass-1 softmax confidence
    #     exceeds `phase4_gate_threshold`. Pass 2 still forwards but contributes
    #     gradient only on the "hard" subset, matching Think-at-Hard's finding
    #     that forcing refinement on confident tokens corrupts them.
    phase4_gate_enabled:   bool  = False
    phase4_gate_threshold: float = 0.7
    #
    #  #2 Coconut full-tensor feedback (arxiv:2412.06769)
    #     When > 0, pass 2 receives the last K positions of pass-1's final_hidden
    #     as a [B, K, D] prepend instead of the pooled [B, D] mean. 0 = legacy.
    phase4_prepend_k: int = 0
    #
    #  #3 HELM upstream-of-RMSNorm classifier (arxiv:2505.24722)
    #     When True, the Apollonian memory's curvature proxy is computed on
    #     the pre-RMSNorm hidden state (which still has radial dynamic range)
    #     instead of final_hidden (where RMSNorm has flattened it to the sphere).
    #     The stored embedding is still post-norm; only the classifier signal
    #     moves upstream.
    #
    #     2026-04-16: flipped default to True. Empirical diagnosis of the
    #     2026-04-16 overnight_opus46 run showed that with upstream=False the
    #     post-RMSNorm curvature metric collapses to a ~[0.99, 1.01] band
    #     (unit-sphere artifact), making the fixed 0.5 threshold misfire in
    #     Phase 4+ when memory population is active. Pre-RMSNorm hidden state
    #     preserves the dynamic range the L2-norm proxy needs.
    phase4_classifier_upstream: bool = True
    #
    #  #4 Titans-style surprise classifier (arxiv:2501.00663)
    #     "curvature"         = legacy L2-norm proxy
    #     "ce_surprise"       = per-token CE loss from pass 2 (cheap grad-surprise proxy)
    phase4_classifier_mode: str = "curvature"
    #
    #  #6 SpiralThinker progressive alignment (arxiv:2511.08983)
    #     When > 0, adds a cosine-alignment penalty between pass-1 and pass-2
    #     hidden states in addition to (not instead of) the legacy MSE consistency.
    phase4_alignment_weight: float = 0.0

    # ----- Vision (placeholder for SigLIP2) -----
    vision_dim: int = 1152
    enable_vision: bool = False  # disabled for FANT 2 v2.0

    # ----- Fuzzification -----
    fuzz_alpha_init: float = 0.5
    fuzz_beta_init: float = 0.1

    # ----- Training defaults (overridden per-phase) -----
    bf16: bool = True
    grad_checkpoint: bool = True
    use_muon: bool = True
    muon_lr: float = 1e-3
    adamw_lr: float = 3e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    # ----- Apollonian / criticality targets -----
    target_avalanche_exponent: float = 1.5
    avalanche_tolerance: float = 0.1   # τ ∈ [1.4, 1.6]

    # ----- Router collapse defenses -----
    router_aux_loss_free_gamma: float = 1e-3   # DeepSeek aux-loss-free bias step size
    router_z_loss_alpha: float = 1e-3          # OLMoE
    router_fep_kl_beta_init: float = 0.1       # FEP prior weight, annealed
    router_fep_kl_beta_max: float = 1.0
    router_tikkun_threshold: float = 0.30      # if mega-pool exceeds this fraction → repair
    router_bipartition_floor: float = 1.05     # IIT phi-irreducibility minimum

    # ----- Misc -----
    dropout: float = 0.0
    attention_dropout: float = 0.0
    init_std: float = 0.02

    # ----- Derived properties -----
    def __post_init__(self):
        assert self.dim % self.n_heads == 0, f"dim {self.dim} not divisible by n_heads {self.n_heads}"
        assert self.n_heads % self.n_kv_heads == 0, f"n_heads {self.n_heads} not divisible by n_kv_heads {self.n_kv_heads}"
        assert self.n_fractal == self.n_megapools * self.n_per_megapool, \
            f"n_fractal {self.n_fractal} != n_megapools * n_per_megapool ({self.n_megapools * self.n_per_megapool})"
        assert self.head_dim == self.dim // self.n_heads
        # 3-level Kron must produce (dim, moe_hidden)-shaped effective weights
        eff_p = self.kron_A_p * self.kron_B_p
        eff_q = self.kron_A_q * self.kron_B_q * self.kron_C_q // self.kron_B_q
        # The exact shape is asserted in kron3.py at construction time

    @property
    def n_query_heads_per_kv(self) -> int:
        return self.n_heads // self.n_kv_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def hub_dim(self) -> int:
        return int(self.dim * self.hub_dim_mult)

    def summary(self) -> str:
        return (
            f"FANT2Config: dim={self.dim}, layers={self.n_layers} ({self.n_dense_layers} dense), "
            f"heads={self.n_heads}q/{self.n_kv_heads}kv, head_dim={self.head_dim}\n"
            f"  MoE: {self.n_megapools}×{self.n_per_megapool}={self.n_fractal} fractal "
            f"+ {self.n_special} special + 1 shared (hidden={self.shared_expert_hidden}), top-k={self.top_k}\n"
            f"  Kron A({self.kron_A_p},{self.kron_A_q}) ⊗ B({self.kron_B_p},{self.kron_B_q}) "
            f"⊗ C({self.kron_C_p},{self.kron_C_q})\n"
            f"  Vocab={self.vocab_size}, max_seq={self.max_seq_len}\n"
            f"  Hub: {self.n_hub_tokens} tokens (dim={self.hub_dim}), window={self.local_window}, "
            f"sinks={self.n_attention_sinks}\n"
            f"  Cerebellum: {self.cerebellum_in_dim}→{self.cerebellum_expand_dim}→{self.cerebellum_out_dim}\n"
            f"  Apollonian: α_cap={self.apollonian_alpha_cap}, β_cap={self.apollonian_beta_cap}, "
            f"retrieve@{self.apollonian_retrieval_layers}"
        )


# ----- Convenience presets -----

def fant2_default() -> FANT2Config:
    """The locked 60M / 200M configuration from the spec."""
    return FANT2Config()


def fant2_tiny() -> FANT2Config:
    """A tiny 5M-stored config for fast tests on CPU."""
    return FANT2Config(
        dim=128,
        n_layers=4,
        n_dense_layers=1,
        n_heads=4,
        n_kv_heads=2,
        head_dim=32,
        n_megapools=4,
        n_per_megapool=4,
        n_fractal=16,
        top_k=2,
        n_total_experts=18,
        shared_expert_hidden=64,
        moe_hidden=256,
        kron_A_p=8,
        kron_A_q=4,
        kron_B_p=16,
        kron_B_q=8,
        kron_C_p=16,
        kron_C_q=8,
        max_seq_len=128,
        n_hub_tokens=4,
        local_window=32,
        cerebellum_in_dim=128,
        cerebellum_expand_dim=512,
        cerebellum_out_dim=128,
        apollonian_alpha_cap=128,
        apollonian_beta_cap=128,
        apollonian_retrieval_layers=(2, 3),
    )


def fant2_750m() -> FANT2Config:
    """
    ~750M stored configuration — fits RTX 3060 12GB for training.

    Scaling from 60M default:
        dim:        768 → 2048  (2.67x)
        n_layers:    12 → 24    (2x, 3 dense + 21 MoE)
        n_fractal:   72 → 128   (1.78x, 8 megapools × 16 per pool)
        top_k:        4 → 4     (same)
        moe_hidden: 1280 → 2048 (1.60x)
        max_seq_len: 1024 → 2048

    VRAM estimate (bf16 + grad ckpt + 8-bit AdamW):
        Params:      ~1.5 GB
        Gradients:   ~1.5 GB
        Optimizer:   ~1.5 GB
        Activations: ~1-3 GB (with grad checkpoint)
        Total:       ~6-8 GB → fits RTX 3060 12GB
    """
    return FANT2Config(
        dim=2048,
        n_layers=26,
        n_dense_layers=3,
        n_heads=16,
        n_kv_heads=4,
        head_dim=128,

        # MoE: 8 megapools × 16 experts = 128 fractal
        n_megapools=8,
        n_per_megapool=16,
        n_fractal=128,
        n_special=2,
        top_k=4,
        n_total_experts=130,
        shared_expert_hidden=768,
        moe_hidden=2048,

        # 3-level Kronecker: A(64,16) ⊗ B(32,32) ⊗ C(48,64)
        kron_A_p=64,
        kron_A_q=16,
        kron_B_p=32,
        kron_B_q=32,
        kron_C_p=48,
        kron_C_q=64,

        vocab_size=32768,
        max_seq_len=2048,

        n_hub_tokens=48,
        hub_dim_mult=2.0,
        local_window=192,
        n_attention_sinks=6,

        # Cerebellum (5x expansion)
        cerebellum_in_dim=2048,
        cerebellum_expand_dim=10240,
        cerebellum_out_dim=2048,
        cerebellum_layers=4,

        # Apollonian memory
        apollonian_alpha_cap=10000,
        apollonian_beta_cap=10000,
        apollonian_curvature_threshold=0.5,
        apollonian_retrieval_layers=(24, 25),  # last 2 of 26

        bf16=True,
        grad_checkpoint=True,
        init_std=0.015,
    )


def fant2_2b() -> FANT2Config:
    """
    2B stored / ~3.6B active configuration for serious training.

    Scaling ratios from the 60M default:
        dim:        768 → 3584  (4.67x)
        n_layers:    12 → 28    (2.33x, 3 dense + 25 MoE)
        n_fractal:   72 → 256   (3.56x, 16 megapools × 16 per pool)
        top_k:        4 → 6
        moe_hidden: 1280 → 4608 (3.60x)
        max_seq_len: 1024 → 4096

    Kron factor sizing:
        A(96,16): per-expert, 3 × 96 × 16 = 4,608 scalars each
        B(48,48): per-layer, shared across experts in a layer
        C(64,112): global, shared across all layers

    Expert granularity G = 2 × 3584 / 4608 = 1.56 (optimal range 1.3–2.0)
    Per-expert Kron compression: 4,608 stored → ~10.6M active = 2,304×

    References:
        - DeepSeek-V3: 671B/37B, 256 experts, dim=7168
        - Kimi K2: 1.04T/32B, 384 experts, dim=7168
        - MoE scaling laws (2024-2025): G ∈ [1.3, 2.0] optimal
    """
    return FANT2Config(
        # Core dimensions
        dim=3584,
        n_layers=28,
        n_dense_layers=3,
        n_heads=28,
        n_kv_heads=4,
        head_dim=128,

        # MoE: 16 megapools × 16 experts = 256 fractal
        n_megapools=16,
        n_per_megapool=16,
        n_fractal=256,
        n_special=2,
        top_k=6,
        n_total_experts=258,     # 256 fractal + 2 special
        shared_expert_hidden=896,  # ~dim/4
        moe_hidden=4608,

        # 3-level Kronecker: A(96,16) ⊗ B(48,48) ⊗ C(64,112)
        kron_A_p=96,
        kron_A_q=16,
        kron_B_p=48,
        kron_B_q=48,
        kron_C_p=64,
        kron_C_q=112,

        # Vocabulary & sequence
        vocab_size=32768,
        max_seq_len=4096,

        # Hub attention
        n_hub_tokens=64,
        hub_dim_mult=2.0,
        local_window=256,
        n_attention_sinks=8,

        # Cerebellum (5x expansion at this scale)
        cerebellum_in_dim=3584,
        cerebellum_expand_dim=17920,
        cerebellum_out_dim=3584,
        cerebellum_layers=6,

        # Apollonian memory (scaled for 2B)
        apollonian_alpha_cap=20000,
        apollonian_beta_cap=20000,
        apollonian_curvature_threshold=0.5,
        apollonian_retrieval_layers=(25, 26, 27),  # last 3 of 28 layers

        # Training defaults for 2B scale
        bf16=True,
        grad_checkpoint=True,
        init_std=0.01,  # smaller init for larger model
    )
