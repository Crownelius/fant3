"""
FANT 2 constants — special tokens, magic numbers, reserved slots.

The 32 reserved special-token slots at the top of the vocabulary are
defined here to prevent embedding-layer breakage when adding new ones later.
"""

# ----- Vocabulary layout -----
VOCAB_SIZE = 32768
RESERVED_SPECIAL_TOKENS = 32  # top of vocab
BPE_VOCAB_TARGET = VOCAB_SIZE - RESERVED_SPECIAL_TOKENS  # 32736 BPE merges

# ----- Special token names and IDs -----
# IDs are assigned starting at (VOCAB_SIZE - RESERVED_SPECIAL_TOKENS)
SPECIAL_TOKENS = {
    # Core
    "<|pad|>":          VOCAB_SIZE - 32,  # 32736
    "<|bos|>":          VOCAB_SIZE - 31,  # 32737
    "<|eos|>":          VOCAB_SIZE - 30,  # 32738
    "<|unk|>":          VOCAB_SIZE - 29,  # 32739
    # ChatML role tokens
    "<|im_start|>":     VOCAB_SIZE - 28,  # 32740
    "<|im_end|>":       VOCAB_SIZE - 27,  # 32741
    "<|system|>":       VOCAB_SIZE - 26,  # 32742
    "<|user|>":         VOCAB_SIZE - 25,  # 32743
    "<|assistant|>":    VOCAB_SIZE - 24,  # 32744
    # Reasoning markers (Phase 5 GRPO)
    "<|think|>":        VOCAB_SIZE - 23,  # 32745
    "<|/think|>":       VOCAB_SIZE - 22,  # 32746
    "<|answer|>":       VOCAB_SIZE - 21,  # 32747
    "<|/answer|>":      VOCAB_SIZE - 20,  # 32748
    # Tool calling
    "<|tool_call|>":    VOCAB_SIZE - 19,  # 32749
    "<|/tool_call|>":   VOCAB_SIZE - 18,  # 32750
    "<|tool_result|>":  VOCAB_SIZE - 17,  # 32751
    "<|/tool_result|>": VOCAB_SIZE - 16,  # 32752
    # Vision (placeholder for SigLIP2 feature insertion)
    "<|image|>":        VOCAB_SIZE - 15,  # 32753
    "<|/image|>":       VOCAB_SIZE - 14,  # 32754
    "<|patch|>":        VOCAB_SIZE - 13,  # 32755
    # Fill-in-the-middle
    "<|fim_prefix|>":   VOCAB_SIZE - 12,  # 32756
    "<|fim_middle|>":   VOCAB_SIZE - 11,  # 32757
    "<|fim_suffix|>":   VOCAB_SIZE - 10,  # 32758
    # Apollonian retrieval markers
    "<|alpha|>":        VOCAB_SIZE - 9,   # 32759
    "<|beta|>":         VOCAB_SIZE - 8,   # 32760
    # Reserved for future use
    "<|reserved_0|>":   VOCAB_SIZE - 7,   # 32761
    "<|reserved_1|>":   VOCAB_SIZE - 6,   # 32762
    "<|reserved_2|>":   VOCAB_SIZE - 5,   # 32763
    "<|reserved_3|>":   VOCAB_SIZE - 4,   # 32764
    "<|reserved_4|>":   VOCAB_SIZE - 3,   # 32765
    "<|reserved_5|>":   VOCAB_SIZE - 2,   # 32766
    "<|reserved_6|>":   VOCAB_SIZE - 1,   # 32767
}

# Convenience IDs
PAD_ID = SPECIAL_TOKENS["<|pad|>"]
BOS_ID = SPECIAL_TOKENS["<|bos|>"]
EOS_ID = SPECIAL_TOKENS["<|eos|>"]
UNK_ID = SPECIAL_TOKENS["<|unk|>"]
THINK_ID = SPECIAL_TOKENS["<|think|>"]
THINK_END_ID = SPECIAL_TOKENS["<|/think|>"]
ANSWER_ID = SPECIAL_TOKENS["<|answer|>"]
ANSWER_END_ID = SPECIAL_TOKENS["<|/answer|>"]
IM_START_ID = SPECIAL_TOKENS["<|im_start|>"]
IM_END_ID = SPECIAL_TOKENS["<|im_end|>"]

# ----- Routing constants -----
N_MEGAPOOLS = 8
N_PER_MEGAPOOL = 9
N_FRACTAL_EXPERTS = N_MEGAPOOLS * N_PER_MEGAPOOL  # 72
N_SPECIAL_EXPERTS = 2  # zero + copy
N_TOTAL_EXPERTS = N_FRACTAL_EXPERTS + N_SPECIAL_EXPERTS  # 74 (shared expert is separate)
TOP_K_FRACTAL = 4

# ----- Apollonian memory -----
ALPHA_CAP = 5000
BETA_CAP = 5000
ALPHA_CURVATURE_THRESHOLD = 0.5  # above → α (instances), below → β (schemas)

# ----- Hub attention -----
N_HUB_TOKENS = 32
LOCAL_WINDOW = 128
N_ATTENTION_SINKS = 4

# ----- Telemetry collection cadence -----
TELEMETRY_EVERY_N_STEPS = 500
TIKKUN_CHECK_EVERY_N_STEPS = 200
BIPARTITION_CHECK_EVERY_N_STEPS = 1000

# ----- Streaming data -----
MAX_DISK_GB = 10.0
MAX_HF_CACHE_GB = 5.0
