"""
FANT 2 command-line entry points.

Provides a single unified `fant2` CLI with subcommands for every phase
and utility. See the individual modules for full argument lists.

    fant2 train-phase0 [args]   # train the BPE tokenizer
    fant2 train-phase1 [args]   # JEPA pretraining
    fant2 train-phase2 [args]   # MoE specialization
    fant2 train-phase3 [args]   # Active-layer calibration
    fant2 train-phase4 [args]   # Self-refinement + STaR
    fant2 train-phase5 [args]   # Dr.GRPO RL (stub)
    fant2 train-phase6 [args]   # SimPO + KTO (stub)
    fant2 generate     [args]   # text generation
    fant2 chat         [args]   # interactive chat
    fant2 eval-ppl     [args]   # perplexity evaluation
    fant2 eval-gsm8k   [args]   # GSM8K accuracy
    fant2 eval-arc     [args]   # ARC multiple choice
    fant2 eval-hellaswag [args] # HellaSwag multiple choice
    fant2 info         [args]   # print model + tokenizer summary
"""

from .main import main

__all__ = ["main"]
