"""
Phase 5: Dr.GRPO RL on procedurally-generated math problems.

This is the real entry point for Phase 5 (Dr.GRPO done right). The trainer
hook in `trainer._phase5_grpo_forward` handles per-step rollout / reward /
loss; this script wires up the data stream, the frozen reference policy,
and runs `FANT2Trainer.train()`.

**Training data: NO public benchmarks.** Per the user-imposed constraint,
the spec's nominal Phase 5 targets (GSM8K / MATH / HumanEval) are reserved
for *evaluation only*. The training stream is `ProceduralMathStream`, which
generates math word problems on the fly from random templates and sampled
values. No external dataset is touched.

Spec references (still apply for hyperparams and acceptance gates):
  * §8 Phase 5: G=16 rollouts, ε_hi=0.28 clip-higher, lr=5e-7
  * §11 #10:    GSM8K pass@1 ≥ 25% (eval-time only — see bench/ harness)

Usage
-----

    python -m fant2.training.phase5_grpo \\
        --resume output/phase4/final.pt \\
        --preset default \\
        --n-steps 5000 \\
        --out-dir output/phase5 \\
        --batch-size 2 \\
        --grpo-n-rollouts 16
"""

import copy
import sys

import torch

from .phase_common import make_phase_parser, build_tokenizer, build_model
from .phase5_rollout import ProceduralMathStream, Phase5BatchStream
from .trainer import TrainConfig, FANT2Trainer


PHASE = 5
DESCRIPTION = "Dr.GRPO RL on procedurally-generated math problems"


def main() -> int:
    parser = make_phase_parser(PHASE, DESCRIPTION)
    parser.add_argument("--grpo-n-rollouts", type=int, default=8,
                        help="rollouts per prompt for Dr.GRPO group "
                             "(spec default: 16; lower for tiny preset / CPU)")
    parser.add_argument("--grpo-max-new-tokens", type=int, default=96)
    parser.add_argument("--grpo-temperature", type=float, default=0.9)
    parser.add_argument("--grpo-top-p", type=float, default=0.95)
    parser.add_argument("--grpo-clip-eps", type=float, default=0.20)
    parser.add_argument("--grpo-clip-eps-hi", type=float, default=0.28,
                        help="DAPO clip-higher upper bound (spec §8 Phase 5)")
    parser.add_argument("--problem-seed", type=int, default=0,
                        help="seed for ProceduralMathStream")
    parser.add_argument("--max-value", type=int, default=20,
                        help="max integer value sampled in math templates")
    args = parser.parse_args()

    print(f"=== FANT 2 Phase {PHASE}: {DESCRIPTION} ===")
    print(f"  Training data: ProceduralMathStream (no public benchmark)")
    print(f"  G = {args.grpo_n_rollouts}, eps = ({args.grpo_clip_eps}, "
          f"+{args.grpo_clip_eps_hi})")

    # ----- Tokenizer + model -----
    tokenizer = build_tokenizer(args)
    model, cfg = build_model(args)
    if tokenizer.vocab_size != cfg.vocab_size:
        print(
            f"  WARNING: tokenizer.vocab_size={tokenizer.vocab_size} != "
            f"cfg.vocab_size={cfg.vocab_size}."
        )

    # ----- Phase 5 data stream (procedural math, NOT a benchmark) -----
    problem_stream = ProceduralMathStream(
        seed=args.problem_seed,
        max_value=args.max_value,
    )
    batch_stream = Phase5BatchStream(
        tokenizer=tokenizer,
        problems=problem_stream,
        batch_size=args.batch_size,
        device="cpu",  # the trainer moves dummy tensors to device on its own
    )

    train_cfg = TrainConfig(
        phase=PHASE,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        grad_accum=args.grad_accum,
        muon_lr=args.muon_lr,
        adam_lr=args.adam_lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        use_8bit_adam=(not args.no_8bit_adam) and args.device == "cuda",
        # FEP keeps running at the final annealed value (not directly used by
        # the GRPO hook but kept for any auxiliary calls)
        z_loss_alpha=1e-3,
        fep_kl_beta_init=1.0,
        fep_kl_beta_max=1.0,
        fep_kl_anneal_steps=1,
        # Phase 5 GRPO knobs
        grpo_n_rollouts=args.grpo_n_rollouts,
        grpo_max_new_tokens=args.grpo_max_new_tokens,
        grpo_temperature=args.grpo_temperature,
        grpo_top_p=args.grpo_top_p,
        grpo_clip_eps=args.grpo_clip_eps,
        grpo_clip_eps_hi=args.grpo_clip_eps_hi,
        # Cadence
        log_every=args.log_every,
        save_every=args.save_every,
        # Checkpoint
        out_dir=args.out_dir,
        resume_from=args.resume,
        # Hardware
        device=args.device,
        bf16=args.bf16,
        grad_checkpoint=(not args.no_grad_checkpoint),
    )

    trainer = FANT2Trainer(model, train_cfg, batch_stream)

    # ----- Frozen reference policy for Dr.GRPO old_logps -----
    # We deep-copy the model AFTER loading the resume checkpoint (which
    # FANT2Trainer.__init__ already did) so the ref captures the starting
    # weights. The reference is held in eval mode and never updated.
    print("  Cloning frozen reference policy for Dr.GRPO old_logps...")
    ref_model = copy.deepcopy(trainer.model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    trainer.ref_model = ref_model

    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
