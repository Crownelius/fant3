"""
Phase 4: Self-refinement + STaR + Apollonian fill.

This phase teaches the model to:

  1. Judge its own outputs via the success_estimator head (STaR-style
     self-teaching: if the top-1 argmax matches the true next token, that
     position is "correct", else "incorrect")
  2. Fill the Apollonian α / β memory packs with final-hidden embeddings
     classified by curvature (high-curvature = α = instance memory,
     low-curvature = β = schema memory)

Two extensions are planned but deferred to follow-ups:

  * True two-pass refinement where pass-2 sees pass-1's JEPA prediction
    prepended as a virtual token
  * Soft-label distillation from the frozen Phase 3 checkpoint

Usage
-----

    python -m fant2.training.phase4_refine \\
        --resume output/phase3/final.pt \\
        --preset default \\
        --n-steps 10000 \\
        --refine-weight 0.5 \\
        --out-dir output/phase4
"""

import sys

from .phase_common import make_phase_parser, build_everything
from .trainer import TrainConfig, FANT2Trainer


PHASE = 4
DESCRIPTION = "Self-refinement + STaR success estimator + Apollonian fill"


def main() -> int:
    parser = make_phase_parser(PHASE, DESCRIPTION)
    parser.add_argument("--refine-weight", type=float, default=0.5,
                        help="weight on the success-estimator BCE loss")
    args = parser.parse_args()

    print(f"=== FANT 2 Phase {PHASE}: {DESCRIPTION} ===")
    model, cfg, stream, _tok = build_everything(args)

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
        # FEP keeps running at the final annealed value
        z_loss_alpha=1e-3,
        fep_kl_beta_init=1.0,
        fep_kl_beta_max=1.0,
        fep_kl_anneal_steps=1,
        # Refinement knob
        refine_weight=args.refine_weight,
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

    trainer = FANT2Trainer(model, train_cfg, stream)
    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
