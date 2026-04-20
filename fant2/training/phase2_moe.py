"""
Phase 2: MoE expert specialization via the FEP unified loss.

This is the core FANT 2 training phase — where the 72 fractal experts learn
to specialize under the hierarchical Apollonian router. Loss components:

    L_FEP = CE(next_token)
          + α · router_z_loss                       (OLMoE numerical stability)
          + β · KL(router_dist ‖ uniform_prior)     (FEP expected free energy)

The DeepSeek aux-loss-free router bias is updated inside the trainer after
every backward pass (gradient-free, prevents softmax interference with MoE
specialization — the fix for FANT 350M's router collapse failure mode).

Expected behaviour at convergence:

  * mean pairwise JSD between domain routings ≥ 0.30  (the success metric)
  * Parisi P(q) entropy in (1.0, ln(72)) — a non-trivial ultrametric tree
  * box-counting dimension of routing decisions in (0.5, 2.0)

Usage
-----

    python -m fant2.training.phase2_moe \\
        --resume output/phase1/final.pt \\
        --preset default \\
        --n-steps 40000 \\
        --batch-size 8 --seq-len 1024 \\
        --use-hf \\
        --out-dir output/phase2
"""

import sys

from .phase_common import make_phase_parser, build_everything
from .trainer import TrainConfig, FANT2Trainer


PHASE = 2
DESCRIPTION = "MoE expert specialization (FEP unified loss)"


def main() -> int:
    parser = make_phase_parser(PHASE, DESCRIPTION)
    # Phase 2 specific: FEP annealing schedule
    parser.add_argument("--z-loss-alpha", type=float, default=1e-3,
                        help="coefficient on router z-loss")
    parser.add_argument("--fep-kl-beta-init", type=float, default=0.1,
                        help="initial FEP KL weight")
    parser.add_argument("--fep-kl-beta-max", type=float, default=1.0,
                        help="final (post-anneal) FEP KL weight")
    parser.add_argument("--fep-kl-anneal-steps", type=int, default=5000,
                        help="linear anneal steps for FEP KL weight")
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
        # FEP loss hyperparams
        z_loss_alpha=args.z_loss_alpha,
        fep_kl_beta_init=args.fep_kl_beta_init,
        fep_kl_beta_max=args.fep_kl_beta_max,
        fep_kl_anneal_steps=args.fep_kl_anneal_steps,
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
