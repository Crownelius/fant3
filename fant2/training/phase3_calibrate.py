"""
Phase 3: Active-layer calibration (rank / condition-number repair).

This phase adds a `calibration_loss` on randomly-materialized fractal expert
weights to the base FEP loss. The goal is to keep the A_expert / B_layer
kron factors well-conditioned so that the effective top-k-materialized
SwiGLU weights never suffer rank collapse or blow-up singular values.

Specifically, for each training step we:

  1. Pick `calib_n_samples` random (layer, expert) pairs
  2. Materialize their kron(A, B) W_gate / W_up / W_down
  3. Compute effective rank and condition number of each
  4. Penalize:
        * max(0, 0.9 - eff_rank / full_rank)   (rank collapse)
        * max(0, (σ_max/σ_min) - 100) / 100    (ill-conditioning)

Gradients flow back through the kron op into A and B.

Usage
-----

    python -m fant2.training.phase3_calibrate \\
        --resume output/phase2/final.pt \\
        --preset default \\
        --n-steps 10000 \\
        --calib-weight 0.1 \\
        --out-dir output/phase3
"""

import sys

from .phase_common import make_phase_parser, build_everything
from .trainer import TrainConfig, FANT2Trainer


PHASE = 3
DESCRIPTION = "Active-layer calibration (rank + condition number repair)"


def main() -> int:
    parser = make_phase_parser(PHASE, DESCRIPTION)
    parser.add_argument("--calib-weight", type=float, default=0.1,
                        help="weight on the calibration loss term")
    parser.add_argument("--calib-n-samples", type=int, default=4,
                        help="how many random (layer, expert) pairs per step")
    parser.add_argument("--calib-rank-target-frac", type=float, default=0.9,
                        help="target (effective rank / full rank) floor")
    parser.add_argument("--calib-max-condition", type=float, default=100.0,
                        help="maximum allowed condition number before penalty")
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
        # Calibration knobs (new in Phase 3)
        calib_weight=args.calib_weight,
        calib_n_samples=args.calib_n_samples,
        calib_rank_target_frac=args.calib_rank_target_frac,
        calib_max_condition=args.calib_max_condition,
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
