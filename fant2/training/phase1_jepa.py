"""
Phase 1: LLM-JEPA + SIGReg self-supervised pretraining.

This phase trains the FANT 2 model with:

    L = CE(next_token)  +  JEPA(pred, detach(target))  +  SIGReg(variance)

The JEPA target is the mean-pooled embedding of the *second half* of the
sequence; the predictor sees the mean of the *first half*. This gives the
model a bootstrapped self-distillation signal without needing an EMA
teacher network (saves ~100M VRAM on RTX 3060).

SIGReg (variance-invariance-covariance regularization, simplified) ensures
the predictor does not collapse to a constant by penalizing `max(0, 1 - std)`.

The unified driver `FANT2Trainer.train_step()` already handles phase==1:
it runs `llm_jepa_loss()` and adds it to the CE.

Usage
-----

    python -m fant2.training.phase1_jepa \\
        --preset default \\
        --n-steps 20000 \\
        --batch-size 8 --seq-len 1024 \\
        --use-hf \\
        --out-dir output/phase1
"""

import os
import sys

from .phase_common import make_phase_parser, build_everything
from .trainer import TrainConfig, FANT2Trainer


PHASE = 1
DESCRIPTION = "LLM-JEPA self-supervised pretraining (+ SIGReg variance penalty)"


def main() -> int:
    parser = make_phase_parser(PHASE, DESCRIPTION)
    # Phase 1 specific: SIGReg + JEPA have no tunables exposed here
    parser.add_argument("--jepa-warmup", type=int, default=200,
                        help="steps before JEPA loss is added at full weight "
                             "(not implemented yet; placeholder)")
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
        # FEP: Phase 1 does not apply the FEP KL prior (JEPA is the main signal)
        # but we still keep z_loss active because it's cheap.
        z_loss_alpha=1e-3,
        fep_kl_beta_init=0.0,
        fep_kl_beta_max=0.0,
        fep_kl_anneal_steps=1,
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
