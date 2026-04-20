"""
Phase 6: SimPO + KTO preference optimization.

Final alignment phase. Uses two complementary methods:

  * SimPO (Meng et al. 2024): length-normalized preference optimization
    that does NOT need a reference model. Cheap and stable.

  * KTO (Ethayarajh et al. 2024): Kahneman-Tversky Optimization, which
    uses Prospect Theory's asymmetric value function. Unlike DPO, KTO
    does not require strict (chosen, rejected) pairs — it can use
    independent positive and negative examples with separate weights.

Both losses are implemented in `fant2.training.losses`. The trainer hook
in `_phase6_simpo_kto_forward` computes the composite per-step; this
script wires up the data stream, the frozen reference policy, and runs
`FANT2Trainer.train()`.

**Training data: NO public benchmarks.** Per the user-imposed constraint,
the spec's nominal Phase 6 sources (Tulu 3 / Magpie-Pro / UltraFeedback)
are reserved for *evaluation only*. The training stream is
`SyntheticPreferenceStream`, which derives (chosen, rejected) triples
from the same procedural math problems used in Phase 5. The chosen
response is well-formatted and correct; the rejected response is one of
three failure modes (wrong number, unformatted, unhelpful). No external
data is touched.

Spec references (still apply for hyperparams):
  * §8 Phase 6: SimPO + KTO composite, β=2.0, γ=1.6, KTO β=0.1
  * §9 / losses.py: simpo_loss, kto_loss

Usage
-----

    python -m fant2.training.phase6_simpo_kto \\
        --resume output/phase5/final.pt \\
        --preset default \\
        --n-steps 5000 \\
        --out-dir output/phase6 \\
        --batch-size 2
"""

import copy
import sys

from .phase_common import make_phase_parser, build_tokenizer, build_model
from .phase6_pref import SyntheticPreferenceStream, Phase6BatchStream
from .trainer import TrainConfig, FANT2Trainer


PHASE = 6
DESCRIPTION = "SimPO + KTO preference optimization (alignment)"


def main() -> int:
    parser = make_phase_parser(PHASE, DESCRIPTION)
    parser.add_argument("--simpo-beta", type=float, default=2.0,
                        help="SimPO temperature β")
    parser.add_argument("--simpo-gamma", type=float, default=1.6,
                        help="SimPO target margin γ")
    parser.add_argument("--kto-beta", type=float, default=0.1,
                        help="KTO temperature β")
    parser.add_argument("--kto-weight", type=float, default=0.5,
                        help="weight on KTO term in composite loss")
    parser.add_argument("--problem-seed", type=int, default=0,
                        help="seed for SyntheticPreferenceStream")
    parser.add_argument("--max-value", type=int, default=20,
                        help="max integer value sampled in math templates")
    args = parser.parse_args()

    print(f"=== FANT 2 Phase {PHASE}: {DESCRIPTION} ===")
    print(f"  Training data: SyntheticPreferenceStream (no public benchmark)")
    print(f"  SimPO: β={args.simpo_beta}, γ={args.simpo_gamma}")
    print(f"  KTO:   β={args.kto_beta}, weight={args.kto_weight}")

    # ----- Tokenizer + model -----
    tokenizer = build_tokenizer(args)
    model, cfg = build_model(args)
    if tokenizer.vocab_size != cfg.vocab_size:
        print(
            f"  WARNING: tokenizer.vocab_size={tokenizer.vocab_size} != "
            f"cfg.vocab_size={cfg.vocab_size}."
        )

    # ----- Phase 6 data stream (synthetic preference, NOT a benchmark) -----
    pair_stream = SyntheticPreferenceStream(
        seed=args.problem_seed,
        max_value=args.max_value,
    )
    batch_stream = Phase6BatchStream(
        tokenizer=tokenizer,
        pairs=pair_stream,
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
        # FEP keeps running at the final annealed value (not used by hook,
        # but kept for any auxiliary calls)
        z_loss_alpha=1e-3,
        fep_kl_beta_init=1.0,
        fep_kl_beta_max=1.0,
        fep_kl_anneal_steps=1,
        # Phase 6 SimPO + KTO knobs
        simpo_beta=args.simpo_beta,
        simpo_gamma=args.simpo_gamma,
        kto_beta=args.kto_beta,
        kto_weight=args.kto_weight,
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

    # ----- Frozen reference policy for KTO -----
    # Deep-copy the model AFTER loading the resume checkpoint (which
    # FANT2Trainer.__init__ already did) so the ref captures the
    # post-Phase-5 weights. The reference is held in eval mode and never
    # updated.
    print("  Cloning frozen reference policy for KTO...")
    ref_model = copy.deepcopy(trainer.model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)
    trainer.ref_model = ref_model

    trainer.train()
    return 0


if __name__ == "__main__":
    sys.exit(main())
