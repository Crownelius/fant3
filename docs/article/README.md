# The FANT story

How FANT 3 came to look the way it does. This is the narrative companion to the README and the theory guide — it reads in order, while those are references.

## Prologue: the 60 M question

In early 2026 the FANT 2 workspace was running 60 M stored / 200 M active on a single RTX 3060 12 GB. It worked. The HierarchicalApollonianRouter (HAR) solved the FANT 350 M single-mega-pool collapse (94.5% load on one expert). Peak VRAM stayed under 11 GB. Training was stable. Evaluation was unambitious but honest.

And yet the numbers weren't going anywhere. GSM8K at 5 M parameters sits at chance. Even at 60 M, the model had learned some arithmetic patterns but nothing that would transfer outside its training distribution. The Chinchilla-optimal token budget for 60 M is ~1.2 B tokens, and we'd only trained on ~100 M. Scaling up was the obvious answer. But scaling up meant leaving the 3060.

## Chapter 1: the preset bug

The first version of FANT 3 was written with the same design grammar as FANT 2 — a preset function per size, named after the intended parameter count. `fant3_742m()` was supposed to return a 742 M model. `fant3_1b()` was supposed to return a 1 B model.

They didn't.

On 2026-04-19, running a VRAM audit before the first Colab A100 launch, the real parameter counts came back: **6.6 B for `fant3_742m`, roughly 7 B for `fant3_1b`**. The root cause was that `MatryoshkaMoEFFN` materializes full-rank expert weight matrices, and the `kron_*` fields in the config (meant to factor the weights as Kronecker products) were unused by the model code. At 128 experts × dim=2048 × moe_hidden=2048 × 2 × 4 MoE blocks, the experts alone were 6.4 B.

The fix was to shrink the preset dimensions: 742m became dim=1024, n_layers=16, n_megapools=4, n_per_megapool=8 (32 experts), moe_hidden=1792 → real **770.88 M stored**, verified by `sum(p.numel() for p in model.parameters())`. 1b became dim=1024, n_layers=20, moe_hidden=2304 → real **986.62 M stored**.

The lesson became a rule: never trust a preset name. Always count.

## Chapter 2: the gradient checkpointing OOM

The fixed 742m preset launched on Colab A100 80 GB. It OOMed at step 10 with 93 GB of 94.97 GB allocated. The Colab estimate had said 35 GB; the allocator actually wanted 93 GB. The missing factor was activation memory, which scales linearly with sequence length, and the first launch was at T=1024 which was 2x what the estimate assumed.

The fix was gradient checkpointing. Wrap each `DenseBlock` and `MoEBlock` in `torch.utils.checkpoint.checkpoint(use_reentrant=False)`; checkpoint each MoR pass independently. Verified a 2.65x VRAM reduction (from 4.24 GB to 1.60 GB at a smaller configuration) and bit-exact loss (10.5625 → 10.5625).

After that the 742m Tier C run ran clean. 78.6 minutes, best CE 5.72, no NaN, no OOM.

## Chapter 3: the chirality starvation

The Spinor Apollonian memory was designed to split hidden states into two packs: α for instance memory (high chirality), β for schema memory (low chirality). The first implementation used a scalar-curvature classifier with a learned threshold.

At every scale the threshold misbehaved. Either α got 100% of tokens (β starved) or β got 100% (α starved). The model learned to keep only one pack. The whole point of having two packs — the instance/schema split — evaporated.

The fix came from Kocik 2001.05866, "Spinors and Descartes." The idea: classify hidden states by the sign of the Descartes invariant, which is the Minkowski quadratic form in signature (1,3), computed over a 4-vector projection of the hidden state. The classification is fundamental, not threshold-based. The invariant is zero on the Apollonian packing surface and nonzero inside/outside.

On 2026-04-19, the spinor classifier was verified across all scales with chirality balance:
- 5m: 0.266
- 40m: 0.447
- 150m: 0.500
- 350m: 0.719

All within the 0.2–0.8 healthy band. No starvation.

## Chapter 4: the curriculum question

By mid-April 2026 the architecture was solid. The training loop was stable. The question became: what do we feed it?

The default mix was flat — all 11 sources interleaved from step 0. FineWeb-Edu 20%, NVIDIA reasoning 60%, Opus + Sonnet + Kimi distillations 20%. It worked, but it didn't obviously win.

Then on 2026-04-24 two arxiv papers came in for review:

- **arxiv:2604.16278 DeepInsightTheorem** (Li et al., CityU HK + Tsinghua + Ke Holdings): three-stage progressive SFT curriculum (Apprentice/Journeyman/Expert) on hierarchically annotated proofs. 7 B reaches RL-trained performance through SFT alone. **1 B – 3 B sees disproportionate boost.**

- **arxiv:2604.16004 AgentV-RL** (Fudan + ByteDance + HUST + HKU): Qwen3-4B agentic verifier (Forward + Backward Plan-Validate-Verdict) with GRPO + DAPO asymmetric clip. 4 B verifier beats a 70 B ORM by +25.2 pp on MATH500 at N=128 Best-of-N.

The first paper spoke directly to us: FANT 3's target band is exactly 1 B–3 B. A cross-check against MemPalace surfaced three independent prior-art papers (arxiv:2510.14865 midtraining, arxiv:2510.01631 synthetic mixtures, arxiv:2510.25741 Ouro + arxiv:2511.07384 Retrofitted Recurrence) all pointing the same direction.

The landing on 2026-04-24: `fant3/training/curriculum.py` with three named presets. Default `legacy_2phase` is bit-identical to the pre-curriculum mix (backward compat). `deepinsight_3phase` is the paper's schedule mapped to our 11-source registry. `flat_1phase` is a no-curriculum control arm.

The A/B test is the next training run.

## Chapter 5: what comes next

Fix 3 is parked in [docs/fix3_rl_plan.md](../fix3_rl_plan.md). AgentV-RL showed that an agentic verifier adds +25.2 pp on MATH at N=128. Our inference budget is N=1–8, so the transfer is unknown. The plan: post-pretrain, distill the Qwen3-4B verifier into a 1B FANT student via rejection-sampling SFT, then GRPO with a composite reward (outcome + verifier).

TurboQuant (arxiv:2504.19874) is on the queue for post-training KV-cache compression. TRIM-KV (arxiv:2512.03324) for Apollonian memory eviction. A BendVM experiment lives at `bendvm/` — a virtual machine whose state is a Pauli spinor and whose instructions are SL(2,Z) matrices — as a thought experiment about operating programs while compressed.

The repository exists to answer one question: what would a language model look like if you designed it for exactly one GPU, one developer, and the 1 B – 3 B band where different things start to matter?

This is the work-in-progress answer.

---

## Further reading

- [README.md](../../README.md) — workspace-level overview
- [ACHIEVEMENTS.md](../../ACHIEVEMENTS.md) — measured results, reverse chronological
- [THEORY/README.md](../THEORY/README.md) — the mathematics per component
- [fix3_rl_plan.md](../fix3_rl_plan.md) — the RL plan
