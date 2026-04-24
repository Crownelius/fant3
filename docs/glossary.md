# Glossary

Terms, abbreviations, and idioms used across the FANT 3 codebase and documentation.

## Architecture

**AHN — Artificial Hippocampus Network.** A sliding FIFO of the last `short_window` hidden states plus a compressed long-term buffer, applied as a zero-initialized gated residual before the final layer norm. Source: ByteDance 2025.

**α-pack / β-pack.** The two memory packs in Spinor Apollonian memory. α holds instance (high-chirality) tokens; β holds schema (low-chirality) tokens. Split by the sign of the Descartes invariant.

**Apollonian memory.** A memory module organized around the Apollonian packing structure. FANT 3 uses Kocik tangency spinors (Cl(2,1)) rather than the scalar-curvature classifier of earlier drafts.

**Cerebellum.** A fixed 25 M-parameter echo-state reservoir (spectral radius 0.95) with a trainable linear Purkinje readout. Active at 742m and 1b scales.

**Chirality.** The sign of the Descartes invariant of a hidden state's 4-vector projection. Determines α vs β pack assignment in Spinor Apollonian memory. Healthy balance: 0.2–0.8 across a training batch.

**Dense block.** A standard transformer block (dense FFN, no MoE). FANT 3 uses 2 dense blocks as the first two layers, then MoE blocks for the rest.

**ETF — Equiangular Tight Frame.** A matrix whose rows are unit vectors with equal pairwise inner product. Used to initialize routers; `W W^T = I` with off-diagonal `-1/(n-1)`. Routers are frozen after `etf_freeze_after_step` steps.

**MASA — Multi-head Attention with Shared Atoms.** All layers share a dictionary of `n_attention_atoms` basis matrices; per-layer attention is a rank-`masa_coef_rank` linear combination. Cuts attention parameter count dramatically.

**Matryoshka MoE.** Mixture of Experts where experts are organized as nested megapools. Inference can use any prefix of the expert sequence, so one trained checkpoint serves multiple compute budgets. Source: Wang et al., arXiv:2509.26520.

**MoE — Mixture of Experts.** An FFN variant where only `top_k` of `N` experts are activated per token. Here, top_k=1 or top_k=2, depending on the preset.

**MoR — Mixture of Recursions.** A mechanism where each token chooses how many times to loop through a shared recursion block (1, 2, or 3 passes). Implemented with contractive-alpha decay for convergence guarantees.

**Purkinje readout.** The trainable linear layer in the Cerebellum module. Named after the Purkinje cells in the mammalian cerebellum which read out the fixed reservoir's response.

**Spinor Apollonian memory.** See Apollonian memory. The "spinor" qualifier distinguishes the Kocik chirality-based classifier from earlier scalar-curvature versions.

**Stored parameters.** Total parameter count materialized in VRAM/disk. Distinct from **active parameters** which is what's computed per token.

## Training

**8-bit AdamW.** AdamW optimizer with int8 state via bitsandbytes. Halves optimizer memory vs fp32 state.

**bf16 — bfloat16.** 16-bit float with 8-bit exponent (same range as fp32) and 7-bit mantissa. Numerically stable for training; half the memory of fp32.

**Cross-entropy loss (CE).** Standard LM objective: `-log P(token_i | prefix_i)` averaged over non-pad positions. Reported in nats (natural log).

**Gradient checkpointing.** Recomputes activations during backward instead of storing them forward. Trades ~30% wall time for 2–3x VRAM reduction. Mandatory at 742m+ scales.

**LR schedule.** Learning rate schedule. FANT 3 uses Litim compact-support: `1 - (t/T)^2` clamped to [0, 1] after warmup. Smoother than cosine at the right endpoint. Source: Phys. Rev. D 64 105007.

**NaN step.** A training step where all micro-batch losses evaluate to NaN. Usually caused by bf16 overflow at bad LR schedules. Counter `consec_nan` stops training after `max_nan_steps` consecutive NaN steps.

**Peak LR.** The maximum learning rate, reached at the end of linear warmup. Typical: 1.5e-4 to 3e-4 for 50m+.

**Warmup steps.** Linear LR ramp from 0 to peak_lr over `warmup_steps` steps. Typical: 500–1,500 steps.

**z-loss / z-coef.** Router entropy regularizer. Coefficient `z_coef` multiplies the log-sum-exp of router logits; prevents logit magnitudes from exploding.

## Data

**13-gram SHA-1 filter.** The decontamination cache. Every 13-token window is hashed with SHA-1; if the hash appears in the test-set cache (457,910 hashes across GSM8K, MMLU, MATH-500), the training sample is rejected.

**Chinchilla-optimal.** The token-to-parameter ratio (~20 tokens per parameter) at which training loss is optimal for a given compute budget. Hoffmann et al. 2022. FANT 3 at 742m Tier C was trained at 82 M tokens, which is ~190x under Chinchilla-optimal.

**FineWeb-Edu.** The CC-BY-licensed filtered web corpus used as a general-language anchor in every curriculum phase.

**MIX v3.** The NVIDIA-heavy 11-source training mix. 60% NVIDIA, 20% FineWeb, 12% Sonnet 4.6, 8% Opus 4.6.

**MIX v4.** The chat-focused 12-source mix for small scales. 22% Sonnet 4.6, plus Cascade-2 chat/IF, FineTome, Daring-Anteater.

**Tokenizer v2.** The BPE tokenizer trained on 82K documents from the 6-source distillation mix. 32,768 tokens. 10–18% compression gain over v1.

## Concepts

**A/B test.** Two training runs with the same recipe except for one lever (e.g. curriculum preset). Compare CE and quality probes.

**Backward compat.** A rule for this repository: new levers default to reproducing prior behavior bit-exactly. See [feedback_backward_compat_refactors](../CLAUDE.md#conventions-for-changes).

**CE probe.** A held-out CE evaluation on a fixed set of batches, run every `ce_probe_every` steps during training. Measures generalization, not just training-set fit.

**Curriculum.** The phase-dependent data weighting schedule. Named presets in `fant3/training/curriculum.py`: `legacy_2phase`, `deepinsight_3phase`, `flat_1phase`.

**Smoke test.** A fast (under 2 seconds) end-to-end dry-run that validates wiring without requiring real training. FANT 3 uses `scripts/smoke_*.py` scripts paired with pytest.

**Tier A/B/C/D.** Training recipe tiers, calibrated for specific VRAM envelopes. Tier C is 742m-native: B=1 T=1024 accum=8 steps=10000 warmup=1500 peak_lr=1.5e-4.

**Wilson 95% CI.** The Wilson score interval for a binomial proportion. Used for benchmark accuracy reporting because it doesn't degenerate near 0% or 100%.

## Hardware

**A100 40 GB / 80 GB / 96 GB.** The NVIDIA Ampere-class GPUs FANT 3 is tuned for. Colab Pro+ provides A100 40 GB; RunPod offers 80 GB and 96 GB variants.

**Colab.** Google Colab. Secondary training target. Run via the notebook.

**CUDA OOM.** Out-of-memory error from the CUDA allocator. Diagnose with actual param count + activation size + optimizer state. See [DEVELOPER_GUIDE](./DEVELOPER_GUIDE/README.md#cuda-out-of-memory).

**RTX 3060 12 GB.** The original FANT target hardware; FANT 2's design is calibrated for it. FANT 3 can train at 150m on this card with bf16 + 8-bit AdamW + gradient checkpointing.

**RunPod.** The primary training target. GPU rental service; A100 pods start under $2/h.

**VRAM.** GPU memory. Sized against stored parameters + activation memory + optimizer state.

## People and papers

**Chinchilla.** Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models."

**DAPO.** Yu et al. 2025 — asymmetric clip variant of GRPO. Used in Fix 3 plan.

**DeepInsightTheorem.** Li et al., CityU HK + Tsinghua + Ke Holdings, April 2026. arxiv:2604.16278.

**Delétang et al.** 2023, "Language Models are Compressors." The framing.

**GRPO.** Group Relative Policy Optimization. An RL algorithm in the PPO family. Used by AgentV-RL.

**Kocik.** Jerzy Kocik, arXiv:2001.05866. Tangency spinors in Minkowski space; foundational for Spinor Apollonian memory.

**Litim.** Daniel Litim. "Optimised regulator" in Phys. Rev. D 64 105007. Source of the compact-support LR schedule.

**Parisi RSB.** Replica symmetry breaking, Giorgio Parisi. arXiv:2604.11921 extends the analysis up to the de Almeida-Thouless line; theoretical grounding for MoE routing diversity.

**TRIM-KV.** arXiv:2512.03324. Retention-gate KV cache eviction; planned for Apollonian memory.

**TurboQuant.** arxiv:2504.19874 (ICLR 2026). 6x KV cache reduction via Haar rotation + per-coordinate Beta quantizer.
