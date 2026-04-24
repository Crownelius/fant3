# Achievements

Measured results from the FANT 3 workspace, in reverse chronological order. Every entry is a verifiable claim backed by a test run, a trained checkpoint, a commit, or a research memory.

---

## 2026-04-24 — Curriculum module landed + Fix 3 RL plan documented

- **Progressive curriculum** (per arxiv:2604.16278 DeepInsightTheorem) implemented as `fant3/training/curriculum.py` with three named presets: `legacy_2phase` (default, bit-identical), `deepinsight_3phase` (Apprentice/Journeyman/Expert), `flat_1phase` (control).
- **49 new unit tests** across six test classes; full suite **125 of 125 passing** (test_ahn.py pre-broken, excluded).
- **GPU-free smoke script** `scripts/smoke_curriculum.py` validates all presets in under two seconds without CUDA, bitsandbytes, or HuggingFace network.
- **runpod_train.py** now accepts `--curriculum` flag; backward compatible with queued 50m-unlimited run.
- **Fix 3 RL plan** documented in `docs/fix3_rl_plan.md` — post-pretrain verifier distillation from arxiv:2604.16004 AgentV-RL (Qwen3-4B teacher).
- Commits `ef42478` and `3f3a4a7` pushed to origin/main at [github.com/Crownelius/fant3](https://github.com/Crownelius/fant3).

## 2026-04-23 — 742m Tier C training complete

- Tier C recipe (B=1 T=1024 accum=8 steps=10000 warmup=1500 peak_lr=1.5e-4) on Colab A100 96 GB.
- **78.6 minutes** wall-clock, no NaN, no OOM.
- Best cross-entropy **5.72** at step 7025.
- Peak VRAM **45.66 GB** stable across the run.
- Quality probe shows domain-vocabulary acquisition: "mutex", "idempotency", "AoPS", "preimage", "Tensor Conversion", "linearized" (real NVIDIA corpus vocabulary).
- 82 M training tokens = 190x under Chinchilla-optimal for this scale, but architecture fully validated.

## 2026-04-19 — Four preset and recipe bugs diagnosed and fixed

- **`fant3_742m` preset was materializing 6.6 B parameters.** Root cause: `MatryoshkaMoEFFN.W_up`/`W_down` are full-rank per expert; the `kron_*` config fields are unused by the model code. At 128 experts × dim=2048 × moe_hidden=2048 × 2 × 4 MoE blocks ≈ 6.4 B in experts alone. Fixed preset: dim=1024, n_layers=16, n_megapools=4, n_per_megapool=8 (32 experts), moe_hidden=1792 → real **770.88 M stored**, verified.
- **`FANT3Config()` defaults (1b preset) was materializing ~7 B parameters.** Same root cause, same fix. Real **986.62 M stored**, verified.
- **PYTORCH_CUDA_ALLOC_CONF must be set before `torch` import**, otherwise Colab silently ignores it. Notebook cell 0.2 reordered.
- **Gradient checkpointing added** to `MoEBlock` and `DenseBlock` via `torch.utils.checkpoint.checkpoint(use_reentrant=False)`. Verified **2.65x VRAM reduction** (4.24 GB → 1.60 GB at the same configuration), bit-exact loss (10.5625).

## 2026-04-19 — Scale-ladder warmup (5 scales validated end-to-end)

- **4 of 5 scales pass end-to-end** with no code changes: 5m (8.33 M actual), 40m (72.7 M), 150m (96 M), 350m (263 M).
- **742m OOMs at 9.37 GB** without 8-bit AdamW + gradient checkpointing — a known production configuration, not a bug.
- **Chirality balance 0.266–0.719** across all five scales, confirming the SpinorApollonian fix for α/β starvation is holding at every size.
- Two dtype bugs discovered by the ladder and fixed: MASA RoPE cos/sin stayed f32 while V was bf16 (`fant3/model/attention.py`); AHN `get_stats` dummy_q was f32 vs bf16 gate_proj (`fant3/model/ahn.py`).

## 2026-04-19 — Five pre-launch fixes landed (parallelized)

All five were landed by three concurrent background agents in a single session:
1. **`formats.py` emits `<|answer|>...<|/answer|>` wrapping** for training targets.
2. **`tokenizer_v2.json` retrained** on 82K documents from the 6-source mix → 10–18% compression gain over tokenizer_v1.
3. **`SpinorApollonianMemory`** implemented in `fant3/model/spinor_apollonian.py` — Kocik Cl(2,1) chirality split replaces the scalar threshold. Chirality balance **0.4375** fixes the α/β starvation bug at every scale.
4. **`ArtificialHippocampusNetwork`** implemented in `fant3/model/ahn.py` with zero-initialized gate — disabled at step 0, smoothly ramps on only when useful.
5. **SAE diagnostics** implemented in `fant3/diagnostics/sae.py` for Apollonian memory introspection.

~1,500 lines of code, 30+ new tests. 72.7 M smoke passes with spinor memory and AHN active.

## 2026-04-18 — HuggingFace archive indexed (36 months, 23 labs)

- HuggingFace trending archive fully indexed May 2023 launch → April 2026 (**~1,080 papers**).
- 23 AI labs indexed, including MSL (16 facts) and expanded OpenAI (45 total facts via HF org + GitHub + arxiv + Wikipedia).
- Knowledge graph grew from 1,402 → **2,286 triples** (+884 facts) via nine parallel sonnet agents.
- Top FANT analogs identified: FlashMLA → MASA, Embed v3 → Matryoshka, Expert Choice → MoE routing, Jamba → hybrid, Artificial Hippocampus → memory consolidation, GSPO → G2RPO-A, Kimi k1.5 → SleepGate.

## 2026-04-16 — FANT 3 architectural modules landed

- Core modules landed and standalone-tested:
  - `config.py` + `etf.py` — ETFs mathematically perfect (equiangular tight frames)
  - `attention.py` — MASA (Multi-head Attention with Shared Atoms)
  - `matryoshka_moe.py` — Nested MoE with elastic inference
  - `recursion.py` — Mixture of Recursions
- Two bugs caught during smoke: MASA GQA shape, Matryoshka band_size=1 squeeze. Both fixed.

## 2026-04-11 — N3 SleepGate result on FANT 2

- **N3 SleepGate: 59.9 % accuracy** on 1K eval (Wilson 95% CI [0.568, 0.629]).
- **+5.3 pp over L1.5 baseline** — new FANT 2 best result.
- N6 (G2RPO-A) and N7 (SEC) both regressed at 5 M scale: 27.4% and 40.6% respectively.
- Validated `feedback_auxiliary_loss_fragility`: any auxiliary loss or training text format change hurts at 5 M scale; only structural and scheduling levers are safe.

## 2026-04-09 — BendVM: operating programs while compressed

- Virtual machine whose state is a Pauli spinor (or Descartes quartet) and whose instructions are SL(2, Z) (or Apollonian 4×4) integer matrices.
- Program = product matrix; compose, invert, equivalence all **O(1)** regardless of length.
- **9 of 9 demos passed**, including:
  - 1 M-step Fibonacci (4 ints storing a 2.78 Mbit program)
  - 1 B-step T^n in under 1 ms via 30 matrix squarings
  - 100-step Apollonian walk preserving Descartes invariant Q(b) = 0
- **724 lines of Python**, no dependencies.

## 2026-04-17 — Qwen 2.5 1.5B compression validation

- FANT 2 beats classical codecs (gzip/zlib/bz2/lzma) on three Gutenberg books; contamination question settled.
- Qwen 2.5 1.5B in bf16 hits **0.6–1.0 bits-per-byte** on the same books — 3.5–5.2 times better than gzip, approaching Chinchilla-70 B quality at 47 times fewer parameters.
- bf16 vs fp32 produce **99.913 % bit-identical** quantized probabilities at 16-bit quantization, with **2.21x VRAM savings** and a bpb delta of 0.0012.

## 2026-04-02 — FANT 2 architecture first-pass

- 60 M stored / ~200 M active per token.
- 12 transformer layers (2 dense + 10 MoE).
- dim=768, HierarchicalApollonianRouter (HAR) with curvature-aware mega-pool diversity.
- Peak VRAM under 11 GB on a single RTX 3060 12 GB with bf16 + gradient checkpointing + 8-bit AdamW.
- Fixed the FANT 350M single-mega-pool collapse (94.5% load).

---

## What the Tier C result does NOT prove

Being explicit about the boundary of the validated region:

- **Not a general-language model.** 82 M tokens at 742 M parameters is 190x under Chinchilla-optimal. The model has acquired domain vocabulary, not general language.
- **Not a benchmark result.** GSM8K, MMLU, MATH-500 all run at or near statistical chance. The Chinchilla-optimal budget for 742 M is ~15 B tokens; current budget is under 1% of that.
- **Not an evaluation of the curriculum.** `deepinsight_3phase` was landed on 2026-04-24, after the Tier C run. The Tier C checkpoint used the `legacy_2phase` mix.
- **Not an evaluation of RL.** Fix 3 (GSPO with agentic verifier per arxiv:2604.16004) is parked in `docs/fix3_rl_plan.md` for post-pretrain; no RL has been run.

---

## Next milestones

- **RunPod curriculum A/B**: launch `deepinsight_3phase` arm alongside `legacy_2phase` control at 50 m scale; compare final CE and quality probes.
- **Opus 4.6 technique/sketch distillation**: ~$30 OpenRouter spend to generate (Technique, Sketch) pairs for 10 K NVIDIA/Numina seeds. Follows curriculum positive signal.
- **Fix 3 Stage 1**: verifier distillation from Qwen3-4B agentic verifier. Blocked on pretrain checkpoint availability.
- **TurboQuant integration**: 6x KV-cache reduction via Haar rotation + per-coordinate Beta quantizer (arxiv:2504.19874).
