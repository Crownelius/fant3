# Scripts — Catalog

All runnable scripts live in the `scripts/` directory. Run each with `PYTHONPATH=.` from the project root (`/d/FANT_TRAINING_D_Drive/fant2/`).

---

## Training scripts

### `train_overnight.py`
Overnight FANT 2 training run on a local RTX 3060. Trains the 60M default preset for ~30 000 steps using the 8-source Opus 4.6 distillation mix (Kimi 25%, Crownelius 15%, TeichAI 10%, Superior 10%, NuminaMath 10%, FineTome 10%, FineWeb 10%, Kimi-Math 10%). N3 SleepGate is enabled (memory consolidation every 200 steps). Saves checkpoints to `output/overnight_opus46/`.

**When to run:** local RTX 3060 overnight pretraining on FANT 2. Superseded by the Colab notebook for FANT 3.

**Arguments:**
```
--smoke              20 steps with synthetic data
--scale SCALE        model size preset (default: default/60M)
--n-steps N          training steps (default: 30000)
```

**Output:** `output/overnight_opus46/step_NNNN.pt`, `output/overnight_opus46/final.pt`

---

### `train_2b.py`
Training script for the 2.025B parameter FANT 2 model. Requires at least 24 GB VRAM. Uses 8-bit AdamW and gradient checkpointing. Not suitable for RTX 3060.

**When to run:** Only on a GPU with ≥24 GB VRAM (A100/H100). Superseded by FANT 3 Colab notebook.

**Output:** `output/train_2b/`

---

### `run_campaign_n.py`
Unified runner for a single Campaign N variant (N1–N7). Selects the variant by `--variant` flag and delegates to the appropriate implementation in `campaign_n.py`. Variants include N3 SleepGate, N6 gold reasoning, and N7 SEC (Semantic Entropy Calibration).

**When to run:** ablation experiments comparing Campaign N variants at 5M scale.

**Arguments:**
```
--variant N1|N2|...|N7
--n-steps N
```

**Output:** `output/campaign_n/variant_N*/`

---

### `run_all_campaign_n.py`
Runs all 7 Campaign N variants in sequence (or a configurable subset). Intended for overnight full sweeps when the RTX 3060 is idle.

**When to run:** full Campaign N sweep. Approximately 7× the time of a single variant.

**Output:** `output/campaign_n/` (one subdirectory per variant)

---

## Evaluation scripts

### `eval_benchmarks.py`
Unified GSM8K, MMLU, and MATH-500 evaluation for FANT 3 checkpoints. Loads a `.pt` checkpoint, rebuilds the model config from the saved `cfg_dict`, and runs greedy generation (GSM8K, MATH-500) or letter-logit comparison (MMLU). Reports accuracy with 95% Wilson CI (Confidence Interval).

**When to run:** after training completes; also invoked from Colab notebook cell 26.

**Arguments:**
```
--ckpt PATH          path to .pt checkpoint (required)
--tokenizer PATH     tokenizer JSON (default: output/tokenizer/tokenizer_v2.json)
--benchmark gsm8k|mmlu|math500  (required)
--n N                problems to evaluate (default: 100)
--device cuda|cpu
--dtype bf16|fp16|fp32
```

**Output:** prints accuracy + CI to stdout.

See [`docs/evaluation/BENCHMARKS.md`](../evaluation/BENCHMARKS.md) for full details on answer extraction and the Wilson CI formula.

---

### `eval_1k.py`
Evaluates a FANT 2 checkpoint on 1 000 ProceduralMathStream problems (arithmetic word problems generated on the fly). Reports accuracy with Wilson CI. Used for Campaign N head-to-head comparisons — 1 000 samples gives ~6pp CI width vs ~14pp for 200 samples.

**When to run:** comparing FANT 2 architecture variants at 5M scale.

**Arguments:**
```
--ckpt PATH
```

**Output:** prints accuracy, CI, and per-problem breakdown to stdout.

---

### `eval_1k_default.py`
Adapter version of `eval_1k.py` that works with the default (60M) FANT 2 config rather than requiring a scale argument. Used for mid-training quality probes on the `overnight_default` checkpoint.

**When to run:** mid-training spot checks on the 60M overnight model.

**Output:** stdout accuracy report.

---

### `test_step500.py`
Qualitative probe on the `step_500.pt` checkpoint from the overnight distillation run. Generates completions for a small set of math questions and prints them. Not a pass/fail test — used to eyeball generation quality.

**When to run:** one-off quality inspection after the first 500 steps of overnight training.

**Output:** stdout completions.

---

### `test_step2000_qualitative.py`
Qualitative probe on the `step_2000.pt` checkpoint. Generates completions for 5 arithmetic probes and inspects whether `<|answer|>` tags are present. Identified the template-echo and 1.8.8.8. loop failure modes at CE 4–5.

**When to run:** mid-training qualitative check at step 2000.

**Output:** stdout completions + tag-presence summary.

---

### `template_accuracy_analysis.py`
Analyses a batch of model completions to distinguish template-following failures (missing answer tags, format errors) from genuine arithmetic errors. Useful for diagnosing whether low accuracy is due to format or reasoning.

**When to run:** post-hoc analysis of a saved batch of completions.

**Output:** breakdown table of error types to stdout.

---

## Compression scripts

### `compress_test.py`
Phase 0 compression benchmark. Measures CE (Cross-Entropy) bits-per-byte of the FANT 2 `step_3000.pt` checkpoint on three short corpora (prose, Python code, JSON config) and compares to gzip / zlib / bz2 / lzma at maximum compression. Validates the Delétang 2023 equivalence: CE bpb equals the theoretical limit of an arithmetic coder driven by the model.

**When to run:** one-off validation after a training run to check if the model is gzip-competitive on prose.

**Arguments:** none (uses hardcoded `CKPT` and `TOK` paths at the top of the file).

**Output:** per-corpus comparison table to stdout.

---

### `compress_book.py`
Extends the compression benchmark to three Project Gutenberg books (`data/gutenberg/alice.txt`, `data/gutenberg/tale_two_cities.txt`, `data/gutenberg/frankenstein.txt`). These are 19th-century English text that is entirely out-of-distribution, settling the training-contamination question. Strips Gutenberg headers/footers before measurement.

**When to run:** out-of-distribution compression validation. Requires the three Gutenberg text files to exist under `data/gutenberg/`.

**Arguments:**
```
--n-bytes N    trim each book to N chars (default: 20000; 0 = full book)
--device cuda|cpu
--ckpt PATH    override checkpoint path
```

**Output:** per-book comparison table + summary to stdout.

---

### `compress_qwen.py`
Benchmarks Qwen 2.5 1.5B compression on the same Gutenberg books for head-to-head comparison with FANT 2 (84.8M params). Also includes a precision-invariance test: loads Qwen in BF16 and F32, quantises softmax probabilities to a 65 536-level integer grid, and verifies that 99.913% of entries are bit-identical — confirming that BF16 inference is sufficient for a BendVM-compatible arithmetic coder.

**When to run:** benchmarking Qwen as a compression baseline; validating BF16 precision sufficiency.

**Arguments:**
```
--dtype bf16|f32|fp16
--n-bytes N
--precision-test    run the BF16 vs F32 comparison instead of the book benchmark
```

**Output:** comparison tables + precision-invariance report to stdout.

---

### `bendvm_demo.py`
End-to-end demonstration of the BendVM virtual machine. Runs 9 demos: Fibonacci correctness, program compression (10⁶ steps in 4 integers), fast-exponentiation speedup, program composition (O(1) matrix multiply), exact inversion, SL(2, Z) equivalence relations, Euclidean GCD, Apollonian walk with Descartes invariant verification, and T^10⁹ fast power. All 9 demos must pass.

**When to run:** sanity-check the BendVM library after changes to `bendvm/`.

**Output:** per-demo pass/fail + timing to stdout.

---

## Data scripts

### `build_distillation_corpus.py`
Queries the OpenRouter Elephant-Alpha teacher model (or any OpenRouter-compatible model) to generate `<|think|>...<|/think|><|answer|>...<|/answer|>` completions for prompts drawn from NuminaMath, Superior Reasoning stage 1, OpenR1 logic puzzles, and Kimi K2.5 PhD-science. Caches responses to a JSONL file that the trainer reads via `LocalJSONLStream`.

**When to run:** building a custom distillation cache. Requires `OPENROUTER_PROVISIONING_KEY` environment variable and an inference key file (created by `openrouter_keys.py`).

**Arguments:**
```
--sources numina,superior1,logic,kimi-sci
--per-source N     max prompts per source (default: 200)
--n-samples N      total new samples to generate (default: 150)
--cache PATH       JSONL output file (default: data/distill_cache/elephant_alpha.jsonl)
--key-file PATH    path to inference key JSON (default: .openrouter_key)
--model MODEL      OpenRouter model string (default: openrouter/elephant-alpha)
--per-minute N     rate cap (default: 18, free-tier max is 20)
--per-day N        daily cap (default: 195, free-tier max is 200)
```

**Output:** JSONL appended to `--cache` path.

---

### `openrouter_keys.py`
Creates, lists, inspects, and deletes throwaway OpenRouter inference keys using the provisioning API. Keys are scoped to a spend budget and expire automatically. The key secret is written to a JSON file that `build_distillation_corpus.py` reads.

**When to run:** before and after a distillation corpus collection session.

**Arguments (subcommands):**
```
create --label NAME --budget USD --expires-hours H --out-file PATH
list
inspect --hash HASH
delete --hash HASH
disable --hash HASH
```

**Requires:** `OPENROUTER_PROVISIONING_KEY` environment variable (set permanently with `setx` on Windows).

**Output:** key secret printed to stdout + JSON file written to `--out-file`.

---

### `validate_opus46_datasets.py`
Pre-launch validator for the Opus 4.6 training recipe. Pulls 2 samples from `opus46-crownelius-3300x` and `opus46-teichai-887x`, checks that `extract_text()` returns a non-empty ChatML string, and runs the full `InterleavedMultiDatasetStream` for 20 iterations. Exits non-zero on any error.

**When to run:** before starting a training run that uses the Opus 4.6 datasets, to catch registry or format wiring bugs early.

**Output:** pass/fail report to stdout.

---

## Decontamination scripts

### `decontaminate.py`
Implements the 13-gram SHA-1 decontamination filter. See [`docs/datasets/DECONTAMINATION.md`](../datasets/DECONTAMINATION.md) for the full algorithm description.

**When to run (report mode):** `python scripts/decontaminate.py --n-docs 2000` scans 2 000 documents per source and prints contamination rates.

**When to run (cache build):** `python scripts/decontaminate.py --rebuild-cache` re-downloads benchmark test sets and rebuilds `output/decontamination/ngram_hashes.json`.

**When to use as a library:** `from scripts.decontaminate import is_contaminated` — returns `True` if a text string contains any benchmark 13-gram.

**Output:** `output/decontamination/ngram_hashes.json` (cache) + stdout report.

---

## Smoke / validation scripts

### `smoke_fant3.py`
Builds a FANT 3 model at a chosen scale (smoke / 742m / 1b), runs a forward pass, backward pass, and 5 optimizer steps on synthetic data, and verifies no NaN loss or NaN gradients. Also tests ETF (Equiangular Tight Frame) router freezing if enabled.

**When to run:** after any change to `fant3/` model code. Run before starting a real training job.

**Arguments:**
```
--scale smoke|742m|1b   (default: smoke)
--steps N               optimizer steps (default: 5)
--device cuda|cpu
--bf16                  use BF16
```

**Output:** per-step loss + NaN status to stdout. Exits with code 2 if NaN detected.

---

### `scale_ladder_smoke.py`
Runs the full scale ladder (5M → 40M → 150M → 350M → 742M) in sequence, verifying that each rung completes 3 optimizer steps without NaN, OOM (Out of Memory), or shape errors. Reports param count, VRAM peak, forward/backward time, loss, and chirality balance (SpinorApollonianMemory α/β split) for each rung.

**When to run:** after landing a multi-rung architectural change (e.g., the five 2026-04-19 fixes). Requires CUDA.

**Output:** per-rung summary table + pass/fail count to stdout.

---

## Legacy option scripts (`option_*.py`)

These scripts implement specific training experiments from the FANT 2 research campaign. They are not intended for general use and are preserved for reproducibility.

| Script | Experiment |
|--------|-----------|
| `option_b_real_text.py` | Phase 2 with real text (web + math) |
| `option_c_gpu_smoke.py` | GPU smoke test at Phase 2 |
| `option_d_phase4.py` | Phase 4 ramp (classifier + math) |
| `option_e_phase5_grpo.py` | Phase 5 G2RPO (Group Relative Policy Optimisation) |
| `option_f_phase6_simpo.py` | Phase 6 SimPO+KTO alignment |
| `option_g_benchmark_rampup.py` | Benchmark ramp-up evaluation |
| `option_h_real_benchmarks.py` | Post-H baseline benchmarks (200 problems) |
| `option_h2_postk_benchmarks.py` | Post-K benchmarks |
| `option_h3_postl1_benchmarks.py` | Post-L1 benchmarks |
| `option_h4_postl1_5_benchmarks.py` | Post-L1.5 benchmarks (1K eval) |
| `option_i_real_pretrain.py` | Real pretraining (option I) |
| `option_k_procedural_ramp.py` | Procedural math ramp (option K) |
| `option_l1_phase4_ramp.py` | Phase 4 ramp with L1 improvements |
| `option_l1_5_curvature_fix.py` | L1.5 curvature-score fix |
| `option_m_ablation.py` | M-series ablation runner |
| `option_m1_think_at_hard.py` | M1: think-token only on hard problems |
| `option_m2_full_tensor.py` | M2: full-tensor attention |
| `option_m3_classifier_fix.py` | M3: Apollonian classifier upstream fix |
| `option_m4_synthesis.py` | M4: synthesis of M1+M2+M3 (51.2%) |
| `option_m4_ebm.py` | M4-EBM: energy-based memory (6.4%, collapsed) |
| `option_n1_ortho_var.py` | N1: orthogonal initialisation variant |
| `overfit_sanity.py` | Overfit a single batch to verify the loss can reach zero |
