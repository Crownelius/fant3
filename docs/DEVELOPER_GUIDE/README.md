# FANT 3 Developer Guide

For contributors and architects. Covers where concepts live in code, the testing protocol, how to add a new preset or curriculum, and how to diagnose the common failure modes.

## Where concepts live

Map from abstract concept to concrete file:

| Concept | File | Lines (approx) |
|---|---|---|
| Config and presets | `fant3/config.py` | 300 |
| MASA attention | `fant3/model/attention.py` | 250 |
| Matryoshka MoE | `fant3/model/matryoshka_moe.py` | 200 |
| Mixture of Recursions | `fant3/model/recursion.py` | 350 |
| Spinor Apollonian memory | `fant3/model/spinor_apollonian.py` | 400 |
| Artificial Hippocampus Network | `fant3/model/ahn.py` | 300 |
| Cerebellum reservoir | `fant3/model/cerebellum.py` | 200 |
| ETF frame and freezing | `fant3/model/etf.py` | 150 |
| Top-level model assembly | `fant3/model/fant3_model.py` | 500 |
| Training loop | `scripts/runpod_train.py` | 500 |
| Curriculum scheduler | `fant3/training/curriculum.py` | 260 |
| LR schedulers | `fant3/training/schedulers.py` | 60 |
| SAE diagnostics | `fant3/diagnostics/sae.py` | 350 |
| Dataset registry | `fant2/data/registry.py` | 440 |
| Format handlers | `fant2/data/formats.py` | 200 |
| Interleaved streaming | `fant2/data/streaming.py` | 260 |

Reasoning for the two-tier `fant2/` + `fant3/` layout: the data plumbing is stable and shared; the architecture is under active development. Do not modify `fant2/` unless explicitly asked.

## Testing protocol

```bash
# Full suite (125 tests, ~30 s on CPU)
python -m pytest tests/ --ignore=tests/test_ahn.py -v

# Subset by component
python -m pytest tests/test_smoke.py                  # Import + forward/backward
python -m pytest tests/test_spinor_apollonian.py      # 10 tests
python -m pytest tests/test_mor_lti.py                # Mixture of Recursions
python -m pytest tests/test_monotonic_dynamic_k.py    # 22 tests, ISRM
python -m pytest tests/test_curriculum.py             # 49 tests
python -m pytest tests/test_router_collapse.py        # FANT 2 regression canary
python -m pytest tests/test_sae.py                    # SAE introspection
python -m pytest tests/test_trainer_integration.py    # Trainer end-to-end
```

`test_ahn.py` has a pre-existing fixture bug from the initial commit and is excluded until rewritten.

### GPU-free smoke scripts

Paired with the pytest suite, each major subsystem ships a `scripts/smoke_*.py` dry-run that validates wiring in under 2 seconds without CUDA:

- `scripts/smoke_curriculum.py` — all 3 curriculum presets + legacy override
- `scripts/smoke_fant3.py` — model forward/backward at the `fant3_smoke` preset
- `scripts/scale_ladder_smoke.py` — all 5 scales end-to-end (requires a GPU for 742m/1b but smokes the configs at all sizes)

When adding a new subsystem, add a corresponding smoke script. See [feedback_gpu_free_smoke_pattern](../../memory-index.md) for the pattern.

## How to add a new scale preset

1. Edit `fant3/config.py`, add `fant3_NEWm()` returning a `FANT3Config`.
2. Add to `preset_map` in `scripts/runpod_train.py`.
3. Add a choice to the `--scale` argparse choices.
4. Verify param count: `python -c "from fant3.config import fant3_NEWm; from fant3.model.fant3_model import FANT3Model; m = FANT3Model(fant3_NEWm()); print(sum(p.numel() for p in m.parameters()))"`.
5. Run `scripts/smoke_fant3.py --scale NEWm` to confirm forward/backward works.
6. Add a training recipe row to [USER_GUIDE/README.md](../USER_GUIDE/README.md#scale-ladder).

Preset names have lied about their size before (742m materialized 6.6 B, 1b materialized ~7 B before the 2026-04-19 fix). Always verify.

## How to add a new dataset

1. Edit `fant2/data/registry.py`, add a `DatasetEntry` in `TRAINING_DATASETS`.
2. Confirm the HF streaming probe works: `python -c "from fant2.data.registry import get_dataset; ds = get_dataset('your-key'); print(next(iter(ds)))"`.
3. If the dataset has a novel schema, add a format handler in `fant2/data/formats.py`.
4. Run `python scripts/decontaminate.py --check your-key` and note the contamination rate in the commit message.
5. Add to at least one curriculum preset in `fant3/training/curriculum.py`.

Contamination budget: aim for under 1% 13-gram overlap with GSM8K + MMLU + MATH-500 test sets. Worst-source observed so far: 1.80% (NVIDIA OpenMathInstruct-2), which we accept for its size.

## How to add a new curriculum preset

1. Edit `fant3/training/curriculum.py`.
2. Define a new `Curriculum` with `PhaseSpec` tuples summing weights to 1.0 per phase.
3. Add to the `PRESETS` dict.
4. Unit tests in `tests/test_curriculum.py::TestPresets` will automatically cross-check dataset keys against `TRAINING_DATASETS`.
5. Run `scripts/smoke_curriculum.py` to verify phase walk and stream construction.
6. Add to the `--curriculum` argparse choices (automatic via `sorted(CURRICULUM_PRESETS.keys())`).

Backward compatibility: the `legacy_2phase` preset must remain bit-identical to the pre-curriculum hardcoded mix. There is a unit test (`test_legacy_2phase_matches_original_runpod_train`) that enforces this.

## Diagnosing common failure modes

### Router collapse (single expert takes >85% of load)

Symptom: `max|rtr|` in the training log keeps climbing; `chirality_balance` stuck at 1.0 or 0.0; MoE throughput drops as one expert becomes a bottleneck.

Check:
- Chirality balance in `mstats["chirality_balance"]`: healthy is 0.4–0.6.
- Router logit magnitude (`mp_logits` in `router_infos`): healthy is under 30.
- ETF freeze step: routers should freeze at `cfg.etf_freeze_after_step` (default 500 for 50m+).

Fix:
- Increase `cfg.z_coef` (router entropy regularizer).
- Check `scripts/check_router_collapse.py` against a trained checkpoint.
- Memory: [research_spinor_apollonian_2026_04_16](../../.claude/projects/C--FANT/memory/) describes the chirality-balance fix.

### NaN loss

Symptom: `[NaN] step=X all N micros NaN` appears in the log; `consec_nan` counter climbs.

Check:
- `max|logit|` in the log before the NaN: if >1000, numerical overflow.
- Learning rate: if peak_lr > 3e-4 and warmup is too short, bf16 overflow at step ~672.
- Tokenizer consistency: mismatch between training tokenizer and model vocab will NaN immediately.

Fix:
- Drop peak_lr (successful 150m run at 1.5e-4 after a 3e-4 divergence).
- Extend warmup_steps to at least 5% of total_steps.
- Tighten `cfg.lm_head_logit_cap` (default 30.0 for 50m+).

### CUDA out of memory

Symptom: OOM at step 10 despite plausible VRAM budget.

Check:
- Actual param count vs preset name: `fant3_742m` once materialized 6.6 B.
- `cfg.use_gradient_checkpointing` should be True at 742m+.
- `BATCH_SIZE=4, T=512` is the floor for A100 80 GB at 742m with gradient checkpointing.

Fix:
- Halve BATCH_SIZE, double grad_accum (keeps effective batch constant).
- Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` BEFORE `torch` is imported.
- Drop SEQ_LEN 1024 → 512 and double steps.

### Chirality starvation

Symptom: `chirality_balance` stuck at 0 or 1; Spinor Apollonian memory degenerates to single-pack behavior.

This was the pre-fix behavior of FANT 3 before 2026-04-19. The scalar-curvature classifier had fragile thresholds and systematically starved one pack. The Kocik spinor fix in `spinor_apollonian.py` resolves it natively. If you see starvation in a new branch: check you did not regress to the old classifier.

## Commit conventions

- Imperative subject line under 70 chars: `"curriculum module: 3-phase deepinsight preset + tests"`.
- Blank line.
- Body: bullet list of what and why. Cite papers by arxiv ID.
- No Claude identifiers (no Co-Authored-By, no Generated-with lines) for this repository.

## Pull request conventions

Not applicable at present — FANT 3 is a single-author repository. If that changes, a CONTRIBUTING.md will be added here.

## See also

- [ADR/](../ADR/) — architectural decision records (why we chose MoR over fixed depth, Matryoshka MoE over standard, etc.)
- [architecture/](../architecture/) — per-component deep dives
- [testing/](../testing/) — testing protocol in depth
