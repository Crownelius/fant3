# Claude session context for FANT 3

This file is loaded automatically at the start of any Claude Code session run inside this repository. Its purpose is to orient an AI collaborator in seconds — what matters, where things live, what to be careful about.

If you are a human reader: nothing here is secret, but the file is written for machine use. The [README](./README.md) is the right starting point for you.

---

## What this repository is

FANT 3 is a compact Mixture-of-Experts language model research workspace. One GPU, one developer, seven preset scales from 1 M to 1 B parameters. The code is organized in three tiers:

1. **`fant3/`** — the current architecture. This is the active codebase. Modify things here.
2. **`fant2/`** — the legacy runtime. Kept for comparison probes and as a known-working reference. Do not modify unless explicitly asked; the working memory system has several references to FANT 2 numbers (60 M stored / 200 M active, HAR router, 94.5% collapse fix) that should stay accurate.
3. **`bendvm/`** — a research aside on operating programs while compressed. Self-contained; unrelated to the main training loop.

---

## Where to start

- **Trying to understand what FANT 3 is:** read [README.md](./README.md).
- **Trying to run training:** read [docs/USER_GUIDE/README.md](./docs/USER_GUIDE/README.md).
- **Trying to contribute code:** read [docs/DEVELOPER_GUIDE/README.md](./docs/DEVELOPER_GUIDE/README.md).
- **Trying to understand why a design choice was made:** read [docs/ADR/](./docs/ADR/) or [docs/THEORY/README.md](./docs/THEORY/README.md).
- **Trying to remember what was measured when:** read [ACHIEVEMENTS.md](./ACHIEVEMENTS.md).

---

## Operational facts an AI collaborator needs

- **The project root is on D: drive:** `D:\FANT_TRAINING_D_Drive\fant2\`. This is a Git repository; the remote is `github.com/Crownelius/fant3`.
- **Python 3.10+**, one Ampere-class GPU recommended.
- **bf16 weights + 8-bit AdamW + gradient checkpointing** at 742m and 1b presets.
- **Tokenizer is v2** (`output/tokenizer/tokenizer_v2.json`), 32,768 BPE tokens from the 6-source mix.
- **RunPod is the primary training target** (via `scripts/runpod_train.py`). Colab A100 is the secondary target (via the notebook). The zipped code artifact is `fant_code.zip` (rebuild via `scripts/build_fant_zip.py`).
- **Public benchmarks are eval-only.** GSM8K, MMLU, MATH-500 are never to be trained on. A 13-gram SHA-1 decontamination cache at `output/decontamination/ngram_hashes.json` (457,910 hashes) filters training streams. Worst-source contamination rate observed: 1.80% (NVIDIA OpenMathInstruct-2).

## Things that have burned us before

These are the scars. An AI collaborator should respect them without re-learning:

- **Do not add auxiliary losses or change training text format at scale.** Campaign N (2026-04-11) showed N6 and N7 regressed badly (27.4% and 40.6% vs 59.9% N3 baseline). Only structural and scheduling levers are safe at 5 M+; keep this in mind at 50 M and above.
- **Preset names lie about their size.** `fant3_742m` once materialized 6.6 B parameters. Always verify with `sum(p.numel() for p in model.parameters())` before trusting a VRAM budget.
- **PYTORCH_CUDA_ALLOC_CONF must precede `torch` import.** Setting it after is silently ignored on Colab.
- **LF vs CRLF line endings** — git complains on commit; the warnings are harmless and can be ignored.
- **`tests/test_ahn.py` has a pre-existing fixture bug** (fixture `ahn` not found, from initial commit). It errors on collection; use `--ignore=tests/test_ahn.py` until that file is rewritten.

## Conventions for changes

- **Backward compatibility first.** When adding a new lever (flag, preset, curriculum), the default must reproduce the prior behavior bit-exactly. Example: `--curriculum legacy_2phase` is the default of `runpod_train.py` so existing commands are unaffected.
- **Pair infra changes with a GPU-free smoke script.** `scripts/smoke_*.py` files should validate the new code path without CUDA, bitsandbytes, or HuggingFace network. Example: `scripts/smoke_curriculum.py` runs in under two seconds locally and catches preset typos before RunPod GPU burn.
- **Test files in `tests/`**, using pytest. Target: new module → new test file with at least validation + edge-case + integration coverage.
- **Commit messages**: imperative subject line under 70 chars, blank line, bullet body explaining what and why. See recent history for the style. No Claude identifiers in commit messages or PR bodies for this repository.

---

## The four scopes an AI collaborator should respect

1. **Do not rewrite FANT 2 code.** Memory references its internals.
2. **Do not touch `output/`** unless explicitly asked — it contains trained checkpoints, tokenizer, and decontamination hashes.
3. **Do not regenerate `fant_code.zip` unless explicitly asked** — it is the RunPod upload artifact; rebuild only when the user wants a new upload.
4. **Do not train on public benchmarks.** GSM8K / MMLU / MATH-500 are eval-only. If in doubt, check `scripts/decontaminate.py`.

## Ongoing work

- **Current run**: 50 m unlimited (`project_50m_unlimited_2026_04_23` memory). Not yet launched on RunPod as of the landing of the curriculum module. The curriculum A/B is the next launch.
- **Queued**: Fix 3 (RL via GSPO + verifier distillation from Qwen3-4B per arxiv:2604.16004). Parked in `docs/fix3_rl_plan.md`. Blocked on pretrain checkpoint availability.
- **Queued**: Opus 4.6 technique/sketch distillation, follows positive curriculum signal.
- **Deferred**: Nested `<|technique|>...<|sketch|>` tokens in tokenizer v3. Only pilot at 150 m after the curriculum is validated.

---

## The papers that shape this repository

- **arxiv:2604.16278 DeepInsightTheorem** — progressive Apprentice/Journeyman/Expert curriculum. Landed as `deepinsight_3phase` preset in `fant3/training/curriculum.py`.
- **arxiv:2604.16004 AgentV-RL** — agentic verifier (Qwen3-4B) + GRPO with DAPO asymmetric clip. Documented in `docs/fix3_rl_plan.md`.
- **Kocik, arXiv:2001.05866** — tangency spinors in Cl(2,1) Minkowski. Theoretical basis for `SpinorApollonianMemory`.
- **Wang et al., arXiv:2509.26520** — Matryoshka MoE.
- **arxiv:2510.14865** — midtraining bridges pretraining and posttraining distributions. Independent support for the progressive curriculum.
- **arxiv:2510.01631** — synthetic data mixtures beat pure synthetic. Independent support for the FineWeb anchor in every curriculum phase.
- **Parisi RSB / de Almeida-Thouless, arXiv:2604.11921** — theoretical grounding for MoE routing diversity.
- **Delétang et al. (2023)** — Language Models are Compressors. The framing.

---

## If you are the AI collaborator

Before changing anything:

1. Check [MEMORY.md](../.claude/projects/C--FANT/memory/MEMORY.md) if it is in your environment. It is loaded automatically in Claude Code sessions and indexes the full auto-memory.
2. Check [ACHIEVEMENTS.md](./ACHIEVEMENTS.md) for what has been measured recently.
3. Check [docs/ADR/](./docs/ADR/) for why a design choice was made.
4. Check `git log --oneline -20` for what has changed in code recently.

If you are about to do something that affects shared state (push, overwrite a checkpoint, modify `output/`), confirm with the user first. This project moves fast in private; an accidental force-push costs hours.
