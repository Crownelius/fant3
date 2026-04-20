# =============================================================================
# FANT 2 Makefile
# Fractal Atomic Neural Topology — second generation
# =============================================================================
#
# All targets default to `python -m fant2 ...` so the package layout is
# self-contained. Override `PY` to use a venv or specific interpreter.
# =============================================================================

PY        ?= python
PIP       ?= $(PY) -m pip
PYTEST    ?= $(PY) -m pytest

# -- model preset (tiny | default) --------------------------------------------
PRESET    ?= default
# -- output directory (per-phase subdirs are created automatically) -----------
OUT       ?= output
# -- device (cuda | cpu) -------------------------------------------------------
DEVICE    ?= cuda
# -- tokenizer artifact (output of train-phase0) ------------------------------
TOKENIZER ?= $(OUT)/phase0/tokenizer.json
# -- per-phase resume points --------------------------------------------------
CKPT1     ?= $(OUT)/phase1/final.pt
CKPT2     ?= $(OUT)/phase2/final.pt
CKPT3     ?= $(OUT)/phase3/final.pt
CKPT4     ?= $(OUT)/phase4/final.pt
CKPT5     ?= $(OUT)/phase5/final.pt
CKPT6     ?= $(OUT)/phase6/final.pt
# -- training config knobs (overridable from CLI) -----------------------------
N_STEPS   ?= 50000
BATCH     ?= 8
SEQ_LEN   ?= 1024


# -----------------------------------------------------------------------------
# Top-level help
# -----------------------------------------------------------------------------
.PHONY: help
help:
	@echo "FANT 2 build / train / test targets"
	@echo "==================================="
	@echo ""
	@echo "  make install              install python dependencies"
	@echo "  make test                 run pytest test suite (smoke + router + integration)"
	@echo "  make smoke                run smoke tests only"
	@echo "  make router-canary        run the FANT 350M collapse regression test"
	@echo "  make info                 print model info for the current PRESET"
	@echo ""
	@echo "Training pipeline (run in order):"
	@echo "  make phase0               train BPE tokenizer  -> \$$(TOKENIZER)"
	@echo "  make phase1               LLM-JEPA + SIGReg pretrain"
	@echo "  make phase2               MoE specialization (FEP unified loss)"
	@echo "  make phase3               Active-layer calibration"
	@echo "  make phase4               Self-refinement + STaR + Apollonian fill"
	@echo "  make phase5               Dr.GRPO RL (stub)"
	@echo "  make phase6               SimPO + KTO preference (stub)"
	@echo "  make train-all            run phases 0..6 sequentially"
	@echo ""
	@echo "Inference / evaluation:"
	@echo "  make generate PROMPT='Hello world.'"
	@echo "  make chat                 interactive chat with the latest checkpoint"
	@echo "  make eval-ppl             evaluate perplexity"
	@echo "  make eval-gsm8k           evaluate GSM8K accuracy"
	@echo "  make eval-arc             evaluate ARC-Easy multichoice"
	@echo "  make eval-hellaswag       evaluate HellaSwag multichoice"
	@echo "  make eval-all             run all four evals on the final checkpoint"
	@echo ""
	@echo "House-keeping:"
	@echo "  make clean                remove __pycache__ and .pytest_cache"
	@echo "  make clean-output         remove the entire \$$(OUT) directory"
	@echo ""
	@echo "Variables you can override on the command line:"
	@echo "  PRESET=$(PRESET)  OUT=$(OUT)  DEVICE=$(DEVICE)"
	@echo "  N_STEPS=$(N_STEPS)  BATCH=$(BATCH)  SEQ_LEN=$(SEQ_LEN)"


# -----------------------------------------------------------------------------
# Install / test
# -----------------------------------------------------------------------------
.PHONY: install
install:
	$(PIP) install -r requirements.txt

.PHONY: test
test:
	$(PYTEST) tests/ -v

.PHONY: smoke
smoke:
	$(PYTEST) tests/test_smoke.py -v

.PHONY: router-canary
router-canary:
	$(PYTEST) tests/test_router_collapse.py -v

.PHONY: integration
integration:
	$(PYTEST) tests/test_trainer_integration.py -v


# -----------------------------------------------------------------------------
# Info
# -----------------------------------------------------------------------------
.PHONY: info
info:
	$(PY) -m fant2 info --preset $(PRESET)


# -----------------------------------------------------------------------------
# Phase 0 — BPE tokenizer
# -----------------------------------------------------------------------------
.PHONY: phase0
phase0:
	mkdir -p $(OUT)/phase0
	$(PY) -m fant2 train-phase0 \
		--out-dir $(OUT)/phase0 \
		--preset $(PRESET) \
		--seed-repeat 50000


# -----------------------------------------------------------------------------
# Phase 1 — LLM-JEPA + SIGReg pretrain
# -----------------------------------------------------------------------------
.PHONY: phase1
phase1:
	mkdir -p $(OUT)/phase1
	$(PY) -m fant2 train-phase1 \
		--out-dir $(OUT)/phase1 \
		--preset $(PRESET) \
		--tokenizer $(TOKENIZER) \
		--device $(DEVICE) \
		--n-steps $(N_STEPS) \
		--batch-size $(BATCH) \
		--seq-len $(SEQ_LEN)


# -----------------------------------------------------------------------------
# Phase 2 — MoE specialization (FEP unified loss)
# -----------------------------------------------------------------------------
.PHONY: phase2
phase2:
	mkdir -p $(OUT)/phase2
	$(PY) -m fant2 train-phase2 \
		--out-dir $(OUT)/phase2 \
		--preset $(PRESET) \
		--tokenizer $(TOKENIZER) \
		--device $(DEVICE) \
		--n-steps $(N_STEPS) \
		--batch-size $(BATCH) \
		--seq-len $(SEQ_LEN) \
		--resume-from $(CKPT1)


# -----------------------------------------------------------------------------
# Phase 3 — active-layer calibration
# -----------------------------------------------------------------------------
.PHONY: phase3
phase3:
	mkdir -p $(OUT)/phase3
	$(PY) -m fant2 train-phase3 \
		--out-dir $(OUT)/phase3 \
		--preset $(PRESET) \
		--tokenizer $(TOKENIZER) \
		--device $(DEVICE) \
		--n-steps $(N_STEPS) \
		--batch-size $(BATCH) \
		--seq-len $(SEQ_LEN) \
		--resume-from $(CKPT2)


# -----------------------------------------------------------------------------
# Phase 4 — self-refinement + STaR + Apollonian fill
# -----------------------------------------------------------------------------
.PHONY: phase4
phase4:
	mkdir -p $(OUT)/phase4
	$(PY) -m fant2 train-phase4 \
		--out-dir $(OUT)/phase4 \
		--preset $(PRESET) \
		--tokenizer $(TOKENIZER) \
		--device $(DEVICE) \
		--n-steps $(N_STEPS) \
		--batch-size $(BATCH) \
		--seq-len $(SEQ_LEN) \
		--resume-from $(CKPT3)


# -----------------------------------------------------------------------------
# Phase 5 — Dr.GRPO RL (stub: falls back to Phase 2 loss)
# -----------------------------------------------------------------------------
.PHONY: phase5
phase5:
	mkdir -p $(OUT)/phase5
	$(PY) -m fant2 train-phase5 \
		--out-dir $(OUT)/phase5 \
		--preset $(PRESET) \
		--tokenizer $(TOKENIZER) \
		--device $(DEVICE) \
		--n-steps $(N_STEPS) \
		--batch-size $(BATCH) \
		--seq-len $(SEQ_LEN) \
		--resume-from $(CKPT4)


# -----------------------------------------------------------------------------
# Phase 6 — SimPO + KTO preference alignment (stub)
# -----------------------------------------------------------------------------
.PHONY: phase6
phase6:
	mkdir -p $(OUT)/phase6
	$(PY) -m fant2 train-phase6 \
		--out-dir $(OUT)/phase6 \
		--preset $(PRESET) \
		--tokenizer $(TOKENIZER) \
		--device $(DEVICE) \
		--n-steps $(N_STEPS) \
		--batch-size $(BATCH) \
		--seq-len $(SEQ_LEN) \
		--resume-from $(CKPT5)


# -----------------------------------------------------------------------------
# Run the full 7-phase pipeline
# -----------------------------------------------------------------------------
.PHONY: train-all
train-all: phase0 phase1 phase2 phase3 phase4 phase5 phase6
	@echo "================================================================"
	@echo " FANT 2 full training pipeline complete."
	@echo " Final checkpoint: $(CKPT6)"
	@echo "================================================================"


# -----------------------------------------------------------------------------
# Inference
# -----------------------------------------------------------------------------
PROMPT ?= "The Apollonian gasket is"

.PHONY: generate
generate:
	$(PY) -m fant2 generate \
		--tokenizer $(TOKENIZER) \
		--checkpoint $(CKPT6) \
		--preset $(PRESET) \
		--device $(DEVICE) \
		--prompt $(PROMPT) \
		--max-new-tokens 200

.PHONY: chat
chat:
	$(PY) -m fant2 chat \
		--tokenizer $(TOKENIZER) \
		--checkpoint $(CKPT6) \
		--preset $(PRESET) \
		--device $(DEVICE)


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
.PHONY: eval-ppl
eval-ppl:
	$(PY) -m fant2 eval-ppl \
		--tokenizer $(TOKENIZER) \
		--checkpoint $(CKPT6) \
		--preset $(PRESET) \
		--device $(DEVICE) \
		--max-batches 100

.PHONY: eval-gsm8k
eval-gsm8k:
	$(PY) -m fant2 eval-gsm8k \
		--tokenizer $(TOKENIZER) \
		--checkpoint $(CKPT6) \
		--preset $(PRESET) \
		--device $(DEVICE) \
		--max-examples 200

.PHONY: eval-arc
eval-arc:
	$(PY) -m fant2 eval-arc \
		--tokenizer $(TOKENIZER) \
		--checkpoint $(CKPT6) \
		--preset $(PRESET) \
		--device $(DEVICE) \
		--max-examples 200

.PHONY: eval-hellaswag
eval-hellaswag:
	$(PY) -m fant2 eval-hellaswag \
		--tokenizer $(TOKENIZER) \
		--checkpoint $(CKPT6) \
		--preset $(PRESET) \
		--device $(DEVICE) \
		--max-examples 200

.PHONY: eval-all
eval-all: eval-ppl eval-gsm8k eval-arc eval-hellaswag


# -----------------------------------------------------------------------------
# House-keeping
# -----------------------------------------------------------------------------
.PHONY: clean
clean:
	-find . -type d -name __pycache__ -exec rm -rf {} +
	-rm -rf .pytest_cache

.PHONY: clean-output
clean-output:
	rm -rf $(OUT)
