#!/usr/bin/env bash
# =============================================================================
# FANT 2 — full 7-phase training driver
#
# Usage:
#   ./train.sh [phase]      # run a single phase (0..6) or "all"
#   ./train.sh              # equivalent to ./train.sh all
#
# Environment variables (override on the command line):
#   PRESET    model preset      (default: default)
#   OUT       output dir        (default: output)
#   DEVICE    cuda|cpu          (default: cuda)
#   N_STEPS   training steps    (default: 50000)
#   BATCH     batch size        (default: 8)
#   SEQ_LEN   sequence length   (default: 1024)
#   PY        python binary     (default: python)
#
# Examples:
#   ./train.sh 1
#   PRESET=tiny DEVICE=cpu N_STEPS=200 ./train.sh all
# =============================================================================

set -euo pipefail

# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------
: "${PY:=python}"
: "${PRESET:=default}"
: "${OUT:=output}"
: "${DEVICE:=cuda}"
: "${N_STEPS:=50000}"
: "${BATCH:=8}"
: "${SEQ_LEN:=1024}"

TOKENIZER="${OUT}/phase0/tokenizer.json"
CKPT1="${OUT}/phase1/final.pt"
CKPT2="${OUT}/phase2/final.pt"
CKPT3="${OUT}/phase3/final.pt"
CKPT4="${OUT}/phase4/final.pt"
CKPT5="${OUT}/phase5/final.pt"
CKPT6="${OUT}/phase6/final.pt"

PHASE="${1:-all}"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
banner() {
    echo
    echo "================================================================"
    echo " $*"
    echo "================================================================"
}

ensure_dir() {
    mkdir -p "$1"
}

# -----------------------------------------------------------------------------
# Phase entrypoints
# -----------------------------------------------------------------------------
run_phase0() {
    banner "Phase 0 — BPE tokenizer training"
    ensure_dir "${OUT}/phase0"
    "${PY}" -m fant2 train-phase0 \
        --out-dir "${OUT}/phase0" \
        --preset "${PRESET}" \
        --seed-repeat 50000
}

run_phase1() {
    banner "Phase 1 — LLM-JEPA + SIGReg pretrain"
    ensure_dir "${OUT}/phase1"
    "${PY}" -m fant2 train-phase1 \
        --out-dir "${OUT}/phase1" \
        --preset "${PRESET}" \
        --tokenizer "${TOKENIZER}" \
        --device "${DEVICE}" \
        --n-steps "${N_STEPS}" \
        --batch-size "${BATCH}" \
        --seq-len "${SEQ_LEN}"
}

run_phase2() {
    banner "Phase 2 — MoE specialization (FEP unified loss)"
    ensure_dir "${OUT}/phase2"
    "${PY}" -m fant2 train-phase2 \
        --out-dir "${OUT}/phase2" \
        --preset "${PRESET}" \
        --tokenizer "${TOKENIZER}" \
        --device "${DEVICE}" \
        --n-steps "${N_STEPS}" \
        --batch-size "${BATCH}" \
        --seq-len "${SEQ_LEN}" \
        --resume-from "${CKPT1}"
}

run_phase3() {
    banner "Phase 3 — active-layer calibration"
    ensure_dir "${OUT}/phase3"
    "${PY}" -m fant2 train-phase3 \
        --out-dir "${OUT}/phase3" \
        --preset "${PRESET}" \
        --tokenizer "${TOKENIZER}" \
        --device "${DEVICE}" \
        --n-steps "${N_STEPS}" \
        --batch-size "${BATCH}" \
        --seq-len "${SEQ_LEN}" \
        --resume-from "${CKPT2}"
}

run_phase4() {
    banner "Phase 4 — self-refinement + STaR + Apollonian fill"
    ensure_dir "${OUT}/phase4"
    "${PY}" -m fant2 train-phase4 \
        --out-dir "${OUT}/phase4" \
        --preset "${PRESET}" \
        --tokenizer "${TOKENIZER}" \
        --device "${DEVICE}" \
        --n-steps "${N_STEPS}" \
        --batch-size "${BATCH}" \
        --seq-len "${SEQ_LEN}" \
        --resume-from "${CKPT3}"
}

run_phase5() {
    banner "Phase 5 — Dr.GRPO RL (stub falling back to FEP loss)"
    ensure_dir "${OUT}/phase5"
    "${PY}" -m fant2 train-phase5 \
        --out-dir "${OUT}/phase5" \
        --preset "${PRESET}" \
        --tokenizer "${TOKENIZER}" \
        --device "${DEVICE}" \
        --n-steps "${N_STEPS}" \
        --batch-size "${BATCH}" \
        --seq-len "${SEQ_LEN}" \
        --resume-from "${CKPT4}"
}

run_phase6() {
    banner "Phase 6 — SimPO + KTO preference (stub)"
    ensure_dir "${OUT}/phase6"
    "${PY}" -m fant2 train-phase6 \
        --out-dir "${OUT}/phase6" \
        --preset "${PRESET}" \
        --tokenizer "${TOKENIZER}" \
        --device "${DEVICE}" \
        --n-steps "${N_STEPS}" \
        --batch-size "${BATCH}" \
        --seq-len "${SEQ_LEN}" \
        --resume-from "${CKPT5}"
}

run_all() {
    run_phase0
    run_phase1
    run_phase2
    run_phase3
    run_phase4
    run_phase5
    run_phase6
    banner "FANT 2 full training pipeline complete."
    echo " Final checkpoint: ${CKPT6}"
}

# -----------------------------------------------------------------------------
# Dispatcher
# -----------------------------------------------------------------------------
case "${PHASE}" in
    0)   run_phase0 ;;
    1)   run_phase1 ;;
    2)   run_phase2 ;;
    3)   run_phase3 ;;
    4)   run_phase4 ;;
    5)   run_phase5 ;;
    6)   run_phase6 ;;
    all) run_all ;;
    *)
        echo "Unknown phase: ${PHASE}"
        echo "Usage: $0 [0|1|2|3|4|5|6|all]"
        exit 2
        ;;
esac
