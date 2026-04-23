#!/usr/bin/env bash
# RunPod one-shot setup for the FANT 3 training pipeline.
# Intended usage: SSH into a freshly-started RunPod pod (A100 80 GB PCIe
# recommended), then run `bash scripts/runpod_setup.sh` from the repo root.
# Idempotent — safe to re-run.

set -euo pipefail

REPO_URL="https://github.com/rsfitzgibbon/fant3.git"
REPO_DIR="${REPO_DIR:-/workspace/fant3}"
CKPT_DIR="${CKPT_DIR:-/workspace/ckpts}"
CKPT_NAME="${CKPT_NAME:-step_00500.pt}"

echo "=== FANT 3 RunPod setup ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true

# ----- OS deps (usually pre-installed on runpod/pytorch images) -----
if ! command -v git >/dev/null 2>&1; then
    apt-get update -q && apt-get install -yq --no-install-recommends git
fi

# ----- Clone or update repo -----
if [ -d "${REPO_DIR}/.git" ]; then
    echo "Repo present at ${REPO_DIR}; pulling latest"
    git -C "${REPO_DIR}" pull --ff-only
else
    echo "Cloning ${REPO_URL} -> ${REPO_DIR}"
    git clone "${REPO_URL}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"

# ----- Python deps -----
# The base pytorch image ships torch; we add only the deltas.
PY="${PY:-python}"
${PY} -m pip install --quiet --upgrade pip
${PY} -m pip install --quiet \
    bitsandbytes \
    datasets \
    tokenizers \
    scipy \
    huggingface_hub \
    numpy \
    matplotlib

# ----- Sanity: print torch / cuda info -----
${PY} - <<'PY'
import torch, sys
print(f'python     : {sys.version.split()[0]}')
print(f'torch      : {torch.__version__}')
print(f'cuda avail : {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'cuda device: {torch.cuda.get_device_name(0)}')
    print(f'cuda mem   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
PY

# ----- Checkpoint directory -----
mkdir -p "${CKPT_DIR}"
echo
echo "=== Checkpoint transfer ==="
echo "Upload your Colab checkpoint to: ${CKPT_DIR}/${CKPT_NAME}"
echo
echo "Three ways to do this:"
echo "  1. RunPod web UI 'Upload' button on the pod's File Browser."
echo "     Drag ${CKPT_NAME} (~4 GB) from your local machine into ${CKPT_DIR}."
echo "  2. SCP from local:"
echo "       scp -P <pod_ssh_port> /path/to/${CKPT_NAME} root@<pod_host>:${CKPT_DIR}/"
echo "  3. gdown from a public Google Drive share link:"
echo "       pip install gdown"
echo "       gdown 'https://drive.google.com/uc?id=<FILE_ID>' -O ${CKPT_DIR}/${CKPT_NAME}"
echo
echo "Once the file is in place, resume training with:"
echo "  cd ${REPO_DIR}"
echo "  python scripts/runpod_train.py --resume ${CKPT_DIR}/${CKPT_NAME}"
echo

# ----- Decontamination + tokenizer sanity check (files must be in the repo) -----
if [ ! -f output/tokenizer/tokenizer_v2.json ]; then
    echo "WARN: output/tokenizer/tokenizer_v2.json missing; pipeline will fail at tokenizer load."
fi
if [ ! -f output/decontamination/ngram_hashes.json ]; then
    echo "WARN: output/decontamination/ngram_hashes.json missing; decontamination filter will rebuild (slow)."
fi

echo "=== setup complete ==="
