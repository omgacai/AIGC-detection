#!/usr/bin/env bash
# One-epoch DINOv3-Forensic smoke run on the known Apptainer-capable GPU node.
# Submit from the repository root with: sbatch cluster/slurm_dinov3_smoke.sh
#SBATCH --job-name=dinov3-smoke
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-dinov3-%j.out

set -eu

REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CHECKPOINT_ROOT="${AIGC_CHECKPOINT_ROOT:-${STORAGE_ROOT}/checkpoints}"
OUTPUT_ROOT="${AIGC_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs}"
MANIFEST="${AIGC_MANIFEST:-${DATA_ROOT}/manifests/sid_all.csv}"
PYTHON_DEPS="${CACHE_ROOT}/container-python-deps"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"

mkdir -p "${CACHE_ROOT}/apptainer" "${CACHE_ROOT}/huggingface" "${CACHE_ROOT}/pip" "${PYTHON_DEPS}" "${CHECKPOINT_ROOT}" "${OUTPUT_ROOT}"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

if [ ! -f "${MANIFEST}" ]; then
  echo "ERROR: Dataset manifest not found: ${MANIFEST}" >&2
  echo "Download and verify SID before submitting training." >&2
  exit 2
fi

echo "Host: $(hostname)"
echo "Manifest: ${MANIFEST}"
/usr/bin/apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" \
  --pwd /workspace \
  "${IMAGE}" \
  bash -lc '
    set -eu
    export PYTHONPATH="'"${PYTHON_DEPS}"':/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
    export PIP_CACHE_DIR="'"${CACHE_ROOT}"'/pip"
    export HF_HOME="'"${CACHE_ROOT}"'/huggingface"
    export AIGC_DATA_ROOT="'"${DATA_ROOT}"'"
    export AIGC_CACHE_ROOT="'"${CACHE_ROOT}"'"
    export AIGC_CHECKPOINT_ROOT="'"${CHECKPOINT_ROOT}"'"
    export AIGC_OUTPUT_ROOT="'"${OUTPUT_ROOT}"'"
    python --version
    python -c "import torch; print(\"torch=\", torch.__version__); print(\"gpu=\", torch.cuda.get_device_name(0))"
    if ! python -c "import transformers, albumentations, sklearn, tensorboard"; then
      python -m pip install --target "'"${PYTHON_DEPS}"'" --retries 20 --timeout 300 transformers albumentations scikit-learn tensorboard
    fi
    python -c "import transformers, albumentations, sklearn, tensorboard; print(\"Python dependencies available\")"
    python scripts/train.py --config configs/dinov3_forensic.toml --manifest "'"${MANIFEST}"'" --epochs 1
  '
