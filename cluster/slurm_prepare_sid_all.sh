#!/usr/bin/env bash
# Decode all downloaded SID rows into a separate resumable image directory.
#SBATCH --job-name=sid-decode-all
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:55:00
#SBATCH --output=slurm-sid-decode-all-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
DATASET_NAME="${SID_ALL_DATASET_NAME:-sid_all}"
RESUME_FLAG="${SID_DECODE_RESUME:+--resume}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
mkdir -p "${CACHE_ROOT}/apptainer"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

/usr/bin/apptainer exec --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PYTHONPATH=/workspace/src
  export AIGC_DATA_ROOT="'"${DATA_ROOT}"'"
  . "'"${CONTAINER_VENV}"'/bin/activate"
  python -c "import pyarrow" 2>/dev/null || python -m pip install --retries 20 --timeout 300 pyarrow
  python scripts/prepare_sid.py \
    --decode-all '"${RESUME_FLAG}"' \
    --output-dir "'"${DATA_ROOT}"'/'"${DATASET_NAME}"'_images" \
    --manifest-dir "'"${DATA_ROOT}"'/manifests" \
    --manifest-name "'"${DATASET_NAME}"'"
'
