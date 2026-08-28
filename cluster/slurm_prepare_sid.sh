#!/usr/bin/env bash
# Prepare a larger balanced SID subset from the already-downloaded Parquet shards.
#SBATCH --job-name=sid-prepare-full
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-sid-prepare-full-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
MAX_PER_CLASS="${SID_MAX_PER_CLASS:-10000}"
DATASET_NAME="${SID_DATASET_NAME:-sid_balanced_20k}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
mkdir -p "${CACHE_ROOT}/apptainer"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

/usr/bin/apptainer exec --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PYTHONPATH=/workspace/src
  export AIGC_DATA_ROOT="'"${DATA_ROOT}"'"
  . "'"${CONTAINER_VENV}"'/bin/activate"
  python scripts/prepare_sid.py \
    --output-dir "'"${DATA_ROOT}"'/'"${DATASET_NAME}"'_images" \
    --manifest-dir "'"${DATA_ROOT}"'/manifests" \
    --manifest-name "'"${DATASET_NAME}"'" \
    --max-per-class "'"${MAX_PER_CLASS}"'"
'
