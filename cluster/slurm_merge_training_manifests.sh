#!/usr/bin/env bash
# Combine only the three approved training datasets into the full manifest.
#SBATCH --job-name=aigc-merge-manifests
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-merge-manifests-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
for manifest in sid_balanced_20k_all.csv cifake_all.csv wildfake_all.csv; do
  [ -f "${DATA_ROOT}/manifests/${manifest}" ] || { echo "ERROR: Missing ${DATA_ROOT}/manifests/${manifest}" >&2; exit 2; }
done
mkdir -p "${CACHE_ROOT}/apptainer"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

/usr/bin/apptainer exec --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PYTHONPATH=/workspace/src
  . "'"${CONTAINER_VENV}"'/bin/activate"
  python scripts/merge_manifests.py \
    --manifest "'"${DATA_ROOT}"'/manifests/sid_balanced_20k_all.csv" \
    --manifest "'"${DATA_ROOT}"'/manifests/cifake_all.csv" \
    --manifest "'"${DATA_ROOT}"'/manifests/wildfake_all.csv" \
    --output-dir "'"${DATA_ROOT}"'/manifests" \
    --name aigc_mixed
'
