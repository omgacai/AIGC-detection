#!/usr/bin/env bash
# Generate WildFake manifests after inspecting and confirming its extracted layout.
#SBATCH --job-name=wildfake-prepare
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-wildfake-prepare-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
mkdir -p "${CACHE_ROOT}/apptainer" "${DATA_ROOT}/manifests"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"
RAW_ROOT="${DATA_ROOT}/wildfake/raw"
if find "${RAW_ROOT}" -type f \( -iname '*coco*' -o -iname '*dalle*' \) -print -quit | grep -q .; then
  echo "ERROR: COCO/DALL-E organiser benchmark material was found below ${RAW_ROOT}. Refusing to create a training manifest." >&2
  exit 2
fi
/usr/bin/apptainer exec --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PYTHONPATH=/workspace/src
  . "'"${CONTAINER_VENV}"'/bin/activate"
  python scripts/register_directory_dataset.py \
    --dataset wildfake \
    --data-dir "'"${RAW_ROOT}"'" \
    --manifest-dir "'"${DATA_ROOT}"'/manifests"
'
