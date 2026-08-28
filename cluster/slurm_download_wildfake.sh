#!/usr/bin/env bash
# Download WildFake from ModelScope. Layout inspection/preparation follows download.
#SBATCH --job-name=wildfake-download
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=slurm-wildfake-download-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
mkdir -p "${DATA_ROOT}/wildfake" "${CACHE_ROOT}/apptainer" "${CACHE_ROOT}/pip" "${CACHE_ROOT}/modelscope"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

/usr/bin/apptainer exec --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PIP_CACHE_DIR="'"${CACHE_ROOT}"'/pip"
  export MODELSCOPE_CACHE="'"${CACHE_ROOT}"'/modelscope"
  . "'"${CONTAINER_VENV}"'/bin/activate"
  command -v modelscope >/dev/null 2>&1 || python -m pip install --retries 20 --timeout 300 modelscope
  modelscope download --dataset hy2628982280/WildFake --local_dir "'"${DATA_ROOT}"'/wildfake"
  echo "[INFO] WildFake downloaded. Inspect its top-level layout before generating labels:"
  find "'"${DATA_ROOT}"'/wildfake" -maxdepth 3 -type d | head -100
'
