#!/usr/bin/env bash
# Download CIFAKE through Kaggle and generate its initial manifest.
#SBATCH --job-name=cifake-download
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-cifake-download-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
if [ ! -s "${HOME}/.kaggle/kaggle.json" ] && [ -z "${KAGGLE_API_TOKEN:-}" ] && [ -z "${KAGGLE_KEY:-}" ]; then
  echo "ERROR: Configure Kaggle authentication without committing credentials." >&2
  echo "Use ~/.kaggle/kaggle.json (mode 600) or export KAGGLE_API_TOKEN." >&2
  exit 2
fi
mkdir -p "${CACHE_ROOT}/apptainer" "${CACHE_ROOT}/pip"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

/usr/bin/apptainer exec --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PYTHONPATH=/workspace/src
  export PIP_CACHE_DIR="'"${CACHE_ROOT}"'/pip"
  export AIGC_DATA_ROOT="'"${DATA_ROOT}"'"
  export AIGC_CACHE_ROOT="'"${CACHE_ROOT}"'"
  . "'"${CONTAINER_VENV}"'/bin/activate"
  python -c "import kaggle" 2>/dev/null || python -m pip install --retries 20 --timeout 300 kaggle
  python scripts/download_datasets.py --dataset cifake --output-dir "'"${DATA_ROOT}"'"
'
