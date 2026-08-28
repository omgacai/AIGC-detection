#!/usr/bin/env bash
# Decode a balanced 2,000-image SID subset for the first DINOv3 training test.
#SBATCH --job-name=sid-prepare
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-sid-prepare-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
PYTHON_DEPS="${CACHE_ROOT}/container-python-deps"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
mkdir -p "${CACHE_ROOT}/apptainer" "${CACHE_ROOT}/pip" "${PYTHON_DEPS}"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

/usr/bin/apptainer exec --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PYTHONPATH="'"${PYTHON_DEPS}"':/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
  export PIP_CACHE_DIR="'"${CACHE_ROOT}"'/pip"
  export AIGC_DATA_ROOT="'"${DATA_ROOT}"'"
  if ! python -c "import pyarrow"; then
    python -m pip install --target "'"${PYTHON_DEPS}"'" --retries 20 --timeout 300 pyarrow
  fi
  python scripts/prepare_sid.py --max-per-class 1000
'
