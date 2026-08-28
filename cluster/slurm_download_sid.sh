#!/usr/bin/env bash
# Download SID_Set and create manifests in persistent cluster storage.
# Submit from the repository root with: sbatch cluster/slurm_download_sid.sh
#SBATCH --job-name=sid-download
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-sid-download-%j.out

set -eu

REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
PYTHON_DEPS="${CACHE_ROOT}/container-python-deps"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"

mkdir -p "${DATA_ROOT}" "${CACHE_ROOT}/apptainer" "${CACHE_ROOT}/pip" "${PYTHON_DEPS}"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

echo "Host: $(hostname)"
echo "Downloading SID to: ${DATA_ROOT}/sid"
/usr/bin/apptainer exec \
  --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" \
  --pwd /workspace \
  "${IMAGE}" \
  bash -lc '
    set -eu
    export PYTHONPATH="'"${PYTHON_DEPS}"':/workspace/src${PYTHONPATH:+:$PYTHONPATH}"
    export PIP_CACHE_DIR="'"${CACHE_ROOT}"'/pip"
    export AIGC_DATA_ROOT="'"${DATA_ROOT}"'"
    export AIGC_CACHE_ROOT="'"${CACHE_ROOT}"'"
    export HF_HOME="'"${CACHE_ROOT}"'/huggingface"
    if ! python -c "import huggingface_hub, yaml, sklearn"; then
      python -m pip install --target "'"${PYTHON_DEPS}"'" --retries 20 --timeout 300 huggingface_hub pyyaml scikit-learn
    fi
    python scripts/download_datasets.py --dataset sid --output-dir "'"${DATA_ROOT}"'"
  '
