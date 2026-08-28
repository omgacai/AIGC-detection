#!/usr/bin/env bash
# Download one approved WildFake ZIP per Slurm-array element.  COCO and DALL-E
# are deliberately absent from cluster/wildfake_archives.py because they are
# reserved organiser-benchmark data and must not enter training data.
#SBATCH --job-name=wildfake-download
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=03:00:00
#SBATCH --array=0-12%1
#SBATCH --output=slurm-wildfake-download-%A_%a.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
NODE_TEMP_ROOT="${SLURM_TMPDIR:-/tmp/${USER}-wildfake}"

mkdir -p "${CACHE_ROOT}/apptainer" "${CACHE_ROOT}/pip" "${DATA_ROOT}/wildfake/raw" "${NODE_TEMP_ROOT}"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"
ARCHIVE="$(python3 "${REPO_ROOT}/cluster/wildfake_archives.py" --index "${SLURM_ARRAY_TASK_ID}")"
echo "[INFO] array_task=${SLURM_ARRAY_TASK_ID} archive=${ARCHIVE}"

/usr/bin/apptainer exec \
  --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT},${NODE_TEMP_ROOT}:${NODE_TEMP_ROOT}" \
  --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PYTHONPATH=/workspace/src
  export PIP_CACHE_DIR="'"${CACHE_ROOT}"'/pip"
  export MODELSCOPE_CACHE="'"${CACHE_ROOT}"'/modelscope"
  . "'"${CONTAINER_VENV}"'/bin/activate"
  python -c "import modelscope" 2>/dev/null || python -m pip install --retries 20 --timeout 300 modelscope
  python cluster/download_wildfake_archive.py \
    --archive "'"${ARCHIVE}"'" \
    --raw-root "'"${DATA_ROOT}"'/wildfake/raw" \
    --temporary-root "'"${NODE_TEMP_ROOT}"'"
'
