#!/usr/bin/env bash
# Full DINOv3 curriculum, executed in bounded resumable epoch chunks.
#SBATCH --job-name=dinov3-full
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:55:00
#SBATCH --output=slurm-train-full-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CHECKPOINT_ROOT="${AIGC_CHECKPOINT_ROOT:-${STORAGE_ROOT}/checkpoints}"
OUTPUT_ROOT="${AIGC_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs}"
MANIFEST="${AIGC_MANIFEST:-${DATA_ROOT}/manifests/aigc_mixed_all.csv}"
MODEL_CONFIG="${AIGC_CONFIG:-configs/dinov3_forensic.toml}"
EPOCHS_THIS_JOB="${AIGC_EPOCHS_THIS_JOB:-4}"
RESUME="${AIGC_RESUME:-}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
for required in "${MANIFEST}" "${MODEL_CONFIG}"; do
  [ -f "${required}" ] || { echo "ERROR: Required file missing: ${required}" >&2; exit 2; }
done
if [ -n "${RESUME}" ] && [ ! -f "${RESUME}" ]; then
  echo "ERROR: Resume checkpoint missing: ${RESUME}" >&2; exit 2
fi
mkdir -p "${CACHE_ROOT}/apptainer" "${CHECKPOINT_ROOT}" "${OUTPUT_ROOT}"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

/usr/bin/apptainer exec --nv --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PYTHONPATH=/workspace/src
  export HF_HOME="'"${CACHE_ROOT}"'/huggingface"
  export AIGC_DATA_ROOT="'"${DATA_ROOT}"'"
  export AIGC_CACHE_ROOT="'"${CACHE_ROOT}"'"
  export AIGC_CHECKPOINT_ROOT="'"${CHECKPOINT_ROOT}"'"
  export AIGC_OUTPUT_ROOT="'"${OUTPUT_ROOT}"'"
  . "'"${CONTAINER_VENV}"'/bin/activate"
  set -- \
    --config "'"${MODEL_CONFIG}"'" \
    --manifest "'"${MANIFEST}"'" \
    --epochs-this-job "'"${EPOCHS_THIS_JOB}"'"
  if [ -n "'"${RESUME}"'" ]; then
    set -- "$@" --resume "'"${RESUME}"'"
  fi
  python scripts/train.py "$@"
'
