#!/usr/bin/env bash
# Frozen DINOv3 layer/probe robustness diagnostic. It never reads split=test.
#SBATCH --job-name=dinov3-layer-probe
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:55:00
#SBATCH --output=slurm-layer-probe-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
OUTPUT_ROOT="${AIGC_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs}"
MANIFEST="${AIGC_MANIFEST:-${DATA_ROOT}/manifests/sid_balanced_20k_all.csv}"
PROBE_OUTPUT="${AIGC_PROBE_OUTPUT:-${OUTPUT_ROOT}/dinov3_layer_probe_sid}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"

[ -f "${MANIFEST}" ] || { echo "ERROR: Manifest not found: ${MANIFEST}" >&2; exit 2; }
mkdir -p "${CACHE_ROOT}/apptainer" "${PROBE_OUTPUT}"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"

/usr/bin/apptainer exec --nv --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PYTHONPATH=/workspace/src
  export HF_HOME="'"${CACHE_ROOT}"'/huggingface"
  . "'"${CONTAINER_VENV}"'/bin/activate"
  python scripts/probe_dinov3_layers.py \
    --manifest "'"${MANIFEST}"'" \
    --output-dir "'"${PROBE_OUTPUT}"'" \
    --batch-size 32 \
    --num-workers 2 \
    --layers all
'
