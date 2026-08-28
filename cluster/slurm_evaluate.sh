#!/usr/bin/env bash
# Robustness evaluation: clean plus every configured real-world transformation.
#SBATCH --job-name=aigc-eval
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-eval-%j.out

set -eu

REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit this job from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CHECKPOINT_ROOT="${AIGC_CHECKPOINT_ROOT:-${STORAGE_ROOT}/checkpoints}"
OUTPUT_ROOT="${AIGC_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs}"
MANIFEST="${AIGC_MANIFEST:-${DATA_ROOT}/manifests/sid_smoke_all.csv}"
MODEL_CONFIG="${AIGC_CONFIG:-configs/google_vit_large_forensic_smoke.toml}"
EVALUATION_CONFIG="${AIGC_EVAL_CONFIG:-configs/evaluation.toml}"
EVALUATION_SPLIT="${AIGC_EVAL_SPLIT:-internal_val}"
CHECKPOINT="${AIGC_CHECKPOINT:-${CHECKPOINT_ROOT}/google_vit_large_forensic_smoke/best.pt}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"

for required in "${MANIFEST}" "${MODEL_CONFIG}" "${EVALUATION_CONFIG}" "${CHECKPOINT}" "${CONTAINER_VENV}/bin/python"; do
  if [ ! -e "${required}" ]; then
    echo "ERROR: Required path not found: ${required}" >&2
    exit 2
  fi
done

mkdir -p "${OUTPUT_ROOT}" "${CACHE_ROOT}/apptainer"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"
echo "Host: $(hostname)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Manifest: ${MANIFEST}"
echo "Model config: ${MODEL_CONFIG}"
echo "Evaluation config: ${EVALUATION_CONFIG}"
echo "Split: ${EVALUATION_SPLIT}"

/usr/bin/apptainer exec --nv \
  --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" \
  --pwd /workspace \
  "${IMAGE}" \
  bash -lc '
    set -eu
    export PYTHONPATH=/workspace/src
    export HF_HOME="'"${CACHE_ROOT}"'/huggingface"
    export AIGC_DATA_ROOT="'"${DATA_ROOT}"'"
    export AIGC_CACHE_ROOT="'"${CACHE_ROOT}"'"
    export AIGC_CHECKPOINT_ROOT="'"${CHECKPOINT_ROOT}"'"
    export AIGC_OUTPUT_ROOT="'"${OUTPUT_ROOT}"'"
    . "'"${CONTAINER_VENV}"'/bin/activate"
    python scripts/evaluate.py \
      --checkpoint "'"${CHECKPOINT}"'" \
      --manifest "'"${MANIFEST}"'" \
      --model-config "'"${MODEL_CONFIG}"'" \
      --evaluation-config "'"${EVALUATION_CONFIG}"'" \
      --split "'"${EVALUATION_SPLIT}"'"
  '
