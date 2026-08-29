#!/usr/bin/env bash
# Native A100 version of full DINOv3 training; keeps the Titan path unchanged.
#SBATCH --job-name=dinov3-vitb-a100
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=02:55:00
#SBATCH --output=slurm-train-vitb-a100-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CHECKPOINT_ROOT="${AIGC_CHECKPOINT_ROOT:-${STORAGE_ROOT}/checkpoints}"
OUTPUT_ROOT="${AIGC_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs}"
MANIFEST="${AIGC_MANIFEST:-${DATA_ROOT}/manifests/aigc_mixed_all.csv}"
MODEL_CONFIG="${AIGC_CONFIG:-configs/dinov3_multiscale_full_mixed.toml}"
EPOCHS_THIS_JOB="${AIGC_EPOCHS_THIS_JOB:-2}"
RESUME="${AIGC_RESUME:-}"
RUN_DIRECTORY="${AIGC_RUN_ID:-}"
VENV="${AIGC_A100_VENV:-${CACHE_ROOT}/venvs/a100-cu121-py312}"

for required in "$MANIFEST" "$MODEL_CONFIG" "$VENV/.ready"; do
  [ -f "$required" ] || { echo "ERROR: Required path missing: $required" >&2; exit 2; }
done
if [ -n "$RESUME" ] && [ ! -f "$RESUME" ]; then echo "ERROR: Resume checkpoint missing: $RESUME" >&2; exit 2; fi
mkdir -p "$CHECKPOINT_ROOT" "$OUTPUT_ROOT" "$CACHE_ROOT/huggingface"
. "$VENV/bin/activate"
export PYTHONPATH="$REPO_ROOT/src"
export HF_HOME="$CACHE_ROOT/huggingface"
export AIGC_DATA_ROOT="$DATA_ROOT" AIGC_CACHE_ROOT="$CACHE_ROOT"
export AIGC_CHECKPOINT_ROOT="$CHECKPOINT_ROOT" AIGC_OUTPUT_ROOT="$OUTPUT_ROOT"
echo "[INFO] host=$(hostname) python=$(python --version) gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
set -- --config "$MODEL_CONFIG" --manifest "$MANIFEST" --epochs-this-job "$EPOCHS_THIS_JOB"
if [ -n "$RESUME" ]; then set -- "$@" --resume "$RESUME"; elif [ -n "$RUN_DIRECTORY" ]; then set -- "$@" --run-directory "$RUN_DIRECTORY"; fi
python "$REPO_ROOT/scripts/train.py" "$@"
