#!/usr/bin/env bash
# Native A100 version of robustness evaluation.
#SBATCH --job-name=aigc-eval-a100
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-eval-a100-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CHECKPOINT_ROOT="${AIGC_CHECKPOINT_ROOT:-${STORAGE_ROOT}/checkpoints}"
OUTPUT_ROOT="${AIGC_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs}"
MANIFEST="${AIGC_MANIFEST:-${DATA_ROOT}/manifests/aigc_mixed_all.csv}"
MODEL_CONFIG="${AIGC_CONFIG:-configs/dinov3_multiscale_full_mixed.toml}"
EVALUATION_CONFIG="${AIGC_EVAL_CONFIG:-configs/evaluation.toml}"
EVALUATION_SPLIT="${AIGC_EVAL_SPLIT:-internal_val}"
CHECKPOINT="${AIGC_CHECKPOINT:?Set AIGC_CHECKPOINT to the exact checkpoint to evaluate}"
VENV="${AIGC_A100_VENV:-${CACHE_ROOT}/venvs/a100-cu121-py312}"

for required in "$MANIFEST" "$MODEL_CONFIG" "$EVALUATION_CONFIG" "$CHECKPOINT" "$VENV/.ready"; do
  [ -f "$required" ] || { echo "ERROR: Required path missing: $required" >&2; exit 2; }
done
mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT/huggingface"
. "$VENV/bin/activate"
export PYTHONPATH="$REPO_ROOT/src" HF_HOME="$CACHE_ROOT/huggingface"
export AIGC_DATA_ROOT="$DATA_ROOT" AIGC_CACHE_ROOT="$CACHE_ROOT"
export AIGC_CHECKPOINT_ROOT="$CHECKPOINT_ROOT" AIGC_OUTPUT_ROOT="$OUTPUT_ROOT"
echo "[INFO] host=$(hostname) checkpoint=$CHECKPOINT split=$EVALUATION_SPLIT"
python "$REPO_ROOT/scripts/evaluate.py" --checkpoint "$CHECKPOINT" --manifest "$MANIFEST" --model-config "$MODEL_CONFIG" --evaluation-config "$EVALUATION_CONFIG" --split "$EVALUATION_SPLIT"
