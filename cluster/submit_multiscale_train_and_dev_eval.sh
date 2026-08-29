#!/usr/bin/env bash
# Submit one resumable training chunk and queue development robustness eval.
# Held-out test / organiser data are deliberately NOT submitted here.
set -eu

REPO_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$REPO_ROOT"
STORAGE_ROOT="${HOME}/aigc-storage"
MANIFEST="${AIGC_MANIFEST:-${STORAGE_ROOT}/data/manifests/aigc_mixed_all.csv}"
CONFIG="${AIGC_CONFIG:-configs/dinov3_multiscale_full_mixed.toml}"
RUN_ID="${AIGC_RUN_ID:-$(date +%Y%m%d-%H%M%S)_dinov3_multiscale_full_mixed}"
EPOCHS_THIS_JOB="${AIGC_EPOCHS_THIS_JOB:-2}"
RESUME="${AIGC_RESUME:-}"
CHECKPOINT_ROOT="${AIGC_CHECKPOINT_ROOT:-${STORAGE_ROOT}/checkpoints}"
OUTPUT_ROOT="${AIGC_OUTPUT_ROOT:-${STORAGE_ROOT}/outputs}"

[ -f "$MANIFEST" ] || { echo "ERROR: mixed manifest is unavailable: $MANIFEST" >&2; exit 2; }
[ -f "$CONFIG" ] || { echo "ERROR: config is unavailable: $CONFIG" >&2; exit 2; }

if [ -n "$RESUME" ]; then
  RUN_ID="$(basename "$(dirname "$RESUME")")"
fi

TRAIN_JOB="$({
  AIGC_MANIFEST="$MANIFEST" AIGC_CONFIG="$CONFIG" AIGC_EPOCHS_THIS_JOB="$EPOCHS_THIS_JOB" \
  AIGC_RESUME="$RESUME" AIGC_RUN_ID="$RUN_ID" \
  sbatch --parsable --export=ALL,AIGC_RUN_ID "$REPO_ROOT/cluster/slurm_train_full.sh"
} | tail -n 1)"

CHECKPOINT="${CHECKPOINT_ROOT}/${RUN_ID}/best_tpr_at_1_fpr.pt"
DEV_OUTPUT_ROOT="${OUTPUT_ROOT}/${RUN_ID}_internal_val"
EVAL_JOB="$({
  AIGC_MANIFEST="$MANIFEST" AIGC_CONFIG="$CONFIG" AIGC_CHECKPOINT="$CHECKPOINT" \
  AIGC_EVAL_SPLIT="internal_val" AIGC_OUTPUT_ROOT="$DEV_OUTPUT_ROOT" \
  sbatch --parsable --dependency="afterok:${TRAIN_JOB}" "$REPO_ROOT/cluster/slurm_evaluate.sh"
} | tail -n 1)"

printf '%s\n' "RUN_ID=${RUN_ID}" "TRAIN_JOB=${TRAIN_JOB}" "DEV_EVAL_JOB=${EVAL_JOB}" \
  "Train log: slurm-train-vitb-full-${TRAIN_JOB}.out" \
  "Eval log:  slurm-eval-${EVAL_JOB}.out" \
  "Dev results: ${DEV_OUTPUT_ROOT}/robustness_eval_v1/robustness_summary.csv"
