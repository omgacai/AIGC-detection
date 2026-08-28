#!/usr/bin/env bash
# Phase 1+ template only. Do not submit until train.py exists.
#SBATCH --job-name=aigc-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=aigc-train-%j.out
#SBATCH --error=aigc-train-%j.err
#SBATCH --partition=<set-if-required>
#SBATCH --account=<set-if-required>

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
: "${AIGC_DATA_ROOT:?Set cluster dataset storage}"
: "${AIGC_CACHE_ROOT:?Set cluster cache storage}"
export HF_HOME="$AIGC_CACHE_ROOT/huggingface"
export TORCH_HOME="$AIGC_CACHE_ROOT/torch"
export TRANSFORMERS_CACHE="$HF_HOME"
source .venv/bin/activate
# Phase 1+: python train.py --config configs/train.yaml
