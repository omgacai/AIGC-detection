#!/usr/bin/env bash
# Generic template: inspect sinfo/scontrol and set partition/account/qos only if your cluster requires them.
#SBATCH --job-name=aigc-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=aigc-smoke-%j.out
#SBATCH --error=aigc-smoke-%j.err
#SBATCH --partition=<set-if-required>
#SBATCH --account=<set-if-required>

set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$REPO_ROOT"
: "${AIGC_DATA_ROOT:?Set AIGC_DATA_ROOT to cluster-accessible dataset storage}"
: "${AIGC_CACHE_ROOT:?Set AIGC_CACHE_ROOT to large cluster storage, not HOME}"
export HF_HOME="$AIGC_CACHE_ROOT/huggingface"
export TORCH_HOME="$AIGC_CACHE_ROOT/torch"
export TRANSFORMERS_CACHE="$HF_HOME"
source .venv/bin/activate
python scripts/check_gpu.py
: "${AIGC_MANIFEST:?Set AIGC_MANIFEST to a generated manifest CSV}"
python scripts/dataloader_smoke_test.py --manifest "$AIGC_MANIFEST"
python scripts/check_dinov2.py
