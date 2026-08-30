#!/usr/bin/env bash
# Build an uncapped, training-safe three-source manifest. The trainer still
# consumes only split=train; validation/test rows remain held out.
#SBATCH --job-name=aigc-merge-full-manifests
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-merge-full-manifests-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
DATA_ROOT="${AIGC_DATA_ROOT:-${HOME}/aigc-storage/data}"
# The system Python on normal nodes has no PyTorch.  Importing the current
# data package loads the dataset module, so use the already-bootstrapped A100
# environment even though this manifest job itself is CPU-only.
VENV="${AIGC_A100_VENV:-${HOME}/aigc-storage/cache/venvs/a100-cu121-py312}"
for manifest in sid_all_all.csv cifake_all.csv wildfake_all.csv; do
  [ -f "${DATA_ROOT}/manifests/${manifest}" ] || { echo "ERROR: Missing ${DATA_ROOT}/manifests/${manifest}" >&2; exit 2; }
done
[ -f "${VENV}/bin/activate" ] || { echo "ERROR: Python environment missing: ${VENV}" >&2; exit 2; }
. "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src"
echo "[INFO] python=$(python --version)"
python3 "${REPO_ROOT}/scripts/merge_manifests.py" \
  --manifest "${DATA_ROOT}/manifests/sid_all_all.csv" \
  --manifest "${DATA_ROOT}/manifests/cifake_all.csv" \
  --manifest "${DATA_ROOT}/manifests/wildfake_all.csv" \
  --output-dir "${DATA_ROOT}/manifests" \
  --name aigc_full \
  --max-train-per-dataset-class 0 \
  --max-val-per-dataset-class 0 \
  --max-test-per-dataset-class 0
