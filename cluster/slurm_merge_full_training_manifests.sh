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
for manifest in sid_all_all.csv cifake_all.csv wildfake_all.csv; do
  [ -f "${DATA_ROOT}/manifests/${manifest}" ] || { echo "ERROR: Missing ${DATA_ROOT}/manifests/${manifest}" >&2; exit 2; }
done
export PYTHONPATH="${REPO_ROOT}/src"
python3 "${REPO_ROOT}/scripts/merge_manifests.py" \
  --manifest "${DATA_ROOT}/manifests/sid_all_all.csv" \
  --manifest "${DATA_ROOT}/manifests/cifake_all.csv" \
  --manifest "${DATA_ROOT}/manifests/wildfake_all.csv" \
  --output-dir "${DATA_ROOT}/manifests" \
  --name aigc_full \
  --max-train-per-dataset-class 0 \
  --max-val-per-dataset-class 0 \
  --max-test-per-dataset-class 0
