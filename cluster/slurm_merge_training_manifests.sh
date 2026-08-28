#!/usr/bin/env bash
# Combine only the three approved training datasets into the full manifest.
#SBATCH --job-name=aigc-merge-manifests
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --gres=gpu:nv:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-merge-manifests-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"
# A fast, balanced prototype: 4k train images per source/class gives 24k
# train images across SID, CIFAKE, and WildFake.  Override only after this
# experiment has been validated.
MAX_TRAIN_PER_DATASET_CLASS="${AIGC_MAX_TRAIN_PER_DATASET_CLASS:-4000}"
MAX_VAL_PER_DATASET_CLASS="${AIGC_MAX_VAL_PER_DATASET_CLASS:-500}"
MAX_TEST_PER_DATASET_CLASS="${AIGC_MAX_TEST_PER_DATASET_CLASS:-1000}"
for manifest in sid_balanced_20k_all.csv cifake_all.csv wildfake_all.csv; do
  [ -f "${DATA_ROOT}/manifests/${manifest}" ] || { echo "ERROR: Missing ${DATA_ROOT}/manifests/${manifest}" >&2; exit 2; }
done
mkdir -p "${CACHE_ROOT}/apptainer"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"
echo "[INFO] per-source/class caps: train=${MAX_TRAIN_PER_DATASET_CLASS} val=${MAX_VAL_PER_DATASET_CLASS} test=${MAX_TEST_PER_DATASET_CLASS}"

/usr/bin/apptainer exec --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "${IMAGE}" bash -lc '
  set -eu
  export PYTHONPATH=/workspace/src
  . "'"${CONTAINER_VENV}"'/bin/activate"
  python scripts/merge_manifests.py \
    --manifest "'"${DATA_ROOT}"'/manifests/sid_balanced_20k_all.csv" \
    --manifest "'"${DATA_ROOT}"'/manifests/cifake_all.csv" \
    --manifest "'"${DATA_ROOT}"'/manifests/wildfake_all.csv" \
    --output-dir "'"${DATA_ROOT}"'/manifests" \
    --name aigc_mixed \
    --max-train-per-dataset-class "'"${MAX_TRAIN_PER_DATASET_CLASS}"'" \
    --max-val-per-dataset-class "'"${MAX_VAL_PER_DATASET_CLASS}"'" \
    --max-test-per-dataset-class "'"${MAX_TEST_PER_DATASET_CLASS}"'"
'
