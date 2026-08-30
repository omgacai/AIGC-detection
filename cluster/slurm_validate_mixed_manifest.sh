#!/usr/bin/env bash
# Validate the mixed manifest on the known-working xgpd0 Python container.
# No image is modified or deleted; only a new manifest/report are written.
#SBATCH --job-name=validate-mixed-images
#SBATCH --partition=gpu
#SBATCH --nodelist=xgpd0
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-validate-mixed-images-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
MANIFEST="${AIGC_MANIFEST:-${DATA_ROOT}/manifests/aigc_mixed_all.csv}"
OUTPUT="${AIGC_VALIDATED_MANIFEST:-${DATA_ROOT}/manifests/aigc_mixed_verified_all.csv}"
REPORT="${AIGC_INVALID_IMAGE_REPORT:-${DATA_ROOT}/manifests/aigc_mixed_invalid_images.csv}"
CONTAINER_VENV="${CACHE_ROOT}/venvs/pytorch-2.4-cu121"
IMAGE="docker://pytorch/pytorch:2.4.1-cuda12.1-cudnn9-runtime"

for required in "$MANIFEST" "$CONTAINER_VENV/bin/python"; do
  [ -f "$required" ] || { echo "ERROR: Required path missing: $required" >&2; exit 2; }
done
mkdir -p "${CACHE_ROOT}/apptainer"
export APPTAINER_CACHEDIR="${CACHE_ROOT}/apptainer"
/usr/bin/apptainer exec --bind "${REPO_ROOT}:/workspace:ro,${STORAGE_ROOT}:${STORAGE_ROOT}" --pwd /workspace "$IMAGE" bash -lc '
  set -eu
  export PYTHONPATH=/workspace/src
  . "'"${CONTAINER_VENV}"'/bin/activate"
  python scripts/validate_image_manifest.py \
    --manifest "'"${MANIFEST}"'" \
    --output "'"${OUTPUT}"'" \
    --invalid-report "'"${REPORT}"'"
'
