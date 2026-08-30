#!/usr/bin/env bash
# Read every full-manifest image and create an immutable verified copy.
# This is CPU/I/O work; it never changes or deletes original dataset files.
#SBATCH --job-name=validate-full-images
#SBATCH --partition=long
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm-validate-full-images-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
DATA_ROOT="${AIGC_DATA_ROOT:-${STORAGE_ROOT}/data}"
VENV="${AIGC_A100_VENV:-${STORAGE_ROOT}/cache/venvs/a100-cu121-py312}"
MANIFEST="${AIGC_MANIFEST:-${DATA_ROOT}/manifests/aigc_full_all.csv}"
OUTPUT="${AIGC_VALIDATED_MANIFEST:-${DATA_ROOT}/manifests/aigc_full_verified_all.csv}"
REPORT="${AIGC_INVALID_IMAGE_REPORT:-${DATA_ROOT}/manifests/aigc_full_invalid_images.csv}"

for required in "$MANIFEST" "$VENV/bin/activate"; do
  [ -f "$required" ] || { echo "ERROR: Required path missing: $required" >&2; exit 2; }
done
[ ! -e "$OUTPUT" ] || { echo "ERROR: Refusing to overwrite existing verified manifest: $OUTPUT" >&2; exit 2; }
[ ! -e "$REPORT" ] || { echo "ERROR: Refusing to overwrite existing invalid-image report: $REPORT" >&2; exit 2; }

. "$VENV/bin/activate"
export PYTHONPATH="$REPO_ROOT/src"
python "$REPO_ROOT/scripts/validate_image_manifest.py" \
  --manifest "$MANIFEST" \
  --output "$OUTPUT" \
  --invalid-report "$REPORT" \
  --workers "${SLURM_CPUS_PER_TASK:-1}"
