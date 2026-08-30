#!/usr/bin/env bash
# Build COCO/DALL-E reference manifests on CPU only. It never modifies images.
#SBATCH --job-name=competition-reference-manifest
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-competition-reference-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
MANIFEST_DIR="${AIGC_REFERENCE_MANIFEST_DIR:-${STORAGE_ROOT}/data/manifests}"
REFERENCE_RAW_ROOT="${AIGC_REFERENCE_RAW_ROOT:-${STORAGE_ROOT}/data/competition_reference/raw}"
# ModelsScope archives retain the archive-name directory after extraction.
AIGC_COCO_ROOT="${AIGC_COCO_ROOT:-${REFERENCE_RAW_ROOT}/Images/Real/coco/coco/coco2017}"
AIGC_DALLE_ROOT="${AIGC_DALLE_ROOT:-${REFERENCE_RAW_ROOT}/Images/Diffusion_based/DALLE/DALLE/Advanced/DALLE3/dalle3}"
VENV="${AIGC_A100_VENV:-${STORAGE_ROOT}/cache/venvs/a100-cu121-py312}"
: "${AIGC_COCO_ROOT:?Set AIGC_COCO_ROOT to the authorised COCO val2017 image directory}"
: "${AIGC_DALLE_ROOT:?Set AIGC_DALLE_ROOT to the authorised DALL-E Advanced image directory}"
[ -f "${VENV}/bin/activate" ] || { echo "ERROR: Python environment missing: ${VENV}" >&2; exit 2; }

. "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src"
python3 "${REPO_ROOT}/scripts/prepare_reference_benchmark.py" \
  --coco-root "$AIGC_COCO_ROOT" \
  --dalle-root "$AIGC_DALLE_ROOT" \
  --manifest-dir "$MANIFEST_DIR"
