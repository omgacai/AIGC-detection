#!/usr/bin/env bash
# Export a selected checkpoint to ONNX on a CPU compute node. No dataset or GPU
# is required; the A100 Python environment supplies the installed ML stack.
#SBATCH --job-name=aigc-export-onnx
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=slurm-export-onnx-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
VENV="${AIGC_A100_VENV:-${STORAGE_ROOT}/cache/venvs/a100-cu121-py312}"
CONFIG="${AIGC_CONFIG:-configs/dinov3_multiscale_full_mixed.toml}"
CHECKPOINT="${AIGC_CHECKPOINT:?Set AIGC_CHECKPOINT to the checkpoint to export}"
OUTPUT="${AIGC_ONNX_OUTPUT:-${STORAGE_ROOT}/exports/aigc_detector.onnx}"

for required in "${VENV}/bin/activate" "${CONFIG}" "${CHECKPOINT}"; do
  [ -f "${required}" ] || { echo "ERROR: Required path missing: ${required}" >&2; exit 2; }
done
. "${VENV}/bin/activate"
export PYTHONPATH="${REPO_ROOT}/src"
export HF_HOME="${STORAGE_ROOT}/cache/huggingface"
python -c 'import onnx; print("[INFO] onnx=", onnx.__version__)'
python "${REPO_ROOT}/scripts/export_onnx.py" \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --output "${OUTPUT}"
