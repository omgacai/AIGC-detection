#!/usr/bin/env bash
# One-time native Python environment bootstrap for xgph0 (A100 80 GB).
# xgph0 has no Apptainer, so this intentionally uses its system Python 3.12.
#SBATCH --job-name=a100-env-bootstrap
#SBATCH --partition=gpu
#SBATCH --nodelist=xgph0
#SBATCH --gres=gpu:a100-80:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-a100-bootstrap-%j.out

set -eu
REPO_ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
STORAGE_ROOT="${HOME}/aigc-storage"
CACHE_ROOT="${AIGC_CACHE_ROOT:-${STORAGE_ROOT}/cache}"
VENV="${AIGC_A100_VENV:-${CACHE_ROOT}/venvs/a100-cu121-py312}"
READY_MARKER="${VENV}/.ready"
PYTHON="${AIGC_A100_PYTHON:-python3.12}"
PIP_CACHE_DIR="${CACHE_ROOT}/pip"

command -v "$PYTHON" >/dev/null || { echo "ERROR: Python 3.12 is required but unavailable: $PYTHON" >&2; exit 2; }
mkdir -p "${CACHE_ROOT}" "${PIP_CACHE_DIR}"
export PIP_CACHE_DIR

if [ -e "$READY_MARKER" ]; then
  echo "[INFO] Ready A100 environment already exists: $VENV"
  "$VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
  exit 0
fi
if [ -e "$VENV" ]; then
  echo "ERROR: Refusing to overwrite incomplete environment: $VENV" >&2
  echo "Inspect it first. Remove this exact directory manually only if you decide it is disposable." >&2
  exit 2
fi

"$PYTHON" -m venv "$VENV"
. "$VENV/bin/activate"
python -m pip install --upgrade pip
# Pin the CUDA build so a CPU wheel is never selected accidentally.
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.4.1+cu121' 'torchvision==0.19.1+cu121'
REQUIREMENTS_FILE="${SLURM_TMPDIR:-/tmp}/${USER}-a100-requirements.txt"
grep -Ev '^(torch|torchvision)([<>=!~ ].*)?$' "$REPO_ROOT/requirements.txt" > "$REQUIREMENTS_FILE"
python -m pip install --retries 20 --timeout 300 -r "$REQUIREMENTS_FILE"
python - <<'PY'
import albumentations, sklearn, tensorboard, timm, torch, torchvision, transformers
assert torch.__version__ == '2.4.1+cu121'
assert torchvision.__version__ == '0.19.1+cu121'
assert transformers.__version__ == '4.56.2'
assert torch.cuda.is_available()
print(f'[INFO] torch={torch.__version__} cuda={torch.version.cuda} gpu={torch.cuda.get_device_name(0)}')
PY
touch "$READY_MARKER"
echo "[INFO] A100 environment ready: $VENV"
