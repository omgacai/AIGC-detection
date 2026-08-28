#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$REPO_ROOT/.venv}"

"$PYTHON_BIN" - <<'PY'
import sys
if not ((3, 10) <= sys.version_info < (3, 13)):
    raise SystemExit(f"[ERROR] Python 3.10, 3.11, or 3.12 is required; found {sys.version.split()[0]}")
PY

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip

if python -c 'import torch, torchvision; print("[INFO] Existing PyTorch:", torch.__version__, "CUDA:", torch.version.cuda); print("[INFO] Existing torchvision:", torchvision.__version__)' 2>/dev/null; then
  # Preserve an existing cluster-provided CUDA build; install every other requirement.
  TEMP_REQUIREMENTS="$(mktemp)"
  grep -Ev '^(torch|torchvision)([<>=!~ ].*)?$' "$REPO_ROOT/requirements.txt" > "$TEMP_REQUIREMENTS"
  python -m pip install -r "$TEMP_REQUIREMENTS"
  rm -f "$TEMP_REQUIREMENTS"
else
  echo "[INFO] No PyTorch detected. Installing requirements; on a GPU cluster, use the institution's CUDA-compatible PyTorch instructions if this chooses an unsuitable build."
  python -m pip install -r "$REPO_ROOT/requirements.txt"
fi
python -m pip install -e "$REPO_ROOT"
python - <<'PY'
for module in ("torch", "torchvision", "transformers", "timm", "huggingface_hub", "PIL", "yaml"):
    __import__(module)
    print(f"[INFO] import PASS: {module}")
PY
python "$REPO_ROOT/scripts/check_gpu.py"
