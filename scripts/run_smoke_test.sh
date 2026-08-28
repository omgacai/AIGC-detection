#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python "$REPO_ROOT/scripts/check_gpu.py"
if [[ -n "${AIGC_MANIFEST:-}" ]]; then
  python "$REPO_ROOT/scripts/dataloader_smoke_test.py" --manifest "$AIGC_MANIFEST"
else
  echo "[WARNING] Set AIGC_MANIFEST=/path/to/sid_all.csv to run the DataLoader smoke test."
fi
