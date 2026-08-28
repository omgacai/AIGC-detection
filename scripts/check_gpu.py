#!/usr/bin/env python3
from __future__ import annotations

import platform
import sys


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({platform.platform()})")
    try:
        import torch
    except ImportError:
        print("[ERROR] PyTorch is not installed. Run bash scripts/setup_env.sh or activate the cluster-provided environment.")
        return 1
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("[WARNING] CUDA is unavailable. On a cluster, run this inside an allocated GPU job—not on a login node.")
        return 0
    count = torch.cuda.device_count()
    print(f"GPU count: {count}")
    for index in range(count):
        properties = torch.cuda.get_device_properties(index)
        print(f"GPU {index}: {properties.name}; memory: {properties.total_memory / 2**30:.2f} GiB")
    device = torch.cuda.current_device()
    print(f"Current device: {device}")
    try:
        x = torch.randn(1024, 1024, device="cuda")
        y = x @ x
        torch.cuda.synchronize()
        print(f"Tensor operation: PASS ({tuple(y.shape)})")
    except Exception as error:
        print(f"[ERROR] GPU tensor operation failed: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
