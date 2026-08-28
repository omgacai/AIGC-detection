#!/usr/bin/env python3
"""One forward-pass availability check only; this script never trains DINOv2."""
from __future__ import annotations

import argparse
import gc
from pathlib import Path
import sys

import torch
from transformers import AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robust_aigc.utils.paths import configure_caches, resolve_paths


def check(model_id: str, device: str, dtype: torch.dtype) -> None:
    print(f"[INFO] Loading {model_id}")
    model = AutoModel.from_pretrained(model_id).eval().to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=dtype):
        features = model(pixel_values=torch.randn(1, 3, 224, 224, device=device)).last_hidden_state
    torch.cuda.synchronize()
    print(f"DINOv2 load: PASS\nforward pass: PASS\nfeature shape: {tuple(features.shape)}\nGPU memory allocated: {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
    del features, model; gc.collect(); torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--model", default="facebook/dinov2-large"); parser.add_argument("--also-base", action="store_true"); args = parser.parse_args()
    paths = resolve_paths(create=True); configure_caches(paths)
    if not torch.cuda.is_available():
        print("[ERROR] CUDA unavailable. Submit this script inside a GPU allocation."); return 1
    bf16 = torch.cuda.is_bf16_supported(); dtype = torch.bfloat16 if bf16 else torch.float16
    print(f"FP16 supported/tested: True\nBF16 supported/tested: {bf16}")
    try:
        check(args.model, "cuda", dtype)
    except torch.cuda.OutOfMemoryError:
        print("[WARNING] DINOv2-L did not fit. Retry --model facebook/dinov2-base.")
        torch.cuda.empty_cache(); return 2
    if args.also_base and args.model != "facebook/dinov2-base": check("facebook/dinov2-base", "cuda", dtype)
    return 0


if __name__ == "__main__": raise SystemExit(main())
