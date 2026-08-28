#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robust_aigc.data.registry import build_records_from_directory, load_manifest
from robust_aigc.utils.paths import resolve_paths


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--preview", type=Path)
    args = parser.parse_args()
    paths = resolve_paths(create=True)
    records = load_manifest(args.manifest) if args.manifest else build_records_from_directory((args.data_root or paths.data_root) / args.dataset, args.dataset)
    sizes, formats, valid = [], Counter(), []
    for record in records:
        try:
            with Image.open(record["path"]) as image:
                sizes.append(image.size); formats[image.format or "unknown"] += 1; valid.append(record)
        except Exception as error:
            print(f"[WARNING] Unreadable: {record['path']} ({error})")
    widths, heights = np.array([size[0] for size in sizes]), np.array([size[1] for size in sizes])
    labels, splits, generators = Counter(record["label"] for record in valid), Counter(record["split"] for record in valid), Counter(record.get("generator") or "unknown" for record in valid)
    print(f"images: {len(valid)}\nreal: {labels[0]}\nAI: {labels[1]}\nclass balance: {dict(labels)}\nformats: {dict(formats)}")
    print(f"image size min/median/max: ({widths.min()}, {heights.min()}) / ({int(np.median(widths))}, {int(np.median(heights))}) / ({widths.max()}, {heights.max()})")
    print(f"generator counts: {dict(generators)}\nsplit counts: {dict(splits)}")
    preview = args.preview or paths.output_root / "dataset_preview.png"; preview.parent.mkdir(parents=True, exist_ok=True)
    sample = valid[:16]
    figure, axes = plt.subplots(4, 4, figsize=(12, 12))
    for axis, record in zip(axes.flat, sample):
        with Image.open(record["path"]) as image: axis.imshow(image.convert("RGB"))
        axis.set_title(f"{'AI' if record['label'] else 'real'} | {record.get('generator') or 'unknown'}", fontsize=8); axis.axis("off")
    for axis in axes.flat[len(sample):]: axis.axis("off")
    figure.tight_layout(); figure.savefig(preview, dpi=150); print(f"preview: {preview}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
