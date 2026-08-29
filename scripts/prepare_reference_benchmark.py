#!/usr/bin/env python3
"""Create an immutable evaluation-only COCO-vs-DALL·E reference manifest.

This script deliberately does not create train/internal-validation/test rows.
The benchmark may be inspected and evaluated, but it cannot enter the normal
training manifest merger.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from robust_aigc.data.registry import IMAGE_EXTENSIONS


def image_paths(root: Path) -> list[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    return sorted(path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def records(paths: list[Path], label: int, source: str, generator: str | None) -> list[dict]:
    return [
        {
            "path": str(path),
            "label": label,
            "source_dataset": source,
            "generator": generator or "",
            "split": "organizer_demo",
        }
        for path in paths
    ]


def write_manifest(destination: Path, values: list[dict]) -> None:
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite benchmark manifest: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = ("path", "label", "source_dataset", "generator", "split")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument("--dalle-root", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--expected-coco", type=int, default=4998)
    parser.add_argument("--expected-dalle", type=int, default=8843)
    args = parser.parse_args()
    coco, dalle = image_paths(args.coco_root), image_paths(args.dalle_root)
    if len(coco) != args.expected_coco or len(dalle) != args.expected_dalle:
        raise ValueError(
            "Reference benchmark count mismatch: "
            f"COCO={len(coco)} (expected {args.expected_coco}), "
            f"DALL-E={len(dalle)} (expected {args.expected_dalle}). "
            "Check the roots; do not evaluate a partial or mixed benchmark."
        )
    coco_records = records(coco, 0, "competition_coco_val2017", None)
    dalle_records = records(dalle, 1, "competition_dalle_advanced", "dalle_advanced")
    destination = args.manifest_dir
    write_manifest(destination / "competition_coco_val2017_reference.csv", coco_records)
    write_manifest(destination / "competition_dalle_advanced_reference.csv", dalle_records)
    write_manifest(destination / "competition_coco_dalle_reference.csv", coco_records + dalle_records)
    print(f"[INFO] Wrote evaluation-only benchmark manifests to {destination}")
    print(f"[INFO] COCO real={len(coco)}; DALL-E Advanced AIGC={len(dalle)}; split=organizer_demo")


if __name__ == "__main__":
    main()
