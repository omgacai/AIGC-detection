#!/usr/bin/env python3
"""Materialize all false-positive and false-negative examples from an evaluation.

The evaluator writes one prediction score per image but deliberately retains
only a small number of examples in ``errors_*.json``.  This utility joins the
full prediction file with the immutable manifest and creates a lightweight,
reviewable directory of every error at a fixed decision threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", required=True, help="Manifest split evaluated, e.g. organizer_demo.")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Use symlinks by default to avoid duplicating the source images.",
    )
    return parser.parse_args()


def load_labels(manifest: Path, split: str) -> dict[str, dict[str, str]]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"path", "label", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Manifest must include: {', '.join(sorted(required))}")
    return {row["path"]: row for row in rows if row["split"] == split}


def materialize(source: Path, destination: Path, mode: str) -> None:
    if destination.exists() or destination.is_symlink():
        return
    if mode == "symlink":
        os.symlink(source, destination)
    else:
        shutil.copy2(source, destination)


def main() -> None:
    args = parse_args()
    predictions = json.loads(args.predictions.read_text(encoding="utf-8"))
    labels_by_path = load_labels(args.manifest, args.split)
    output = args.output_dir
    fp_dir, fn_dir = output / "false_positives", output / "false_negatives"
    fp_dir.mkdir(parents=True, exist_ok=True)
    fn_dir.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, str | float | int]] = []
    unseen_paths = 0
    for row in predictions:
        path = row["image_path"]
        record = labels_by_path.get(path)
        if record is None:
            unseen_paths += 1
            continue
        label, score = int(record["label"]), float(row["pred"])
        error_type = "false_positive" if label == 0 and score >= args.threshold else (
            "false_negative" if label == 1 and score < args.threshold else None
        )
        if error_type is None:
            continue
        source = Path(path)
        target_dir = fp_dir if error_type == "false_positive" else fn_dir
        destination = target_dir / f"{len(selected):06d}_{source.name}"
        if not source.is_file():
            raise FileNotFoundError(f"Prediction points to missing image: {source}")
        materialize(source, destination, args.mode)
        selected.append({
            "error_type": error_type,
            "label": label,
            "pred": score,
            "source_dataset": record.get("source_dataset", ""),
            "generator": record.get("generator", ""),
            "original_path": str(source),
            "review_path": str(destination),
        })

    with (output / "error_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["error_type", "label", "pred", "source_dataset", "generator", "original_path", "review_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    counts = {kind: sum(row["error_type"] == kind for row in selected) for kind in ("false_positive", "false_negative")}
    print(f"[INFO] false_positives={counts['false_positive']} false_negatives={counts['false_negative']}")
    print(f"[INFO] wrote review directory: {output}")
    if unseen_paths:
        print(f"[WARNING] {unseen_paths} prediction path(s) were not in split={args.split!r} and were skipped")


if __name__ == "__main__":
    main()
