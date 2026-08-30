#!/usr/bin/env python3
"""Merge prepared datasets without changing their established train/val/test splits."""
from __future__ import annotations

import argparse
import random
from collections import Counter
from pathlib import Path

from robust_aigc.data.registry import load_manifest, write_manifest
from robust_aigc.data.splits import persist_split_manifests, validate_split_isolation


# These paths belong to the organiser's reference benchmark and are evaluation
# only.  Keep the check deliberately specific: other generated images may have
# a directory named "coco" and must not be rejected on that name alone.
RESERVED_COMPETITION_PATH_FRAGMENTS = (
    "/competition_reference/",
    "/images/real/coco/",
    "/images/diffusion_based/dalle/",
)


def merge_manifests(paths: list[Path]) -> list[dict]:
    records = [record for path in paths for record in load_manifest(path)]
    seen: set[str] = set()
    for record in records:
        path = str(Path(record["path"]).expanduser().resolve())
        normalized_path = path.lower()
        source_dataset = str(record.get("source_dataset", "")).lower()
        if source_dataset.startswith("competition_") or any(
            fragment in normalized_path for fragment in RESERVED_COMPETITION_PATH_FRAGMENTS
        ):
            raise ValueError(
                "Competition reference data detected in a training-manifest input; "
                "COCO/DALL·E reference material must remain evaluation-only."
            )
        if path in seen:
            raise ValueError(f"Duplicate image path across input manifests: {path}")
        seen.add(path)
        record["path"] = path
        if record["split"] == "organizer_demo":
            raise ValueError("organizer_demo must remain a separate evaluation manifest")
    validate_split_isolation(records)
    return records


def cap_split_groups(records: list[dict], limits: dict[str, int], seed: int = 42) -> list[dict]:
    grouped: dict[tuple[str, str, int], list[dict]] = {}
    for record in records:
        key = (record["source_dataset"], record["split"], int(record["label"]))
        grouped.setdefault(key, []).append(record)
    selected: list[dict] = []
    for key in sorted(grouped):
        group = sorted(grouped[key], key=lambda record: record["path"])
        random.Random(f"{seed}:{key}").shuffle(group)
        limit = limits.get(key[1], 0)
        selected.extend(group if limit <= 0 else group[:limit])
    validate_split_isolation(selected)
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", required=True, type=Path, help="Repeat once per dataset manifest.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--name", default="aigc_mixed")
    parser.add_argument("--max-train-per-dataset-class", type=int, default=8000)
    parser.add_argument("--max-val-per-dataset-class", type=int, default=1000)
    parser.add_argument("--max-test-per-dataset-class", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = merge_manifests(args.manifest)
    records = cap_split_groups(records, {
        "train": args.max_train_per_dataset_class,
        "internal_val": args.max_val_per_dataset_class,
        "test": args.max_test_per_dataset_class,
    }, args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = write_manifest(records, args.output_dir / f"{args.name}_all.csv")
    persist_split_manifests(records, args.output_dir, args.name)
    counts = Counter((record["source_dataset"], record["split"], int(record["label"])) for record in records)
    print(f"[INFO] Wrote {len(records)} records to {output}")
    for key in sorted(counts):
        print(f"[INFO] dataset={key[0]} split={key[1]} label={key[2]} count={counts[key]}")


if __name__ == "__main__":
    main()
