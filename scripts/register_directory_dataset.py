#!/usr/bin/env python3
"""Register an extracted real/AI directory as deterministic manifests."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from robust_aigc.data.registry import build_records_from_directory, write_manifest
from robust_aigc.data.splits import assign_deterministic_splits, persist_split_manifests, validate_split_isolation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--manifest-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    records = assign_deterministic_splits(
        build_records_from_directory(args.data_dir, args.dataset), seed=args.seed
    )
    validate_split_isolation(records)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    write_manifest(records, args.manifest_dir / f"{args.dataset}_all.csv")
    persist_split_manifests(records, args.manifest_dir, args.dataset)
    counts = Counter((record["split"], int(record["label"])) for record in records)
    print(f"[INFO] Registered {len(records)} images from {args.data_dir}")
    for key in sorted(counts):
        print(f"[INFO] split={key[0]} label={key[1]} count={counts[key]}")


if __name__ == "__main__":
    main()
