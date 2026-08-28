#!/usr/bin/env python3
"""Decode a balanced image-only subset from SID Parquet shards for training."""
from __future__ import annotations

import argparse
import io
import os
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robust_aigc.data.registry import build_records_from_directory, sid_binary_label, write_manifest
from robust_aigc.data.splits import assign_deterministic_splits, persist_split_manifests, validate_split_isolation


def image_payload(value) -> bytes:
    """Extract bytes from Hugging Face's Parquet Image feature."""
    payload = value.get("bytes") if isinstance(value, dict) else value
    if isinstance(payload, memoryview):
        payload = payload.tobytes()
    if not isinstance(payload, bytes):
        raise ValueError("SID image entry did not contain embedded image bytes")
    return payload


def safe_stem(value: object, fallback: str) -> str:
    candidate = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or fallback)).strip("._")
    return candidate or fallback


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(os.environ.get("AIGC_DATA_ROOT", "data")) / "sid")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("AIGC_DATA_ROOT", "data")) / "sid_smoke_images")
    parser.add_argument("--manifest-dir", type=Path, default=Path(os.environ.get("AIGC_DATA_ROOT", "data")) / "manifests")
    parser.add_argument("--manifest-name", default="sid_smoke")
    parser.add_argument("--max-per-class", type=int, default=1000, help="0 decodes every row; default is a balanced 2,000-image smoke set.")
    args = parser.parse_args()
    if args.max_per_class < 0:
        raise ValueError("--max-per-class must be zero or positive")
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required; install it in the container job before preparing SID.") from error
    parquet_files = sorted((args.source_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No SID Parquet files found in {args.source_dir / 'data'}")
    for class_name in ("real", "aigc"):
        (args.output_dir / class_name).mkdir(parents=True, exist_ok=True)
    written = {0: 0, 1: 0}
    for parquet_path in parquet_files:
        reader = pq.ParquetFile(parquet_path)
        for group_index in range(reader.num_row_groups):
            table = reader.read_row_group(group_index, columns=["img_id", "image", "label"])
            for row_index, row in enumerate(table.to_pylist()):
                binary = sid_binary_label(int(row["label"]))
                if args.max_per_class and written[binary] >= args.max_per_class:
                    continue
                class_name = "aigc" if binary else "real"
                stem = safe_stem(row.get("img_id"), f"{parquet_path.stem}_{group_index}_{row_index}")
                destination = args.output_dir / class_name / f"{stem}.jpg"
                if not destination.exists():
                    with Image.open(io.BytesIO(image_payload(row["image"]))) as image:
                        image.convert("RGB").save(destination, format="JPEG", quality=95)
                written[binary] += 1
            if args.max_per_class and all(written[key] >= args.max_per_class for key in written):
                break
        print(f"[INFO] {parquet_path.name}: real={written[0]} aigc={written[1]}", flush=True)
        if args.max_per_class and all(written[key] >= args.max_per_class for key in written):
            break
    if not written[0] or not written[1]:
        raise RuntimeError(f"Could not prepare both classes: {written}")
    records = assign_deterministic_splits(build_records_from_directory(args.output_dir, "sid"))
    validate_split_isolation(records)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    persist_split_manifests(records, args.manifest_dir, args.manifest_name)
    manifest = write_manifest(records, args.manifest_dir / f"{args.manifest_name}_all.csv")
    print(f"[INFO] Prepared {sum(written.values())} images; manifest: {manifest}")


if __name__ == "__main__":
    main()
