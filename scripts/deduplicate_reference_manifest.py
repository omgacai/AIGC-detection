#!/usr/bin/env python3
"""Create a separate exact-content-deduplicated evaluation manifest.

The source manifest is never changed.  Duplicate detection hashes decoded RGB
pixels (including dimensions), so metadata-only or file-encoding differences
do not hide an exact duplicate.  This is deliberately not score-based: model
outputs are neither evidence of duplicate content nor an acceptable filtering
criterion.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from PIL import Image

FIELDS = ("path", "label", "source_dataset", "generator", "split")


def decoded_rgb_sha256(path: str | Path) -> str:
    """Hash the decoded RGB pixels and dimensions, ignoring image metadata."""
    source = Path(path)
    try:
        with Image.open(source) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            payload = width.to_bytes(8, "big") + height.to_bytes(8, "big") + rgb.tobytes()
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not decode {source}: {error}") from error
    return hashlib.sha256(payload).hexdigest()


def deduplicate(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Keep the first representative of each exact decoded-content group."""
    representatives: dict[str, dict] = {}
    retained, removed = [], []
    for record in records:
        digest = decoded_rgb_sha256(record["path"])
        previous = representatives.get(digest)
        if previous is None:
            representatives[digest] = record
            retained.append(record)
            continue
        if int(previous["label"]) != int(record["label"]):
            raise ValueError(
                "Decoded-content duplicate has contradictory labels: "
                f"{previous['path']} (label={previous['label']}) versus "
                f"{record['path']} (label={record['label']})"
            )
        removed.append({
            "sha256_decoded_rgb": digest,
            "kept_path": previous["path"],
            "removed_path": record["path"],
            "label": int(record["label"]),
            "source_dataset": record["source_dataset"],
        })
    return retained, removed


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain: {', '.join(FIELDS)}")
        records = list(reader)
    if not records:
        raise ValueError("Manifest is empty")
    if any(record["split"] != "organizer_demo" for record in records):
        raise ValueError("This tool accepts evaluation-only organizer_demo manifests only")
    return records


def write_manifest(path: Path, records: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({field: record.get(field, "") for field in FIELDS} for record in records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Original, immutable organizer_demo manifest")
    parser.add_argument("--output", required=True, type=Path, help="New deduplicated manifest path")
    parser.add_argument("--report", required=True, type=Path, help="JSON audit report for removed duplicates")
    args = parser.parse_args()

    if args.manifest.resolve() == args.output.resolve():
        raise ValueError("Output must be a new path; the official manifest is immutable")
    records = read_manifest(args.manifest)
    retained, removed = deduplicate(records)
    write_manifest(args.output, retained)
    report = {
        "method": "sha256 of decoded RGB pixel bytes and dimensions",
        "source_manifest": str(args.manifest.resolve()),
        "source_count": len(records),
        "deduplicated_count": len(retained),
        "duplicates_removed": len(removed),
        "removed": removed,
    }
    if args.report.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {args.report}")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] source={len(records)} unique={len(retained)} removed={len(removed)}")
    print(f"[INFO] manifest={args.output}")
    print(f"[INFO] audit_report={args.report}")


if __name__ == "__main__":
    main()
