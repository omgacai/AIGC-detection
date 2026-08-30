#!/usr/bin/env python3
"""Create a non-destructive manifest copy excluding unreadable image files."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from PIL import Image, UnidentifiedImageError


FIELDS = ("path", "label", "source_dataset", "generator", "split")


def is_readable(path: Path) -> tuple[bool, str | None]:
    try:
        # verify catches truncated/corrupt payloads without retaining decoded
        # pixels; reopen/load catches decoder issues verify alone can miss.
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.convert("RGB").load()
        return True, None
    except (OSError, UnidentifiedImageError, ValueError) as error:
        return False, f"{type(error).__name__}: {error}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--invalid-report", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.invalid_report.exists():
        raise FileExistsError("Refusing to overwrite an existing validated manifest or invalid-image report.")
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Manifest needs fields: {', '.join(FIELDS)}")
        source = list(reader)
    valid, invalid = [], []
    for index, row in enumerate(source, 1):
        readable, error = is_readable(Path(row["path"]))
        if readable:
            valid.append(row)
        else:
            invalid.append({**row, "error": error})
        if index % 1000 == 0 or index == len(source):
            print(f"[INFO] verified={index}/{len(source)} valid={len(valid)} invalid={len(invalid)}", flush=True)
    if not valid:
        raise RuntimeError("No readable images remained; refusing to write an empty manifest.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader(); writer.writerows(valid)
    with args.invalid_report.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=(*FIELDS, "error"))
        writer.writeheader(); writer.writerows(invalid)
    print(f"[INFO] wrote validated manifest: {args.output}")
    print(f"[INFO] wrote invalid-image report: {args.invalid_report}")
    if invalid:
        print(f"[WARNING] excluded unreadable images: {len(invalid)}", flush=True)


if __name__ == "__main__":
    main()
