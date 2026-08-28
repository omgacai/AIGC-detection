#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robust_aigc.data.registry import load_manifest
from robust_aigc.data.splits import validate_split_isolation


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); args = parser.parse_args()
    records = load_manifest(args.manifest); bad = []
    for record in records:
        if record["label"] not in (0, 1): bad.append((record["path"], "invalid label")); continue
        try:
            with Image.open(record["path"]) as image: image.convert("RGB")
        except Exception as error: bad.append((record["path"], str(error)))
    validate_split_isolation(records)
    print(f"records: {len(records)}; class counts: {dict(Counter(r['label'] for r in records))}; corrupted: {len(bad)}")
    for path, error in bad[:20]: print(f"[ERROR] {path}: {error}")
    return 1 if bad else 0


if __name__ == "__main__": raise SystemExit(main())
