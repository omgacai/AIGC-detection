#!/usr/bin/env python3
"""Download one official reference archive outside the WildFake training root."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from competition_reference_archives import ARCHIVES


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--temporary-root", required=True, type=Path)
    args = parser.parse_args()
    if args.archive not in ARCHIVES:
        raise ValueError("Only the official COCO/DALL-E reference archives are allowed here.")
    archive_path = Path(args.archive)
    target = args.raw_root / archive_path.with_suffix("")
    marker = target / ".complete.json"
    if marker.exists():
        print(f"[INFO] Already complete: {target}")
        return
    args.temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="competition-reference-", dir=args.temporary_root) as temporary:
        temporary_root = Path(temporary)
        download_root = temporary_root / "download"
        subprocess.run(["modelscope", "download", "--dataset", "hy2628982280/WildFake", "--include", args.archive, "--local_dir", str(download_root)], check=True)
        candidates = list(download_root.rglob(archive_path.name))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one {archive_path.name}, found {len(candidates)}")
        extraction_root = temporary_root / "extracted"
        extraction_root.mkdir()
        with zipfile.ZipFile(candidates[0]) as handle:
            safe_extract(handle, extraction_root)
        expected = extraction_root / archive_path.with_suffix("")
        source = expected if expected.exists() else extraction_root
        staged_target = temporary_root / "target"
        shutil.move(str(source), staged_target)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise FileExistsError(f"Incomplete target exists: {target}; inspect it before retrying")
        shutil.move(str(staged_target), target)
    marker.write_text(json.dumps({"archive": args.archive, "status": "complete", "purpose": "evaluation_only"}, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Complete evaluation-only archive: {target}")


if __name__ == "__main__":
    main()
