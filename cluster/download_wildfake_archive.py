#!/usr/bin/env python3
"""Fetch one approved WildFake ZIP to node-local disk and persist extracted data."""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

from wildfake_archives import ARCHIVES


def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if root not in target.parents and target != root:
            raise ValueError(f"Unsafe ZIP member path: {member.filename}")
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--temporary-root", required=True, type=Path)
    args = parser.parse_args()
    archive_path = Path(args.archive)
    if args.archive not in ARCHIVES:
        raise ValueError(
            "Archive is not in the approved WildFake training allowlist; "
            "organiser benchmark archives must never be downloaded."
        )
    target = args.raw_root / archive_path.with_suffix("")
    marker = target / ".complete.json"
    if marker.exists():
        print(f"[INFO] Already complete: {target}")
        return
    args.temporary_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="wildfake-", dir=args.temporary_root) as temporary:
        temporary_root = Path(temporary)
        download_root = temporary_root / "download"
        subprocess.run([
            "modelscope", "download", "--dataset", "hy2628982280/WildFake",
            "--include", args.archive, "--local_dir", str(download_root),
        ], check=True)
        candidates = list(download_root.rglob(archive_path.name))
        if len(candidates) != 1:
            raise FileNotFoundError(f"Expected one {archive_path.name} below {download_root}, found {len(candidates)}")
        extraction_root = temporary_root / "extracted"
        extraction_root.mkdir()
        with zipfile.ZipFile(candidates[0]) as handle:
            safe_extract(handle, extraction_root)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Archives differ in whether they contain the final folder or the
        # image files directly.  Persist either layout under one archive
        # directory; the manifest scanner recursively discovers images.
        expected_inside_archive = extraction_root / archive_path.with_suffix("")
        staged_target = temporary_root / "target"
        source = expected_inside_archive if expected_inside_archive.exists() else extraction_root
        shutil.move(str(source), staged_target)
        if target.exists():
            raise FileExistsError(f"Incomplete target already exists: {target}; inspect it before retrying")
        shutil.move(str(staged_target), target)
    marker.write_text(json.dumps({"archive": args.archive, "status": "complete"}, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] Complete: {target}")


if __name__ == "__main__":
    main()
