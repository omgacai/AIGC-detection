#!/usr/bin/env python3
"""Download one dataset to configured cluster storage; never to a hard-coded home cache."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robust_aigc.data.registry import DATASETS, build_records_from_directory, write_manifest
from robust_aigc.data.splits import persist_split_manifests, preserve_directory_splits, validate_split_isolation
from robust_aigc.utils.paths import configure_caches, resolve_paths


def available_space(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    print(f"[INFO] Target: {path}; free: {usage.free / 2**30:.1f} GiB")
    subprocess.run(["df", "-h", str(path)], check=False)


def download_sid(destination: Path, cache_root: Path) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=DATASETS["sid"].identifier, repo_type="dataset", local_dir=destination,
                      cache_dir=cache_root / "huggingface", resume_download=True)


def download_cifake(destination: Path) -> None:
    if not (Path.home() / ".kaggle" / "kaggle.json").exists() and not os.environ.get("KAGGLE_USERNAME"):
        raise RuntimeError("Kaggle credentials are missing. Create ~/.kaggle/kaggle.json (chmod 600) or set KAGGLE_USERNAME and KAGGLE_KEY; do not commit either.")
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    api.dataset_download_files(DATASETS["cifake"].identifier, path=destination, unzip=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=DATASETS, required=True)
    parser.add_argument("--output-dir", type=Path, help="Cluster storage root; defaults to AIGC_DATA_ROOT/config.")
    parser.add_argument("--no-manifest", action="store_true")
    args = parser.parse_args()
    paths = resolve_paths(create=True); configure_caches(paths)
    root = (args.output_dir or paths.data_root).expanduser().resolve()
    destination = root / args.dataset
    available_space(root)
    if args.dataset == "sid":
        download_sid(destination, paths.cache_root)
    elif args.dataset == "cifake":
        download_cifake(destination)
    else:
        print("[WARNING] WildFake is not downloaded automatically: ModelScope access/layout must be verified first. See README, then register its extracted directory with inspect_dataset.py.")
        return 2
    print(f"[INFO] Download complete: {destination}")
    if args.dataset == "sid":
        print("[INFO] SID_Set is stored as Parquet shards. Run scripts/prepare_sid.py to decode a chosen image subset and create manifests.")
        return 0
    if not args.no_manifest:
        records = preserve_directory_splits(build_records_from_directory(destination, args.dataset), destination)
        validate_split_isolation(records)
        manifest_dir = root / "manifests"
        persist_split_manifests(records, manifest_dir, args.dataset)
        write_manifest(records, manifest_dir / f"{args.dataset}_all.csv")
        print(f"[INFO] Wrote manifests to {manifest_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
