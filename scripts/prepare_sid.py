#!/usr/bin/env python3
"""Decode a deterministic, balanced training subset from all SID Parquet shards.

SID_Set is distributed as Parquet rather than labelled image directories. This
samples every downloaded shard before decoding images, so an early-shard slice
cannot accidentally become the training data. SID's raw three-way label is
used only here and is never emitted to the final manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import heapq
import io
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robust_aigc.data.registry import sid_binary_label, write_manifest
from robust_aigc.data.splits import assign_deterministic_splits, persist_split_manifests, validate_split_isolation


@dataclass(frozen=True, order=True)
class Candidate:
    """A row locator ranked reproducibly without retaining image payloads."""

    rank: int
    parquet_path: str
    group_index: int
    row_index: int
    raw_label: int
    image_id: str


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


def rank_for(seed: int, parquet_path: Path, group_index: int, row_index: int, image_id: object) -> int:
    """Stable pseudo-random rank independent of download/scan order."""
    key = f"{seed}|{parquet_path.name}|{group_index}|{row_index}|{image_id}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def raw_quotas(raw_counts: Counter[int], max_per_class: int) -> dict[int, int]:
    """Balance real vs AI and preserve SID's two AI-source categories."""
    if max_per_class <= 0:
        raise ValueError("--max-per-class must be positive; choose an explicit safe subset size.")
    if raw_counts[0] < max_per_class:
        raise ValueError(f"SID has only {raw_counts[0]} real rows, fewer than requested {max_per_class}.")
    first = min(raw_counts[1], (max_per_class + 1) // 2)
    second = min(raw_counts[2], max_per_class - first)
    first = min(raw_counts[1], first + max(0, max_per_class - first - second))
    second = min(raw_counts[2], max_per_class - first)
    if first + second < max_per_class:
        raise ValueError("SID does not contain enough AI-labelled rows for the requested balanced subset.")
    return {0: max_per_class, 1: first, 2: second}


def count_raw_labels(parquet_files: list[Path], pq: object) -> Counter[int]:
    counts: Counter[int] = Counter()
    for parquet_path in parquet_files:
        reader = pq.ParquetFile(parquet_path)
        for group_index in range(reader.num_row_groups):
            table = reader.read_row_group(group_index, columns=["label"])
            counts.update(int(value) for value in table.column("label").to_pylist())
        print(f"[INFO] counted {parquet_path.name}: raw={dict(sorted(counts.items()))}", flush=True)
    return counts


def select_candidates(parquet_files: list[Path], pq: object, quotas: dict[int, int], seed: int) -> list[Candidate]:
    """Keep only the lowest deterministic ranks for each raw SID category."""
    # Negative ranks make the heap root the worst retained candidate, keeping
    # memory bounded even while reading the complete SID collection.
    heaps: dict[int, list[tuple[int, Candidate]]] = {label: [] for label in quotas}
    for parquet_path in parquet_files:
        reader = pq.ParquetFile(parquet_path)
        for group_index in range(reader.num_row_groups):
            table = reader.read_row_group(group_index, columns=["img_id", "label"])
            for row_index, row in enumerate(table.to_pylist()):
                raw_label = int(row["label"])
                if raw_label not in quotas:
                    raise ValueError(f"Unexpected SID label {raw_label}; expected 0, 1, or 2")
                candidate = Candidate(
                    rank_for(seed, parquet_path, group_index, row_index, row.get("img_id")),
                    str(parquet_path), group_index, row_index, raw_label, str(row.get("img_id") or ""),
                )
                heap = heaps[raw_label]
                entry = (-candidate.rank, candidate)
                if len(heap) < quotas[raw_label]:
                    heapq.heappush(heap, entry)
                elif entry > heap[0]:
                    heapq.heapreplace(heap, entry)
        print(f"[INFO] sampled {parquet_path.name}", flush=True)
    selected = [candidate for heap in heaps.values() for _, candidate in heap]
    selected.sort()
    selected_counts = Counter(candidate.raw_label for candidate in selected)
    if any(selected_counts[label] != quota for label, quota in quotas.items()):
        raise RuntimeError(f"Selection did not meet quotas: selected={dict(selected_counts)} requested={quotas}")
    return selected


def prepare_destination(destination: Path, resume: bool = False) -> None:
    existing = [path for path in destination.rglob("*") if path.is_file()]
    if existing and not resume:
        raise FileExistsError(
            f"Refusing to mix a new SID sample with {len(existing)} existing file(s) in {destination}. "
            "Choose a new --output-dir (recommended), archive the old dataset, or use --resume for a full decode."
        )
    for class_name in ("real", "aigc"):
        (destination / class_name).mkdir(parents=True, exist_ok=True)


def destination_for(destination_root: Path, parquet_name: str, group_index: int, row_index: int, image_id: object, raw_label: int) -> tuple[Path, int]:
    binary = sid_binary_label(raw_label)
    class_name = "aigc" if binary else "real"
    stem = safe_stem(image_id, f"row_{row_index}")
    filename = f"sid_{Path(parquet_name).stem}_{group_index}_{row_index}_{stem}.jpg"
    return destination_root / class_name / filename, binary


def decode_candidates(selected: list[Candidate], pq: object, destination_root: Path) -> list[dict]:
    requested: dict[tuple[str, int], dict[int, Candidate]] = defaultdict(dict)
    for candidate in selected:
        requested[(candidate.parquet_path, candidate.group_index)][candidate.row_index] = candidate
    records: list[dict] = []
    for (parquet_name, group_index), rows in sorted(requested.items()):
        table = pq.ParquetFile(parquet_name).read_row_group(group_index, columns=["image"])
        images = table.column("image").to_pylist()
        for row_index, candidate in sorted(rows.items()):
            destination, binary = destination_for(destination_root, parquet_name, group_index, row_index, candidate.image_id, candidate.raw_label)
            with Image.open(io.BytesIO(image_payload(images[row_index]))) as image:
                image.convert("RGB").save(destination, format="JPEG", quality=95)
            records.append({"path": str(destination), "label": binary, "source_dataset": "sid", "generator": None, "split": "train"})
        print(f"[INFO] decoded {Path(parquet_name).name} row-group {group_index}", flush=True)
    return records


def decode_all(parquet_files: list[Path], pq: object, destination_root: Path) -> list[dict]:
    """Decode every SID row; existing deterministic files are safely skipped on resume."""
    records: list[dict] = []
    for parquet_path in parquet_files:
        reader = pq.ParquetFile(parquet_path)
        for group_index in range(reader.num_row_groups):
            table = reader.read_row_group(group_index, columns=["img_id", "image", "label"])
            for row_index, row in enumerate(table.to_pylist()):
                destination, binary = destination_for(destination_root, str(parquet_path), group_index, row_index, row.get("img_id"), int(row["label"]))
                if not destination.exists():
                    with Image.open(io.BytesIO(image_payload(row["image"]))) as image:
                        image.convert("RGB").save(destination, format="JPEG", quality=95)
                records.append({"path": str(destination), "label": binary, "source_dataset": "sid", "generator": None, "split": "train"})
            print(f"[INFO] decoded {parquet_path.name} row-group {group_index}", flush=True)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path(os.environ.get("AIGC_DATA_ROOT", "data")) / "sid")
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("AIGC_DATA_ROOT", "data")) / "sid_balanced_images")
    parser.add_argument("--manifest-dir", type=Path, default=Path(os.environ.get("AIGC_DATA_ROOT", "data")) / "manifests")
    parser.add_argument("--manifest-name", default="sid_balanced")
    parser.add_argument("--max-per-class", type=int, default=10000, help="Number of real and AI images each; AI is balanced across SID's two AI sources.")
    parser.add_argument("--decode-all", action="store_true", help="Decode every SID row into a separate directory; use --resume after a time-limited job ends.")
    parser.add_argument("--resume", action="store_true", help="Skip existing deterministic files during --decode-all.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    try:
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError("pyarrow is required; install it in the container job before preparing SID.") from error
    parquet_files = sorted((args.source_dir / "data").glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No SID Parquet files found in {args.source_dir / 'data'}")
    if args.resume and not args.decode_all:
        raise ValueError("--resume is only supported with --decode-all")
    if args.decode_all:
        print(f"[INFO] decoding every row across {len(parquet_files)} SID shards", flush=True)
        prepare_destination(args.output_dir, resume=args.resume)
        records = decode_all(parquet_files, pq, args.output_dir)
    else:
        raw_counts = count_raw_labels(parquet_files, pq)
        quotas = raw_quotas(raw_counts, args.max_per_class)
        print(f"[INFO] selecting across {len(parquet_files)} shards with raw quotas={quotas}", flush=True)
        selected = select_candidates(parquet_files, pq, quotas, args.seed)
        prepare_destination(args.output_dir)
        records = decode_candidates(selected, pq, args.output_dir)
    records = assign_deterministic_splits(records, seed=args.seed)
    validate_split_isolation(records)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)
    persist_split_manifests(records, args.manifest_dir, args.manifest_name)
    manifest = write_manifest(records, args.manifest_dir / f"{args.manifest_name}_all.csv")
    counts = Counter((record["split"], int(record["label"])) for record in records)
    print(f"[INFO] Prepared {len(records)} images; manifest: {manifest}")
    for key in sorted(counts):
        print(f"[INFO] split={key[0]} label={key[1]} count={counts[key]}")


if __name__ == "__main__":
    main()
