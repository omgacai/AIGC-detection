from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .registry import write_manifest

SEED = 42


def assign_deterministic_splits(records: list[dict], seed: int = SEED) -> list[dict]:
    """Assign 80/10/10 splits stratified by label and generator metadata."""
    groups: dict[tuple[int, str | None], list[dict]] = defaultdict(list)
    for record in records:
        groups[(int(record["label"]), record.get("generator"))].append(dict(record))
    output: list[dict] = []
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
        train_end, val_end = round(len(group) * 0.8), round(len(group) * 0.9)
        for index, record in enumerate(group):
            record["split"] = "train" if index < train_end else "internal_val" if index < val_end else "test"
            output.append(record)
    return output


def persist_split_manifests(records: list[dict], manifest_dir: str | Path, prefix: str) -> dict[str, Path]:
    directory = Path(manifest_dir)
    result = {}
    for split in ("train", "internal_val", "test"):
        result[split] = write_manifest([record for record in records if record["split"] == split], directory / f"{prefix}_{split}.csv")
    return result


def validate_split_isolation(records: list[dict]) -> None:
    by_split: dict[str, set[str]] = defaultdict(set)
    for record in records:
        path = record["path"]
        split = record["split"]
        if path in by_split[split]:
            raise ValueError(f"Duplicate path within {split}: {path}")
        by_split[split].add(path)
    for left, right in (("train", "internal_val"), ("train", "test"), ("internal_val", "test")):
        overlap = by_split[left] & by_split[right]
        if overlap:
            raise ValueError(f"Leakage: {left} and {right} share {len(overlap)} path(s), e.g. {next(iter(overlap))}")
    if by_split["train"] & by_split["organizer_demo"]:
        raise ValueError("Leakage: organizer_demo paths must never appear in training.")
