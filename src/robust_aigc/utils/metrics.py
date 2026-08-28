"""Small, dependency-light per-epoch metrics writer."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping


class EpochMetricsWriter:
    """Append one machine-readable metrics record per epoch in JSONL and CSV."""

    def __init__(self, output_root: str | Path, run_name: str):
        self.run_dir = Path(output_root) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"

    def write(self, epoch: int, metrics: Mapping[str, float | int]) -> dict:
        record = {"epoch": epoch, **{key: float(value) for key, value in metrics.items()}}
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        new_file = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(record))
            if new_file:
                writer.writeheader()
            writer.writerow(record)
        return record
