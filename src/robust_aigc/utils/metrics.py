"""Small, dependency-light per-epoch metrics writer."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve


def binary_operating_point_metrics(labels, scores) -> dict[str, float]:
    """Compute binary validation metrics for positive (AI-generated) scores.

    Scores must increase with confidence that the image is AI-generated (label 1).
    `tpr_at_1_fpr` is higher-is-better; `fpr_at_99_tpr` is lower-is-better.
    Values are fractions rather than percent strings so CSV/JSONL stay numerical.
    """
    labels_array = np.asarray(labels, dtype=np.int64)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.ndim != 1 or scores_array.ndim != 1 or len(labels_array) != len(scores_array):
        raise ValueError("labels and scores must be one-dimensional arrays with equal length")
    if len(labels_array) == 0 or set(np.unique(labels_array)) != {0, 1}:
        raise ValueError("operating-point metrics require at least one real (0) and one AI (1) example")
    fpr, tpr, _ = roc_curve(labels_array, scores_array, pos_label=1)
    tpr_at_1_fpr = float(np.max(tpr[fpr <= 0.01]))
    fpr_at_99_tpr = float(np.min(fpr[tpr >= 0.99]))
    return {
        "roc_auc": float(roc_auc_score(labels_array, scores_array)),
        "tpr_at_1_fpr": tpr_at_1_fpr,
        "fpr_at_99_tpr": fpr_at_99_tpr,
    }


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


class TensorBoardMetricsWriter:
    """Write per-epoch scalar metrics to TensorBoard event files."""

    def __init__(self, output_root: str | Path, run_name: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise RuntimeError("TensorBoard is not installed. Run the project setup script.") from error
        self.log_dir = Path(output_root) / run_name / "tensorboard"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(self.log_dir))

    def write(self, epoch: int, metrics: Mapping[str, float | int]) -> None:
        for name, value in metrics.items():
            self.writer.add_scalar(f"epoch/{name}", float(value), global_step=epoch)
        self.writer.flush()

    def close(self) -> None:
        self.writer.close()
