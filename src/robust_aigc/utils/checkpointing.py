"""Resumable, atomic checkpoint handling for future training scripts."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import torch


Mode = Literal["max", "min"]


@dataclass
class CheckpointManager:
    """Keep a resumable `last.pt` and a validation-selected `best.pt` per run."""

    root: Path
    run_name: str
    monitor: str = "internal_val_accuracy"
    mode: Mode = "max"
    best_value: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"max", "min"}:
            raise ValueError("mode must be 'max' or 'min'")
        self.run_dir.mkdir(parents=True, exist_ok=True)

    @property
    def run_dir(self) -> Path:
        return self.root / self.run_name

    def _is_better(self, value: float) -> bool:
        if self.best_value is None:
            return True
        return value > self.best_value if self.mode == "max" else value < self.best_value

    def _atomic_save(self, payload: dict[str, Any], destination: Path) -> None:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, destination)

    def save_epoch(
        self,
        *,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        metrics: dict[str, float],
        training_args: dict[str, Any] | None = None,
    ) -> dict[str, Path | bool]:
        """Save resumable state each epoch and best state only on improvement."""
        if self.monitor not in metrics:
            raise KeyError(f"Checkpoint monitor '{self.monitor}' missing from metrics: {sorted(metrics)}")
        monitor_value = float(metrics[self.monitor])
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "monitor": self.monitor,
            "monitor_value": monitor_value,
            "training_args": training_args or {},
        }
        last_path = self.run_dir / "last.pt"
        self._atomic_save(payload, last_path)
        is_best = self._is_better(monitor_value)
        if is_best:
            self.best_value = monitor_value
            payload["best_value"] = self.best_value
            self._atomic_save(payload, self.run_dir / "best.pt")
        return {"last": last_path, "best": self.run_dir / "best.pt", "is_best": is_best}

    def write_metadata(self) -> Path:
        path = self.run_dir / "checkpoint_config.json"
        import json
        path.write_text(json.dumps(asdict(self), default=str, indent=2) + "\n", encoding="utf-8")
        return path
