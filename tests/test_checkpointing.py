import torch
import pytest

from robust_aigc.utils.checkpointing import CheckpointManager
from robust_aigc.utils.metrics import EpochMetricsWriter, TensorBoardMetricsWriter, binary_operating_point_metrics


def test_checkpoint_manager_keeps_best_and_last(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "checkpoints", "run-1")
    manager.save_epoch(epoch=1, model=model, optimizer=optimizer, scheduler=None,
                       metrics={"internal_val_accuracy": 0.7, "tpr_at_1_fpr": 0.5, "fpr_at_99_tpr": 0.4})
    manager.save_epoch(epoch=2, model=model, optimizer=optimizer, scheduler=None,
                       metrics={"internal_val_accuracy": 0.6, "tpr_at_1_fpr": 0.6, "fpr_at_99_tpr": 0.5})
    assert (tmp_path / "checkpoints" / "run-1" / "last.pt").exists()
    best = torch.load(tmp_path / "checkpoints" / "run-1" / "best.pt", weights_only=False)
    assert best["epoch"] == 1
    assert torch.load(tmp_path / "checkpoints" / "run-1" / "best_tpr_at_1_fpr.pt", weights_only=False)["epoch"] == 2
    assert torch.load(tmp_path / "checkpoints" / "run-1" / "best_fpr_at_99_tpr.pt", weights_only=False)["epoch"] == 1


def test_metrics_writer_appends_epoch_records(tmp_path):
    writer = EpochMetricsWriter(tmp_path / "outputs", "run-1")
    writer.write(1, {"train_loss": 0.3, "internal_val_accuracy": 0.8})
    writer.write(2, {"train_loss": 0.2, "internal_val_accuracy": 0.9})
    assert len(writer.jsonl_path.read_text().splitlines()) == 2
    assert len(writer.csv_path.read_text().splitlines()) == 3


def test_operating_point_metrics_are_correctly_oriented():
    metrics = binary_operating_point_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["tpr_at_1_fpr"] == 1.0
    assert metrics["fpr_at_99_tpr"] == 0.0


def test_tensorboard_writer_creates_event_file(tmp_path):
    pytest.importorskip("tensorboard")
    writer = TensorBoardMetricsWriter(tmp_path / "outputs", "run-1")
    writer.write(1, {"train_loss": 0.3})
    writer.close()
    assert list(writer.log_dir.glob("events.out.tfevents.*"))
