import torch

from robust_aigc.utils.checkpointing import CheckpointManager
from robust_aigc.utils.metrics import EpochMetricsWriter


def test_checkpoint_manager_keeps_best_and_last(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    manager = CheckpointManager(tmp_path / "checkpoints", "run-1")
    manager.save_epoch(epoch=1, model=model, optimizer=optimizer, scheduler=None,
                       metrics={"internal_val_accuracy": 0.7})
    manager.save_epoch(epoch=2, model=model, optimizer=optimizer, scheduler=None,
                       metrics={"internal_val_accuracy": 0.6})
    assert (tmp_path / "checkpoints" / "run-1" / "last.pt").exists()
    best = torch.load(tmp_path / "checkpoints" / "run-1" / "best.pt", weights_only=False)
    assert best["epoch"] == 1


def test_metrics_writer_appends_epoch_records(tmp_path):
    writer = EpochMetricsWriter(tmp_path / "outputs", "run-1")
    writer.write(1, {"train_loss": 0.3, "internal_val_accuracy": 0.8})
    writer.write(2, {"train_loss": 0.2, "internal_val_accuracy": 0.9})
    assert len(writer.jsonl_path.read_text().splitlines()) == 2
    assert len(writer.csv_path.read_text().splitlines()) == 3
