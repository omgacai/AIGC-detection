import pytest

torch = pytest.importorskip("torch")
from robust_aigc.data.dataset import AIGCImageDataset


def test_organizer_demo_rejected_for_training():
    record = {"path": "/not/loaded.jpg", "label": 0, "source_dataset": "demo", "generator": None, "split": "organizer_demo"}
    with pytest.raises(AssertionError, match="evaluation-only"):
        AIGCImageDataset([record])


def test_organizer_demo_allowed_only_for_non_training_evaluation():
    record = {"path": "/not/loaded.jpg", "label": 0, "source_dataset": "demo", "generator": None, "split": "organizer_demo"}
    assert len(AIGCImageDataset([record], for_training=False)) == 1
