import pytest

torch = pytest.importorskip("torch")
from robust_aigc.data.dataset import AIGCImageDataset
from robust_aigc.data.paired_dataset import PairedAIGCImageDataset


def test_organizer_demo_rejected_for_training():
    record = {"path": "/not/loaded.jpg", "label": 0, "source_dataset": "demo", "generator": None, "split": "organizer_demo"}
    with pytest.raises(AssertionError, match="evaluation-only"):
        AIGCImageDataset([record])


def test_organizer_demo_allowed_only_for_non_training_evaluation():
    record = {"path": "/not/loaded.jpg", "label": 0, "source_dataset": "demo", "generator": None, "split": "organizer_demo"}
    assert len(AIGCImageDataset([record], for_training=False)) == 1


def test_paired_training_replaces_corrupt_file_from_same_source_and_label(tmp_path):
    from PIL import Image

    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not an image")
    valid = tmp_path / "valid.png"
    Image.new("RGB", (4, 4), color="red").save(valid)
    records = [
        {"path": str(broken), "label": 1, "source_dataset": "wildfake", "split": "train"},
        {"path": str(valid), "label": 1, "source_dataset": "wildfake", "split": "train"},
    ]
    dataset = PairedAIGCImageDataset(records, image_size=4)
    with pytest.warns(RuntimeWarning, match="Skipping unreadable"):
        item = dataset[0]
    assert item["path"] == str(valid)
    assert item["label"].item() == 1
