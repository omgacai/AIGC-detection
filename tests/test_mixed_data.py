from pathlib import Path

import pytest

from scripts.merge_manifests import cap_split_groups, merge_manifests
from scripts.train import balanced_sample_weights
from robust_aigc.data.registry import write_manifest


def test_joint_dataset_and_class_weights_equalize_groups():
    records = [
        {"source_dataset": "sid", "label": 0},
        {"source_dataset": "sid", "label": 0},
        {"source_dataset": "sid", "label": 1},
        {"source_dataset": "cifake", "label": 1},
    ]
    assert balanced_sample_weights(records, True, True) == [0.5, 0.5, 1.0, 1.0]


def test_merge_rejects_duplicate_paths(tmp_path):
    image = tmp_path / "same.jpg"; image.touch()
    record = {"path": str(image), "label": 0, "source_dataset": "sid", "generator": None, "split": "train"}
    first, second = tmp_path / "one.csv", tmp_path / "two.csv"
    write_manifest([record], first); write_manifest([{**record, "source_dataset": "cifake"}], second)
    with pytest.raises(ValueError, match="Duplicate image path"):
        merge_manifests([first, second])


def test_caps_are_applied_per_dataset_class_and_split():
    records = [
        {"path": f"/{dataset}-{label}-{index}.jpg", "source_dataset": dataset, "label": label, "split": "train"}
        for dataset in ("sid", "cifake") for label in (0, 1) for index in range(3)
    ]
    selected = cap_split_groups(records, {"train": 2})
    counts = {}
    for record in selected:
        key = record["source_dataset"], record["label"]
        counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {2}
