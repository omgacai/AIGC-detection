from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path

from robust_aigc.data.splits import preserve_directory_splits


def _prepare_sid_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_sid.py"
    spec = importlib.util.spec_from_file_location("prepare_sid", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_sid_subset_quota_balances_binary_classes_and_ai_sources() -> None:
    prepare_sid = _prepare_sid_module()
    quotas = prepare_sid.raw_quotas(Counter({0: 100, 1: 60, 2: 80}), max_per_class=40)
    assert quotas == {0: 40, 1: 20, 2: 20}
    assert quotas[1] + quotas[2] == quotas[0]


def test_sid_subset_quota_reallocates_when_one_ai_source_is_small() -> None:
    prepare_sid = _prepare_sid_module()
    quotas = prepare_sid.raw_quotas(Counter({0: 100, 1: 4, 2: 80}), max_per_class=40)
    assert quotas == {0: 40, 1: 4, 2: 36}


def test_directory_test_folder_is_never_reassigned_to_training(tmp_path: Path) -> None:
    root = tmp_path / "cifake"
    records = []
    for source_split, count in (("train", 10), ("test", 4)):
        for label, class_name in ((0, "REAL"), (1, "FAKE")):
            for index in range(count):
                path = root / source_split / class_name / f"{index}.jpg"
                records.append({"path": str(path), "label": label, "source_dataset": "cifake", "generator": None, "split": "train"})
    assigned = preserve_directory_splits(records, root)
    assert all(record["split"] == "test" for record in assigned if "/test/" in record["path"])
    assert not any(record["split"] == "test" for record in assigned if "/train/" in record["path"])
