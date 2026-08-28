from robust_aigc.data.splits import assign_deterministic_splits, validate_split_isolation


def test_splits_are_reproducible_and_isolated():
    records = [{"path": f"/data/{index}.jpg", "label": index % 2, "source_dataset": "sid", "generator": None, "split": "train"} for index in range(30)]
    first = assign_deterministic_splits(records)
    second = assign_deterministic_splits(records)
    assert first == second
    validate_split_isolation(first)
    assert {record["split"] for record in first} == {"train", "internal_val", "test"}
