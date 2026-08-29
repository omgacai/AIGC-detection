from scripts.evaluate import by_source_metrics
from robust_aigc.utils.metrics import binary_classification_metrics


def test_evaluation_metrics_include_operating_points_and_threshold_metrics():
    metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["tpr_at_1_fpr"] == 1.0
    assert metrics["fpr_at_99_tpr"] == 0.0


def test_by_source_metrics_keeps_sources_separate():
    rows = by_source_metrics("clean", "internal_val", [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], ["sid", "sid", "cifake", "cifake"], 0.5)
    assert [row["source_dataset"] for row in rows] == ["cifake", "sid"]
    assert all(row["accuracy"] == 1.0 for row in rows)
