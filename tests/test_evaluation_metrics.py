from scripts.evaluate import by_source_metrics
from scripts.prepare_reference_benchmark import records
from scripts.validate_image_manifest import is_readable
from cluster.competition_reference_archives import ARCHIVES
from robust_aigc.utils.config import load_toml
from robust_aigc.utils.metrics import binary_classification_metrics


def test_evaluation_metrics_include_operating_points_and_threshold_metrics():
    metrics = binary_classification_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["tpr_at_1_fpr"] == 1.0
    assert metrics["tpr_at_5_fpr"] == 1.0
    assert metrics["fpr_at_99_tpr"] == 0.0
    assert metrics["fpr_at_95_tpr"] == 0.0
    assert 0.0 <= metrics["threshold_at_1_fpr"] <= 1.0
    assert 0.0 <= metrics["threshold_at_5_fpr"] <= 1.0


def test_by_source_metrics_keeps_sources_separate():
    rows = by_source_metrics("clean", "internal_val", [0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8], ["sid", "sid", "cifake", "cifake"], 0.5)
    assert [row["source_dataset"] for row in rows] == ["cifake", "sid"]
    assert all(row["accuracy"] == 1.0 for row in rows)


def test_reference_records_are_evaluation_only_and_do_not_mix_labels(tmp_path):
    paths = [tmp_path / "one.jpg", tmp_path / "two.jpg"]
    values = records(paths, 1, "competition_dalle_advanced", "dalle_advanced")
    assert {row["split"] for row in values} == {"organizer_demo"}
    assert {row["label"] for row in values} == {1}


def test_reference_evaluation_is_clean_only():
    config = load_toml("configs/evaluation_reference_clean.toml", validate_experiment=False)
    assert [condition["name"] for condition in config["evaluation"]["conditions"]] == ["clean"]


def test_only_official_reference_archives_are_listed():
    assert ARCHIVES == ("Images/Real/coco.zip", "Images/Diffusion_based/DALLE.zip")


def test_image_readability_detects_a_bad_payload(tmp_path):
    bad = tmp_path / "broken.png"
    bad.write_bytes(b"not an image")
    readable, error = is_readable(bad)
    assert not readable
    assert error is not None
