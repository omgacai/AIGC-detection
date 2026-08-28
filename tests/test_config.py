from pathlib import Path

import pytest

from robust_aigc.utils.config import load_toml, validate_experiment_config


def test_dinov3_forensic_config_is_valid():
    root = Path(__file__).resolve().parents[1]
    config = load_toml(root / "configs" / "dinov3_forensic.toml")
    assert config["model"]["backbone"].startswith("facebook/dinov3")
    assert config["data"]["allow_organizer_demo_for_training"] is False


def test_srm_supcon_robustness_config_is_valid_and_uses_sid_only():
    root = Path(__file__).resolve().parents[1]
    config = load_toml(root / "configs" / "dinov3_srm_supcon.toml")
    assert config["data"]["datasets"] == ["sid"]
    assert config["forensic_head"]["residual_adapter_enabled"] is True
    assert config["loss"]["supcon_weight"] > 0


def test_isolated_robustness_ablation_configs_are_valid():
    root = Path(__file__).resolve().parents[1]
    loss_config = load_toml(root / "configs" / "dinov3_supcon_blur_noise.toml")
    architecture_config = load_toml(root / "configs" / "dinov3_srm_gated_blur_noise.toml")
    assert loss_config["forensic_head"]["residual_adapter_enabled"] is False
    assert architecture_config["forensic_head"]["residual_adapter_enabled"] is True
    assert architecture_config["loss"]["supcon_weight"] == 0


def test_evaluation_toml_loads_without_training_schema_validation():
    root = Path(__file__).resolve().parents[1]
    config = load_toml(root / "configs" / "evaluation.toml", validate_experiment=False)
    assert config["evaluation"]["conditions"][0]["kind"] == "clean"


def test_config_rejects_organizer_demo_training():
    config = {section: {} for section in ("run", "data", "model", "forensic_head", "loss", "optimizer", "scheduler", "training", "metrics", "logging", "curriculum")}
    config["data"] = {"allow_organizer_demo_for_training": True, "train_splits": ["train"]}
    config["model"] = {"parameter_budget": 300_000_000}
    with pytest.raises(ValueError, match="evaluation-only"):
        validate_experiment_config(config)
