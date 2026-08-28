from pathlib import Path

from scripts.train import run_directory_name


def config():
    return {
        "run": {"name": "forensic experiment"},
        "model": {"architecture": "dinov3/vit-l-16"},
        "checkpointing": {"run_directory_template": "{started_at}_{model}_{run_name}"},
    }


def test_new_training_run_name_contains_model_and_never_reuses_directory(tmp_path):
    checkpoint_root, output_root = tmp_path / "checkpoints", tmp_path / "outputs"
    first = run_directory_name(config(), checkpoint_root, output_root, None)
    assert "dinov3_vit-l-16" in first
    (checkpoint_root / first).mkdir(parents=True)
    second = run_directory_name(config(), checkpoint_root, output_root, None)
    assert second == f"{first}-02"


def test_resume_retains_original_run_directory(tmp_path):
    checkpoint = tmp_path / "20260828-133500_dinov3" / "last.pt"
    assert run_directory_name(config(), tmp_path, tmp_path, checkpoint) == checkpoint.parent.name
