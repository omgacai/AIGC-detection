import torch

from robust_aigc.utils.ema import TrainableParameterEMA
from scripts.probe_dinov3_layers import parse_layers, pooled_patch_features
from scripts.train import binary_js_divergence, job_stop_epoch, optimizer_parameter_groups, smooth_binary_targets, supervised_contrastive_loss


def test_binary_label_smoothing_moves_targets_toward_half():
    result = smooth_binary_targets(torch.tensor([0.0, 1.0]), 0.05)
    assert torch.allclose(result, torch.tensor([0.025, 0.975]))


def test_ema_tracks_only_trainable_parameters_and_can_be_applied():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False
    ema = TrainableParameterEMA(model, decay=0.5)
    assert all(name.startswith("1.") for name in ema.shadow)
    before = {name: value.clone() for name, value in ema.shadow.items()}
    with torch.no_grad():
        for parameter in model[1].parameters():
            parameter.add_(2.0)
    ema.update(model)
    for name, value in ema.shadow.items():
        assert torch.allclose(value, before[name] + 1.0)


def test_epoch_chunk_preserves_global_schedule_position():
    assert job_stop_epoch(start=8, total=30, epochs_this_job=4) == 12
    assert job_stop_epoch(start=28, total=30, epochs_this_job=4) == 30


def test_optimizer_uses_lower_learning_rate_for_trainable_backbone():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 1)
    groups = optimizer_parameter_groups(Model(), {"optimizer": {"learning_rate": 2e-4, "backbone_learning_rate": 1e-5}})
    assert [group["lr"] for group in groups] == [2e-4, 1e-5]


def test_prediction_consistency_is_zero_for_equal_logits_and_positive_otherwise():
    logits = torch.tensor([-2.0, 0.5, 3.0])
    assert torch.allclose(binary_js_divergence(logits, logits), torch.tensor(0.0), atol=1e-6)
    assert binary_js_divergence(logits, -logits) > 0


def test_supervised_contrastive_uses_clean_augmented_and_class_positives():
    clean = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    augmented = clean.clone()
    labels = torch.tensor([0.0, 0.0, 1.0, 1.0])
    loss = supervised_contrastive_loss(clean, augmented, labels, temperature=0.1)
    assert torch.isfinite(loss)
    assert loss >= 0


def test_layer_probe_selects_blocks_and_mean_pools_patch_tokens():
    assert parse_layers("all", 3) == (1, 2, 3)
    assert parse_layers("1,3", 3) == (1, 3)
    hidden = torch.ones(2, 6, 4)  # CLS + one register + four patch tokens
    pooled = pooled_patch_features(hidden, num_register_tokens=1)
    assert tuple(pooled.shape) == (2, 4)
    assert torch.allclose(pooled.norm(dim=1), torch.ones(2))
