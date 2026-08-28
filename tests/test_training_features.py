import torch

from robust_aigc.utils.ema import TrainableParameterEMA
from scripts.train import job_stop_epoch, smooth_binary_targets


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
