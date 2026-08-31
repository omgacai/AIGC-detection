import torch

from scripts.export_onnx import AIGCProbabilityONNXWrapper


class _DictionaryLogitModel(torch.nn.Module):
    def forward(self, image):
        return {"logits": image.mean(dim=(1, 2, 3))}


def test_onnx_wrapper_returns_probability_tensor():
    output = AIGCProbabilityONNXWrapper(_DictionaryLogitModel())(torch.zeros(2, 3, 4, 4))
    assert output.shape == (2,)
    assert torch.allclose(output, torch.full((2,), 0.5))
