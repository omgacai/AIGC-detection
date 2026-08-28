import pytest

torchvision = pytest.importorskip("torchvision")
from PIL import Image
from robust_aigc.data.transforms import basic_eval_transform


def test_basic_transform_is_deterministic_rgb_tensor():
    image = Image.new("RGB", (300, 250), "red")
    transformed = basic_eval_transform(224)(image)
    assert tuple(transformed.shape) == (3, 224, 224)
