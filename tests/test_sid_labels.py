import pytest

from robust_aigc.data.registry import sid_binary_label


@pytest.mark.parametrize(("raw", "expected"), [(0, 0), (1, 1), (2, 1)])
def test_sid_raw_labels_are_mapped_to_one_binary_target(raw, expected):
    assert sid_binary_label(raw) == expected


def test_sid_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        sid_binary_label(3)
