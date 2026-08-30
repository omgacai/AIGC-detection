from pathlib import Path

import pytest

from scripts.merge_manifests import merge_manifests


def test_merge_rejects_competition_reference_data(tmp_path: Path):
    manifest = tmp_path / "competition.csv"
    manifest.write_text(
        "path,label,source_dataset,generator,split\n"
        "/home/s/user/aigc-storage/data/competition_reference/raw/Images/Real/coco/example.jpg,0,competition_coco_val2017,,organizer_demo\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Competition reference data"):
        merge_manifests([manifest])
