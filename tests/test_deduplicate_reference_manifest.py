import csv

from PIL import Image

from scripts.deduplicate_reference_manifest import deduplicate


def test_deduplicate_uses_decoded_content_not_file_path(tmp_path):
    first, second, distinct = (tmp_path / name for name in ("first.png", "second.png", "distinct.png"))
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(first)
    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(second)
    Image.new("RGB", (2, 2), color=(30, 20, 10)).save(distinct)
    records = [
        {"path": str(first), "label": "1", "source_dataset": "competition_dalle_advanced", "generator": "dalle_advanced", "split": "organizer_demo"},
        {"path": str(second), "label": "1", "source_dataset": "competition_dalle_advanced", "generator": "dalle_advanced", "split": "organizer_demo"},
        {"path": str(distinct), "label": "1", "source_dataset": "competition_dalle_advanced", "generator": "dalle_advanced", "split": "organizer_demo"},
    ]

    retained, removed = deduplicate(records)

    assert [record["path"] for record in retained] == [str(first), str(distinct)]
    assert removed[0]["kept_path"] == str(first)
    assert removed[0]["removed_path"] == str(second)
