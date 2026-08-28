from scripts.evaluate import error_examples


def test_error_examples_return_most_confident_mistakes():
    result = error_examples(
        ["real-a.jpg", "real-b.jpg", "ai-a.jpg", "ai-b.jpg"],
        [0, 0, 1, 1],
        [0.8, 0.6, 0.1, 0.4],
        threshold=0.5,
        limit=1,
    )
    assert result["false_positives"] == [{"image_path": "real-a.jpg", "label": 0, "pred": 0.8}]
    assert result["false_negatives"] == [{"image_path": "ai-a.jpg", "label": 1, "pred": 0.1}]
