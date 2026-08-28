#!/usr/bin/env python3
"""Evaluate a checkpoint on clean and deterministic real-world transformations."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from robust_aigc.data.augmentations import build_evaluation_augmentation
from robust_aigc.data.paired_dataset import PairedAIGCImageDataset
from robust_aigc.data.registry import load_manifest
from robust_aigc.models import DINOv3Forensic
from robust_aigc.utils.config import load_toml
from robust_aigc.utils.metrics import binary_classification_metrics
from robust_aigc.utils.paths import configure_caches, resolve_paths
from robust_aigc.utils.seed import set_seed


@torch.inference_mode()
def score_condition(model, records, condition, image_size, batch_size, workers, device, normalization_mean, normalization_std):
    augmentation = build_evaluation_augmentation(condition["kind"], condition.get("value"))
    dataset = PairedAIGCImageDataset(records, image_size, augmentation, normalization_mean, normalization_std, for_training=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    labels, scores, paths = [], [], []
    for batch in loader:
        images = (batch["augmented_image"] if augmentation else batch["image"]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(images)["logits"]
        labels.extend(batch["label"].int().tolist())
        scores.extend(torch.sigmoid(logits).float().cpu().tolist())
        paths.extend(batch["path"])
    return paths, labels, scores


def error_examples(paths, labels, scores, threshold, limit):
    rows = [{"image_path": path, "label": int(label), "pred": float(score)} for path, label, score in zip(paths, labels, scores)]
    false_positives = sorted((row for row in rows if row["label"] == 0 and row["pred"] >= threshold), key=lambda row: row["pred"], reverse=True)
    false_negatives = sorted((row for row in rows if row["label"] == 1 and row["pred"] < threshold), key=lambda row: row["pred"])
    return {"false_positives": false_positives[:limit], "false_negatives": false_negatives[:limit]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-config", type=Path, default=Path("configs/dinov3_forensic.toml"))
    parser.add_argument("--evaluation-config", type=Path, default=Path("configs/evaluation.toml"))
    parser.add_argument("--split", help="Defaults to evaluation.default_split; organizer_demo is evaluation-only.")
    args = parser.parse_args()

    model_config = load_toml(args.model_config)
    evaluation_config = load_toml(args.evaluation_config, validate_experiment=False)["evaluation"]
    paths_config = resolve_paths(create=True); configure_caches(paths_config)
    output_root = Path(os.environ.get("AIGC_OUTPUT_ROOT", paths_config.output_root)) / evaluation_config["name"]
    output_root.mkdir(parents=True, exist_ok=True)
    split = args.split or evaluation_config["default_split"]
    records = [record for record in load_manifest(args.manifest) if record["split"] == split]
    if not records:
        raise ValueError(f"No records with split={split!r} in {args.manifest}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DINOv3Forensic(model_config).to(device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"]); model.eval()
    if state.get("ema_state_dict") is not None:
        parameters = dict(model.named_parameters())
        for name, value in state["ema_state_dict"]["shadow"].items():
            if name in parameters:
                parameters[name].data.copy_(value.to(device))
    model_data = model_config["data"]
    normalization_mean = model_data.get("normalization_mean", (0.485, 0.456, 0.406))
    normalization_std = model_data.get("normalization_std", (0.229, 0.224, 0.225))

    summary = []
    for condition_index, condition in enumerate(evaluation_config["conditions"]):
        set_seed(model_config["run"]["seed"] + condition_index)
        print(f"[INFO] Evaluating {condition['name']} on {len(records)} images")
        image_paths, labels, scores = score_condition(model, records, condition, evaluation_config["image_size"], evaluation_config["batch_size"], evaluation_config["num_workers"], device, normalization_mean, normalization_std)
        metrics = binary_classification_metrics(labels, scores, evaluation_config["threshold"])
        summary.append({"condition": condition["name"], "split": split, "count": len(labels), **metrics})
        (output_root / f"predictions_{condition['name']}.json").write_text(json.dumps([{"image_path": path, "pred": float(score)} for path, score in zip(image_paths, scores)], indent=2), encoding="utf-8")
        (output_root / f"errors_{condition['name']}.json").write_text(json.dumps(error_examples(image_paths, labels, scores, evaluation_config["threshold"], evaluation_config["error_examples_per_type"]), indent=2), encoding="utf-8")
    clean = summary[0]
    if clean["condition"] != "clean":
        raise ValueError("The first evaluation condition must be clean so robustness deltas are well-defined")
    for row in summary:
        for metric in ("accuracy", "balanced_accuracy", "roc_auc", "tpr_at_1_fpr", "fpr_at_99_tpr"):
            row[f"delta_{metric}_vs_clean"] = row[metric] - clean[metric]
    with (output_root / "robustness_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (output_root / "robustness_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[INFO] Wrote clean/transformation comparison and prediction JSON files to {output_root}")


if __name__ == "__main__":
    main()
