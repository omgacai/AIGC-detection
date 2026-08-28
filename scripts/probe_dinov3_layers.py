#!/usr/bin/env python3
"""Layer-wise frozen-DINO robustness probe for real-vs-AIGC detection.

Each linear probe is fit on clean *training* features only.  The validation
heatmap therefore measures how well a frozen representation survives each
real-world transform rather than how well a classifier learns that transform.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from torch.utils.data import DataLoader
from transformers import AutoModel

from robust_aigc.data.augmentations import build_evaluation_augmentation
from robust_aigc.data.paired_dataset import PairedAIGCImageDataset
from robust_aigc.data.registry import load_manifest
from robust_aigc.utils.seed import set_seed


CONDITIONS = (
    ("clean", "clean", None),
    ("jpeg_q50", "jpeg", 50),
    ("blur_sigma_2_0", "gaussian_blur", 2.0),
    ("resize_0_25", "resize", 0.25),
    ("noise_sigma_0_10", "gaussian_noise", 0.10),
)


def parse_layers(value: str, num_blocks: int) -> tuple[int, ...]:
    if value == "all":
        return tuple(range(1, num_blocks + 1))
    layers = tuple(int(item) for item in value.split(","))
    if not layers or any(layer < 1 or layer > num_blocks for layer in layers):
        raise ValueError(f"Layers must be within 1..{num_blocks}; got {value!r}")
    return layers


def pooled_patch_features(hidden: torch.Tensor, num_register_tokens: int) -> torch.Tensor:
    """Mean pool patch tokens only; CLS/register tokens are deliberately excluded."""
    patches = hidden[:, 1 + num_register_tokens:]
    return torch.nn.functional.normalize(patches.mean(dim=1), dim=1)


@torch.inference_mode()
def extract_features(model, records, augmentation, layers, image_size, batch_size, workers, device, mean, std):
    dataset = PairedAIGCImageDataset(records, image_size, augmentation, mean, std, for_training=False)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
    values = {layer: [] for layer in layers}
    labels: list[np.ndarray] = []
    num_register_tokens = int(getattr(model.config, "num_register_tokens", 0))
    for step, batch in enumerate(loader, 1):
        images = (batch["augmented_image"] if augmentation else batch["image"]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(pixel_values=images, output_hidden_states=True)
        for layer in layers:
            values[layer].append(pooled_patch_features(output.hidden_states[layer], num_register_tokens).float().cpu().numpy())
        labels.append(batch["label"].numpy().astype(np.int64))
        if step % 100 == 0 or step == len(loader):
            print(f"[INFO] extracted batch={step}/{len(loader)}", flush=True)
    return {layer: np.concatenate(chunks) for layer, chunks in values.items()}, np.concatenate(labels)


def fit_probe(features: np.ndarray, labels: np.ndarray, seed: int) -> SGDClassifier:
    # L2-normalised pooled DINO features make this a true lightweight linear
    # probe, and SGD avoids a heavyweight CPU optimisation per layer.
    probe = SGDClassifier(loss="log_loss", alpha=1e-4, max_iter=200, tol=1e-3, random_state=seed, early_stopping=False)
    probe.fit(features, labels)
    return probe


def metric_row(name: str, condition: str, probe, features: np.ndarray, labels: np.ndarray) -> dict:
    probabilities = probe.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(np.int64)
    return {
        "probe": name,
        "condition": condition,
        "count": int(len(labels)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }


def write_heatmap(rows: list[dict], destination: Path, conditions: tuple[str, ...]) -> None:
    import matplotlib.pyplot as plt

    probes = [row["probe"] for row in rows if row["condition"] == conditions[0]]
    matrix = np.array([[next(row["accuracy"] for row in rows if row["probe"] == probe and row["condition"] == condition) for condition in conditions] for probe in probes])
    figure, axis = plt.subplots(figsize=(10, max(5, len(probes) * 0.48)))
    image = axis.imshow(matrix, cmap="viridis", vmin=0.0, vmax=1.0, aspect="auto")
    axis.set_xticks(range(len(conditions)), labels=conditions, rotation=25, ha="right")
    axis.set_yticks(range(len(probes)), labels=probes)
    axis.set_title("Frozen DINOv3 ViT-B patch-feature probe accuracy")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, f"{100 * matrix[row_index, column_index]:.1f}", ha="center", va="center", color="white" if matrix[row_index, column_index] < 0.65 else "black", fontsize=8)
    figure.colorbar(image, ax=axis, label="Accuracy")
    figure.tight_layout(); figure.savefig(destination, dpi=220); plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backbone", default="facebook/dinov3-vitb16-pretrain-lvd1689m")
    parser.add_argument("--layers", default="all", help="all, or comma-separated transformer block indices")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    records = load_manifest(args.manifest)
    train_records = [record for record in records if record["split"] == "train"]
    validation_records = [record for record in records if record["split"] == "internal_val"]
    if not train_records or not validation_records:
        raise ValueError("Manifest must contain train and internal_val records; test is deliberately not used.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Run this extraction job on a Slurm GPU node.")
    model = AutoModel.from_pretrained(args.backbone).to(device).eval()
    layers = parse_layers(args.layers, int(getattr(model.config, "num_hidden_layers")))
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    print(f"[INFO] device={torch.cuda.get_device_name(0)} layers={layers} train={len(train_records)} internal_val={len(validation_records)}")
    print("[INFO] extracting clean train features", flush=True)
    train_features, train_labels = extract_features(model, train_records, None, layers, args.image_size, args.batch_size, args.num_workers, device, mean, std)
    probes = {f"L{layer}": fit_probe(train_features[layer], train_labels, args.seed + layer) for layer in layers}
    probes["all_layers_concat"] = fit_probe(np.concatenate([train_features[layer] for layer in layers], axis=1), train_labels, args.seed + 1000)
    rows: list[dict] = []
    condition_names = tuple(item[0] for item in CONDITIONS)
    for condition_name, kind, value in CONDITIONS:
        print(f"[INFO] extracting internal_val condition={condition_name}", flush=True)
        augmentation = build_evaluation_augmentation(kind, value)
        features, labels = extract_features(model, validation_records, augmentation, layers, args.image_size, args.batch_size, args.num_workers, device, mean, std)
        for layer in layers:
            rows.append(metric_row(f"L{layer}", condition_name, probes[f"L{layer}"], features[layer], labels))
        rows.append(metric_row("all_layers_concat", condition_name, probes["all_layers_concat"], np.concatenate([features[layer] for layer in layers], axis=1), labels))
    metrics_path = args.output_dir / "layer_transform_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_heatmap(rows, args.output_dir / "layer_transform_accuracy.png", condition_names)
    protocol = {
        "backbone": args.backbone, "layers": list(layers), "feature": "L2-normalised mean-pooled patch tokens",
        "probe": "SGD logistic regression trained on clean train features only",
        "evaluation_split": "internal_val", "conditions": [{"name": name, "kind": kind, "value": value} for name, kind, value in CONDITIONS],
        "excluded_split": "test",
    }
    (args.output_dir / "probe_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] wrote {metrics_path} and layer_transform_accuracy.png to {args.output_dir}")


if __name__ == "__main__":
    main()
