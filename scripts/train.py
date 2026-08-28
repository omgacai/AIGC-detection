#!/usr/bin/env python3
"""Config-driven trainer for the DINOv3-Forensic architecture."""
from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from robust_aigc.data.augmentations import build_curriculum_augmentation
from robust_aigc.data.paired_dataset import PairedAIGCImageDataset
from robust_aigc.data.registry import load_manifest
from robust_aigc.models import DINOv3Forensic
from robust_aigc.utils.checkpointing import CheckpointManager
from robust_aigc.utils.config import load_toml
from robust_aigc.utils.metrics import EpochReporter, binary_operating_point_metrics
from robust_aigc.utils.paths import configure_caches, resolve_paths
from robust_aigc.utils.seed import set_seed


def curriculum_stage(curriculum: dict, epoch: int) -> dict:
    for stage in curriculum.values():
        if stage["epochs"][0] <= epoch <= stage["epochs"][1]:
            return stage
    raise ValueError(f"No curriculum stage covers epoch {epoch}")


def build_loader(records, image_size, augmentation, batch_size, workers, balanced=False, normalization_mean=(0.485, 0.456, 0.406), normalization_std=(0.229, 0.224, 0.225)):
    dataset = PairedAIGCImageDataset(records, image_size, augmentation, normalization_mean, normalization_std)
    sampler = None
    if balanced:
        counts = Counter(int(r["label"]) for r in records)
        sampler = WeightedRandomSampler([1 / counts[int(r["label"])] for r in records], len(records), replacement=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=not balanced, sampler=sampler, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def train_one_epoch(model, loader, optimizer, scaler, device, accumulation, lambda_consistency, clip_norm):
    model.train(); sums = Counter(); optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        image, augmented, label = (batch["image"].to(device, non_blocking=True), batch["augmented_image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True))
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(image, augmented)
            classification = (F.binary_cross_entropy_with_logits(outputs["logits"], label) + F.binary_cross_entropy_with_logits(outputs["augmented_logits"], label)) / 2
            consistency = 1 - F.cosine_similarity(outputs["projection"], outputs["augmented_projection"], dim=1).mean()
            loss = classification + lambda_consistency * consistency
        scaler.scale(loss / accumulation).backward()
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        size = label.numel(); sums.update({"count": size, "train_loss": loss.item() * size, "train_classification_loss": classification.item() * size, "train_consistency_loss": consistency.item() * size})
    return {key: sums[key] / sums["count"] for key in ("train_loss", "train_classification_loss", "train_consistency_loss")}


@torch.inference_mode()
def validate(model, loader, device):
    model.eval(); labels, scores, total_loss = [], [], 0.0
    for batch in loader:
        image, label = batch["image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16): logits = model(image)["logits"]
        total_loss += F.binary_cross_entropy_with_logits(logits, label).item() * label.numel()
        labels.extend(label.int().cpu().tolist()); scores.extend(torch.sigmoid(logits).float().cpu().tolist())
    metrics = binary_operating_point_metrics(labels, scores)
    targets, probabilities = torch.tensor(labels), torch.tensor(scores)
    metrics.update({"internal_val_loss": total_loss / len(labels), "internal_val_accuracy": float(((probabilities >= 0.5).int() == targets).float().mean())})
    return metrics


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=Path("configs/dinov3_forensic.toml")); parser.add_argument("--manifest", type=Path); parser.add_argument("--resume", type=Path); parser.add_argument("--epochs", type=int, help="Optional cap for a smoke run; does not modify the TOML config.")
    args = parser.parse_args(); config = load_toml(args.config); set_seed(config["run"]["seed"])
    paths = resolve_paths(create=True); configure_caches(paths)
    output_root = Path(os.environ.get("AIGC_OUTPUT_ROOT", paths.output_root)); checkpoint_root = Path(os.environ.get("AIGC_CHECKPOINT_ROOT", paths.checkpoint_root))
    manifest = args.manifest or Path(os.path.expandvars(config["data"]["manifest"])); records = load_manifest(manifest)
    train_records = [r for r in records if r["split"] in config["data"]["train_splits"]]
    val_records = [r for r in records if r["split"] == config["data"]["validation_split"]]
    if not train_records or not val_records: raise ValueError("Manifest needs both configured train and internal_val records")
    if any(r["split"] == "organizer_demo" for r in train_records): raise AssertionError("organizer_demo must never be used for training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Submit training through Slurm on a GPU node; CUDA is unavailable.")
    training, data = config["training"], config["data"]
    model = DINOv3Forensic(config).to(device)
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
    epochs, warmup = training["epochs"], config["scheduler"]["warmup_epochs"]
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1")
        epochs = min(epochs, args.epochs)
    scheduler = LambdaLR(optimizer, lambda e: min(1.0, (e + 1) / warmup) if e < warmup else 0.5 * (1 + torch.cos(torch.tensor(torch.pi * (e - warmup) / max(1, epochs - warmup))).item()))
    scaler = torch.amp.GradScaler("cuda")
    reporter = EpochReporter(output_root, config["run"]["name"], tensorboard=config["logging"]["tensorboard"])
    checkpoints = CheckpointManager(checkpoint_root, config["run"]["name"], monitor=config["metrics"]["primary_checkpoint_metric"], mode=config["metrics"]["primary_checkpoint_mode"])
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False); model.load_state_dict(state["model_state_dict"]); optimizer.load_state_dict(state["optimizer_state_dict"]); scheduler.load_state_dict(state["scheduler_state_dict"]); start = state["epoch"] + 1
    normalization_mean = data.get("normalization_mean", (0.485, 0.456, 0.406))
    normalization_std = data.get("normalization_std", (0.229, 0.224, 0.225))
    val_loader = build_loader(val_records, data["image_size"], None, training["batch_size"], training["num_workers"], normalization_mean=normalization_mean, normalization_std=normalization_std)
    for epoch in range(start, epochs):
        stage = curriculum_stage(config["curriculum"], epoch); reporter.logger.info("epoch=%d curriculum=%s", epoch, stage["name"])
        train_loader = build_loader(train_records, data["image_size"], build_curriculum_augmentation(stage), training["batch_size"], training["num_workers"], data["balance_classes_per_batch"], normalization_mean, normalization_std)
        metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, training["gradient_accumulation_steps"], config["loss"]["consistency_weight"], config["optimizer"]["gradient_clip_norm"])
        metrics.update(validate(model, val_loader, device)); metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        reporter.report(epoch, metrics); checkpoints.save_epoch(epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, metrics=metrics, training_args={"config": config, "config_path": str(args.config)}); scheduler.step()
    reporter.close()


if __name__ == "__main__": main()
