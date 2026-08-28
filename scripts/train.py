#!/usr/bin/env python3
"""Config-driven trainer for the DINOv3-Forensic architecture."""
from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, WeightedRandomSampler

from robust_aigc.data.augmentations import build_blur_noise_augmentation, build_curriculum_augmentation, build_single_transform_augmentation
from robust_aigc.data.paired_dataset import PairedAIGCImageDataset
from robust_aigc.data.registry import load_manifest
from robust_aigc.models import DINOv3Forensic
from robust_aigc.utils.checkpointing import CheckpointManager
from robust_aigc.utils.config import load_toml
from robust_aigc.utils.ema import TrainableParameterEMA
from robust_aigc.utils.metrics import EpochReporter, binary_operating_point_metrics
from robust_aigc.utils.paths import configure_caches, resolve_paths
from robust_aigc.utils.seed import set_seed


def run_directory_name(config: dict, checkpoint_root: Path, output_root: Path, resume: Path | None) -> str:
    """Create a non-overwriting run ID, or retain the original directory on resume."""
    if resume is not None:
        return resume.expanduser().resolve().parent.name
    checkpoint_config = config.get("checkpointing", {})
    template = checkpoint_config.get("run_directory_template", "{started_at}_{model}_{run_name}")
    values = {
        "started_at": datetime.now().astimezone().strftime("%Y%m%d-%H%M%S"),
        "model": config["model"]["architecture"],
        "run_name": config["run"]["name"],
    }
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", template.format(**values)).strip("._")
    candidate, suffix = base, 2
    while (checkpoint_root / candidate).exists() or (output_root / candidate).exists():
        candidate = f"{base}-{suffix:02d}"
        suffix += 1
    return candidate


def curriculum_stage(curriculum: dict, epoch: int) -> dict:
    for stage in curriculum.values():
        if stage["epochs"][0] <= epoch <= stage["epochs"][1]:
            return stage
    raise ValueError(f"No curriculum stage covers epoch {epoch}")


def job_stop_epoch(start: int, total: int, epochs_this_job: int | None) -> int:
    return min(total, start + epochs_this_job) if epochs_this_job is not None else total


def balanced_sample_weights(records, balance_classes: bool, balance_datasets: bool):
    if not balance_classes and not balance_datasets:
        return None
    def group(record):
        dataset = record["source_dataset"] if balance_datasets else "all"
        label = int(record["label"]) if balance_classes else "all"
        return dataset, label
    counts = Counter(group(record) for record in records)
    return [1.0 / counts[group(record)] for record in records]


def optimizer_parameter_groups(model, config: dict) -> list[dict]:
    """Use a conservative LR for fine-tuned DINO blocks and normal LR for heads."""
    optimizer_config = config["optimizer"]
    head, backbone = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (backbone if name.startswith("backbone.") else head).append(parameter)
    if not head:
        raise ValueError("No trainable forensic-head parameters found")
    groups = [{"params": head, "lr": optimizer_config["learning_rate"]}]
    if backbone:
        groups.append({"params": backbone, "lr": optimizer_config.get("backbone_learning_rate", optimizer_config["learning_rate"] * 0.05)})
    return groups


def build_loader(records, image_size, augmentation, batch_size, workers, balance_classes=False, balance_datasets=False, normalization_mean=(0.485, 0.456, 0.406), normalization_std=(0.229, 0.224, 0.225)):
    dataset = PairedAIGCImageDataset(records, image_size, augmentation, normalization_mean, normalization_std)
    weights = balanced_sample_weights(records, balance_classes, balance_datasets)
    sampler = WeightedRandomSampler(weights, len(records), replacement=True) if weights is not None else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=sampler is None, sampler=sampler, num_workers=workers, pin_memory=True, persistent_workers=workers > 0)


def smooth_binary_targets(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("label smoothing must be in [0, 1)")
    return labels * (1.0 - smoothing) + 0.5 * smoothing


def binary_js_divergence(first_logits: torch.Tensor, second_logits: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence between two Bernoulli predictions."""
    first = torch.sigmoid(first_logits).clamp(1e-6, 1 - 1e-6)
    second = torch.sigmoid(second_logits).clamp(1e-6, 1 - 1e-6)
    midpoint = (first + second) / 2
    def kl(probability, target):
        return probability * torch.log(probability / target) + (1 - probability) * torch.log((1 - probability) / (1 - target))
    return ((kl(first, midpoint) + kl(second, midpoint)) / 2).mean()


def supervised_contrastive_loss(clean_projection: torch.Tensor, augmented_projection: torch.Tensor, labels: torch.Tensor, temperature: float) -> torch.Tensor:
    """SupCon across the clean/augmented pair and same-class batch examples."""
    if temperature <= 0:
        raise ValueError("supcon_temperature must be positive")
    features = F.normalize(torch.cat((clean_projection, augmented_projection), dim=0), dim=1)
    targets = torch.cat((labels, labels), dim=0).long()
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.shape[0], device=logits.device, dtype=torch.bool)
    positive_mask = targets[:, None].eq(targets[None, :]) & ~self_mask
    logits = logits.masked_fill(self_mask, float("-inf"))
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive_mask.sum(dim=1)
    valid = positive_count > 0
    if not valid.any():
        return logits.new_zeros(())
    return -(log_probability.masked_fill(~positive_mask, 0).sum(dim=1)[valid] / positive_count[valid]).mean()


def train_one_epoch(model, loader, optimizer, scaler, device, accumulation, loss_config, clip_norm, label_smoothing, ema=None, logger=None, epoch=None, log_every_steps=0):
    model.train(); sums = Counter(); optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader, 1):
        image, augmented, label = (batch["image"].to(device, non_blocking=True), batch["augmented_image"].to(device, non_blocking=True), batch["label"].to(device, non_blocking=True))
        smoothed_label = smooth_binary_targets(label, label_smoothing)
        with torch.autocast("cuda", dtype=torch.float16):
            outputs = model(image, augmented)
            classification = (F.binary_cross_entropy_with_logits(outputs["logits"], smoothed_label) + F.binary_cross_entropy_with_logits(outputs["augmented_logits"], smoothed_label)) / 2
            consistency = 1 - F.cosine_similarity(outputs["projection"], outputs["augmented_projection"], dim=1).mean()
            prediction_consistency = binary_js_divergence(outputs["logits"], outputs["augmented_logits"])
            supcon = supervised_contrastive_loss(outputs["projection"], outputs["augmented_projection"], label, float(loss_config.get("supcon_temperature", 0.1))) if loss_config.get("supcon_weight", 0.0) else classification.new_zeros(())
            loss = (classification + float(loss_config.get("consistency_weight", 0.0)) * consistency
                    + float(loss_config.get("prediction_consistency_weight", 0.0)) * prediction_consistency
                    + float(loss_config.get("supcon_weight", 0.0)) * supcon)
        scaler.scale(loss / accumulation).backward()
        if step % accumulation == 0 or step == len(loader):
            scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            if ema is not None: ema.update(model)
        size = label.numel()
        batch_correct = ((torch.sigmoid(outputs["logits"]) >= 0.5) == label.bool()).sum().item()
        sums.update({"count": size, "correct": batch_correct, "train_loss": loss.item() * size, "train_classification_loss": classification.item() * size, "train_consistency_loss": consistency.item() * size, "train_prediction_consistency_loss": prediction_consistency.item() * size, "train_supcon_loss": supcon.item() * size})
        if "residual_gate" in outputs:
            sums.update({"residual_gate_sum": outputs["residual_gate"].mean().item() * size})
        if logger is not None and log_every_steps and (step % log_every_steps == 0 or step == len(loader)):
            logger.info(
                "epoch=%d batch=%d/%d loss=%.6f batch_accuracy=%.4f running_accuracy=%.4f lr=%.8f",
                epoch, step, len(loader), loss.item(), batch_correct / size, sums["correct"] / sums["count"], optimizer.param_groups[0]["lr"],
            )
    result = {key: sums[key] / sums["count"] for key in ("train_loss", "train_classification_loss", "train_consistency_loss", "train_prediction_consistency_loss", "train_supcon_loss")}
    result["train_accuracy"] = sums["correct"] / sums["count"]
    if sums["residual_gate_sum"]:
        result["residual_gate_mean"] = sums["residual_gate_sum"] / sums["count"]
    return result


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
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=Path("configs/dinov3_forensic.toml")); parser.add_argument("--manifest", type=Path); parser.add_argument("--resume", type=Path); parser.add_argument("--epochs", type=int, help="Optional absolute epoch cap for a smoke run."); parser.add_argument("--epochs-this-job", type=int, help="Run at most this many complete epochs, preserving the 30-epoch schedule when resumed.")
    args = parser.parse_args(); config = load_toml(args.config); set_seed(config["run"]["seed"])
    paths = resolve_paths(create=True); configure_caches(paths)
    output_root = Path(os.environ.get("AIGC_OUTPUT_ROOT", paths.output_root))
    checkpoint_config = config.get("checkpointing", {})
    configured_checkpoint_root = os.path.expandvars(checkpoint_config.get("root", str(paths.checkpoint_root)))
    checkpoint_root = Path(os.environ.get("AIGC_CHECKPOINT_ROOT", configured_checkpoint_root))
    run_directory = run_directory_name(config, checkpoint_root, output_root, args.resume)
    manifest = args.manifest or Path(os.path.expandvars(config["data"]["manifest"])); records = load_manifest(manifest)
    train_records = [r for r in records if r["split"] in config["data"]["train_splits"]]
    val_records = [r for r in records if r["split"] == config["data"]["validation_split"]]
    if not train_records or not val_records: raise ValueError("Manifest needs both configured train and internal_val records")
    if any(r["split"] == "organizer_demo" for r in train_records): raise AssertionError("organizer_demo must never be used for training")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda": raise RuntimeError("Submit training through Slurm on a GPU node; CUDA is unavailable.")
    training, data = config["training"], config["data"]
    model = DINOv3Forensic(config).to(device)
    optimizer = AdamW(optimizer_parameter_groups(model, config), weight_decay=config["optimizer"]["weight_decay"])
    epochs, warmup = training["epochs"], config["scheduler"]["warmup_epochs"]
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be at least 1")
        epochs = min(epochs, args.epochs)
    if args.epochs_this_job is not None and args.epochs_this_job < 1:
        raise ValueError("--epochs-this-job must be at least 1")
    minimum_ratio = config["scheduler"]["min_learning_rate"] / config["optimizer"]["learning_rate"]
    def learning_rate_factor(epoch: int) -> float:
        if epoch < warmup:
            return min(1.0, (epoch + 1) / max(1, warmup))
        progress = (epoch - warmup) / max(1, epochs - warmup - 1)
        cosine = 0.5 * (1.0 + torch.cos(torch.tensor(torch.pi * progress)).item())
        return minimum_ratio + (1.0 - minimum_ratio) * cosine
    scheduler = LambdaLR(optimizer, learning_rate_factor)
    scaler = torch.amp.GradScaler("cuda")
    ema = TrainableParameterEMA(model, config["optimizer"]["ema_decay"]) if config["optimizer"].get("ema_decay") is not None else None
    reporter = EpochReporter(output_root, run_directory, tensorboard=config["logging"]["tensorboard"])
    reporter.logger.info("run_directory=%s", run_directory)
    fine_tuning_scope = "frozen" if model.backbone_frozen else (f"last_blocks={model.unfrozen_backbone_blocks}" if model.unfrozen_backbone_blocks else "full_backbone")
    reporter.logger.info("fine_tuning scope=%s trainable_parameters=%d", fine_tuning_scope, sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad))
    checkpoints = None
    if checkpoint_config.get("enabled", True):
        tracked_metrics = checkpoint_config.get("tracked_metric_modes", {"tpr_at_1_fpr": "max", "fpr_at_99_tpr": "min"})
        checkpoints = CheckpointManager(
            checkpoint_root,
            run_directory,
            monitor=checkpoint_config.get("monitor", config["metrics"]["primary_checkpoint_metric"]),
            mode=checkpoint_config.get("mode", config["metrics"]["primary_checkpoint_mode"]),
            tracked_metrics=tracked_metrics,
            save_last=checkpoint_config.get("save_last", True),
            save_primary_best=checkpoint_config.get("save_primary_best", True),
            include_optimizer_state=checkpoint_config.get("include_optimizer_state", True),
            include_scheduler_state=checkpoint_config.get("include_scheduler_state", True),
        )
        checkpoints.write_metadata()
    start = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False); model.load_state_dict(state["model_state_dict"])
        if state.get("optimizer_state_dict") is not None: optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict") is not None: scheduler.load_state_dict(state["scheduler_state_dict"])
        if ema is not None and state.get("ema_state_dict") is not None: ema.load_state_dict(state["ema_state_dict"], device)
        start = state["epoch"] + 1
    stop_epoch = job_stop_epoch(start, epochs, args.epochs_this_job)
    normalization_mean = data.get("normalization_mean", (0.485, 0.456, 0.406))
    normalization_std = data.get("normalization_std", (0.229, 0.224, 0.225))
    val_loader = build_loader(val_records, data["image_size"], None, training["batch_size"], training["num_workers"], normalization_mean=normalization_mean, normalization_std=normalization_std)
    for epoch in range(start, stop_epoch):
        stage = curriculum_stage(config["curriculum"], epoch); reporter.logger.info("epoch=%d curriculum=%s", epoch, stage["name"])
        augmentation_mode = data.get("training_augmentation")
        augmentation = (
            build_single_transform_augmentation(stage) if augmentation_mode == "single_transform"
            else build_blur_noise_augmentation(stage) if augmentation_mode == "blur_noise_single"
            else build_curriculum_augmentation(stage)
        )
        train_loader = build_loader(train_records, data["image_size"], augmentation, training["batch_size"], training["num_workers"], data["balance_classes_per_batch"], data.get("balance_datasets", False), normalization_mean, normalization_std)
        metrics = train_one_epoch(model, train_loader, optimizer, scaler, device, training["gradient_accumulation_steps"], config["loss"], config["optimizer"]["gradient_clip_norm"], training["label_smoothing"], ema, reporter.logger, epoch, config["logging"].get("batch_log_every_steps", 0))
        if ema is not None:
            with ema.average_parameters(model):
                metrics.update(validate(model, val_loader, device))
        else:
            metrics.update(validate(model, val_loader, device))
        metrics["learning_rate"] = optimizer.param_groups[0]["lr"]
        reporter.report(epoch, metrics)
        # Advance before serializing: a resumed job must start with the next
        # epoch's LR rather than repeating the just-completed epoch's LR.
        scheduler.step()
        if checkpoints:
            checkpoints.save_epoch(epoch=epoch, model=model, optimizer=optimizer, scheduler=scheduler, metrics=metrics, training_args={"config": config, "config_path": str(args.config)}, ema_state_dict=ema.state_dict() if ema is not None else None)
    reporter.logger.info("job_finished completed_epochs=%d next_epoch=%d total_epochs=%d", max(0, stop_epoch - start), stop_epoch, epochs)
    if checkpoints and stop_epoch < epochs:
        reporter.logger.info("resume_next_job=%s", checkpoints.run_dir / "last.pt")
    reporter.close()


if __name__ == "__main__": main()
