from __future__ import annotations

import math
import re
import torch
from torch import nn
from transformers import AutoModel


class DINOv3Forensic(nn.Module):
    """DINOv3 ViT-L/16 with three tapped multi-scale forensic branches."""
    def __init__(self, config: dict):
        super().__init__()
        model_cfg, head_cfg = config["model"], config["forensic_head"]
        self.backbone = AutoModel.from_pretrained(model_cfg["backbone"])
        if model_cfg.get("gradient_checkpointing", False):
            if not hasattr(self.backbone, "gradient_checkpointing_enable"):
                raise RuntimeError("This backbone does not support gradient checkpointing")
            self.backbone.gradient_checkpointing_enable()
        self.tapped_layers = tuple(model_cfg["tapped_layers"])
        self.num_register_tokens = int(getattr(self.backbone.config, "num_register_tokens", 0))
        hidden_size, local_dim = int(self.backbone.config.hidden_size), int(head_cfg["local_feature_dim"])
        self.local_branches = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_size, local_dim, kernel_size=head_cfg["local_aggregation_kernel"], padding=head_cfg["local_aggregation_padding"]), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()) for _ in self.tapped_layers])
        self.cls_branches = nn.ModuleList([nn.Sequential(nn.Linear(hidden_size, head_cfg["classifier_hidden_dim"]), nn.GELU(), nn.Dropout(head_cfg["classifier_dropout"]), nn.Linear(head_cfg["classifier_hidden_dim"], local_dim)) for _ in self.tapped_layers])
        fused_dim = 2 * local_dim * len(self.tapped_layers)
        self.normalizer = nn.LayerNorm(fused_dim)
        self.classifier = nn.Sequential(nn.Linear(fused_dim, head_cfg["classifier_hidden_dim"]), nn.GELU(), nn.Dropout(head_cfg["classifier_dropout"]), nn.Linear(head_cfg["classifier_hidden_dim"], 1))
        self.projector = nn.Sequential(nn.Linear(fused_dim, head_cfg["projection_hidden_dim"]), nn.GELU(), nn.Linear(head_cfg["projection_hidden_dim"], head_cfg["projection_output_dim"]))
        self.backbone_frozen = bool(model_cfg.get("freeze_backbone", False))
        self.unfrozen_backbone_blocks: tuple[int, ...] = ()
        requested_blocks = int(model_cfg.get("unfreeze_last_blocks", 0))
        if requested_blocks:
            if self.backbone_frozen:
                raise ValueError("freeze_backbone and unfreeze_last_blocks cannot both be enabled")
            self.unfrozen_backbone_blocks = self._unfreeze_last_blocks(requested_blocks)
        elif self.backbone_frozen:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def _unfreeze_last_blocks(self, count: int) -> tuple[int, ...]:
        """Fine-tune only the final transformer blocks, not the whole ViT."""
        if count < 1:
            raise ValueError("unfreeze_last_blocks must be positive")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        blocks: dict[int, list[nn.Parameter]] = {}
        pattern = re.compile(r"(?:encoder\.(?:layer|layers)|blocks)\.(\d+)\.")
        for name, parameter in self.backbone.named_parameters():
            match = pattern.search(name)
            if match:
                blocks.setdefault(int(match.group(1)), []).append(parameter)
        if not blocks:
            raise RuntimeError("Could not identify DINO transformer blocks for partial fine-tuning; refusing a silent frozen run.")
        selected = tuple(sorted(blocks)[-count:])
        for index in selected:
            for parameter in blocks[index]:
                parameter.requires_grad = True
        # Train the final normalization with the final blocks when present.
        for name, parameter in self.backbone.named_parameters():
            if name.startswith(("layernorm", "norm")):
                parameter.requires_grad = True
        return selected

    def train(self, mode: bool = True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def _encode(self, images: torch.Tensor) -> torch.Tensor:
        requires_grad = any(parameter.requires_grad for parameter in self.backbone.parameters())
        with torch.set_grad_enabled(requires_grad): outputs = self.backbone(pixel_values=images, output_hidden_states=True)
        features = []
        for branch_index, layer_index in enumerate(self.tapped_layers):
            hidden = outputs.hidden_states[layer_index]
            cls_token, patches = hidden[:, 0], hidden[:, 1 + self.num_register_tokens:]
            side = int(math.isqrt(patches.shape[1]))
            if side * side != patches.shape[1]: raise ValueError(f"Patch count {patches.shape[1]} is not square")
            patch_map = patches.transpose(1, 2).reshape(images.shape[0], patches.shape[-1], side, side)
            features.extend((self.local_branches[branch_index](patch_map), self.cls_branches[branch_index](cls_token)))
        return self.normalizer(torch.cat(features, dim=1))

    def forward(self, image: torch.Tensor, augmented_image: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        clean_features = self._encode(image)
        result = {"logits": self.classifier(clean_features).squeeze(1), "projection": self.projector(clean_features)}
        if augmented_image is not None:
            augmented_features = self._encode(augmented_image)
            result["augmented_logits"] = self.classifier(augmented_features).squeeze(1)
            result["augmented_projection"] = self.projector(augmented_features)
        return result
