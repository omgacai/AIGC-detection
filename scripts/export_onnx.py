#!/usr/bin/env python3
"""Export a trained DINOv3 forensic checkpoint as an inference-only ONNX model.

The ONNX graph accepts preprocessed float32 images of shape ``[B, 3, H, W]``
and emits an AIGC probability in ``[0, 1]`` for every image.  Image decoding,
resize/crop, and ImageNet normalization intentionally remain outside the graph
and are written to a sidecar metadata file for reproducible deployment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

from robust_aigc.models import DINOv3Forensic
from robust_aigc.utils.config import load_toml


class AIGCProbabilityONNXWrapper(nn.Module):
    """Expose the model's inference probability rather than its training dict."""

    def __init__(self, model: DINOv3Forensic):
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.model(image)["logits"])


def load_ema_checkpoint(model: nn.Module, checkpoint_path: Path) -> int:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    # Evaluation uses EMA weights when they are available; export the same
    # weights so ONNX predictions match project evaluation predictions.
    if state.get("ema_state_dict") is not None:
        parameters = dict(model.named_parameters())
        for name, value in state["ema_state_dict"]["shadow"].items():
            if name in parameters:
                parameters[name].data.copy_(value.cpu())
    return int(state.get("epoch", -1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.config.is_file():
        raise FileNotFoundError(args.config)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite {args.output}; pass --overwrite explicitly")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    config = load_toml(args.config)
    image_size = int(config["data"]["image_size"])
    model = DINOv3Forensic(config)
    epoch = load_ema_checkpoint(model, args.checkpoint)
    wrapper = AIGCProbabilityONNXWrapper(model.eval())
    dummy = torch.zeros((1, 3, image_size, image_size), dtype=torch.float32)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            dummy,
            args.output,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["aigc_probability"],
            dynamic_axes={"image": {0: "batch_size"}, "aigc_probability": {0: "batch_size"}},
        )

    try:
        import onnx
    except ImportError as error:
        raise RuntimeError("ONNX export completed but model validation requires the 'onnx' package. Install it with: python -m pip install onnx") from error
    onnx.checker.check_model(onnx.load(args.output))

    metadata = {
        "format": "ONNX",
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_epoch": epoch,
        "output": "aigc_probability; one float32 value in [0, 1] per input image",
        "input": {
            "name": "image",
            "dtype": "float32",
            "layout": "NCHW",
            "shape": ["batch_size", 3, image_size, image_size],
            "preprocessing": {
                "decode": "RGB",
                "resize": image_size,
                "center_crop": [image_size, image_size],
                "normalization_mean": config["data"].get("normalization_mean", [0.485, 0.456, 0.406]),
                "normalization_std": config["data"].get("normalization_std", [0.229, 0.224, 0.225]),
            },
        },
    }
    metadata_path = args.output.with_suffix(args.output.suffix + ".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"[INFO] exported={args.output}")
    print(f"[INFO] metadata={metadata_path}")


if __name__ == "__main__":
    main()
