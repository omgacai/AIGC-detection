from __future__ import annotations

from typing import Any


def build_curriculum_augmentation(stage: dict[str, Any]):
    """Build Albumentations for one TOML curriculum stage."""
    import albumentations as A
    import cv2

    transforms = []
    qualities = stage.get("jpeg_quality", [])
    if qualities:
        low, high = min(qualities), max(qualities)
        try: transforms.append(A.ImageCompression(quality_range=(low, high), p=1.0))
        except TypeError: transforms.append(A.ImageCompression(quality_lower=low, quality_upper=high, p=1.0))
    sigmas = stage.get("gaussian_blur_sigma", [])
    if sigmas: transforms.append(A.GaussianBlur(blur_limit=(3, 7), sigma_limit=(min(sigmas), max(sigmas)), p=1.0))
    scales = stage.get("resize_scale", [])
    if scales:
        low, high = min(scales), max(scales)
        try: transforms.append(A.Downscale(scale_range=(low, high), interpolation_pair={"downscale": cv2.INTER_AREA, "upscale": cv2.INTER_LINEAR}, p=1.0))
        except TypeError: transforms.append(A.Downscale(scale_min=low, scale_max=high, interpolation=cv2.INTER_AREA, p=1.0))
    noise = stage.get("gaussian_noise_sigma", [])
    if noise:
        try: transforms.append(A.GaussNoise(std_range=(min(noise), max(noise)), p=1.0))
        except TypeError: transforms.append(A.GaussNoise(var_limit=((255 * min(noise)) ** 2, (255 * max(noise)) ** 2), p=1.0))
    jitter = max(stage.get("color_jitter_strength", [0.0]))
    if jitter: transforms.append(A.ColorJitter(brightness=jitter, contrast=jitter, saturation=jitter, hue=0.0, p=1.0))
    return A.Compose(transforms)
