from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class PairedAIGCImageDataset(Dataset):
    """Return a clean view and an independently augmented view of the same image."""

    def __init__(self, records: list[dict], image_size: int, augmentation=None,
                 normalization_mean=(0.485, 0.456, 0.406), normalization_std=(0.229, 0.224, 0.225),
                 for_training: bool = True):
        if for_training and any(record.get("split") == "organizer_demo" for record in records):
            raise AssertionError("organizer_demo is evaluation-only and must never be used for training")
        self.records, self.image_size, self.augmentation = records, image_size, augmentation
        self.normalization_mean, self.normalization_std = tuple(normalization_mean), tuple(normalization_std)

    @staticmethod
    def _to_tensor(image: np.ndarray, image_size: int, mean, std) -> torch.Tensor:
        pil = Image.fromarray(image).convert("RGB")
        pil = TF.resize(pil, image_size)
        pil = TF.center_crop(pil, [image_size, image_size])
        return TF.normalize(TF.to_tensor(pil), mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record["path"]) as image:
            clean = np.asarray(image.convert("RGB"))
        augmented = self.augmentation(image=clean)["image"] if self.augmentation else clean.copy()
        return {
            "image": self._to_tensor(clean, self.image_size, self.normalization_mean, self.normalization_std),
            "augmented_image": self._to_tensor(augmented, self.image_size, self.normalization_mean, self.normalization_std),
            "label": torch.tensor(int(record["label"]), dtype=torch.float32),
            "path": record["path"],
            # Metadata only: never passed into the model. It enables a
            # source-wise robustness report without mixing source labels into
            # image features or training targets.
            "source_dataset": record.get("source_dataset", "unknown"),
        }
