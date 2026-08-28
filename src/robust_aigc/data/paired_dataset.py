from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class PairedAIGCImageDataset(Dataset):
    """Return a clean view and an independently augmented view of the same image."""

    def __init__(self, records: list[dict], image_size: int, augmentation=None):
        if any(record.get("split") == "organizer_demo" for record in records):
            raise AssertionError("organizer_demo is evaluation-only and must never be used for training")
        self.records, self.image_size, self.augmentation = records, image_size, augmentation

    @staticmethod
    def _to_tensor(image: np.ndarray, image_size: int) -> torch.Tensor:
        pil = Image.fromarray(image).convert("RGB")
        pil = TF.resize(pil, image_size)
        pil = TF.center_crop(pil, [image_size, image_size])
        return TF.normalize(TF.to_tensor(pil), mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record["path"]) as image:
            clean = np.asarray(image.convert("RGB"))
        augmented = self.augmentation(image=clean)["image"] if self.augmentation else clean.copy()
        return {"image": self._to_tensor(clean, self.image_size), "augmented_image": self._to_tensor(augmented, self.image_size), "label": torch.tensor(int(record["label"]), dtype=torch.float32), "path": record["path"]}
