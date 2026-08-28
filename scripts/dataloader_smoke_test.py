#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from robust_aigc.data.dataset import AIGCImageDataset
from robust_aigc.data.registry import load_manifest
from robust_aigc.data.transforms import basic_eval_transform


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--split", default="train"); args = parser.parse_args()
    records = [record for record in load_manifest(args.manifest) if record["split"] == args.split]
    if not records: raise ValueError(f"No records for split={args.split} in {args.manifest}")
    loader = DataLoader(AIGCImageDataset(records, basic_eval_transform()), batch_size=min(8, len(records)), shuffle=True, num_workers=4, pin_memory=True)
    batch = next(iter(loader)); print(f"images: {tuple(batch['image'].shape)}\nlabels: {tuple(batch['label'].shape)}\nmin/max: {batch['image'].min().item():.3f}/{batch['image'].max().item():.3f}\nlabels: {batch['label'].tolist()}\npaths: {list(batch['path'])}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
