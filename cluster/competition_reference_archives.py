#!/usr/bin/env python3
"""Fixed evaluation-only WildFake archives for the competition reference set."""
from __future__ import annotations

import argparse


ARCHIVES = (
    "Images/Real/Coco.zip",
    "Images/Diffusion_based/DALLE.zip",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    if args.index is None:
        print("\n".join(ARCHIVES))
        return
    if not 0 <= args.index < len(ARCHIVES):
        raise SystemExit(f"index must be in [0, {len(ARCHIVES) - 1}]")
    print(ARCHIVES[args.index])


if __name__ == "__main__":
    main()
