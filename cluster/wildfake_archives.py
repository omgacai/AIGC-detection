#!/usr/bin/env python3
"""Approved WildFake training archives; organiser benchmark archives are absent."""
from __future__ import annotations

import argparse

ARCHIVES = (
    "Images/Diffusion_based/ADM.zip",
    "Images/Diffusion_based/DDIM.zip",
    "Images/Diffusion_based/DDPM.zip",
    "Images/Diffusion_based/Imagen.zip",
    "Images/Diffusion_based/VQDM.zip",
    "Images/GAN_based.zip",
    "Images/Other_based.zip",
    "Images/Real/afhq.zip",
    "Images/Real/celebahq.zip",
    "Images/Real/church.zip",
    "Images/Real/ffhq.zip",
    "Images/Real/imagenet.zip",
    "Images/Real/laion5b.zip",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
