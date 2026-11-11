"""Build the half-resolution training images used by every method.

Reads data/downloads/scenes/<scene>/images_full/*.jpg (training views only,
written by scripts/download.sh) and writes data/downloads/scenes/<scene>/images/
<stem>.png, downsampled by an integer 2x2 box filter (crop to even size,
average, round half up). The evaluator applies the identical filter to its
held-out ground truth, so training and evaluation see the same preprocessing.

Usage: python data/prepare_scenes.py [workspace_root]
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def box_downsample_uint8(img: np.ndarray, factor: int) -> np.ndarray:
    h, w = (img.shape[0] // factor) * factor, (img.shape[1] // factor) * factor
    x = img[:h, :w].astype(np.uint32).reshape(h // factor, factor, w // factor, factor, -1)
    n = factor * factor
    return ((x.sum(axis=(1, 3)) + n // 2) // n).astype(np.uint8)


def main(root: str) -> None:
    with open(os.path.join(HERE, "scenes.json")) as f:
        cfg = json.load(f)
    factor = int(cfg["downsample"])
    for scene in cfg["scenes"]:
        base = os.path.join(root, "data", "downloads", "scenes", scene["name"])
        src = os.path.join(base, "images_full")
        dst = os.path.join(base, "images")
        if not os.path.isdir(src):
            raise SystemExit(f"{src} missing; run scripts/download.sh first")
        os.makedirs(dst, exist_ok=True)
        names = sorted(os.listdir(src))
        made = 0
        for name in names:
            out = os.path.join(dst, os.path.splitext(name)[0] + ".png")
            if os.path.exists(out):
                continue
            with Image.open(os.path.join(src, name)) as im:
                arr = np.asarray(im.convert("RGB"))
            Image.fromarray(box_downsample_uint8(arr, factor)).save(out + ".tmp.png")
            os.replace(out + ".tmp.png", out)
            made += 1
        print(f"[prepare] {scene['name']}: {len(names)} training images ({made} new) at 1/{factor} resolution")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(HERE))
