"""Ground-truth loading and image metrics for the held-out test views.

Metric definitions follow the reference 3DGS evaluation script: renders are
clamped to [0, 1] and quantised to 8 bits (as when saved to PNG), PSNR and SSIM
(11x11 Gaussian window, sigma 1.5) are computed per image, LPIPS uses VGG-16
features with the v0.1 linear heads on [0, 1] inputs (the convention of the 3DGS
code base). Ground truth is the released JPG, decoded and box-downsampled by 2.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def box_downsample_uint8(img: np.ndarray, factor: int) -> np.ndarray:
    """Integer box filter: crop to a multiple of factor, average, round half up."""
    h, w = (img.shape[0] // factor) * factor, (img.shape[1] // factor) * factor
    x = img[:h, :w].astype(np.uint32).reshape(h // factor, factor, w // factor, factor, -1)
    s = x.sum(axis=(1, 3))
    n = factor * factor
    return ((s + n // 2) // n).astype(np.uint8)


def load_gt(path: str, factor: int) -> np.ndarray:
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return box_downsample_uint8(arr, factor)


def quantize(img: torch.Tensor) -> torch.Tensor:
    """HxWx3 float render -> 8-bit values in [0, 1] (torchvision save_image rule)."""
    return torch.clamp(torch.floor(img * 255.0 + 0.5), 0, 255) / 255.0


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.double() - b.double()) ** 2).item()
    if mse <= 0:
        return 100.0
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def _gaussian_window(size=11, sigma=1.5, device="cpu"):
    g = torch.tensor([math.exp(-((x - size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(size)])
    g = (g / g.sum()).unsqueeze(1)
    w2 = (g @ g.t()).float()
    return w2.expand(3, 1, size, size).contiguous().to(device)


def ssim(a: torch.Tensor, b: torch.Tensor) -> float:
    """a, b: HxWx3 in [0, 1]."""
    x = a.permute(2, 0, 1)[None].float()
    y = b.permute(2, 0, 1)[None].float()
    w = _gaussian_window(device=x.device)
    pad = 5
    mu1 = F.conv2d(x, w, padding=pad, groups=3)
    mu2 = F.conv2d(y, w, padding=pad, groups=3)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    s1 = F.conv2d(x * x, w, padding=pad, groups=3) - mu1_sq
    s2 = F.conv2d(y * y, w, padding=pad, groups=3) - mu2_sq
    s12 = F.conv2d(x * y, w, padding=pad, groups=3) - mu12
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu12 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s1 + s2 + c2))
    return float(m.mean())


class LPIPSVGG(torch.nn.Module):
    """LPIPS v0.1 with VGG-16 features (weights verified by sha256 before use)."""

    TARGETS = (4, 9, 16, 23, 30)
    CHANNELS = (64, 128, 256, 512, 512)

    def __init__(self, vgg_path: str, lin_path: str):
        super().__init__()
        import torchvision

        vgg = torchvision.models.vgg16(weights=None)
        vgg.load_state_dict(torch.load(vgg_path, map_location="cpu", weights_only=True))
        self.layers = vgg.features[:30]
        self.lins = torch.nn.ModuleList([torch.nn.Conv2d(c, 1, 1, 1, 0, bias=False) for c in self.CHANNELS])
        lin = torch.load(lin_path, map_location="cpu", weights_only=True)
        for i, conv in enumerate(self.lins):
            conv.weight.data.copy_(lin[f"lin{i}.model.1.weight"])
        self.register_buffer("shift", torch.tensor([-0.030, -0.088, -0.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([0.458, 0.448, 0.450])[None, :, None, None])
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def features(self, img: torch.Tensor):
        x = (img.permute(2, 0, 1)[None].float() - self.shift) / self.scale
        feats = []
        for i, layer in enumerate(self.layers, 1):
            x = layer(x)
            if i in self.TARGETS:
                feats.append(x / (torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True)) + 1e-10))
        return feats

    @torch.no_grad()
    def distance(self, feats_a, feats_b) -> float:
        total = 0.0
        for fa, fb, lin in zip(feats_a, feats_b, self.lins):
            total += float(lin((fa - fb) ** 2).mean())
        return total
