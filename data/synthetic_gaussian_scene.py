"""Procedural 2D Gaussian splatting scene used for the compactness benchmark.

The real paper evaluates 3D novel-view synthesis scenes. This tracked generator
provides a small deterministic substitute with the same optimization object:
fit a radiance field represented as a sum of Gaussian primitives from sampled
pixels. It is intentionally method-agnostic and is shared by baseline and method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SceneSamples:
    train_xy: torch.Tensor
    train_rgb: torch.Tensor
    test_xy: torch.Tensor
    test_rgb: torch.Tensor
    image_xy: torch.Tensor
    image_rgb: torch.Tensor
    image_size: int


def _target_field(xy: torch.Tensor) -> torch.Tensor:
    centers = xy.new_tensor(
        [
            [-0.72, -0.55],
            [-0.25, 0.42],
            [0.18, -0.18],
            [0.58, 0.48],
            [0.74, -0.42],
            [-0.58, 0.12],
            [0.05, 0.72],
        ]
    )
    scales = xy.new_tensor(
        [
            [0.08, 0.18],
            [0.15, 0.07],
            [0.10, 0.10],
            [0.09, 0.16],
            [0.06, 0.09],
            [0.13, 0.06],
            [0.07, 0.11],
        ]
    )
    colors = xy.new_tensor(
        [
            [0.90, 0.12, 0.08],
            [0.05, 0.65, 0.95],
            [0.96, 0.82, 0.12],
            [0.38, 0.12, 0.92],
            [0.05, 0.90, 0.36],
            [0.98, 0.42, 0.06],
            [0.14, 0.30, 0.95],
        ]
    )
    diff = xy[:, None, :] - centers[None, :, :]
    exponent = -0.5 * ((diff / scales[None, :, :]) ** 2).sum(dim=-1)
    weights = torch.exp(exponent)
    rgb = weights @ colors
    rgb = rgb / (weights.sum(dim=-1, keepdim=True) + 0.35)
    ripple = 0.04 * torch.sin(11.0 * xy[:, :1]) * torch.cos(9.0 * xy[:, 1:])
    return torch.clamp(rgb + ripple, 0.0, 1.0)


def _grid(size: int) -> torch.Tensor:
    lin = torch.linspace(-1.0, 1.0, size)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")
    return torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=-1)


def make_scene(
    image_size: int = 96,
    n_train: int = 4096,
    n_test: int = 4096,
    seed: int = 7,
    device: str | torch.device = "cpu",
) -> SceneSamples:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    train_xy = 2.0 * torch.rand(n_train, 2, generator=generator) - 1.0
    test_xy = 2.0 * torch.rand(n_test, 2, generator=generator) - 1.0
    image_xy = _grid(image_size)
    train_rgb = _target_field(train_xy)
    test_rgb = _target_field(test_xy)
    image_rgb = _target_field(image_xy)
    return SceneSamples(
        train_xy=train_xy.to(device),
        train_rgb=train_rgb.to(device),
        test_xy=test_xy.to(device),
        test_rgb=test_rgb.to(device),
        image_xy=image_xy.to(device),
        image_rgb=image_rgb.to(device),
        image_size=image_size,
    )


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = torch.mean((pred - target) ** 2).clamp_min(1e-12)
    return float(10.0 * torch.log10(torch.tensor(1.0, device=mse.device) / mse))


def ssim_global(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Small, dependency-free global SSIM proxy for RGB samples."""
    c1 = 0.01**2
    c2 = 0.03**2
    x = pred.reshape(-1)
    y = target.reshape(-1)
    mux, muy = x.mean(), y.mean()
    vx = ((x - mux) ** 2).mean()
    vy = ((y - muy) ** 2).mean()
    cov = ((x - mux) * (y - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2)))


def l1_error(pred: torch.Tensor, target: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(pred - target)))


def flops_proxy(num_points: int, num_samples: int) -> int:
    # For each sample/point pair, the renderer evaluates a 2D Gaussian and RGB
    # contribution. This avoids hardware-dependent wall-clock scoring.
    return int(num_points * num_samples * 18)

