"""Shared differentiable 2D Gaussian field for the local splatting benchmark."""

from __future__ import annotations

import torch
import torch.nn.functional as F


class GaussianField2D(torch.nn.Module):
    def __init__(self, n_points: int, seed: int, device: torch.device):
        super().__init__()
        g = torch.Generator(device="cpu")
        g.manual_seed(seed)
        self.xy = torch.nn.Parameter((2.0 * torch.rand(n_points, 2, generator=g) - 1.0).to(device))
        self.log_scale = torch.nn.Parameter(torch.full((n_points, 2), -2.2, device=device))
        self.opacity_logit = torch.nn.Parameter(torch.full((n_points, 1), -1.5, device=device))
        self.color_logit = torch.nn.Parameter(torch.randn(n_points, 3, generator=g, device=device) * 0.4)

    @property
    def num_points(self) -> int:
        return int(self.xy.shape[0])

    def forward(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        centers = self.xy
        scales = F.softplus(self.log_scale) + 0.015
        opacity = torch.sigmoid(self.opacity_logit)
        color = torch.sigmoid(self.color_logit)
        diff = xy[:, None, :] - centers[None, :, :]
        exponent = -0.5 * ((diff / scales[None, :, :]) ** 2).sum(dim=-1)
        basis = torch.exp(exponent) * opacity.T
        denom = basis.sum(dim=-1, keepdim=True) + 0.25
        rgb = (basis @ color) / denom
        return torch.clamp(rgb, 0.0, 1.0), basis, diff, scales

    def replace_points(
        self,
        keep: torch.Tensor,
        new_xy: torch.Tensor,
        new_log_scale: torch.Tensor,
        new_opacity_logit: torch.Tensor,
        new_color_logit: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            self.xy = torch.nn.Parameter(torch.cat([self.xy.detach()[keep], new_xy], dim=0))
            self.log_scale = torch.nn.Parameter(torch.cat([self.log_scale.detach()[keep], new_log_scale], dim=0))
            self.opacity_logit = torch.nn.Parameter(torch.cat([self.opacity_logit.detach()[keep], new_opacity_logit], dim=0))
            self.color_logit = torch.nn.Parameter(torch.cat([self.color_logit.detach()[keep], new_color_logit], dim=0))


def opacity_halved_logits(parent_logits: torch.Tensor) -> torch.Tensor:
    opacity = torch.sigmoid(parent_logits) * 0.5
    opacity = opacity.clamp(1e-5, 1 - 1e-5)
    return torch.log(opacity / (1 - opacity))
