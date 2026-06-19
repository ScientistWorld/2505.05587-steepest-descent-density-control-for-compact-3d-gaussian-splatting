"""Scaled SteepGS reproduction on a differentiable 2D Gaussian renderer.

This implements the paper's density-control computation: accumulated splitting
matrices, least-eigenvalue filtering, two offspring along the least eigenvector,
and opacity halving. The scene is 2D to keep the benchmark cheap, but the
algorithmic densification step mirrors Steepest Density Control.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.synthetic_gaussian_scene import flops_proxy, l1_error, make_scene, psnr, ssim_global


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

    def replace_points(self, keep: torch.Tensor, new_xy: torch.Tensor, new_log_scale: torch.Tensor, new_opacity_logit: torch.Tensor, new_color_logit: torch.Tensor) -> None:
        with torch.no_grad():
            self.xy = torch.nn.Parameter(torch.cat([self.xy.detach()[keep], new_xy], dim=0))
            self.log_scale = torch.nn.Parameter(torch.cat([self.log_scale.detach()[keep], new_log_scale], dim=0))
            self.opacity_logit = torch.nn.Parameter(torch.cat([self.opacity_logit.detach()[keep], new_opacity_logit], dim=0))
            self.color_logit = torch.nn.Parameter(torch.cat([self.color_logit.detach()[keep], new_color_logit], dim=0))


def _opacity_halved_logits(parent_logits: torch.Tensor) -> torch.Tensor:
    opacity = torch.sigmoid(parent_logits) * 0.5
    opacity = opacity.clamp(1e-5, 1 - 1e-5)
    return torch.log(opacity / (1 - opacity))


def compute_splitting_matrices(
    basis: torch.Tensor,
    diff: torch.Tensor,
    scales: torch.Tensor,
    dloss_dbasis: torch.Tensor,
) -> torch.Tensor:
    """Accumulate S_i = E[(dL/d sigma_i) * Hessian_center(sigma_i)]."""
    inv_var = 1.0 / (scales.detach() ** 2)
    outer = diff[:, :, :, None] * diff[:, :, None, :] * inv_var[None, :, :, None] * inv_var[None, :, None, :]
    hessian = basis.detach()[:, :, None, None] * (outer - torch.diag_embed(inv_var)[None, :, :, :])
    splitting = torch.einsum("bn,bnij->nij", dloss_dbasis.detach(), hessian) / basis.shape[0]
    splitting = 0.5 * (splitting + splitting.transpose(-1, -2))
    return splitting


def steepest_densify(
    model: GaussianField2D,
    splitting_accum: torch.Tensor,
    grad_accum: torch.Tensor,
    interval_count: int,
    epsilon: float,
    split_threshold: float,
    grad_threshold: float,
    max_points: int,
) -> dict[str, float]:
    S = splitting_accum / max(1, interval_count)
    G = grad_accum / max(1, interval_count)
    mean_grad_norm = G.norm(dim=-1)
    eigvals, eigvecs = torch.linalg.eigh(S)
    lambda_min = eigvals[:, 0]
    v_min = eigvecs[:, :, 0]
    eligible = (lambda_min < split_threshold) & (mean_grad_norm <= grad_threshold)
    capacity = max(0, max_points - model.num_points)
    if int(eligible.sum()) > capacity:
        idx = torch.where(eligible)[0]
        chosen_order = torch.argsort(lambda_min[idx])[:capacity]
        selected = torch.zeros_like(eligible)
        selected[idx[chosen_order]] = True
    else:
        selected = eligible
    selected_idx = torch.where(selected)[0]
    if selected_idx.numel() == 0:
        return {
            "splits": 0,
            "min_lambda": float(lambda_min.min()),
            "min_mean_grad_norm": float(mean_grad_norm.min()),
            "eligible": int(eligible.sum()),
        }

    parent_xy = model.xy.detach()[selected_idx]
    direction = v_min[selected_idx]
    new_xy = torch.cat([parent_xy + epsilon * direction, parent_xy - epsilon * direction], dim=0)
    new_log_scale = model.log_scale.detach()[selected_idx].repeat(2, 1)
    new_opacity_logit = _opacity_halved_logits(model.opacity_logit.detach()[selected_idx]).repeat(2, 1)
    new_color_logit = model.color_logit.detach()[selected_idx].repeat(2, 1)
    keep = torch.ones(model.num_points, dtype=torch.bool, device=model.xy.device)
    keep[selected_idx] = False
    model.replace_points(keep, new_xy, new_log_scale, new_opacity_logit, new_color_logit)
    return {
        "splits": int(selected_idx.numel()),
        "min_lambda": float(lambda_min.min()),
        "min_mean_grad_norm": float(mean_grad_norm.min()),
        "eligible": int(eligible.sum()),
    }


def train(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    scene = make_scene(seed=args.seed, device=device, image_size=args.image_size, n_train=args.n_train, n_test=args.n_test)
    model = GaussianField2D(args.init_points, args.seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    split_accum = torch.zeros(args.init_points, 2, 2, device=device)
    grad_accum = torch.zeros(args.init_points, 2, device=device)
    interval_count = 0
    total_splits = 0
    history = []

    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 11)
    for step in range(1, args.steps + 1):
        idx = torch.randint(scene.train_xy.shape[0], (args.batch_size,), generator=generator, device=device)
        xy = scene.train_xy[idx]
        target = scene.train_rgb[idx]
        optimizer.zero_grad(set_to_none=True)
        pred, basis, diff, scales = model(xy)
        loss = F.l1_loss(pred, target) + 0.2 * F.mse_loss(pred, target)
        dloss_dbasis = torch.autograd.grad(loss, basis, retain_graph=True)[0]
        loss.backward()
        S = compute_splitting_matrices(basis, diff, scales, dloss_dbasis)
        grad_vector = model.xy.grad.detach().clone()
        optimizer.step()
        split_accum = split_accum + S
        grad_accum = grad_accum + grad_vector
        interval_count += 1

        if step >= args.densify_from and step % args.densify_interval == 0 and model.num_points < args.max_points:
            info = steepest_densify(
                model,
                split_accum,
                grad_accum,
                interval_count,
                args.epsilon,
                args.split_threshold,
                args.grad_threshold,
                args.max_points,
            )
            total_splits += info["splits"]
            history.append({"step": step, "points": model.num_points, **info})
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            split_accum = torch.zeros(model.num_points, 2, 2, device=device)
            grad_accum = torch.zeros(model.num_points, 2, device=device)
            interval_count = 0

    with torch.no_grad():
        test_pred = model(scene.test_xy)[0]
        image_pred = model(scene.image_xy)[0]
    metrics = {
        "psnr": psnr(test_pred, scene.test_rgb),
        "ssim": ssim_global(test_pred, scene.test_rgb),
        "l1_error": l1_error(test_pred, scene.test_rgb),
        "num_points": model.num_points,
        "point_reduction_vs_adc": math.nan,
        "splits": total_splits,
        "flops_proxy": flops_proxy(model.num_points, scene.test_xy.shape[0]),
    }
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "predictions.npz",
        pred=test_pred.detach().cpu().numpy(),
        target=scene.test_rgb.detach().cpu().numpy(),
        image_pred=image_pred.detach().cpu().numpy(),
        image_target=scene.image_rgb.detach().cpu().numpy(),
    )
    (out_dir / "metrics.json").write_text(json.dumps({"metrics": metrics, "history": history, "args": vars(args)}, indent=2))
    return metrics


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/steepgs2d")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--steps", type=int, default=900)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--n-train", type=int, default=4096)
    p.add_argument("--n-test", type=int, default=4096)
    p.add_argument("--image-size", type=int, default=96)
    p.add_argument("--init-points", type=int, default=12)
    p.add_argument("--max-points", type=int, default=144)
    p.add_argument("--lr", type=float, default=0.045)
    p.add_argument("--densify-from", type=int, default=150)
    p.add_argument("--densify-interval", type=int, default=75)
    p.add_argument("--epsilon", type=float, default=0.045)
    p.add_argument("--split-threshold", type=float, default=-1e-6)
    p.add_argument("--grad-threshold", type=float, default=0.035)
    return p


if __name__ == "__main__":
    result = train(build_parser().parse_args())
    print(json.dumps(result, indent=2))
