"""Original-ADC-style heuristic baseline for the 2D splatting benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.synthetic_gaussian_scene import flops_proxy, l1_error, make_scene, psnr, ssim_global
from method.sdc2d.steepgs2d import GaussianField2D


def _inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp_min(1e-6)
    return x + torch.log(-torch.expm1(-x))


def adc_densify(
    model: GaussianField2D,
    grad_norm_accum: torch.Tensor,
    interval_count: int,
    max_points: int,
    epsilon: float,
    grad_threshold: float,
    scale_threshold: float,
    generator: torch.Generator,
) -> int:
    if model.num_points >= max_points:
        return 0
    avg_grad_norm = grad_norm_accum / max(1, interval_count)
    eligible = avg_grad_norm >= grad_threshold
    capacity = max_points - model.num_points
    if int(eligible.sum()) > capacity:
        idx = torch.where(eligible)[0]
        selected_idx = idx[torch.argsort(avg_grad_norm[idx], descending=True)[:capacity]]
    else:
        selected_idx = torch.where(eligible)[0]
    if selected_idx.numel() == 0:
        return 0

    directions = model.xy.grad.detach()[selected_idx] if model.xy.grad is not None else model.xy.detach()[selected_idx]
    directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
    parent_xy = model.xy.detach()[selected_idx]
    parent_log_scale = model.log_scale.detach()[selected_idx]
    parent_scales = torch.nn.functional.softplus(parent_log_scale) + 0.015
    large = parent_scales.max(dim=-1).values > scale_threshold

    clone_xy = torch.stack([parent_xy + epsilon * directions, parent_xy - epsilon * directions], dim=1)
    noise = torch.randn(selected_idx.numel(), 2, 2, generator=generator, device=parent_xy.device)
    split_xy = parent_xy[:, None, :] + noise * parent_scales[:, None, :]
    new_xy = torch.where(large[:, None, None], split_xy, clone_xy).reshape(-1, 2)

    shrunk_scale_param = _inverse_softplus((0.8 * parent_scales - 0.015).clamp_min(0.01))
    offspring_log_scale = torch.where(large[:, None], shrunk_scale_param, parent_log_scale)
    new_log_scale = offspring_log_scale.repeat_interleave(2, dim=0)
    new_opacity_logit = model.opacity_logit.detach()[selected_idx].repeat_interleave(2, dim=0)
    new_color_logit = model.color_logit.detach()[selected_idx].repeat_interleave(2, dim=0)
    keep = torch.ones(model.num_points, dtype=torch.bool, device=model.xy.device)
    keep[selected_idx] = False
    model.replace_points(keep, new_xy, new_log_scale, new_opacity_logit, new_color_logit)
    return int(selected_idx.numel())


def train(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    torch.manual_seed(args.seed)
    scene = make_scene(seed=args.seed, device=device, image_size=args.image_size, n_train=args.n_train, n_test=args.n_test)
    model = GaussianField2D(args.init_points, args.seed, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    grad_norm_accum = torch.zeros(args.init_points, device=device)
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
        pred, _, _, _ = model(xy)
        loss = F.l1_loss(pred, target) + 0.2 * F.mse_loss(pred, target)
        loss.backward()
        grad_norm_accum = grad_norm_accum + model.xy.grad.detach().norm(dim=-1)
        interval_count += 1
        if step >= args.densify_from and step % args.densify_interval == 0 and model.num_points < args.max_points:
            splits = adc_densify(
                model,
                grad_norm_accum,
                interval_count,
                args.max_points,
                args.epsilon,
                args.grad_threshold,
                args.scale_threshold,
                generator,
            )
            total_splits += splits
            history.append({"step": step, "points": model.num_points, "splits": splits})
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            grad_norm_accum = torch.zeros(model.num_points, device=device)
            interval_count = 0
        else:
            optimizer.step()

    with torch.no_grad():
        test_pred = model(scene.test_xy)[0]
        image_pred = model(scene.image_xy)[0]
    metrics = {
        "psnr": psnr(test_pred, scene.test_rgb),
        "ssim": ssim_global(test_pred, scene.test_rgb),
        "l1_error": l1_error(test_pred, scene.test_rgb),
        "num_points": model.num_points,
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
    p.add_argument("--output", default="outputs/adc2d")
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
    p.add_argument("--grad-threshold", type=float, default=0.001)
    p.add_argument("--scale-threshold", type=float, default=0.13)
    return p


if __name__ == "__main__":
    result = train(build_parser().parse_args())
    print(json.dumps(result, indent=2))
