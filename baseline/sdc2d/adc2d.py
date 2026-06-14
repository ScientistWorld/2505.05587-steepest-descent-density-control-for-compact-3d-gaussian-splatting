"""Original-ADC-style heuristic baseline for the 2D splatting benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.synthetic_gaussian_scene import flops_proxy, l1_error, make_scene, psnr, ssim_global
from method.sdc2d.steepgs2d import GaussianField2D, _opacity_halved_logits


def adc_densify(model: GaussianField2D, max_points: int, epsilon: float) -> int:
    if model.xy.grad is None or model.num_points >= max_points:
        return 0
    grad_norm = model.xy.grad.detach().norm(dim=-1)
    capacity = max_points - model.num_points
    k = min(capacity, max(1, model.num_points // 4))
    selected_idx = torch.argsort(grad_norm, descending=True)[:k]
    directions = model.xy.grad.detach()[selected_idx]
    directions = directions / (directions.norm(dim=-1, keepdim=True) + 1e-8)
    parent_xy = model.xy.detach()[selected_idx]
    new_xy = torch.cat([parent_xy + epsilon * directions, parent_xy - epsilon * directions], dim=0)
    new_log_scale = model.log_scale.detach()[selected_idx].repeat(2, 1)
    new_opacity_logit = _opacity_halved_logits(model.opacity_logit.detach()[selected_idx]).repeat(2, 1)
    new_color_logit = model.color_logit.detach()[selected_idx].repeat(2, 1)
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
        if step >= args.densify_from and step % args.densify_interval == 0 and model.num_points < args.max_points:
            splits = adc_densify(model, args.max_points, args.epsilon)
            total_splits += splits
            history.append({"step": step, "points": model.num_points, "splits": splits})
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
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
    return p


if __name__ == "__main__":
    result = train(build_parser().parse_args())
    print(json.dumps(result, indent=2))
