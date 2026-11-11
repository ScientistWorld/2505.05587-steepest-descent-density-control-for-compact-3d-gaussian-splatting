"""Audit training variants; only method code changes, never evaluator code."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import types


def main():
    p = argparse.ArgumentParser()
    p.add_argument("variant", choices=["reference", "better_decay", "weaker_steps", "shortcut_fixed_gaussians", "shortcut_bruteforce_gaussians"])
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", required=True)
    cfg = p.parse_args()
    source = Path("method/sdc2d/steepgs2d.py").read_text()
    if cfg.variant == "better_decay":
        source = source.replace(
            "        optimizer.step()",
            "        for group in optimizer.param_groups:\n"
            "            group['lr'] = args.lr * 0.1 ** (max(0, step - 600) / 300)\n"
            "        optimizer.step()",
        )
    if cfg.variant == "shortcut_bruteforce_gaussians":
        source = source.replace(
            "    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)",
            "    with torch.no_grad():\n"
            "        axis = torch.linspace(-1.0, 1.0, 12, device=device)\n"
            "        yy, xx = torch.meshgrid(axis, axis, indexing='ij')\n"
            "        model.xy.copy_(torch.stack([xx.flatten(), yy.flatten()], dim=1))\n"
            "    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)",
            1,
        )
    module = types.ModuleType("audit_training_variant")
    exec(compile(source, "method/sdc2d/steepgs2d.py", "exec"), module.__dict__)
    # Hold observations fixed so the three seeds measure optimization variance.
    original_scene = module.make_scene
    def fixed_scene(*args, **kwargs):
        kwargs["seed"] = 7
        return original_scene(*args, **kwargs)
    module.make_scene = fixed_scene
    args = module.build_parser().parse_args([
        "--device", "cpu", "--output", cfg.output,
        "--seed", str(cfg.seed), "--max-points", "72", "--split-threshold=-2e-5",
    ])
    if cfg.variant == "weaker_steps":
        args.steps = 150
    if cfg.variant == "shortcut_fixed_gaussians":
        args.init_points = args.max_points = 72
    if cfg.variant == "shortcut_bruteforce_gaussians":
        args.init_points = args.max_points = 144
    result = module.train(args)
    path = Path(cfg.output) / "metrics.json"
    metadata = json.loads(path.read_text())
    metadata["args"]["seed"] = 7
    metadata["optimization_seed"] = cfg.seed
    metadata["audit_variant"] = cfg.variant
    path.write_text(json.dumps(metadata, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
