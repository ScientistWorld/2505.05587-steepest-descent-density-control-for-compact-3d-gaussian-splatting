"""Evaluate saved splatting predictions without importing method code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def psnr(pred: np.ndarray, target: np.ndarray) -> float:
    mse = np.mean((pred - target) ** 2)
    return float(10.0 * np.log10(1.0 / max(mse, 1e-12)))


def ssim_global(pred: np.ndarray, target: np.ndarray) -> float:
    x = pred.reshape(-1)
    y = target.reshape(-1)
    c1 = 0.01**2
    c2 = 0.03**2
    mux, muy = x.mean(), y.mean()
    vx = ((x - mux) ** 2).mean()
    vy = ((y - muy) ** 2).mean()
    cov = ((x - mux) * (y - muy)).mean()
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux**2 + muy**2 + c1) * (vx + vy + c2)))


def l1_error(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - target)))


def load_run(path: Path) -> dict[str, float]:
    with np.load(path / "predictions.npz") as arr:
        pred = arr["pred"]
        target = arr["target"]
    metadata = json.loads((path / "metrics.json").read_text())
    metrics = metadata["metrics"]
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim_global(pred, target),
        "l1_error": l1_error(pred, target),
        "num_points": int(metrics["num_points"]),
        "flops_proxy": int(metrics["flops_proxy"]),
        "splits": int(metrics["splits"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steepgs-output", default="outputs/steepgs2d")
    parser.add_argument("--adc-output", default="outputs/adc2d")
    parser.add_argument("--scores", default="scoring/scores.json")
    args = parser.parse_args()

    steep = load_run(Path(args.steepgs_output))
    adc = load_run(Path(args.adc_output))
    steep["point_reduction_vs_adc"] = float(1.0 - steep["num_points"] / adc["num_points"])
    adc["point_reduction_vs_adc"] = 0.0
    scores = {
        "compact_synthetic_splatting": {
            "results": {
                "adc2d_baseline": adc,
                "steepgs2d": steep,
            }
        }
    }
    Path(args.scores).write_text(json.dumps({"experiments": scores}, indent=2))
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
