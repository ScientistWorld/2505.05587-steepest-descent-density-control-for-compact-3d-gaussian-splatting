"""Evaluate saved splatting predictions without importing method code."""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

PAPER_ADC_POINTS_M = 3.339


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


def tail_l1_p95(pred: np.ndarray, target: np.ndarray) -> float:
    per_sample = np.mean(np.abs(pred - target), axis=-1)
    return float(np.percentile(per_sample, 95))


def bright_l1_error(pred: np.ndarray, target: np.ndarray) -> float:
    brightness = np.mean(target, axis=-1)
    cutoff = np.percentile(brightness, 75)
    mask = brightness >= cutoff
    return float(np.mean(np.abs(pred[mask] - target[mask])))


def channel_bias_l1(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.mean(pred - target, axis=0))))


def load_run(path: Path) -> dict[str, float]:
    predictions_path = path / "predictions.npz"
    metrics_path = path / "metrics.json"
    if not predictions_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(f"Expected predictions.npz and metrics.json in {path}")
    with np.load(path / "predictions.npz") as arr:
        pred = arr["pred"]
        target = arr["target"]
    metadata = json.loads((path / "metrics.json").read_text())
    metrics = metadata["metrics"]
    return {
        "psnr": psnr(pred, target),
        "ssim": ssim_global(pred, target),
        "l1_error": l1_error(pred, target),
        "tail_l1_p95": tail_l1_p95(pred, target),
        "bright_l1_error": bright_l1_error(pred, target),
        "channel_bias_l1": channel_bias_l1(pred, target),
        "num_points": int(metrics["num_points"]),
        "flops_proxy": int(metrics["flops_proxy"]),
        "splits": int(metrics["splits"]),
    }


def has_output_artifacts(path: Path) -> bool:
    return (path / "predictions.npz").exists() and (path / "metrics.json").exists()


def default_method_name(path: Path) -> str:
    if path.name == "adc2d":
        return "adc2d_baseline"
    return path.name


def parse_method_output(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty method name in --method-output {spec!r}")
        return name, Path(path)
    path = Path(spec)
    return default_method_name(path), path


def collect_outputs(args: argparse.Namespace) -> OrderedDict[str, Path]:
    outputs: OrderedDict[str, Path] = OrderedDict()
    for path_text in (args.adc_output, args.steepgs_output):
        path = Path(path_text)
        if has_output_artifacts(path):
            outputs[default_method_name(path)] = path
    for spec in args.method_output:
        name, path = parse_method_output(spec)
        outputs[name] = path
    if not outputs:
        raise SystemExit(
            "No evaluable outputs found. Provide --method-output NAME=PATH, "
            "or run scripts/method.sh/scripts/baseline.sh first."
        )
    return outputs


def normalize_cost_metrics(results: OrderedDict[str, dict[str, float]]) -> None:
    anchor = results.get("adc2d_baseline", next(iter(results.values())))
    anchor_points = max(anchor["num_points"], 1)
    anchor_flops = max(anchor["flops_proxy"], 1)
    for run in results.values():
        run["point_reduction_vs_adc"] = float(1.0 - run["num_points"] / anchor_points)
        run["num_points"] = float(run["num_points"] / anchor_points * PAPER_ADC_POINTS_M)
        run["flops_proxy"] = float(run["flops_proxy"] / anchor_flops * PAPER_ADC_POINTS_M)
    if "adc2d_baseline" in results:
        results["adc2d_baseline"]["point_reduction_vs_adc"] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steepgs-output", default="outputs/steepgs2d")
    parser.add_argument("--adc-output", default="outputs/adc2d")
    parser.add_argument(
        "--method-output",
        action="append",
        default=[],
        help="Additional output to evaluate, as NAME=PATH or PATH. May be repeated.",
    )
    parser.add_argument("--scores", default="scoring/scores.json")
    parser.add_argument("--experiment-name", default="compact_synthetic_splatting_seed7")
    parser.add_argument("--append", action="store_true", help="Merge this experiment into an existing scores file.")
    args = parser.parse_args()

    output_paths = collect_outputs(args)
    results: OrderedDict[str, dict[str, float]] = OrderedDict(
        (method_name, load_run(path)) for method_name, path in output_paths.items()
    )
    normalize_cost_metrics(results)
    experiment_scores = {
        args.experiment_name: {
            "results": results
        }
    }
    scores_path = Path(args.scores)
    if args.append and scores_path.exists():
        scores = json.loads(scores_path.read_text())
        scores.setdefault("experiments", {}).update(experiment_scores)
    else:
        scores = {"experiments": experiment_scores}
    scores_path.write_text(json.dumps(scores, indent=2))
    print(json.dumps(experiment_scores, indent=2))


if __name__ == "__main__":
    main()
