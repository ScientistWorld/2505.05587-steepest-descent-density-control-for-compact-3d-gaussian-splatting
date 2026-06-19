"""Score the visible train slice from saved splatting output artifacts.

This module is method-agnostic: it reads `predictions.npz` and `metrics.json`
files produced by any method, applies a deterministic half-split over rendered
samples, and writes scores without importing training or method code.
"""

from __future__ import annotations

import argparse
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np

from data.synthetic_gaussian_scene import make_scene


EXPERIMENT = "compact_synthetic_splatting"
PAPER_ADC_POINTS_M = 3.339
SLICE = "train"
SPLIT_SEED = 250505587


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


def row_l1_errors(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - target), axis=1)


def tail_l1_p95(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.quantile(row_l1_errors(pred, target), 0.95))


def bright_l1_error(pred: np.ndarray, target: np.ndarray) -> float:
    intensity = np.mean(target, axis=1)
    cutoff = np.quantile(intensity, 0.75)
    mask = intensity >= cutoff
    if not np.any(mask):
        return l1_error(pred, target)
    return l1_error(pred[mask], target[mask])


def channel_bias_l1(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.mean(pred - target, axis=0))))


def method_name_for_path(path: Path) -> str:
    if path.name == "adc2d":
        return "adc2d_baseline"
    return path.name


def default_path_for_method(name: str, outputs_root: Path) -> Path:
    if name == "adc2d_baseline":
        return outputs_root / "adc2d"
    return outputs_root / name


def parse_method_output(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return method_name_for_path(path), path
    name, raw_path = spec.split("=", 1)
    if not name:
        raise ValueError(f"Method output spec has an empty method name: {spec!r}")
    return name, Path(raw_path)


def has_output_artifact(path: Path) -> bool:
    return (path / "predictions.npz").is_file() and (path / "metrics.json").is_file()


def enumerate_outputs(args: argparse.Namespace) -> "OrderedDict[str, Path]":
    outputs: "OrderedDict[str, Path]" = OrderedDict()
    for path in (Path(args.adc_output), Path(args.steepgs_output)):
        if has_output_artifact(path):
            outputs[method_name_for_path(path)] = path

    root = Path(args.outputs_root)
    if root.is_dir():
        for path in sorted(root.iterdir()):
            if path.is_dir() and has_output_artifact(path):
                outputs.setdefault(method_name_for_path(path), path)

    for spec in args.method_output:
        name, path = parse_method_output(spec)
        outputs[name] = path

    if not outputs:
        raise SystemExit(
            "No method outputs found. Expected directories with predictions.npz "
            "and metrics.json under outputs/ or passed via --method-output NAME=PATH."
        )
    return outputs


def sample_rows(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 0:
        raise ValueError("Prediction arrays must contain at least one sample")
    if arr.ndim >= 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim > 1 and arr.shape[-1] in (1, 3, 4):
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr.reshape(arr.shape[0], -1)


def split_indices(n_samples: int) -> np.ndarray:
    if n_samples < 2:
        raise ValueError("Need at least two samples to form train/test evaluation slices")
    rng = np.random.default_rng(SPLIT_SEED)
    order = rng.permutation(n_samples)
    return np.sort(order[: n_samples // 2])


def load_run(path: Path) -> tuple[dict[str, float], dict[str, int]]:
    if not has_output_artifact(path):
        raise FileNotFoundError(f"Missing predictions.npz or metrics.json in {path}")

    metadata = json.loads((path / "metrics.json").read_text())
    with np.load(path / "predictions.npz") as arr:
        pred = sample_rows(arr["pred"])
    target = sample_rows(load_expected_target(metadata, pred.shape[0]))
    if pred.shape != target.shape:
        raise ValueError(f"Prediction and target shape mismatch in {path}: {pred.shape} != {target.shape}")

    indices = split_indices(pred.shape[0])
    pred_slice = pred[indices]
    target_slice = target[indices]

    raw_metrics = metadata.get("metrics", metadata)
    metrics = {
        "psnr": psnr(pred_slice, target_slice),
        "ssim": ssim_global(pred_slice, target_slice),
        "l1_error": l1_error(pred_slice, target_slice),
        "tail_l1_p95": tail_l1_p95(pred_slice, target_slice),
        "bright_l1_error": bright_l1_error(pred_slice, target_slice),
        "channel_bias_l1": channel_bias_l1(pred_slice, target_slice),
        "num_points": int(raw_metrics["num_points"]),
        "flops_proxy": int(raw_metrics["flops_proxy"]),
        "splits": int(raw_metrics.get("splits", 0)),
    }
    slice_info = {
        "available_samples": int(pred.shape[0]),
        "scored_samples": int(indices.shape[0]),
    }
    return metrics, slice_info


def load_expected_target(metadata: dict, n_test: int) -> np.ndarray:
    args = metadata.get("args", {})
    scene = make_scene(
        seed=int(args.get("seed", 7)),
        image_size=int(args.get("image_size", 96)),
        n_train=int(args.get("n_train", 4096)),
        n_test=int(args.get("n_test", n_test)),
        device="cpu",
    )
    target = scene.test_rgb.cpu().numpy()
    if target.shape[0] != n_test:
        raise ValueError(f"Prediction sample count {n_test} does not match benchmark target count {target.shape[0]}")
    return target


def normalize_cost_metrics(results: "OrderedDict[str, dict[str, float]]") -> None:
    if "adc2d_baseline" in results:
        anchor = results["adc2d_baseline"]
    else:
        anchor = next(iter(results.values()))

    adc_raw_points = max(float(anchor["num_points"]), 1.0)
    adc_raw_flops = max(float(anchor["flops_proxy"]), 1.0)
    for run in results.values():
        run["point_reduction_vs_adc"] = float(1.0 - float(run["num_points"]) / adc_raw_points)
        run["num_points"] = float(float(run["num_points"]) / adc_raw_points * PAPER_ADC_POINTS_M)
        run["flops_proxy"] = float(float(run["flops_proxy"]) / adc_raw_flops * PAPER_ADC_POINTS_M)
    if "adc2d_baseline" in results:
        results["adc2d_baseline"]["point_reduction_vs_adc"] = 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steepgs-output", default="outputs/steepgs2d")
    parser.add_argument("--adc-output", default="outputs/adc2d")
    parser.add_argument("--outputs-root", default="outputs")
    parser.add_argument(
        "--method-output",
        action="append",
        default=[],
        help="Additional method output as NAME=PATH, or PATH to use the directory name.",
    )
    parser.add_argument("--scores", default="scoring/scores_train.json")
    parser.add_argument("--manifest", default="scoring/evaluation_manifest_train.json")
    args = parser.parse_args()

    outputs = enumerate_outputs(args)
    results: "OrderedDict[str, dict[str, float]]" = OrderedDict()
    manifest_entries = []
    for method_name, path in outputs.items():
        metrics, slice_info = load_run(path)
        results[method_name] = metrics
        manifest_entries.append(
            {
                "slice": SLICE,
                "experiment": EXPERIMENT,
                "method": method_name,
                "path": str(path),
                **slice_info,
            }
        )

    normalize_cost_metrics(results)

    scores = {
        "metadata": {"slice": SLICE, "split_seed": SPLIT_SEED},
        "experiments": {EXPERIMENT: {"slice": SLICE, "results": results}},
    }
    scores_path = Path(args.scores)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores, indent=2))

    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "slice": SLICE,
                "split_seed": SPLIT_SEED,
                "scores": str(scores_path),
                "entries": manifest_entries,
            },
            indent=2,
        )
    )

    print(json.dumps(scores["experiments"], indent=2))


if __name__ == "__main__":
    main()
