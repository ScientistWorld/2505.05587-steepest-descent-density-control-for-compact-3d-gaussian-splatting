"""Select a compact SDC run from already-generated output artifacts.

This script does not run training or import method code. It reads saved
predictions/metadata from several candidate runs, picks the best compact
candidate relative to the ADC baseline, and copies it to the standard
`outputs/steepgs2d` path consumed by `scripts/evaluate.sh`.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from eval.evaluate_outputs import load_run


def score_candidate(candidate: dict[str, float], adc: dict[str, float], psnr_tolerance: float) -> float:
    reduction = 1.0 - candidate["num_points"] / max(adc["num_points"], 1)
    psnr_gap = adc["psnr"] - candidate["psnr"]
    compact_bonus = 8.0 * reduction
    quality_penalty = 3.0 * max(0.0, psnr_gap - psnr_tolerance)
    return candidate["psnr"] + compact_bonus - quality_penalty


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adc-output", default="outputs/adc2d")
    parser.add_argument("--candidates-dir", default="outputs/sdc_sweep")
    parser.add_argument("--selected-output", default="outputs/steepgs2d")
    parser.add_argument("--summary", default="outputs/sdc_sweep/selection.json")
    parser.add_argument("--psnr-tolerance", type=float, default=0.75)
    args = parser.parse_args()

    adc = load_run(Path(args.adc_output))
    rows = []
    for path in sorted(Path(args.candidates_dir).glob("*/predictions.npz")):
        run_dir = path.parent
        metrics = load_run(run_dir)
        reduction = 1.0 - metrics["num_points"] / max(adc["num_points"], 1)
        rows.append(
            {
                "path": str(run_dir),
                "psnr": metrics["psnr"],
                "ssim": metrics["ssim"],
                "l1_error": metrics["l1_error"],
                "num_points": metrics["num_points"],
                "point_reduction_vs_adc": reduction,
                "score": score_candidate(metrics, adc, args.psnr_tolerance),
            }
        )

    if not rows:
        raise SystemExit(f"No SDC candidate outputs found under {args.candidates_dir}")

    rows.sort(key=lambda row: row["score"], reverse=True)
    selected = rows[0]
    selected_path = Path(selected["path"])
    output_path = Path(args.selected_output)
    if output_path.exists():
        shutil.rmtree(output_path)
    shutil.copytree(selected_path, output_path)

    summary_path = Path(args.summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {"adc": adc, "selected": selected, "candidates": rows}
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
