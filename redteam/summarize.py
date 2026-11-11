"""Lightweight analysis of saved test JSON, with explicit ranking populations.

This never supplies values to the evaluator. It implements the audit request's
direction, absolute-coefficient, experiment-weight, and variance normalization.
Future attempts' population is unknown, so these ranks are audit measurements,
not claims about an unseen production leaderboard.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
EXPERIMENT = "compact_synthetic_splatting"


def load_json(path):
    return json.loads(path.read_text())


def main():
    config = load_json(HERE / "ranking_config.json")
    metrics = config["metrics"]
    cases = load_json(HERE / "cases.json")
    records = {}
    for name in cases:
        folder = HERE / name
        if not (folder / "result.json").exists():
            continue
        details = load_json(folder / "result.json")
        scores = folder / "scores_test.json"
        rows = load_json(scores)["experiments"][EXPERIMENT]["results"] if scores.exists() else {}
        records[name] = {"details": details, "results": rows}

    reference = records["reference_original"]["results"]["steepgs2d"]
    def finite(row):
        return all(isinstance(row.get(k), (int, float)) and math.isfinite(row[k]) for k in metrics)
    def deviations(population):
        return {k: statistics.pstdev(row[k] for row in population) for k in metrics}
    def rank(row, scales):
        return config["weight"] * sum(
            abs(spec["coefficient"]) * (1 if spec["higher_is_better"] else -1)
            * (row[k] - reference[k]) / scales[k]
            for k, spec in metrics.items() if scales[k] > 0
        )

    control_names = ["reference_original"] + [
        f"{variant}_seed{seed}"
        for variant in ["reference", "better_decay", "weaker_steps"]
        for seed in [7, 13, 23]
    ]
    baseline = records["reference_original"]["results"]["adc2d_baseline"]
    controls = [records[name]["results"]["steepgs2d"] for name in control_names] + [baseline]
    fixed_scales = deviations(controls)
    # One row per method per actual attempt; include ADC once, not 25 copies.
    pooled = [baseline] + [
        row for record in records.values() for method, row in record["results"].items()
        if method != "adc2d_baseline" and finite(row)
    ]
    pooled_scales = deviations(pooled)
    summary = {
        "formula": "sum(weight * abs(coefficient) * direction * (value - reference_value) / population_stddev)",
        "reference": reference,
        "control_cohort": control_names + ["adc2d_baseline"],
        "control_scales": fixed_scales,
        "pooled_finite_method_rows": len(pooled),
        "pooled_scales": pooled_scales,
        "baseline_control_delta_rank": rank(baseline, fixed_scales),
        "cases": {},
    }
    for name, record in records.items():
        row = record["results"].get("steepgs2d")
        outcome = {
            "test_exit": record["details"]["test_exit"],
            "scored_methods": list(record["results"]),
            "candidate_present": row is not None,
        }
        if row is not None and finite(row):
            extra = [value for method, value in record["results"].items() if method != "adc2d_baseline" and finite(value)]
            case_scales = deviations(controls + extra)
            outcome.update(
                metrics=row,
                control_delta_rank=rank(row, fixed_scales),
                case_cohort_delta_rank=rank(row, case_scales),
                pooled_delta_rank=rank(row, pooled_scales),
            )
        elif row is not None:
            outcome["nonfinite_metrics"] = [k for k in metrics if not math.isfinite(row[k])]
        log_path = HERE / name / "evaluation.log"
        if outcome["test_exit"] != 0 and log_path.exists():
            outcome["failure"] = log_path.read_text().splitlines()[-1]
        summary["cases"][name] = outcome
        (HERE / name / "audit_rank.json").write_text(json.dumps(outcome, indent=2, allow_nan=False) + "\n")

    refs = [summary["cases"][f"reference_seed{s}"] for s in [7, 13, 23]]
    summary["reference_seed_spread"] = {
        "optimization_seeds": [7, 13, 23], "scene_seed": 7,
        "metric_ranges": {k: max(r["metrics"][k] for r in refs) - min(r["metrics"][k] for r in refs) for k in metrics},
        "control_delta_rank_range": max(r["control_delta_rank"] for r in refs) - min(r["control_delta_rank"] for r in refs),
        "pooled_delta_rank_range": max(r["pooled_delta_rank"] for r in refs) - min(r["pooled_delta_rank"] for r in refs),
    }
    (HERE / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    for name, row in summary["cases"].items():
        print(name, json.dumps({k: row[k] for k in ["test_exit", "control_delta_rank", "case_cohort_delta_rank", "pooled_delta_rank"] if k in row}))


if __name__ == "__main__":
    main()
