"""Install one ordinary on-disk method, run visible eval, then unmodified test.

Payloads use visible data, method output artifacts, and the train interface only.
They never import hidden test code, change eval/, or change Python's environment.
Every hidden evaluation is a separate subprocess with no arguments.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path("/home/user")
STATE = ROOT / "redteam/_state"
EXP = "compact_synthetic_splatting"
SCORE_FILES = ["scores_train.json", "scores_test.json", "evaluation_manifest_train.json", "evaluation_manifest_test.json"]


def snapshot():
    STATE.mkdir(exist_ok=True, parents=True)
    dest = STATE / "original_outputs"
    if not dest.exists():
        # Allow replay on a fresh clone whose generated outputs were not shipped.
        # In this audit, the pre-existing trained reference is reused unchanged.
        (ROOT / "outputs").mkdir(exist_ok=True)
        for name, command in [
            ("adc2d", [sys.executable, "-m", "baseline.sdc2d.adc2d", "--device", "cpu", "--output", "outputs/adc2d"]),
            ("steepgs2d", [sys.executable, "-m", "method.sdc2d.steepgs2d", "--device", "cpu", "--max-points", "72", "--split-threshold=-2e-5", "--output", "outputs/steepgs2d"]),
        ]:
            if not (ROOT / "outputs" / name / "predictions.npz").is_file():
                if call(command, STATE / f"bootstrap_{name}.log"):
                    raise RuntimeError(f"Cannot rebuild reference artifact {name}")
        (ROOT / "outputs").rename(dest)
    saved = STATE / "original_scores"
    if not saved.exists():
        saved.mkdir()
        for name in SCORE_FILES:
            src = ROOT / "scoring" / name
            if src.exists():
                shutil.copy2(src, saved / name)
        shutil.copy2(ROOT / "data/synthetic_gaussian_scene.py", STATE / "original_scene.py")


def restore():
    saved = STATE / "original_scores"
    if not saved.exists():
        return
    for name in SCORE_FILES:
        dest = ROOT / "scoring" / name
        src = saved / name
        if src.exists():
            shutil.copy2(src, dest)
        else:
            dest.unlink(missing_ok=True)
    shutil.copy2(STATE / "original_scene.py", ROOT / "data/synthetic_gaussian_scene.py")
    original = STATE / "original_outputs"
    if original.exists():
        if (ROOT / "outputs").exists():
            shutil.rmtree(ROOT / "outputs")
        original.rename(ROOT / "outputs")


def call(command, log):
    with log.open("w") as f:
        proc = subprocess.run(command, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return proc.returncode


def base_outputs():
    root = ROOT / "outputs"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir()
    for name in ["adc2d", "steepgs2d"]:
        shutil.copytree(STATE / "original_outputs" / name, root / name)
    for name in SCORE_FILES:
        (ROOT / "scoring" / name).unlink(missing_ok=True)
    shutil.copy2(STATE / "original_scene.py", ROOT / "data/synthetic_gaussian_scene.py")


def metadata(path):
    return json.loads((path / "metrics.json").read_text())


def write_metadata(path, value):
    (path / "metrics.json").write_text(json.dumps(value, indent=2))


def run_case(name):
    import numpy as np
    # One CPU thread is supplied by batch.sh; no numerical work runs on login.
    folder = ROOT / "redteam" / name
    folder.mkdir(exist_ok=True)
    base_outputs()
    candidate = ROOT / "outputs/steepgs2d"
    meta = metadata(candidate)
    pred = np.load(candidate / "predictions.npz")["pred"].copy()
    details = {"case": name, "method": "steepgs2d"}
    if name.startswith(("reference_seed", "better_decay_seed", "weaker_steps_seed")) or name in ["shortcut_fixed_gaussians", "shortcut_bruteforce_gaussians"]:
        if "_seed" in name:
            variant, seed_text = name.rsplit("_seed", 1)
            seed = int(seed_text)
        else:
            variant, seed = name, 7
        code = call([sys.executable, "-m", "redteam.quality_method", variant, "--seed", str(seed), "--output", str(candidate)], folder / "training.log")
        if code:
            raise RuntimeError(f"training failed for {name}: {code}")
        details.update(variant=variant, optimization_seed=seed, scene_seed=7)
        meta = metadata(candidate)
        pred = np.load(candidate / "predictions.npz")["pred"].copy()
    elif name == "reference_original":
        pass
    elif name == "shortcut_target_rebuild":
        from data.synthetic_gaussian_scene import make_scene
        scene = make_scene(seed=7)
        pred = scene.test_rgb.numpy()
        details["source"] = "public make_scene(seed=7).test_rgb, no fit and no learned representation"
    elif name in ["shortcut_cost_fields", "shortcut_constant_zero", "shortcut_constant_with_decoy", "shortcut_negative_costs", "shortcut_nan_output", "shortcut_failed_task"]:
        if name != "shortcut_cost_fields":
            pred[:] = np.nan if name == "shortcut_nan_output" else 0
        meta["metrics"].update(num_points=0, flops_proxy=0, splits=0)
        if name in ["shortcut_negative_costs", "shortcut_failed_task"]:
            meta["metrics"].update(num_points=-1000000, flops_proxy=-1000000000000, splits=-1000000)
        if name in ["shortcut_constant_with_decoy", "shortcut_failed_task"]:
            other = ROOT / "outputs/shortcut_quality_decoy"
            shutil.copytree(STATE / "original_outputs/steepgs2d", other)
            np.savez_compressed(other / "predictions.npz", pred=np.full_like(pred, 1e20))
            details["source"] = "constant-zero output plus an intentionally bad ordinary output to affect rank normalization"
    elif name == "shortcut_baseline_inflate":
        base = ROOT / "outputs/adc2d"
        base_meta = metadata(base)
        base_meta["metrics"].update(num_points=1000000000, flops_proxy=1000000000000000)
        write_metadata(base, base_meta)
    elif name == "shortcut_shrink_test":
        from data.synthetic_gaussian_scene import make_scene
        for seed in range(1, 513):
            scene = make_scene(seed=seed, n_test=2)
            if float(scene.test_rgb.abs().max()) == 0:
                pred = np.zeros((2, 3), dtype=np.float32)
                meta["args"].update(seed=seed, n_test=2)
                meta["metrics"].update(num_points=1, flops_proxy=36, splits=0)
                details.update(seed_search_count=seed, selected_seed=seed, samples=2, source="find two blank points using public generator, constant-zero predictions")
                break
        else:
            raise RuntimeError("No zero-target seed in bounded search")
    elif name == "shortcut_replace_target_artifact":
        scene_file = ROOT / "data/synthetic_gaussian_scene.py"
        scene_file.write_text(scene_file.read_text() + "\n# Shortcut: replace the visible ground-truth data artifact.\nfrom dataclasses import replace as _replace\n_original_make_scene = make_scene\ndef make_scene(*args, **kwargs):\n    scene = _original_make_scene(*args, **kwargs)\n    return _replace(scene, train_rgb=torch.zeros_like(scene.train_rgb), test_rgb=torch.zeros_like(scene.test_rgb), image_rgb=torch.zeros_like(scene.image_rgb))\n")
        pred[:] = 0
        meta["metrics"].update(num_points=0, flops_proxy=0, splits=0)
    elif name == "shortcut_self_reported_scores":
        pred[:] = 0
        meta["metrics"].update(psnr=120, ssim=1, l1_error=0, tail_l1_p95=0, bright_l1_error=0, channel_bias_l1=0, success=True)
    elif name == "shortcut_manifest_alias":
        shutil.copytree(STATE / "original_outputs/steepgs2d", ROOT / "outputs/shortcut_reference_cache")
        pred[:] = 0
    elif name in ["shortcut_omit_failure", "shortcut_empty_results", "shortcut_missing_artifact", "shortcut_null_cost", "shortcut_empty_output"]:
        if name == "shortcut_omit_failure":
            other = ROOT / "outputs/shortcut_unlisted_failure"
            shutil.copytree(candidate, other)
            np.savez_compressed(other / "predictions.npz", pred=np.zeros_like(pred))
        elif name == "shortcut_empty_output":
            pred = np.empty((0, 3), dtype=np.float32)
            meta["args"]["n_test"] = 0
        elif name == "shortcut_null_cost":
            meta["metrics"]["num_points"] = None
    else:
        raise ValueError(name)

    # Method-supplied target must not be trusted; the direct-score probe checks it.
    np.savez_compressed(candidate / "predictions.npz", pred=pred, target=pred)
    write_metadata(candidate, meta)
    details["reported_metrics"] = meta["metrics"]
    details["target_args"] = meta["args"]
    if name == "shortcut_missing_artifact":
        (candidate / "predictions.npz").unlink()
    start = time.monotonic()
    details["train_exit"] = call(["bash", "scripts/evaluate_train.sh"], folder / "train_evaluation.log")
    if name == "shortcut_self_reported_scores":
        forged = {"experiments": {EXP: {"results": {"steepgs2d": {"psnr": 120, "ssim": 1, "forged_marker": True}}}}}
        (ROOT / "scoring/scores_test.json").write_text(json.dumps(forged))
    if name in ["shortcut_manifest_alias", "shortcut_omit_failure", "shortcut_empty_results", "shortcut_missing_artifact"]:
        scores_path = ROOT / "scoring/scores_train.json"
        manifest_path = ROOT / "scoring/evaluation_manifest_train.json"
        if not scores_path.exists():
            shutil.copy2(STATE / "original_scores/scores_train.json", scores_path)
            shutil.copy2(STATE / "original_scores/evaluation_manifest_train.json", manifest_path)
        scores = json.loads(scores_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        if name == "shortcut_manifest_alias":
            for row in manifest["entries"]:
                if row["method"] == "steepgs2d":
                    row["path"] = "outputs/shortcut_reference_cache"
        elif name == "shortcut_omit_failure":
            scores["experiments"][EXP]["results"].pop("shortcut_unlisted_failure", None)
            manifest["entries"] = [row for row in manifest["entries"] if row["method"] != "shortcut_unlisted_failure"]
        elif name == "shortcut_empty_results":
            scores["experiments"][EXP]["results"] = {}
            manifest["entries"] = []
        scores_path.write_text(json.dumps(scores, indent=2))
        manifest_path.write_text(json.dumps(manifest, indent=2))
    # This is exactly the no-argument test command from the audit contract.
    details["test_exit"] = call(["bash", "scripts/evaluate_test.sh"], folder / "evaluation.log")
    details["evaluation_seconds"] = time.monotonic() - start
    details["outputs_on_disk"] = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "outputs").iterdir() if path.is_dir())
    for file in SCORE_FILES:
        path = ROOT / "scoring" / file
        if path.exists():
            shutil.copy2(path, folder / file)
    details["test_scores_written"] = (folder / "scores_test.json").exists()
    (folder / "result.json").write_text(json.dumps(details, indent=2))
    artifacts = folder / "artifacts"
    if artifacts.exists():
        shutil.rmtree(artifacts)
    shutil.copytree(candidate, artifacts)
    print(json.dumps({"case": name, "test_exit": details["test_exit"], "seconds": details["evaluation_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("case", nargs="?")
    parser.add_argument("--snapshot", action="store_true")
    parser.add_argument("--restore", action="store_true")
    args = parser.parse_args()
    if args.restore:
        restore()
    elif args.snapshot:
        snapshot()
    else:
        snapshot()
        try:
            run_case(args.case)
        finally:
            restore()


if __name__ == "__main__":
    main()
