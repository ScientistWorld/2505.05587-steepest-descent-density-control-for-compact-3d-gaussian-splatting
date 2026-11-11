"""Train-side (validation) evaluator for the compact view-synthesis benchmark.

Every immediate sub-directory of the outputs root is a submitted method. For
each benchmark scene the method must provide ``<method>/<scene>/point_cloud.ply``
(a 3D Gaussian splatting representation). The evaluator

1. parses the PLY itself and counts its Gaussians (nothing the method writes
   about itself is read);
2. re-renders it with its own renderer at the validation cameras, whose
   poses live in ``views.json`` next to this file;
3. compares against ground-truth images that only exist under ``eval/train/gt``
   (sha256-verified) with PSNR / SSIM / LPIPS-VGG;
4. applies the evaluator-fixed feasibility rules of ``protocol.json``.

A run that is missing a scene, has an unreadable or non-finite representation,
exceeds the Gaussian budget, or falls below the reconstruction-quality floor on
any scene is FAILED and receives the worst value of every scored metric (the
feasibility bound), so failing can never score better than any feasible run.

Usage: python -m eval.train.evaluate_outputs [--outputs-root outputs]
                                             [--out scoring/scores_train.json]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch  # noqa: E402

from .gs_render import Camera, InvalidSubmission, read_gaussian_ply, render
from .image_metrics import LPIPSVGG, file_sha256, load_gt, psnr, quantize, ssim

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT = "compact_view_synthesis"
METRICS = ("psnr", "ssim", "lpips", "num_points_ratio")


def _load_json(name):
    with open(os.path.join(HERE, name)) as f:
        return json.load(f)


def _bounds(protocol):
    """Per-scene feasibility rules derived from the fixed protocol constants."""
    out = {}
    for scene, c in protocol["scenes"].items():
        out[scene] = {
            "max_points": int(math.floor(protocol["point_cap_fraction_of_adc"] * c["adc_points"])),
            "min_psnr": c["adc_psnr"] - protocol["psnr_margin_db"],
            "min_ssim": c["adc_ssim"] - protocol["ssim_margin"],
            "max_lpips": c["adc_lpips"] + protocol["lpips_margin"],
            "adc_points": c["adc_points"],
        }
    return out


def worst_values(protocol):
    b = _bounds(protocol)
    n = len(b)
    return {
        "psnr": sum(v["min_psnr"] for v in b.values()) / n,
        "ssim": sum(v["min_ssim"] for v in b.values()) / n,
        "lpips": sum(v["max_lpips"] for v in b.values()) / n,
        "num_points_ratio": float(protocol["point_cap_fraction_of_adc"]),
    }


def _integrity(views, protocol):
    """Return (problem or None). Missing files abort; modified files fail every run."""
    for scene, vs in views["scenes"].items():
        for v in vs:
            p = os.path.join(HERE, "gt", scene, v["name"])
            if not os.path.exists(p):
                raise SystemExit(f"missing ground truth {p}; run scripts/download.sh first")
            if file_sha256(p) != v["sha256"]:
                return f"ground-truth file {scene}/{v['name']} does not match its recorded hash"
    for key, spec in protocol["lpips_weights"].items():
        p = os.path.join(HERE, spec["file"])
        if not os.path.exists(p):
            raise SystemExit(f"missing LPIPS weights {p}; run scripts/download.sh first")
        if file_sha256(p) != spec["sha256"]:
            return f"LPIPS weight file {spec['file']} does not match its recorded hash"
    return None


def _pick_device(requested):
    """Requested device, else the visible GPU with the most free memory, else CPU."""
    if requested:
        return torch.device(requested)
    if not torch.cuda.is_available():
        return torch.device("cpu")
    best = 0
    try:
        import subprocess

        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True, timeout=30).stdout
        rows = [tuple(int(x) for x in line.split(",")) for line in q.strip().splitlines()]
        if len(rows) == torch.cuda.device_count():
            best = max(rows, key=lambda r: (r[1], -r[0]))[0]
    except Exception:
        best = 0
    return torch.device(f"cuda:{best}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--outputs-root", default="outputs")
    ap.add_argument("--out", default="scoring/scores_train.json")
    ap.add_argument("--raw", action="store_true",
                    help="report raw per-scene metrics without feasibility rules (protocol calibration only)")
    ap.add_argument("--device", default=None)
    args = ap.parse_args(argv)

    t0 = time.time()
    views = _load_json("views.json")
    protocol = _load_json("protocol.json")
    calibrated = all(c.get("adc_points") for c in protocol["scenes"].values())
    if not args.raw and not calibrated:
        raise SystemExit("protocol.json is not calibrated")
    device = _pick_device(args.device)
    factor = int(views["downsample"])
    scenes = list(protocol["scenes"].keys())
    bounds = _bounds(protocol) if calibrated else None
    integrity_problem = _integrity(views, protocol)

    root = args.outputs_root
    methods = sorted(d for d in os.listdir(root) if not d.startswith(".") and os.path.isdir(os.path.join(root, d))) \
        if os.path.isdir(root) else []
    print(f"[eval/train] device={device} methods={methods}", flush=True)

    diag = {m: {"status": "ok", "reasons": [], "per_scene": {}} for m in methods}

    def fail(m, reason):
        if diag[m]["status"] != "failed":
            diag[m]["status"] = "failed"
        diag[m]["reasons"].append(reason)

    if integrity_problem:
        for m in methods:
            fail(m, integrity_problem)

    lp = None
    if methods and not integrity_problem:
        lp = LPIPSVGG(os.path.join(HERE, protocol["lpips_weights"]["vgg16"]["file"]),
                      os.path.join(HERE, protocol["lpips_weights"]["lin"]["file"])).to(device)

    # Pass 1: every scene file of every method is parsed and checked before any
    # rendering, so structurally invalid or over-budget runs fail cheaply.
    for m in methods:
        for scene in scenes:
            if diag[m]["status"] == "failed" and not args.raw:
                break
            path = os.path.join(root, m, scene, "point_cloud.ply")
            if not os.path.isfile(path):
                fail(m, f"{scene}: missing point_cloud.ply")
                continue
            try:
                n_pts = read_gaussian_ply(path, int(protocol["max_ply_points"])).num_points
            except InvalidSubmission as exc:
                fail(m, f"{scene}: {exc}")
                continue
            diag[m]["per_scene"][scene] = {"num_points": n_pts}
            if bounds and not args.raw and n_pts > bounds[scene]["max_points"]:
                fail(m, f"{scene}: {n_pts} Gaussians exceeds the budget {bounds[scene]['max_points']}")

    # Pass 2: render and score the remaining runs scene by scene.
    for scene in scenes:
        live = {}
        for m in methods:
            if diag[m]["status"] == "failed" and not args.raw:
                continue
            if scene not in diag[m]["per_scene"]:
                continue
            path = os.path.join(root, m, scene, "point_cloud.ply")
            try:
                cloud = read_gaussian_ply(path, int(protocol["max_ply_points"]))
            except InvalidSubmission as exc:
                fail(m, f"{scene}: {exc}")
                continue
            if cloud.num_points != diag[m]["per_scene"][scene]["num_points"]:
                fail(m, f"{scene}: point_cloud.ply changed during evaluation")
                continue
            live[m] = cloud.to(device)
        if not live or lp is None:
            continue
        sums = {m: [0.0, 0.0, 0.0] for m in live}
        for v in views["scenes"][scene]:
            gt_np = load_gt(os.path.join(HERE, "gt", scene, v["name"]), factor)
            gt = torch.from_numpy(gt_np).to(device).float() / 255.0
            cam = Camera(v["R"], v["T"], v["fovx"], v["fovy"], gt.shape[1], gt.shape[0], device)
            gt_feats = lp.features(gt)
            for m in list(live):
                try:
                    img = render(live[m], cam, int(protocol["max_pairs_per_view"]))
                except InvalidSubmission as exc:
                    fail(m, f"{scene}/{v['name']}: {exc}")
                    del live[m]
                    continue
                if not torch.isfinite(img).all():
                    fail(m, f"{scene}/{v['name']}: non-finite render")
                    del live[m]
                    continue
                img = quantize(img)
                sums[m][0] += psnr(img, gt)
                sums[m][1] += ssim(img, gt)
                sums[m][2] += lp.distance(lp.features(img), gt_feats)
            del gt_feats
        nv = len(views["scenes"][scene])
        for m in live:
            ps = diag[m]["per_scene"][scene]
            ps["psnr"], ps["ssim"], ps["lpips"] = (s / nv for s in sums[m])
            if bounds:
                ps["num_points_ratio"] = ps["num_points"] / bounds[scene]["adc_points"]
            if bounds and not args.raw:
                b = bounds[scene]
                if not (ps["psnr"] >= b["min_psnr"]):
                    fail(m, f"{scene}: PSNR {ps['psnr']:.3f} below the floor {b['min_psnr']:.3f}")
                if not (ps["ssim"] >= b["min_ssim"]):
                    fail(m, f"{scene}: SSIM {ps['ssim']:.4f} below the floor {b['min_ssim']:.4f}")
                if not (ps["lpips"] <= b["max_lpips"]):
                    fail(m, f"{scene}: LPIPS {ps['lpips']:.4f} above the ceiling {b['max_lpips']:.4f}")
        print(f"[eval/train] {scene} done ({time.time() - t0:.0f}s)", flush=True)
        del live

    results = {}
    worst = worst_values(protocol) if calibrated else None
    for m in methods:
        d = diag[m]
        complete = all(s in d["per_scene"] and "psnr" in d["per_scene"][s] for s in scenes)
        if args.raw:
            if complete:
                results[m] = {k: sum(d["per_scene"][s][k] for s in scenes) / len(scenes)
                              for k in ("psnr", "ssim", "lpips")}
            continue
        if d["status"] == "ok" and complete:
            vals = {k: sum(d["per_scene"][s][k] for s in scenes) / len(scenes) for k in METRICS}
            if all(math.isfinite(x) for x in vals.values()):
                results[m] = vals
                continue
            fail(m, "non-finite aggregate metric")
        if d["status"] == "ok":
            fail(m, "incomplete evaluation")
        results[m] = dict(worst)

    out = {
        "metadata": {
            "slice": "train",
            "experiment": EXPERIMENT,
            "views_per_scene": {s: len(views["scenes"][s]) for s in scenes},
            "outputs_root": root,
            "raw": bool(args.raw),
            "device": str(device),
            "seconds": round(time.time() - t0, 1),
        },
        "experiments": {EXPERIMENT: {"results": results}},
        "_diagnostics": diag,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    tmp = args.out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(out, f, indent=2)
    os.replace(tmp, args.out)
    for m in methods:
        print(f"[eval/train] {m}: {diag[m]['status']} {results.get(m)} {diag[m]['reasons'][:3]}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
