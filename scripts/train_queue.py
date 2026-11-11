#!/usr/bin/env python
"""Train 3D Gaussian splatting runs for the benchmark on every visible GPU.

Each run trains one scene with the official SteepGS / 3DGS code base
(method/SteepGS/train.py) and copies the final representation to
``<dest>/<scene>/point_cloud.ply``. Finished runs (destination exists) are
skipped, so the script can be re-launched after an interruption.

Groups:
  reference  outputs/steepgs   SteepGS (steepest density control), seed 0
             outputs/adc       3DGS adaptive density control, seed 0
             outputs/adc_thres ADC whose densification stops at SteepGS's
                               Gaussian count on the same scene (paper's
                               "3DGS + Thres." baseline)
  audit      hardening/runs/*  extra seeds and variants used to check the gym

Usage: python scripts/train_queue.py [--groups reference audit] [--scenes ...]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN = os.path.join(ROOT, "method", "SteepGS", "train.py")
SCENES = ["train", "truck", "drjohnson", "playroom"]
WORK = os.path.join(ROOT, "checkpoints", "train_runs")

STEEP = ["--densify_strategy", "steepest"]
ADC = ["--densify_strategy", "adc"]

# name -> (destination dir, extra args, seed, iterations, group)
RUNS = {
    "steepgs": ("outputs/steepgs", STEEP, 0, 30000, "reference"),
    "adc": ("outputs/adc", ADC, 0, 30000, "reference"),
    "adc_thres": ("outputs/adc_thres", ADC, 0, 30000, "reference"),
    "steepgs_seed1": ("hardening/runs/steepgs_seed1", STEEP, 1, 30000, "audit"),
    "steepgs_seed2": ("hardening/runs/steepgs_seed2", STEEP, 2, 30000, "audit"),
    "no_densify": ("hardening/runs/no_densify", ADC + ["--densify_until_iter", "0"], 0, 30000, "audit"),
    "better_long": ("hardening/runs/better_long", STEEP, 0, 60000, "audit"),
    "better_moresteps": ("hardening/runs/better_moresteps", STEEP + ["--densify_until_iter", "25000"], 0, 40000, "audit"),
    "ablate_random_offsets": ("hardening/runs/ablate_random_offsets", ["--densify_strategy", "steepest", "no_eig_upd"], 0, 30000, "audit"),
    "steepgs_halved_opacity": ("hardening/runs/steepgs_halved_opacity", ["--densify_strategy", "steepest", "no_div_opacity"], 0, 30000, "audit"),
    "better_long_seed1": ("hardening/runs/better_long_seed1", STEEP, 1, 60000, "audit"),
    "adc_thres_seed1": ("hardening/runs/adc_thres_seed1", ADC, 1, 30000, "audit"),
}
# Priority order of (run, scene) tasks.
ORDER = ["steepgs", "adc", "adc_thres", "steepgs_seed1", "steepgs_seed2", "no_densify", "better_long",
         "better_moresteps", "adc_thres_seed1", "ablate_random_offsets", "steepgs_halved_opacity",
         "better_long_seed1"]


def ply_count(path: str) -> int:
    with open(path, "rb") as f:
        for _ in range(200):
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                return int(line.split()[2])
            if line == "end_header":
                break
    raise ValueError(f"no vertex count in {path}")


def dest_ply(name, scene):
    return os.path.join(ROOT, RUNS[name][0], scene, "point_cloud.ply")


def deps(name, scene):
    if name.startswith("adc_thres"):
        return [("steepgs", scene)]
    return []


def run_task(name, scene, gpu, log):
    dest, extra, seed, iters, _ = RUNS[name]
    work = os.path.join(WORK, name, scene)
    os.makedirs(work, exist_ok=True)
    args = list(extra)
    if name.startswith("adc_thres"):
        args += ["--densify_max_points", str(ply_count(dest_ply("steepgs", scene)))]
    saves = sorted({7000, iters})
    cmd = [sys.executable, TRAIN, "-s", os.path.join(ROOT, "data", "downloads", "scenes", scene),
           "-i", "images", "-r", "1", "-m", work, "--S_estimator", "inv_cov", "--seed", str(seed),
           "--gpu", str(gpu), "--iterations", str(iters), "--test_iterations", "-1",
           "--save_iterations", *map(str, saves), *args]
    env = dict(os.environ, TQDM_DISABLE="1", PYTHONUNBUFFERED="1")
    t0 = time.time()
    with open(os.path.join(work, "stdout.log"), "w") as fo:
        fo.write(" ".join(cmd) + "\n")
        fo.flush()
        rc = subprocess.call(cmd, stdout=fo, stderr=subprocess.STDOUT, cwd=os.path.join(ROOT, "method", "SteepGS"), env=env)
    final = os.path.join(work, "point_cloud", f"iteration_{iters}", "point_cloud.ply")
    if rc != 0 or not os.path.exists(final):
        log(f"FAILED {name}/{scene} rc={rc} after {time.time() - t0:.0f}s (see {work}/stdout.log)")
        return False
    out = os.path.join(ROOT, dest, scene, "point_cloud.ply")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    shutil.copyfile(final, out + ".tmp")
    os.replace(out + ".tmp", out)
    if name == "steepgs":
        early = os.path.join(work, "point_cloud", "iteration_7000", "point_cloud.ply")
        wk = os.path.join(ROOT, "hardening", "runs", "weaker_7k", scene, "point_cloud.ply")
        os.makedirs(os.path.dirname(wk), exist_ok=True)
        shutil.copyfile(early, wk)
    log(f"done {name}/{scene} gpu{gpu} {time.time() - t0:.0f}s points={ply_count(out)}")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", nargs="+", default=["reference"])
    ap.add_argument("--runs", nargs="+", default=None)
    ap.add_argument("--scenes", nargs="+", default=SCENES)
    ap.add_argument("--deadline-hours", type=float, default=5.2,
                    help="do not start new runs after this many hours")
    args = ap.parse_args()
    import torch

    n_gpu = max(1, torch.cuda.device_count())
    names = [n for n in ORDER if (args.runs is None and RUNS[n][4] in args.groups) or (args.runs and n in args.runs)]
    tasks = [(n, s) for n in names for s in args.scenes if not os.path.exists(dest_ply(n, s))]
    lock = threading.Lock()
    state = {t: "todo" for t in tasks}
    start = time.time()

    def log(msg):
        with lock:
            print(f"[queue {time.time() - start:7.0f}s] {msg}", flush=True)

    def pick():
        with lock:
            for t in tasks:
                if state[t] != "todo":
                    continue
                ready = True
                for d in deps(*t):
                    if os.path.exists(dest_ply(*d)):
                        continue
                    ready = False
                    if state.get(d) in (None, "failed") and not os.path.exists(dest_ply(*d)):
                        state[t] = "skipped"
                if state[t] == "todo" and ready:
                    state[t] = "running"
                    return t
            return None

    def worker(gpu):
        while True:
            if time.time() - start > args.deadline_hours * 3600:
                log(f"gpu{gpu}: deadline reached, not starting new runs")
                return
            t = pick()
            if t is None:
                with lock:
                    pending = any(v in ("todo", "running") for v in state.values())
                    waiting = any(v == "todo" for v in state.values())
                if not waiting or not pending:
                    return
                time.sleep(20)
                continue
            log(f"start {t[0]}/{t[1]} on gpu{gpu}")
            ok = run_task(t[0], t[1], gpu, log) or run_task(t[0], t[1], gpu, log)
            with lock:
                state[t] = "done" if ok else "failed"

    log(f"{len(tasks)} tasks on {n_gpu} GPU(s): {tasks}")
    threads = [threading.Thread(target=worker, args=(g,)) for g in range(n_gpu)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    log(f"finished: { {k: v for k, v in state.items()} }")


if __name__ == "__main__":
    main()
