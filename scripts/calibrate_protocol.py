#!/usr/bin/env python
"""One-time gym construction step: freeze the ADC reference constants.

Reads raw (ungated) evaluations of the ADC reference run on the validation and
held-out test views and writes the per-scene constants into
eval/train/protocol.json and eval/test/protocol.json. Not part of any scoring
path; the frozen constants are committed with the evaluator.

Usage: python scripts/calibrate_protocol.py <raw_train.json> <raw_test.json> [adc_method_name]
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main(raw_train, raw_test, method="adc"):
    for side, raw_path in (("train", raw_train), ("test", raw_test)):
        raw = json.load(open(raw_path))
        per_scene = raw["_diagnostics"][method]["per_scene"]
        ppath = os.path.join(ROOT, "eval", side, "protocol.json")
        proto = json.load(open(ppath))
        for scene in proto["scenes"]:
            ps = per_scene[scene]
            proto["scenes"][scene] = {
                "adc_points": int(ps["num_points"]),
                "adc_psnr": float(ps["psnr"]),
                "adc_ssim": float(ps["ssim"]),
                "adc_lpips": float(ps["lpips"]),
            }
        with open(ppath, "w") as f:
            json.dump(proto, f, indent=2)
        print(f"[calibrate] {side}: {proto['scenes']}")


if __name__ == "__main__":
    main(*sys.argv[1:])
