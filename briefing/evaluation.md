# Evaluation

## What is submitted

Every immediate sub-directory of `outputs/` is a method. For each scene in
{train, truck, drjohnson, playroom} it must contain `<scene>/point_cloud.ply`: a
binary little-endian PLY with one vertex per Gaussian and the standard 3DGS
properties `x y z f_dc_0..2 [f_rest_0..44] opacity scale_0..2 rot_0..3`
(opacity as a logit, scales as logs, rotation as a quaternion w,x,y,z;
spherical-harmonic degree 0-3 inferred from the number of `f_rest_*`). Other
properties and any other files are ignored.

## How it is scored

The evaluator (`eval/test`, held-out test views; `eval/train`, validation views)
reads each PLY, counts its Gaussians, and re-renders it with its own renderer at
the held-out cameras. The renderer reproduces the forward pass of the standard
tile-based 3DGS rasteriser (EWA projection with 0.3 px low-pass, 3-sigma tiles,
depth-sorted alpha compositing, black background); it matches the reference
CUDA rasteriser to within ~85 dB PSNR. Renders are clamped and quantised to 8
bits and compared with the ground-truth images (released JPGs, 2x box-filtered
exactly like the training images).

Metrics, averaged per scene over its views and then over the four scenes:

| metric | direction | meaning |
|---|---|---|
| `psnr` (primary) | higher | per-view PSNR in dB |
| `ssim` | higher | per-view SSIM, 11x11 Gaussian window |
| `lpips` | lower | per-view LPIPS, VGG-16 v0.1 heads on [0,1] inputs |
| `num_points_ratio` | lower | Gaussians / Gaussians of the evaluator's fixed standard-3DGS reference on that scene |

Rank coefficients (`scoring/reference.json`): psnr 1.0, ssim 0.5, lpips 0.5,
num_points_ratio 1.0.

## Feasibility (fixed by the evaluator, `eval/*/protocol.json`)

Per scene, relative to the evaluator's frozen standard-3DGS (adaptive density
control) reference run on the same views:

- at most 0.85x its Gaussian count;
- PSNR at least its PSNR - 0.5 dB, SSIM at least its SSIM - 0.015, LPIPS at most
  its LPIPS + 0.03 (quality comparable to the standard pipeline).

A run that misses a scene, has an unreadable, empty or non-finite PLY, exceeds
the budget, or misses a quality floor on any scene is **failed** and gets the
worst value of every metric (the feasibility bounds averaged over scenes), which
is never better than any feasible run.

## Running it

`bash scripts/evaluate_test.sh` and `bash scripts/evaluate_train.sh` take no
arguments, score every method under `outputs/`, and write
`scoring/scores_test.json` / `scoring/scores_train.json` (per-scene values and
failure reasons are under `_diagnostics`). A GPU is strongly recommended:
on one A100 a method takes about 45 s per slice; the CPU fallback produces the
same numbers but is two to three orders of magnitude slower.

The evaluator needs `scripts/download.sh` to have populated the ground truth and
LPIPS weights; their sha256 hashes are checked and a modified file fails every
run.
