# Overview

- **Paper ID:** 2505.05587
- **Title:** Steepest Descent Density Control for Compact 3D Gaussian Splatting
- **Domain:** Computer Vision / Neural Rendering
- **TL;DR:** Grow 3D Gaussian Splatting models by splitting only primitives whose
  splitting matrix has a negative eigenvalue, into two half-opacity off-springs
  along its least eigenvector: about half the Gaussians at the same quality.

## The gym

- Data: the paper's Tanks&Temples (train, truck) and Deep Blending (drjohnson,
  playroom) scenes from the public 3DGS archive, half resolution; every 8th view
  held out for the hidden test (paper protocol), views at index 4 mod 8 for the
  validation slice, the rest for training.
- Submission: one standard 3DGS PLY per scene in `outputs/<method>/<scene>/`.
- Scoring: the evaluator re-renders each PLY at the held-out cameras, measures
  PSNR / SSIM / LPIPS and the Gaussian count relative to a frozen standard-3DGS
  run, and fails runs that exceed the Gaussian budget or lose quality relative
  to standard 3DGS on any scene.

## Key results

| | Gaussians vs 3DGS | PSNR | SSIM | LPIPS |
|---|---:|---:|---:|---:|
| paper, same 4 scenes, full res (Tables 2-5) | 0.488 | 26.82 (3DGS 26.72) | 0.873 (0.876) | 0.223 (0.211) |
| this gym, half res, held-out test | 0.465 | 26.98 (3DGS 26.66) | 0.893 (0.891) | 0.145 (0.140) |

Standard densification stopped at the same count loses 2.5 dB on the *train*
scene and fails the quality floor (paper: -1.3 dB on Tanks&Temples).
