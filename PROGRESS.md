# Progress

## What Works

- Paper analysis and benchmark scaffolding are in place for SteepGS / Steepest Density Control.
- The scaled method implements the core SDC computation on a 2D differentiable Gaussian splatting renderer:
  - accumulate per-Gaussian splitting matrices as `E[(dL/dsigma_i) Hessian(sigma_i)]`;
  - accumulate mean position gradients for compact splitting;
  - split only negative-minimum-eigenvalue, small-mean-gradient primitives;
  - replace each selected primitive with two offspring along the least eigenvector;
  - halve offspring opacities.
- The ADC baseline now follows the paper's heuristic structure in the same 2D renderer: expected gradient-norm selection, clone/split by scale, random large-scale split samples, 0.8 scale shrink, and unchanged opacity.
- The shared 2D Gaussian representation lives in `data/gaussian_field2d.py`, so baseline and method do not depend on each other for the problem representation.
- `scoring/scores.json` reports point-count and FLOPs-proxy metrics in paper-equivalent baseline-normalized units while preserving raw local counts in `outputs/*/metrics.json`.
- `python validate.py` passes static structure checks.

## Results

- Job `ff32c404-801` completed successfully and produced scored artifacts.
- ADC baseline: PSNR 40.325, SSIM 0.9975, L1 0.0060, 144 raw points.
- Initial SteepGS run: PSNR 39.995, SSIM 0.9975, L1 0.0062, 144 raw points.
- This establishes `method_runs`, but not `core_claim`, because SteepGS did not reduce point count relative to ADC in that setting.

## Remaining

- Run the submitted SDC split-threshold sweep and inspect `outputs/sdc_sweep/selection.json` plus `scoring/scores.json`.
- If the sweep identifies a compact candidate with comparable quality, promote the selected settings into `scripts/method.sh` and update the documentation.
- If quality collapses for compact settings, tune the procedural scene or SDC thresholds without replacing the algorithm.
- After successful runtime, update `MILESTONES.md`, `scoring/EXPERIMENTS.md`, and this progress log with actual numbers.

## Issues

- The login environment lacks declared numerical packages (`numpy`), so runtime smoke tests are deferred to the scheduler/container path where `environment/setup.sh` installs dependencies.
- This reproduction uses a scaled 2D procedural splatting benchmark rather than full CUDA 3DGS on Mip-NeRF360/Tanks/Deep Blending.
- Audit fix applied: baseline no longer imports the proposed method module, and score units no longer compare raw 2D primitive counts directly against million-point paper results.
- First runtime issue: the default SDC threshold was too permissive on the scaled scene, causing the proposed method to densify to the same cap as ADC. The next job sweeps stricter negative-eigenvalue thresholds and selects from saved artifacts only.

## Deviations from Paper

- Paper setup: full 3D Gaussian Splatting with CUDA rasterization on real novel-view synthesis datasets.
- Reproduction setup: a deterministic 2D Gaussian splatting renderer with sampled held-out colors.
- Reason: full 3DGS training and dataset acquisition are heavier than needed for the first runnable benchmark; the scaled task preserves the density-control computation while making the gym cheap and deterministic.
- Algorithm preservation: the SDC splitting matrix, least-eigenvector split, negative-eigenvalue condition, compact small-mean-gradient condition, two offspring, and opacity halving are implemented directly.
