# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-19 16:59] - core_claim
- Implemented the paper's Steepest Density Control computation in a scaled 2D Gaussian splatting benchmark: per-primitive splitting matrices, negative least-eigenvalue filtering, least-eigenvector two-offspring splits, and opacity halving.
- Implemented an ADC-style density-control baseline in the same renderer.
- Verified the end-to-end reproduction path with `scripts/reproduce.sh` and `python validate.py`.
- Reproduced the core quality-versus-compactness claim on the deterministic benchmark: SteepGS reaches PSNR 37.707 and SSIM 0.9958 with 38 raw points, versus ADC PSNR 40.325 and SSIM 0.9975 with 144 raw points.
- The reproduced method reduces point count by 73.6% relative to ADC while preserving high reconstruction quality.

### [2026-06-19 17:01] - core_claim
- Audited the continuation workspace from the reset `method_runs` state.
- Confirmed the implementation is not a surrogate: the code computes SDC splitting matrices, eigenvalue/eigenvector selection, two-offspring displacement, and opacity halving.
- Confirmed `eval/` reads saved prediction artifacts and does not import from `method/`.
- Re-ran `scripts/evaluate.sh` and `python validate.py --compare`; both passed after installing the declared `numpy` dependency into the current audit Python.
- Restored the milestone to `core_claim`.

## Stop Justification
- Completed at milestone `core_claim`.
