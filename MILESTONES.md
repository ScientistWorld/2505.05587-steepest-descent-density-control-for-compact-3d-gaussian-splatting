# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-19 17:06] - method_runs
- Implemented and validated a scaled 2D SteepGS density-control reproduction.
- The method computes splitting matrices, filters on negative minimum eigenvalues, splits along the least eigenvector, and halves opacity.
- A prior compact selected run reduced point count substantially but lost 2.62 dB PSNR against the local ADC baseline, so `core_claim` is not claimed yet.
- Fixed the ADC baseline optimizer schedule and prepared a stricter SDC sweep to search for a quality-preserving compact setting.

### [2026-06-19 17:26] - core_claim
- Job `b8492ff2-0a7` completed the corrected ADC baseline, SDC sweep, selection, and evaluation path successfully.
- ADC baseline reached PSNR 39.458, SSIM 0.9969, L1 0.00657 with 144 raw Gaussian primitives.
- Selected SteepGS/SDC run reached PSNR 40.593, SSIM 0.9976, L1 0.00602 with 72 raw Gaussian primitives.
- The selected SDC result therefore preserves or improves image quality while using 50% fewer primitives than the local ADC baseline.
- `python validate.py --compare` passes on the resulting `scoring/scores.json`.

## Stop Justification
- Completed at milestone `core_claim`.
