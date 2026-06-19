# Reproduction Milestones

**Current: method_runs**

## Progress Log

### [2026-06-19 17:06] - method_runs
- Implemented and validated a scaled 2D SteepGS density-control reproduction.
- The method computes splitting matrices, filters on negative minimum eigenvalues, splits along the least eigenvector, and halves opacity.
- A prior compact selected run reduced point count substantially but lost 2.62 dB PSNR against the local ADC baseline, so `core_claim` is not claimed yet.
- Fixed the ADC baseline optimizer schedule and prepared a stricter SDC sweep to search for a quality-preserving compact setting.

## Stop Justification
- Not stopped; next submitted job searches for a `core_claim` result.
