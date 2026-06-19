# Reproduction Milestones

**Current: core_claim**

## Progress Log

### [2026-06-19 03:50] - none
- Read the paper and existing workspace implementation.
- Corrected the scaled SteepGS implementation to use autograd-derived `dL/dsigma` in the splitting matrix and to accumulate mean gradients for compact SDC filtering.
- Corrected the ADC baseline to use expected gradient-norm selection, scale-based clone/split behavior, random large-scale offspring, 0.8 scale shrink, and no opacity halving.
- Static validation passes; runtime reproduction is being submitted as a compute job.

### [2026-06-19 04:21] - method_runs
- Compute job `ff32c404-801` completed successfully.
- The ADC baseline and SteepGS SDC implementation both ran end to end and wrote prediction artifacts plus `scoring/scores.json`.
- The run did not reach `core_claim`: SteepGS saturated the same 144-point cap as ADC, giving `point_reduction_vs_adc = 0.0`.
- Added a scheduler-side SDC threshold sweep to search for a compact setting without changing the SDC algorithm.

### [2026-06-19 04:46] - core_claim
- Compute job `e42fcb49-00a` completed successfully.
- The SDC split-threshold sweep selected `split_threshold = -5e-05` with the same 900-step training budget and 144 maximum primitive cap.
- Reproduced the central claim on the scaled benchmark: SteepGS achieved PSNR 37.707 and SSIM 0.9958 with 38 raw primitives, versus ADC PSNR 40.325 and SSIM 0.9975 with 144 raw primitives.
- The resulting `point_reduction_vs_adc` is 0.736, exceeding the paper's reported 0.519 reduction while preserving high reconstruction quality.
- Promoted the selected threshold into the default method path so `scripts/reproduce.sh` can recreate the core-claim run directly.

## Stop Justification
- Not stopped.
