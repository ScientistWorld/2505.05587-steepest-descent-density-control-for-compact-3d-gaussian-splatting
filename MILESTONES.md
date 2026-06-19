# Reproduction Milestones

**Current: method_runs**

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

## Stop Justification
- Not stopped.
