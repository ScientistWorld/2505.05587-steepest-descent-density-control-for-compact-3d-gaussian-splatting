# Evaluation

## Metrics

- PSNR on held-out samples, higher is better.
- Global SSIM proxy on held-out samples, higher is better.
- Mean absolute color error, lower is better.
- Ninety-fifth percentile per-sample L1 error, lower is better.
- Bright-region L1 error on the brightest target quartile, lower is better.
- Mean absolute channel bias, lower is better.
- Final number of Gaussian primitives, lower is better when quality remains competitive. In `scores.json`, this is normalized to paper-equivalent million-point units by mapping the local ADC baseline to the paper's 3.339M-point Mip-NeRF360 ADC reference; raw local counts remain in each run's `outputs/*/metrics.json`.
- Point reduction relative to the adaptive-density baseline, higher is better.
- A hardware-independent rendering-cost proxy proportional to sample-count times primitive-count, lower is better. This score is normalized with the same baseline-relative scale as the point count.
- Total densification splits, lower is better when quality remains competitive.

## Baselines and Targets

The reference target is the paper's Table 1 claim on Mip-NeRF360: the proposed system reports roughly half as many points as the original 3DGS baseline while preserving similar PSNR/SSIM/LPIPS. The local benchmark uses deterministic 2D splatting scenes with an ADC-style baseline and the same output metrics. `compact_synthetic_splatting` is the primary seed, and `compact_synthetic_splatting_seed13` is an additional scene seed for broader validation.

## Evaluation Protocol

Run `scripts/reproduce.sh` to train the baseline and proposed method on the default procedural training samples, then run `scripts/evaluate.sh`. Run `scripts/reproduce_plus.sh` to score the default scene plus the additional seed-13 scene. The evaluator reads saved predictions and metadata from `outputs/` and writes `scoring/scores.json`; it does not import or rerun the method implementation.
