# Evaluation Targets

## Primary: compact_synthetic_splatting_seed7

Tests whether a Gaussian-splatting reconstruction method achieves high held-out reconstruction quality on deterministic benchmark scene seed 7.
- **Metric**: psnr (higher is better)
- **Robustness metrics**: ssim (higher is better), l1_error, tail_l1_p95, bright_l1_error, and channel_bias_l1 (lower is better)

## Constraint: compact_synthetic_splatting_seed7

Tests whether the quality is achieved with a compact primitive set rather than by increasing the number of Gaussians.
- **Metrics**: num_points, flops_proxy, and splits (lower is better), point_reduction_vs_adc (higher is better)

## Primary: compact_synthetic_splatting_seed13

Repeats the same quality target on deterministic benchmark scene seed 13 to check that the density-control result is not a single-seed artifact.
- **Metric**: psnr (higher is better)
- **Robustness metrics**: ssim (higher is better), l1_error, tail_l1_p95, bright_l1_error, and channel_bias_l1 (lower is better)

## Constraint: compact_synthetic_splatting_seed13

Tests whether seed-13 quality is also achieved with a compact primitive set rather than by increasing the number of Gaussians.
- **Metrics**: num_points, flops_proxy, and splits (lower is better), point_reduction_vs_adc (higher is better)
