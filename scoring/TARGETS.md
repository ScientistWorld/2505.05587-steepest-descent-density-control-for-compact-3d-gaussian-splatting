# Evaluation Targets

## Primary: compact_synthetic_splatting, compact_synthetic_splatting_seed13

Tests whether a Gaussian-splatting reconstruction method achieves high held-out reconstruction quality on the benchmark scenes.
- **Metric**: psnr (higher is better)
- **Robustness metrics**: ssim (higher is better), l1_error, tail_l1_p95, bright_l1_error, and channel_bias_l1 (lower is better)

## Constraint: compact_synthetic_splatting, compact_synthetic_splatting_seed13

Tests whether the quality is achieved with a compact primitive set rather than by increasing the number of Gaussians.
- **Metrics**: num_points, flops_proxy, and splits (lower is better), point_reduction_vs_adc (higher is better)
