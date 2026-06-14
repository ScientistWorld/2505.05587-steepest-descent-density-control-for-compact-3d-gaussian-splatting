# Evaluation Targets

## Primary: compact_synthetic_splatting

Tests whether a Gaussian-splatting reconstruction method achieves high held-out reconstruction quality on the benchmark scene.
- **Metric**: psnr (higher is better)

## Constraint: compact_synthetic_splatting

Tests whether the quality is achieved with a compact primitive set rather than by increasing the number of Gaussians.
- **Metrics**: num_points and flops_proxy (lower is better), point_reduction_vs_adc (higher is better)
