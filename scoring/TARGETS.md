# Evaluation Targets

Experiment `compact_view_synthesis` (weight 1.0): four real scenes (Tanks&Temples
train, truck; Deep Blending drjohnson, playroom), half resolution, held-out
views scored by re-rendering the submitted representation. Values are means
over the four scenes.

## Primary (benefit): rendering quality on held-out views

- `psnr` (higher is better, coefficient 1.0) — primary metric.
- `ssim` (higher, 0.5), `lpips` (lower, 0.5).

## Benefit / constraint: compactness

- `num_points_ratio` (lower, 1.0): Gaussians relative to the evaluator-fixed
  standard 3DGS (adaptive density control) run on the same scene.

## Constraints (evaluator-enforced, per scene; failure = worst value on every metric)

- Gaussian budget: at most 0.85x the standard-3DGS count.
- Quality comparable to standard 3DGS: PSNR >= its PSNR - 0.5 dB, SSIM >= its
  SSIM - 0.015, LPIPS <= its LPIPS + 0.03.

## Reference values (held-out test slice, `scoring/scores_test.json`)

| method | psnr | ssim | lpips | num_points_ratio | status |
|---|---:|---:|---:|---:|---|
| standard 3DGS (ADC), raw | 26.662 | 0.8912 | 0.1404 | 1.000 | fails budget |
| reference (paper's method) | 26.982 | 0.8925 | 0.1448 | 0.465 | ok |
| ADC stopped at the reference's count | 19.97 on *train* | 0.719 on *train* | 0.296 on *train* | 0.503 on *train* | fails quality floor |

Seed spread of the reference (3 seeds): PSNR 0.12 dB, SSIM 0.0005, LPIPS 0.0009,
num_points_ratio 0.004. The paper reports, on the same four scenes at full
resolution, 0.488x the 3DGS Gaussian count at +0.11 dB PSNR (see
`reference.json`).
