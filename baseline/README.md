# Baselines

Both baselines run through the same official 3DGS / SteepGS code base as the
method (`method/SteepGS/train.py`), so they share renderer, optimiser and
training schedule and differ only in density control:

- `adc` — standard 3DGS adaptive density control (`--densify_strategy adc`).
- `adc_thres` — ADC whose densification stops once the Gaussian count reaches
  SteepGS's final count on the same scene (`--densify_max_points N`), the
  paper's "3DGS + Thres." matched-budget baseline.

Run with `bash scripts/baseline.sh` (after `scripts/method.sh`, which provides
the SteepGS counts). Outputs: `outputs/adc/<scene>/point_cloud.ply`,
`outputs/adc_thres/<scene>/point_cloud.ply`.
