# Constraints

Hold these fixed; the evaluator enforces the first group, the rest are part of
the task definition and are checked by review.

## Enforced by the evaluator

- Output contract: `outputs/<method>/<scene>/point_cloud.ply` for all four
  scenes, standard 3DGS PLY layout, rendered by the evaluator's renderer (SH
  degree <= 3, black background, 3DGS conventions). What the method reports
  about itself is ignored.
- Gaussian budget per scene: at most 0.85x the frozen standard-3DGS reference
  count (`eval/*/protocol.json`).
- Quality floors per scene: PSNR >= reference - 0.5 dB, SSIM >= reference -
  0.015, LPIPS <= reference + 0.03 (reference = frozen standard-3DGS run on the
  same views). Missing, invalid, over-budget or below-floor runs are failed and
  receive the worst value of every metric.
- Ground truth, cameras and constants live in `eval/` and are hash-checked.

## Task rules

- Train only on the views in `data/downloads/scenes/<scene>/images` with the
  COLMAP model in `sparse/0`; never read `eval/*/gt` or anything else under
  `eval/` for training, selection or tuning.
- Half-resolution images as provided (2x box filter); keep the resolution and
  the black background.
- Keep the representation a 3D Gaussian splatting model rendered by the
  standard rasteriser; per-Gaussian parameters are limited to the standard PLY
  fields.
- Keep the per-scene training budget comparable to the reference (30k
  optimisation steps); improvements should come from how the representation is
  grown and maintained, not from more compute.
- Do not compress a finished model as a separate post-processing stage (see
  `DIRECTION.md`).
