# Experiments

## compact_view_synthesis (weight 1.0)

Establishes whether a Gaussian-splatting reconstruction keeps the novel-view
quality of the standard 3DGS training pipeline while using fewer Gaussians, on
the paper's Tanks&Temples (train, truck) and Deep Blending (drjohnson,
playroom) scenes. Each method submits one representation per scene; the
evaluator renders it at views that were never available for training and
measures image quality (PSNR, SSIM, LPIPS) and compactness (Gaussian count
relative to the standard pipeline's count on the same scene).

The benefit side is compactness at comparable quality and quality at a given
compactness; the constraint side is that quality on every scene stays within a
small, fixed margin of the standard pipeline and that the Gaussian count stays
below a fixed budget. Runs that break the constraint count as failed. The
validation slice (`scores_train.json`) measures the same quantities on a
different set of held-out views of the same scenes.
