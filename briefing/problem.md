# Problem Definition

## Research Question

3D Gaussian Splatting reconstructs a scene from posed photographs as a set of
anisotropic 3D Gaussian primitives that are rasterised to render new views.
Training alternates gradient descent on the primitives with a density-control
procedure that decides when primitives are added (and removed). Standard
density control produces many redundant primitives: memory, storage and
rendering cost grow with the primitive count.

How can a Gaussian-splatting reconstruction reach the novel-view quality of the
standard pipeline while using substantially fewer Gaussians?

## Setting

- Four real scenes used in the paper: Tanks&Temples *train* and *truck*, Deep
  Blending *drjohnson* and *playroom* (images and COLMAP cameras / sparse
  points as released with the 3DGS code base), at half of the released image
  resolution.
- Every 8th view (sorted by name, starting with the first) is held out for the
  hidden test, exactly the paper's protocol; the views at index 4 mod 8 are a
  held-out validation set. Training uses the remaining 6 of every 8 views
  (`data/downloads/scenes/<scene>/images`), plus the COLMAP model in `sparse/0`.
- Output: one 3D Gaussian splatting representation per scene, in the standard
  3DGS PLY layout, at `outputs/<method_name>/<scene>/point_cloud.ply`.

## Why It Matters

Fewer Gaussians mean smaller models, less GPU memory and faster rendering, which
matters for mobile and VR deployment. Reducing redundancy while the model is
trained (rather than compressing a finished model) also lowers training cost.

## Success Criteria

On held-out views, rendering quality (PSNR, SSIM, LPIPS) must stay comparable to
the standard 3DGS pipeline on every scene, and the representation must use as
few Gaussians as possible relative to that pipeline. A method that loses quality
to save Gaussians, or gains quality by adding Gaussians beyond the budget, does
not solve the problem.
