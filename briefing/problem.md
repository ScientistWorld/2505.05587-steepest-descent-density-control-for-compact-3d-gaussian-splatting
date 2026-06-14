# Problem Definition

## Research Question

How can a Gaussian-splatting radiance-field reconstruction system maintain high novel-view or pixel reconstruction quality while using substantially fewer Gaussian primitives than a standard adaptive-density training pipeline?

## Why It Matters

Dense Gaussian point clouds increase model storage, memory use, and rendering cost. A successful compact training procedure makes Gaussian splatting more practical for interactive and resource-constrained rendering.

## Success Criteria

A good solution should reconstruct held-out colors with high PSNR/SSIM and low error while reducing the final number of primitives and the compute implied by rendering those primitives. The quality/compactness tradeoff must be evaluated against a baseline trained under the same data and optimization budget.
