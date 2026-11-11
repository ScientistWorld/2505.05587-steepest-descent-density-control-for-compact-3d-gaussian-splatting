# Research Direction

## Research Problem

Density control for 3D Gaussian splatting: during training, deciding which
Gaussians to add, where to place them, how to set their parameters and when to
remove them, so that the final representation is compact without losing
rendering quality.

## Core Contribution Scope

Improvements should come from the training-time density-control rule (and its
interaction with the optimisation schedule): selection criteria for growth,
offspring count and placement, opacity/scale handling of new primitives, pruning
decisions made during training, and budget allocation across the scene.

## Approach Scope

Any growth/pruning criterion, statistic (first- or second-order) or schedule is
in scope, as long as the method trains a standard 3D Gaussian splatting
representation from the provided training views within a comparable step budget
and submits it through the standard PLY contract.

## Out of Bounds

- Replacing Gaussian splatting with another representation (neural fields,
  meshes, voxels) or changing the renderer conventions.
- Post-hoc compression of a finished model as a separate stage (pruning or
  distilling a trained model afterwards, quantisation, codebooks). The paper's
  question is how density control itself avoids redundancy.
- Using held-out views, the evaluator's data, or extra training data.
- Buying quality with more Gaussians than the budget or with substantially more
  optimisation steps.
