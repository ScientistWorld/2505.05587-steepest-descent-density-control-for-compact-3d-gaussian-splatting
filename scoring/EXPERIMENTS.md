# Experiment Overview

## compact_synthetic_splatting

Tests whether a Gaussian-splatting reconstruction system can preserve held-out color reconstruction quality while reducing the number of primitives and the corresponding rendering-cost proxy. The purpose is to measure the paper's central quality-versus-compactness claim in a small deterministic setting that future agents can rerun quickly.

## compact_synthetic_splatting_seed13

Repeats the same quality-versus-compactness evaluation on a second deterministic scene seed. The purpose is to check that improvements are not tuned only to the default procedural sample draw.
