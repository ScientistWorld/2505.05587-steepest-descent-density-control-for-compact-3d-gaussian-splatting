# Progress

## Current state (hardening iteration, 2026-11-11)

- Gym = compact novel-view synthesis on four of the paper's real scenes; method = official SteepGS code (`method/SteepGS`) with the paper's opacity halving restored; baselines = standard 3DGS ADC and ADC stopped at SteepGS's count (same code base, `baseline/README.md`).
- `scripts/download.sh` fetches the public 3DGS archive and LPIPS weights; held-out views go only to `eval/*/gt`.
- `scripts/run.sh` -> `scripts/reproduce.sh`: prepare images, train SteepGS, ADC, ADC+Thres. (~2.5 GPU-hours on A100), evaluate both slices.
- Evaluator: re-renders submitted PLYs with an evaluator-owned PyTorch renderer (parity with the CUDA rasteriser >= 77.9 dB), frozen per-scene constants and feasibility rules in `eval/*/protocol.json`.
- Results and audit: `HARDENING.md`, `hardening/`.

## Known limitations

- Half resolution, 6/8 training views, 4 of 11 scenes; no Mip-NeRF 360.
- The evaluator wants a GPU (CPU fallback ~25 s per view on 16 cores).
- The random-offset ablation of the released code crashes (shape mismatch in the clone branch of `no_eig_upd`).
