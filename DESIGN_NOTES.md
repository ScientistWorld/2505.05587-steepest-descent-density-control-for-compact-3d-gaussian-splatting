# Design Notes

Current design (hardening iteration): see `HARDENING.md` ("What was rebuilt") and
`briefing/evaluation.md`. Key points:

- Test answers (held-out images) exist only in `eval/test/gt`, validation answers
  only in `eval/train/gt`; both are sha256-pinned in tracked `views.json` files.
- The evaluator re-renders every PLY itself and counts Gaussians from the file;
  nothing a method writes about itself is read.
- Comparison constants (standard-3DGS counts and quality per scene) are frozen
  in `eval/*/protocol.json`; budget 0.85x, floors -0.5 dB / -0.015 SSIM / +0.03 LPIPS.
- Every directory under `outputs/` is scored; failures get the worst value of
  every metric (the feasibility bounds).
- `eval/train` and `eval/test` share no code (duplicated renderer/metrics).
