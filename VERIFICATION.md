# Verification Report

Superseded by the hardening iteration; see `HARDENING.md` for the current gym,
its checks (renderer parity, seed noise, better/weaker variants, shortcut
cases) and its limitations. `python validate.py` passes on the final scores.

- Milestone: core_claim_plus
- Ready for gym use: yes (the hidden test should be given a GPU)
- Key limitation: four scenes at half resolution; post-hoc pruning of a finished
  model cannot be told apart by the evaluator and is ruled out in DIRECTION.md.
