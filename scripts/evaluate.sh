#!/bin/bash
# Standard evaluation: scores every method directory under outputs/ on the
# validation views (scoring/scores_train.json) and on the held-out test views
# (scoring/scores_test.json); scoring/scores.json is the held-out test result,
# i.e. the paper's protocol (every 8th view held out).
# Uses only eval/ (never method/). Needs scripts/download.sh to have run.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
cp scoring/scores_test.json scoring/scores.json
