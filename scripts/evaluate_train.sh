#!/bin/bash
# Train-side (validation views) evaluation of every method directory under
# outputs/. No arguments needed; writes scoring/scores_train.json.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p scoring
python -m eval.train.evaluate_outputs --outputs-root outputs --out scoring/scores_train.json "$@"
