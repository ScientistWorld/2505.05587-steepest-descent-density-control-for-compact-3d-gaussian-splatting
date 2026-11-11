#!/bin/bash
# Held-out test evaluation of every method directory under outputs/.
# No arguments needed; writes scoring/scores_test.json.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p scoring
python -m eval.test.evaluate_outputs --outputs-root outputs --out scoring/scores_test.json "$@"
