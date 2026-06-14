#!/bin/bash
# Run the paper's proposed method.
#
# This script trains/runs the paper's method using code in method/.
#
# Usage:
#   bash scripts/method.sh

set -e

cd /home/user

python -m method.sdc2d.steepgs2d \
  --output outputs/steepgs2d \
  --device cpu \
  "$@"
