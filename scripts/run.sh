#!/bin/bash
set -euo pipefail
cd /home/user
bash redteam/confirm.sh 2>&1 | tee redteam/confirmation.log
