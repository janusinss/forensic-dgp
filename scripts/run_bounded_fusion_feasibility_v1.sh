#!/usr/bin/env bash
set -euo pipefail
test -n "${TMUX:-}" || { echo 'Open the manual tmux session first'; exit 1; }
test "$#" -eq 1
export PYTHONDONTWRITEBYTECODE=1
python -B -u scripts/supervise_codeformer_bounded_fusion_feasibility_v1.py --root . --protocol-sha "$1"
