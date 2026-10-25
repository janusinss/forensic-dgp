#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1
test -n "${TMUX:-}"
python -B -u scripts/run_u96_postfusion_smoothanchors_v1.py --root . --protocol-sha "$1" --supervise
