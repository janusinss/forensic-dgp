#!/usr/bin/env bash
# A failed quality screen stays failed; supervisor still exports retained evidence.
set -uo pipefail
test "$#" -eq 1 || { echo 'Supply the exact protocol SHA256'; exit 1; }
test -n "${TMUX:-}" || { echo 'Launch manually inside tmux'; exit 1; }
test -n "${VIRTUAL_ENV:-}" || { echo 'Activate the retained VM virtual environment'; exit 1; }
python -B -u scripts/run_matched_feature_learning_v1.py --root . --protocol-sha "$1" --supervise
worker_status=$?
printf 'Finite matched supervisor exit code: %s\n' "$worker_status"
exit "$worker_status"
