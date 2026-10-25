#!/usr/bin/env bash
# Manual tmux launcher for Conditioned SFT V2 on NVIDIA L4 VM.
set -uo pipefail
test "$#" -eq 1 || { echo 'Supply the exact protocol SHA256'; exit 1; }
test -n "${TMUX:-}" || { echo 'Launch manually inside tmux'; exit 1; }
test -n "${VIRTUAL_ENV:-}" || { echo 'Activate the retained VM virtual environment'; exit 1; }
python -B -u scripts/run_conditioned_sft_learning_v2.py --root . --protocol-sha "$1" --supervise
worker_status=$?
printf 'Finite conditioned-sft-v2 supervisor exit code: %s\n' "$worker_status"
exit "$worker_status"
