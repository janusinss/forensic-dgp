#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1
test -f protocol.json
python -B -u scripts/supervise_cctv_dgp_actual_steps_v1.py --root "$(pwd -P)" --protocol-sha "$1"
