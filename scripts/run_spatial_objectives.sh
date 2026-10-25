#!/usr/bin/env bash
set -euo pipefail
test "$#" -eq 1
test -n "${TMUX:-}"
cd "$(dirname -- "${BASH_SOURCE[0]}")"
exec python -B -u scripts/supervise_codeformer_spatial_objectives_v1.py --root . --protocol-sha "$1"
