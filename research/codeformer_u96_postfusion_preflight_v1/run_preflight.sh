#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
test "$#" -eq 1
test -n "${TMUX:-}"
test -x "$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
exec "$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python" -B -u \
  scripts/run_u96_postfusion_preflight_v1.py --root . --protocol-sha "$1" --supervise
