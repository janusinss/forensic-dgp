#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9a-f]{64}$ ]]; then
  printf '%s\n' 'Usage: bash research/codeformer_u96_spatial_feasibility_v1/run_feasibility.sh PROTOCOL_SHA256' >&2
  exit 64
fi
if [[ -z "${TMUX:-}" ]]; then
  printf '%s\n' 'Manual launch inside tmux is required.' >&2
  exit 64
fi
task_root="$HOME/forensic-dgp/cctv_codeformer_u96_spatial_feasibility_v1_vm"
if [[ "$(pwd -P)" != "$task_root" ]]; then
  printf '%s\n' 'Use the exact new stage as working directory.' >&2
  exit 64
fi
exec "$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python" -B -u \
  scripts/run_u96_spatial_feasibility_v1.py --root "$task_root" --protocol-sha "$1" --supervise
