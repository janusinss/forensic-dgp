#!/usr/bin/env bash
set -euo pipefail
if [[ "$#" -ne 1 || ! "$1" =~ ^[0-9a-f]{64}$ || -z "${TMUX:-}" ]]; then
  printf '%s\n' 'One protocol SHA and a manual tmux session are required.' >&2
  exit 64
fi
task_root="$HOME/forensic-dgp/cctv_codeformer_u64_png_surrogate_measurement_v1_vm"
if [[ "$(pwd -P)" != "$task_root" ]]; then
  printf '%s\n' 'Use the exact new diagnostic stage.' >&2
  exit 64
fi
exec "$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python" -B -u \
  scripts/run_u64_png_surrogate_measurement_v1.py --root "$task_root" --protocol-sha "$1" --supervise
