#!/usr/bin/env bash
set -uo pipefail
EXPECTED="${1:?Usage: bash scripts/run_completion_codeformer_unified_r2_v2.sh PROTOCOL_SHA256}"
ROOT="$HOME/forensic-dgp/cctv_completion_codeformer_unified_r2_v2_vm"
PY="$HOME/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python"
test -x "$PY" || exit 1
cd "$ROOT" || exit 1
"$PY" -B -u scripts/cctv_completion_codeformer_unified_r2_v2_vm.py --root "$ROOT" --protocol-sha "$EXPECTED" --run
worker_exit=$?
printf 'Unified-R2-v2 worker exit code: %s\n' "$worker_exit"
"$PY" -B -u scripts/export_completion_codeformer_unified_r2_v2.py --root "$ROOT" --protocol-sha "$EXPECTED"
export_exit=$?
printf 'Unified-R2-v2 export exit code: %s\n' "$export_exit"
if test "$export_exit" -ne 0; then exit "$export_exit"; fi
exit "$worker_exit"
