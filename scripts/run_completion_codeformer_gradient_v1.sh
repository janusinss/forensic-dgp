#!/usr/bin/env bash
set -u
export PYTHONDONTWRITEBYTECODE=1
protocol_sha="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
python -B -u "$root/scripts/cctv_completion_codeformer_gradient_v1_vm.py" --root "$root" --protocol-sha "$protocol_sha" --run
worker_exit=$?
printf 'Diagnostic exit code: %s\n' "$worker_exit"
python -B -u "$root/scripts/export_completion_codeformer_gradient_v1.py" --root "$root" --protocol-sha "$protocol_sha"
export_exit=$?
printf 'Export exit code: %s\n' "$export_exit"
if [ "$worker_exit" -ne 0 ]; then exit "$worker_exit"; fi
exit "$export_exit"
