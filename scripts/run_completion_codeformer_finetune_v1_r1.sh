#!/usr/bin/env bash
# Manual bounded pilot. Always attempt return export after a worker exit.
set -u
export PYTHONDONTWRITEBYTECODE=1
if [ "$#" -ne 1 ]; then
  printf 'Usage: bash scripts/run_completion_codeformer_finetune_v1_r1.sh <protocol-sha256>\n' >&2
  exit 64
fi
protocol_sha="$1"
root="$(cd "$(dirname "$0")/.." && pwd)"
python -B -u "$root/scripts/cctv_completion_codeformer_finetune_v1_r1_vm.py" --root "$root" --protocol-sha "$protocol_sha" --run
worker_exit=$?
printf 'Training worker exit code: %s\n' "$worker_exit"
python -B -u "$root/scripts/export_completion_codeformer_finetune_v1_r1.py" --root "$root" --protocol-sha "$protocol_sha"
export_exit=$?
printf 'Evidence export exit code: %s\n' "$export_exit"
if [ "$worker_exit" -ne 0 ]; then exit "$worker_exit"; fi
exit "$export_exit"
