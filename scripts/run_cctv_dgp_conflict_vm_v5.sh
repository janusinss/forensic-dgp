#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
test ! -e outputs/cctv_dgp_conflict_v5
test ! -e pilot_conflict_v5.log
test ! -e environment_conflict_v5.txt
test ! -e cctv-dgp-conflict-v5-results.tar.gz
test ! -e cctv-dgp-conflict-v5-results.tar.gz.sha256
python -c 'import torch, torchvision; assert open("cuda_runtime_before.txt").read().splitlines() == [torch.__version__, torchvision.__version__], "CUDA runtime changed"; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.get_device_properties(0).total_memory >= 8 * 1024**3, "At least 8 GiB VRAM required"'
python -u scripts/run_cctv_dgp_conflict_vm_v5.py --verify --root "$PWD"
python -m pip freeze > environment_conflict_v5.txt
timeout --signal=TERM --kill-after=30s 30m python -u scripts/run_cctv_dgp_conflict_vm_v5.py --root "$PWD" 2>&1 | tee pilot_conflict_v5.log
python -u scripts/audit_cctv_dgp_conflict_results_v5.py --root "$PWD" --results outputs/cctv_dgp_conflict_v5 --verify-recognizer
tar -czf cctv-dgp-conflict-v5-results.tar.gz \
  conflict_protocol_v5.json conflict_protocol_v5.sha256 \
  outputs/cctv_dgp_conflict_v5 environment_conflict_v5.txt \
  pilot_conflict_v5.log cuda_runtime_before.txt
sha256sum cctv-dgp-conflict-v5-results.tar.gz > cctv-dgp-conflict-v5-results.tar.gz.sha256
echo 'V5 training/audit complete. Download cctv-dgp-conflict-v5-results.tar.gz and .sha256; native output review and app promotion remain pending.'
