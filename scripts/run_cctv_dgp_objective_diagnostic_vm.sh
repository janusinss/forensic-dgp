#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
test ! -e outputs/cctv_dgp_objective_diagnostic_v4
test ! -e objective_diagnostic_v4.log
test ! -e objective_diagnostic_environment_v4.txt
test ! -e cctv-dgp-objective-v4-results.tar.gz
test ! -e cctv-dgp-objective-v4-results.tar.gz.sha256
python -c 'import torch, torchvision; assert open("cuda_runtime_before.txt").read().splitlines() == [torch.__version__, torchvision.__version__], "CUDA runtime changed"; assert torch.cuda.is_available(), "CUDA unavailable"'
python -u scripts/run_cctv_dgp_objective_diagnostic_vm.py --verify --root "$PWD"
python -m pip freeze > objective_diagnostic_environment_v4.txt
timeout --signal=TERM --kill-after=30s 10m python -u scripts/run_cctv_dgp_objective_diagnostic_vm.py --root "$PWD" 2>&1 | tee objective_diagnostic_v4.log
python -u scripts/audit_cctv_dgp_objective_diagnostic_v4.py --root "$PWD" --results outputs/cctv_dgp_objective_diagnostic_v4
tar -czf cctv-dgp-objective-v4-results.tar.gz \
  objective_diagnostic_protocol_v4.json objective_diagnostic_protocol_v4.sha256 \
  outputs/cctv_dgp_objective_diagnostic_v4 objective_diagnostic_environment_v4.txt \
  objective_diagnostic_v4.log cuda_runtime_before.txt
sha256sum cctv-dgp-objective-v4-results.tar.gz > cctv-dgp-objective-v4-results.tar.gz.sha256
echo 'Zero-update objective diagnostic complete. Download cctv-dgp-objective-v4-results.tar.gz and .sha256; no model was trained or selected.'
