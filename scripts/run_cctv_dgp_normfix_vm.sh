#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
test ! -e cctv-dgp-normfix-v2-results.tar.gz
test ! -e pilot_normfix_v2.log
python -c 'import torch, torchvision; assert open("cuda_runtime_before.txt").read().splitlines() == [torch.__version__, torchvision.__version__], "CUDA runtime changed"; assert torch.cuda.is_available(), "CUDA unavailable"'
python -m pip freeze > environment_normfix_v2.txt
python -u scripts/run_cctv_dgp_normfix_vm.py "$@" 2>&1 | tee pilot_normfix_v2.log
python -u scripts/audit_cctv_dgp_normfix_results.py --root "$PWD" --results outputs/cctv_dgp_pilot
tar -czf cctv-dgp-normfix-v2-results.tar.gz \
  protocol.json protocol.sha256 outputs/cctv_dgp_pilot \
  --transform='s|^environment_normfix_v2.txt$|environment.txt|' environment_normfix_v2.txt \
  --transform='s|^pilot_normfix_v2.log$|pilot.log|' pilot_normfix_v2.log \
  cuda_runtime_before.txt
sha256sum cctv-dgp-normfix-v2-results.tar.gz > cctv-dgp-normfix-v2-results.tar.gz.sha256
echo 'Corrected pilot and file audit complete. Download cctv-dgp-normfix-v2-results.tar.gz and its .sha256.'
