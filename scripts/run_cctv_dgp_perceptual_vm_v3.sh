#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
test ! -e outputs/cctv_dgp_perceptual_v3
test ! -e pilot_perceptual_v3.log
test ! -e environment_perceptual_v3.txt
test ! -e cctv-dgp-perceptual-v3-results.tar.gz
test ! -e cctv-dgp-perceptual-v3-results.tar.gz.sha256
python -c 'import torch, torchvision; assert open("cuda_runtime_before.txt").read().splitlines() == [torch.__version__, torchvision.__version__], "CUDA runtime changed"; assert torch.cuda.is_available(), "CUDA unavailable"; print("Existing CUDA runtime verified:", torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))'
python -u scripts/run_cctv_dgp_perceptual_vm_v3.py --verify --root "$PWD"
python -m pip freeze > environment_perceptual_v3.txt
timeout --signal=TERM --kill-after=30s 90m python -u scripts/run_cctv_dgp_perceptual_vm_v3.py --root "$PWD" 2>&1 | tee pilot_perceptual_v3.log
python -u scripts/audit_cctv_dgp_perceptual_results_v3.py --root "$PWD" --results outputs/cctv_dgp_perceptual_v3
tar -czf cctv-dgp-perceptual-v3-results.tar.gz \
  perceptual_protocol_v3.json perceptual_protocol_v3.sha256 \
  outputs/cctv_dgp_perceptual_v3 environment_perceptual_v3.txt \
  pilot_perceptual_v3.log cuda_runtime_before.txt
sha256sum cctv-dgp-perceptual-v3-results.tar.gz > cctv-dgp-perceptual-v3-results.tar.gz.sha256
echo 'V3 pilot and file audit complete. Download cctv-dgp-perceptual-v3-results.tar.gz and its .sha256; native review remains required.'
