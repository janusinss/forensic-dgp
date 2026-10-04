#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
source ../cctv_dgp_vm_bundle/.venv/bin/activate
test ! -e outputs/cctv_dgp_targets_v6
test ! -e cctv-dgp-targets-v6-results.tar.gz
nvidia-smi
python - <<'PY'
from pathlib import Path
import torch, torchvision
assert torch.cuda.is_available(), 'CUDA unavailable'
assert 'L4' in torch.cuda.get_device_name(0), 'Existing L4 VM required'
assert Path('../cctv_dgp_vm_bundle/cuda_runtime_before.txt').read_text().splitlines() == [torch.__version__, torchvision.__version__], 'CUDA runtime changed'
print('Existing CUDA runtime verified:', torch.__version__, torchvision.__version__)
PY
python -m pip freeze > environment_v6.txt
python -u scripts/audit_cctv_dgp_targets_v6.py --root "$PWD" --verify-preparation --receipt preparation_audit_vm.json
timeout --signal=INT --kill-after=30s 22m python -u scripts/run_cctv_dgp_targets_vm_v6.py 2>&1 | tee pilot_targets_v6.log
python -u scripts/audit_cctv_dgp_targets_v6.py --root "$PWD" --results outputs/cctv_dgp_targets_v6 --receipt outputs/cctv_dgp_targets_v6/independent_audit_vm.json
tar -czf cctv-dgp-targets-v6-results.tar.gz targets_protocol_v6.json targets_protocol_v6.sha256 preparation_audit_vm.json environment_v6.txt pilot_targets_v6.log outputs/cctv_dgp_targets_v6
sha256sum cctv-dgp-targets-v6-results.tar.gz > cctv-dgp-targets-v6-results.tar.gz.sha256
echo 'V6 finished. Export cctv-dgp-targets-v6-results.tar.gz and .sha256; no production model promoted.'
