#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
# Use the existing VM CUDA torch installation without replacing it.
python3 - <<'PY'
import torch, torchvision
assert torch.cuda.is_available(), 'Activate the existing VM GPU Python environment first'
assert tuple(map(int,torch.__version__.split('+')[0].split('.')[:3])) >= (2,5,1), 'SAM2 needs torch >=2.5.1; stop and report this environment'
print('Existing CUDA runtime:',torch.__version__,torchvision.__version__,torch.cuda.get_device_name())
PY
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install 'hydra-core==1.3.2' 'iopath==0.1.10' 'Pillow>=9.4' 'opencv-python-headless==4.12.0.88'
python -m pip freeze > environment.txt
python train_feature_mixed_vm.py --preflight
