#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

# Reuse the active CUDA environment; install only these five isolated packages.
python3 - <<'PY'
import importlib
import importlib.metadata as metadata
import sys
assert sys.platform == 'linux', 'Use the Linux VM'
import torch
assert torch.cuda.is_available(), 'Activate the existing VM CUDA Python environment first'
from packaging.requirements import Requirement
for text in ('torch>=2.5.1', 'torchvision>=0.9', 'numpy>=1.19.3', 'Pillow>=8',
             'tqdm>=4.42.1', 'requests', 'filelock', 'fsspec>=2023.5.0',
             'packaging>=20.9', 'typing-extensions>=3.7.4.3'):
    requirement = Requirement(text)
    version = metadata.version(requirement.name)
    assert version in requirement.specifier, f'Existing dependency differs: {text} found {version}'
for module in ('torchvision', 'numpy', 'PIL', 'cv2', 'tqdm', 'requests',
               'filelock', 'fsspec', 'packaging', 'typing_extensions'):
    importlib.import_module(module)
print('Existing CUDA runtime:', torch.__version__, torch.cuda.get_device_name())
PY
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

dep_dir=outputs/face_occlusion_dependencies
if [ -e "$dep_dir" ]; then
  echo "Preserve existing dependency directory: $dep_dir; inspect before rerunning" >&2
  exit 1
fi
python3 -m pip install --disable-pip-version-check --no-deps --only-binary=:all: \
  --target "$dep_dir" -r requirements_face_occlusion_vm.txt

python3 - <<'PY'
import json
from pathlib import Path
import sys
target = Path('outputs/face_occlusion_dependencies').resolve()
sys.path.insert(0, str(target))
from scripts.train_face_occlusion_vm import dependency_versions
packages = dependency_versions(target)
import torch, torchvision
assert torch.cuda.is_available() and not Path(torch.__file__).resolve().is_relative_to(target)
assert not Path(torchvision.__file__).resolve().is_relative_to(target)
report = {'packages': packages, 'python': sys.version, 'executable': sys.executable,
          'torch': str(torch.__version__), 'torchvision': str(torchvision.__version__),
          'gpu': torch.cuda.get_device_name(), 'optimizer_updates': 0}
(target/'setup.json').write_text(json.dumps(report, indent=2)+'\n')
print('Five pinned packages imported from isolated target; CUDA runtime reused', flush=True)
PY
