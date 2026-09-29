#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
nvidia-smi
python -u scripts/train_expanded_feature_vm.py --root "$PWD" --preflight
python -m pip freeze > expanded_environment.txt
python -u scripts/train_expanded_feature_vm.py --root "$PWD" 2>&1 | tee expanded_training.log
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/expanded_feature_training')
r=json.loads((p/'results.json').read_text())
assert r['complete'] and set(r['arms'])=={'fixed','anatomical'}
for arm in r['arms'].values():
    assert arm['complete'] and len(arm['history'])==20 and arm['seen']==3540
assert Path('expanded-feature-results.tar.gz').is_file()
PY
echo "Download: $(pwd)/expanded-feature-results.tar.gz"
