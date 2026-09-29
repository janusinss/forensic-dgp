#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
nvidia-smi
python train_feature_mixed_vm.py --preflight
python -u train_feature_mixed_vm.py 2>&1 | tee training.log
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/feature_mixed_training')
r=json.loads((p/'results.json').read_text())
assert r.get('complete') and len(r['history'])==20
assert (p/'final_epoch_20.pth').is_file()
PY
tar -czf feature-mixed-vm-results.tar.gz \
  outputs/feature_mixed_training/final_epoch_20.pth \
  outputs/feature_mixed_training/results.json \
  outputs/feature_mixed_training/protocol.json \
  outputs/feature_mixed_training/sources.json \
  outputs/feature_mixed_training/cache_progress.json \
  training.log environment.txt bundle_inventory.json
echo "DONE: $(pwd)/feature-mixed-vm-results.tar.gz"
