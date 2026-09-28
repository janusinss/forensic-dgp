#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON_CMD:-python3}"
"$PYTHON_CMD" -c 'import torch; assert torch.cuda.is_available(), "CUDA required for this VM recipe"'
"$PYTHON_CMD" -u detector_replay.py \
  --manifest "${MANIFEST:-dataset/detector_expanded_review/manifest.json}" \
  --split "${SPLIT:-outputs/phase4_with_progress/split.json}" \
  --benchmark "${BENCHMARK:-outputs/completion_pretrained_vm}" \
  --checkpoint "${CHECKPOINT:-outputs/completion_pilot/epoch_2.pth}" \
  --output_dir "${OUTPUT:-outputs/detector_replay_vm}" \
  --epochs "${EPOCHS:-10}" --sources 200 --steps 21 --batch_size 8 --lr 0.00001 --seed 42
