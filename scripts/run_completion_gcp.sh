#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON_CMD:-python3}"
"$PYTHON_CMD" -c 'import torch; assert torch.cuda.is_available(), "Completion training requires CUDA on the VM"'
SPLIT_FILE="${SPLIT_FILE:-outputs/phase4_with_progress/split.json}"
if [ ! -f "$SPLIT_FILE" ]; then
    echo "Missing split: $SPLIT_FILE. Restore the original manifest or explicitly supply a new split."
    exit 1
fi
"$PYTHON_CMD" -u train_completion.py \
    --data_dir "${DATA_DIR:-dataset/thumbnails128x128,dataset/asian_faces}" \
    --split_file "$SPLIT_FILE" \
    --restorer "${RESTORATION_CHECKPOINT:-checkpoints/dgp_zamboanga_final.pth}" \
    --output_dir "${OUTPUT_DIR:-outputs/completion_pilot}" "$@"
