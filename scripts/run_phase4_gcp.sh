#!/usr/bin/env bash
# Run on the GPU host from the repository root. Optional: DATA_DIR and OUTPUT_DIR.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON_CMD:-python3}"
"$PYTHON_CMD" -c 'import torch; assert torch.cuda.is_available(), "Phase 4 requires a CUDA-enabled PyTorch environment"'
DATA_DIR="${DATA_DIR:-dataset/thumbnails128x128}"
if [ -d dataset/asian_faces ] && [ "$DATA_DIR" = dataset/thumbnails128x128 ]; then
    DATA_DIR="$DATA_DIR,dataset/asian_faces"
fi
"$PYTHON_CMD" -u train.py \
    --data_dir "$DATA_DIR" \
    --resume_from "${RESUME_FROM:-checkpoints/dgp_zamboanga_final.pth}" \
    --start_epoch 27 --epochs 31 \
    --batch_size 16 --num_workers 2 \
    --heavy_blur_probability 0.35 \
    --validation_fraction 0.05 --seed 42 \
    --lr 5e-5 --lr_backbone 5e-6 \
    --output_dir "${OUTPUT_DIR:-outputs/phase4}" "$@"
