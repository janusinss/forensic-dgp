#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON_CMD:-python3}"
DATA_DIR="${DATA_DIR:-dataset/thumbnails128x128,dataset/asian_faces}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/phase5_identity}"
SPLIT_FILE="${SPLIT_FILE:-outputs/phase4_with_progress/split.json}"
ARCFACE_MODEL="${ARCFACE_MODEL:-$HOME/.insightface/models/buffalo_l/w600k_r50.onnx}"
CACHE_DIR="${CACHE_DIR:-outputs/landmark_cache}"
"$PYTHON_CMD" -c 'import torch, onnx2torch; assert torch.cuda.is_available(), "Phase 5 needs CUDA PyTorch"'
if [ ! -f "$SPLIT_FILE" ]; then
    echo "Missing Phase 4 split: $SPLIT_FILE. Set SPLIT_FILE to the completed run's split.json."
    exit 1
fi
echo "Checking ArcFace conversion and gradients..."
"$PYTHON_CMD" -u scripts/check_phase5_identity.py --model "$ARCFACE_MODEL" --device cuda
echo "Preparing reusable landmarks (first run only needs detection)..."
"$PYTHON_CMD" -u scripts/prepare_phase5_landmarks.py --data_dir "$DATA_DIR" --cache_dir "$CACHE_DIR" --device cuda
echo "Starting a separate two-epoch Phase 5 pilot from Phase 3..."
"$PYTHON_CMD" -u train_phase5.py \
    --data_dir "$DATA_DIR" --split_file "$SPLIT_FILE" \
    --checkpoint "${CHECKPOINT:-checkpoints/dgp_zamboanga_final.pth}" \
    --arcface_model "$ARCFACE_MODEL" --landmark_cache "$CACHE_DIR" \
    --output_dir "$OUTPUT_DIR" --epochs 2 --batch_size 8 --num_workers 2 \
    --lr 1e-5 --lr_backbone 2e-6 --lambda_identity 0.1 --ema_decay 0.999 "$@"
