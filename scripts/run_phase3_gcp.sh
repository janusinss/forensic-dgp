#!/bin/bash
# ==============================================================================
# Phase 3 Fine-Tuning: Demographic Asian Priors & Second-Order Degradation
# Deep Generative Prior (DGP) for Degraded CCTV Video in Zamboanga City
# Host: Google Cloud Platform (VM: dgp-training-vm / forensic-dgp-thesis)
# ==============================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/5] Checking Hardware Acceleration & GPU Environment ==="
nvidia-smi

PYTHON_CMD="python3"
if [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using active Conda environment: $CONDA_DEFAULT_ENV"
fi

$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}, Device Count: {torch.cuda.device_count()}')"

echo "=== [2/5] Verifying Checkpoint dgp_improved_epoch_20.pth ==="
CHECKPOINT_PATH="checkpoints/dgp_improved_epoch_20.pth"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    if [ -f "checkpoints/dgp_improved_epoch_19.pth" ]; then
        CHECKPOINT_PATH="checkpoints/dgp_improved_epoch_19.pth"
    else
        echo "ERROR: Base checkpoint '$CHECKPOINT_PATH' not found!"
        exit 1
    fi
fi
echo "Resuming from checkpoint: $CHECKPOINT_PATH"

echo "=== [3/5] Verifying Asian Face Demographic Dataset ==="
ASIAN_DIR="dataset/asian_faces"
if [ ! -d "$ASIAN_DIR" ] || [ $(ls -1 "$ASIAN_DIR" 2>/dev/null | wc -l) -lt 500 ]; then
    echo "Asian face dataset not found or incomplete. Initiating automated download..."
    $PYTHON_CMD scripts/download_asian_faces.py "$ASIAN_DIR"
fi
ASIAN_COUNT=$(ls -1 "$ASIAN_DIR" | wc -l)
echo "Verified Asian face dataset: $ASIAN_COUNT images in $ASIAN_DIR"

# Auto-detect FFHQ dataset directory
DATA_DIR="dataset/ffhq"
if [ ! -d "$DATA_DIR" ]; then
    if [ -d "dataset/thumbnails128x128" ]; then
        DATA_DIR="dataset/thumbnails128x128"
    elif [ -d "thumbnails128x128" ]; then
        DATA_DIR="thumbnails128x128"
    elif [ -d "/home/janus/ffhq" ]; then
        DATA_DIR="/home/janus/ffhq"
    elif [ -d "/mnt/disks/data/ffhq" ]; then
        DATA_DIR="/mnt/disks/data/ffhq"
    fi
fi
echo "FFHQ directory: $DATA_DIR"
COMBINED_DATA_DIR="$DATA_DIR,$ASIAN_DIR"

echo "=== [4/5] Initiating Phase 3 Training (Epochs 21 -> 26) ==="
echo "Configuration:"
echo "  - Epoch Range: 21 -> 26 (5 Fine-Tuning Epochs)"
echo "  - Datasets: FFHQ + Asian Faces ($COMBINED_DATA_DIR)"
echo "  - Degradation Engine: Second-Order Compound CCTV (Motion + Sensor Noise + Interlacing + Codec)"
echo "  - Generative Head LR: 1e-4"
echo "  - Backbone LR: 1e-5"
echo "  - Scheduler: CosineAnnealingLR (T_max=5, min_lr=1e-6)"
echo "=========================================================="

$PYTHON_CMD train.py \
    --data_dir "$COMBINED_DATA_DIR" \
    --batch_size 16 \
    --start_epoch 21 \
    --epochs 26 \
    --resume_from "$CHECKPOINT_PATH" \
    --lr 1e-4 \
    --lr_backbone 1e-5 \
    --lambda_vgg 0.15 \
    --lambda_color 0.05 \
    --lambda_fan 0.05 \
    --lambda_sobel 0.10 \
    --lambda_fft 0.05

echo "=== [5/5] Phase 3 Fine-Tuning Completed Successfully! ==="
if [ -f "checkpoints/dgp_improved_epoch_26.pth" ]; then
    cp "checkpoints/dgp_improved_epoch_26.pth" "checkpoints/dgp_zamboanga_final.pth"
    echo "Saved final weights to checkpoints/dgp_zamboanga_final.pth"
fi
