#!/bin/bash
# ==============================================================================
# Optimal Face Reconstruction Retraining Script (Epochs 11-20)
# Deep Generative Prior (DGP) for Degraded CCTV Video in Zamboanga City
# Host: Google Cloud Platform (VM: dgp-training-vm)
# ==============================================================================

set -e

# Navigate to project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "=== [1/5] Checking Environment and Hardware Acceleration ==="
nvidia-smi

# Check Python environment
PYTHON_CMD="python3"
if [ -d "venv" ]; then
    echo "Activating virtual environment (venv)..."
    source venv/bin/activate
elif [ -n "$CONDA_DEFAULT_ENV" ]; then
    echo "Using active Conda environment: $CONDA_DEFAULT_ENV"
fi

echo "Python version: $($PYTHON_CMD --version)"
$PYTHON_CMD -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA Available: {torch.cuda.is_available()}, Device Count: {torch.cuda.device_count()}')"

echo "=== [2/5] Verifying Dataset and Checkpoints ==="
CHECKPOINT_PATH="checkpoints/dgp_improved_epoch_10.pth"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "ERROR: Base checkpoint '$CHECKPOINT_PATH' not found!"
    echo "Please ensure dgp_improved_epoch_10.pth is present in the checkpoints/ folder."
    exit 1
fi
echo "Checkpoint verified: $CHECKPOINT_PATH"

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
echo "Dataset directory: $DATA_DIR"

echo "=== [3/5] Starting Phase 2 Retraining (Epochs 11 -> 20) ==="
echo "Configuration:"
echo "  - Start Epoch: 11"
echo "  - End Epoch: 20"
echo "  - Generative Head LR: 5e-5"
echo "  - Backbone LR: 5e-6"
echo "  - Scheduler: CosineAnnealingLR (T_max=10, min_lr=1e-6)"
echo "  - Multi-Loss Components: Component Attention, Multi-Layer VGG19, Color, Sobel, FFT, FAN"
echo "  - Curriculum Degradation: Active (40% sub-32, 30% mid-range, 30% native)"
echo "=========================================================="

# Run training directly in TTY for steady in-place progress bar animation
$PYTHON_CMD train.py \
    --data_dir "$DATA_DIR" \
    --batch_size 16 \
    --start_epoch 11 \
    --epochs 20 \
    --resume_from "$CHECKPOINT_PATH" \
    --lr 5e-5 \
    --lr_backbone 5e-6 \
    --lambda_vgg 0.15 \
    --lambda_color 0.05 \
    --lambda_fan 0.05 \
    --lambda_sobel 0.10 \
    --lambda_fft 0.05

echo "=== [4/5] Retraining Completed Successfully! ==="
echo "Final weights saved in checkpoints/dgp_improved_epoch_20.pth"
