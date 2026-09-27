#!/usr/bin/env bash
# Matched detector-only experiment; generator and reserved test split stay untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON_CMD:-python3}"
MANIFEST="${MANIFEST:-dataset/detector_expanded_review/manifest.json}"
CHECKPOINT="${CHECKPOINT:-outputs/completion_pilot/epoch_2.pth}"
OUTPUT="${OUTPUT:-outputs/real_detector_balanced_vm}"
EPOCHS="${EPOCHS:-10}"
test -f "$MANIFEST"
test -f "$CHECKPOINT"
if [ -e "$OUTPUT" ]; then echo "Choose a new OUTPUT; directory already exists" >&2; exit 1; fi
"$PYTHON_CMD" -c 'import torch; assert torch.cuda.is_available(), "CUDA required for this VM recipe"'
"$PYTHON_CMD" detector_training.py --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
    --output_dir "$OUTPUT" --validate_only --balanced_batches
# Same initialization, batches, seed, learning rate and update budget in both arms.
for arm in balanced_control visible_penalty; do
    penalty=0
    if [ "$arm" = visible_penalty ]; then penalty=0.25; fi
    "$PYTHON_CMD" -u detector_training.py --manifest "$MANIFEST" --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT/$arm" --epochs "$EPOCHS" --batch_size 4 --lr 0.00001 \
        --seed 42 --balanced_batches --background_weight "$penalty" --hard_fraction 0.1
done
echo "Detector experiment complete: $OUTPUT. Review validation and false positives before test evaluation or deployment."
