#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON_CMD:-python3}"
BENCHMARK="${BENCHMARK:-outputs/completion_pretrained_vm}"
SPLIT_FILE="${SPLIT_FILE:-outputs/phase4_with_progress/split.json}"
DATA_DIR="${DATA_DIR:-dataset/thumbnails128x128,dataset/asian_faces}"
DETECTOR="${DETECTOR:-outputs/completion_pilot/epoch_2.pth}"
RESTORER="${RESTORER:-checkpoints/dgp_zamboanga_final.pth}"
"$PYTHON_CMD" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required for this VM benchmark"'
"$PYTHON_CMD" scripts/download_completion_weights.py
"$PYTHON_CMD" completion_benchmark.py prepare --data_dir "$DATA_DIR" --split_file "$SPLIT_FILE" \
    --output_dir "$BENCHMARK" --images_per_source "${IMAGES_PER_SOURCE:-20}" --size 256
for backend in custom codeformer; do
    checkpoint="$DETECTOR"
    if [ "$backend" = codeformer ]; then checkpoint=checkpoints/codeformer_inpainting.pth; fi
    for mode in oracle predicted; do
        "$PYTHON_CMD" -u run_completion_benchmark.py --benchmark "$BENCHMARK" \
            --output_dir "$BENCHMARK/${backend}_${mode}" --backend "$backend" \
            --checkpoint "$checkpoint" --mask_mode "$mode" --detector "$DETECTOR" --restorer "$RESTORER"
    done
done
echo "Benchmark complete: $BENCHMARK. Review outputs and metrics before any training."
