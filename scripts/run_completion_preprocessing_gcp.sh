#!/usr/bin/env bash
# Matched diagnostic comparison, no weight updates or automatic model promotion.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON_CMD="${PYTHON_CMD:-python3}"
BENCHMARK="${BENCHMARK:-outputs/completion_pretrained_vm}"
OUTPUT="${OUTPUT:-outputs/completion_preprocessing_vm}"
DETECTOR="${DETECTOR:-outputs/completion_pilot/epoch_2.pth}"
EYE_DETECTOR="${EYE_DETECTOR:-$HOME/.insightface/models/buffalo_l/det_10g.onnx}"
test -f "$BENCHMARK/manifest.json"
test -f "$EYE_DETECTOR"
test -f "$DETECTOR"
if [ -e "$OUTPUT" ]; then echo "Output exists; set OUTPUT to a new directory" >&2; exit 1; fi
"$PYTHON_CMD" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"'
"$PYTHON_CMD" scripts/download_completion_weights.py
for mode in oracle predicted; do
    for candidate in unaligned selective; do
        extra=()
        if [ "$candidate" = selective ]; then
            extra=(--experimental_alignment --eye_detector "$EYE_DETECTOR")
        fi
        "$PYTHON_CMD" -u run_completion_benchmark.py --benchmark "$BENCHMARK" \
            --output_dir "$OUTPUT/${candidate}_${mode}" --backend codeformer \
            --checkpoint checkpoints/codeformer_inpainting.pth --mask_mode "$mode" \
            --detector "$DETECTOR" "${extra[@]}"
    done
done
echo "Preprocessing comparison complete: $OUTPUT. Review metrics, alignment fallbacks and images before training."
