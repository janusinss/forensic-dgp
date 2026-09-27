# Detector replay experiment

The real-only epoch 9 detector improved real-mask IoU but reduced synthetic IoU
from 0.97469 to 0.91828 and introduced false masks on 9/80 uncovered synthetic
cases. It is not the production detector.

This experiment starts again from completion_pilot/epoch_2.pth. Every batch of
eight contains two real covered faces, two real uncovered faces, two synthetic
covered faces, and two synthetic uncovered faces. The generator stays frozen.
Ten epochs of 21 updates preserve the earlier update count and real-example
exposure; total batch size doubles to add replay. This is an experimental recipe,
not evidence of better output until results are reviewed.

Synthetic training uses 200 deduplicated source images from the original training
split, with fixed clear/degraded variants of lower, eyes, object and irregular
coverings plus uncovered controls. Original validation/test hashes, all known
real-review source/crop hashes, and benchmark source hashes are excluded. Missing
split files or source paths fail explicitly. Byte hashes do not establish
identity-disjointness or detect differently encoded duplicate photos.

Each epoch evaluates the 25 real validation images and existing 400 synthetic
benchmark cases separately. Selection requires higher real IoU without worsening
real visible false positives, empty detections or uncovered false-positive cases.
Synthetic IoU, missed fraction, visible false positives, empty detections and
uncovered false positives must all retain their initialization baseline. There is
no automatic threshold relaxation if no checkpoint qualifies. The seven earlier
real test images have already been inspected; they are not a fresh final test.

## Existing Google Cloud VM

First push the local code changes to main. In Google Cloud SSH:

```bash
cd ~/forensic-dgp
git pull origin main
tmux new -s detector_replay
```

Inside tmux:

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
bash scripts/run_detector_replay_gcp.sh
```

Required existing inputs: dataset/detector_expanded_review/manifest.json and its
100 image/mask pairs; outputs/phase4_with_progress/split.json and all listed images;
outputs/completion_pretrained_vm/{manifest.json,input,mask}; and
outputs/completion_pilot/epoch_2.pth. Override SPLIT, BENCHMARK, CHECKPOINT or OUTPUT
environment variables if their actual VM locations differ. Do not substitute a
smoke split. The launcher refuses an existing output directory.

After completion:

```bash
tar -czf ~/detector-replay-results.tar.gz outputs/detector_replay_vm
echo "$HOME/detector-replay-results.tar.gz"
```

Use Google Cloud SSH's Download File action with that printed absolute path.
Next: review both validation curves, mask overlays and completion outputs. Only
consider generator fine-tuning after reliable covering detection and preservation
of visible features. Collect a new independently reviewed holdout for final
evaluation before deployment.
