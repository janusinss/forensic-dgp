# Detector label audit and next experiment — 28 September 2026

Reviewed contact sheets for all 42 covered training crops and 14 covered validation
crops. These are visual checks of approximate assistant polygons, not independent
expert adjudication. No labels, splits or selection thresholds were changed.

## Findings

- Training already includes dark, patterned, surgical and respirator coverings.
  Their presence alone has not produced robust detection: replay validation masks
  remain fragmented, especially on pleats, printed texture and dark cloth.
- Validation `images/new_covered_03.png` includes an exposed cheek cutout inside
  its polygon. Treat this as a documented label defect, not a reason to weaken
  selection globally. A corrected version requires a versioned manifest and
  re-evaluation of every baseline before comparing results.
- Validation `images/new_covered_40.png` is a mannequin case. Aggregate validation
  is not exclusively photographs of real people. Report this limitation; do not
  silently remove difficult/nonhuman cases after inspecting model scores.
- Fine straps and curved edges are approximate polygons. Main-mask omissions are
  much larger than boundary differences. Label noise alone does not explain the
  64.82% missed coverage at replay epoch 10.

Sheets: local `c:\xampp\htdocs\YEAR 4\Testing\outputs\detector_replay_review\train_label_audit.jpg`
and `validation_label_audit.jpg`; VM counterparts under
`~/forensic-dgp/outputs/detector_replay_review/` only after transfer.

## Bounded experiment

Test one change: frozen-original-detector consistency on synthetic training replay.
The objective adds weight 1 times Bernoulli KL from original detector probabilities
to student probabilities. It is computed only on synthetic training examples;
real samples retain supervised label loss. Teacher weights receive no gradients.
The generator remains frozen. This tests forgetting mitigation, not a proven cure
for fragmented real masks. Weight 1 is an experimental setting, not an optimum.

Use the original completion epoch 2 initialization, 200 source images, seed 42,
batch 8, 21 updates per epoch, 10 epochs, and unchanged real/synthetic gates.
Compare against the completed replay run; verify source lists and manifest hashes
match before attributing differences to consistency. No additional epochs or
automatic promotion if the strict gates fail.

After pushing the reviewed code, paste into VM SSH:

```bash
cd ~/forensic-dgp
git pull --ff-only origin main
tmux new-session -A -s dgp_training
```

Inside tmux (one GPU training process at a time):

```bash
cd ~/forensic-dgp
if [ -d venv ]; then source venv/bin/activate; fi
bash scripts/run_detector_consistency_gcp.sh
```

Export after completion:

```bash
cd ~/forensic-dgp
tar -czf ~/detector-consistency-results.tar.gz outputs/detector_consistency_vm
echo "$HOME/detector-consistency-results.tar.gz"
```

Use SSH Download File with the printed path. Local destination:
`c:\xampp\htdocs\YEAR 4\Testing\outputs\detector-consistency-results.tar.gz`.
Next review: both validation curves and fixed mask overlays; completion images only
after a detector candidate qualifies. Do not reuse the seven previously inspected
test cases to tune this objective.
