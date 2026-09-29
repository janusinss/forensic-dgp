# Matched border-loss experiment — VM only

Upload `C:\xampp\htdocs\YEAR 4\Testing\scripts\train_expanded_border_vm.py`
to VM home using Google Cloud SSH. This single script reuses the original expanded
bundle and frozen feature caches; no git pull or dependency reinstall is needed.

Attach the training terminal:

```bash
tmux new-session -A -s dgp_training
```

Inside tmux:

```bash
cd ~/forensic-dgp/expanded_feature_bundle &&
source ../feature_vm_bundle/.venv/bin/activate &&
python -u ~/train_expanded_border_vm.py --root "$PWD" --preflight &&
python -u ~/train_expanded_border_vm.py --root "$PWD"
```

Preflight verifies original bundle hashes, source membership, every regenerated
synthetic input/target, full schedule coverage and one forward loss; it performs
zero optimizer updates. Training refuses CPU execution and existing output paths.
Both arms initialize from the same fixed-placement epoch-20 parent with fresh
optimizers, 800 updates each, learning rate 1e-4, seed 42, batch 12. Presence head
and encoder are frozen. Control pixel loss is BCE+Dice; treatment doubles existing
degraded synthetic border BCE weights with per-image normalization. Targets,
dataset membership and thresholds stay unchanged. Final checkpoints only.

Three local tests pass: control value/gradient equivalence, normalized weighting
and invalid-border rejection, CPU refusal before file access. The 10-epoch schedule
was independently checked to cover all 3,540 cases. No local optimization occurred.
GPU preflight/training remain pending. Full runtime has not been measured.

This is a segmentation diagnostic. The frozen gate retains known validation
rejections, so this experiment alone cannot satisfy all original retention gates.
Do not promote either checkpoint from training-fit improvement. Return both final
arms for unchanged validation and visual comparison; address presence failures
separately without lowering thresholds against validation examples.

After DONE, download:

```text
/home/janusdominic0/forensic-dgp/expanded_feature_bundle/expanded-border-results.tar.gz
```

Save to `C:\xampp\htdocs\YEAR 4\Testing\outputs\expanded-border-results.tar.gz`.
Output on VM: `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_border_training/`.
Checkpoints: `control_epoch_10.pth`, `border2_epoch_10.pth`. The archive also includes
protocol, loss histories, original training-fit metrics, region counts and script.
Next: verify matched provenance and evaluate both arms against unchanged safeguards.
Generator/application remain on their existing baselines.
