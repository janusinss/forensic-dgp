# Coverage comparison results — 30 September 2026

**No checkpoint qualifies for promotion.** Neither arm passed all original
real/synthetic gates at any of ten epochs. Do not repeat this recipe unchanged.

Archive: `outputs/coverage-results.tar.gz`, 197,277,663 bytes, SHA256
`a4a29bc18b5e73a70c92be31e36dd4aa0ec72a50b3902b137c72dab10c5eec1a`.
Safely extracted 8,590 file/directory entries to `outputs/downloaded_coverage/`.
VM reported NVIDIA L4, PyTorch2.9.1+cu129, 210 updates/arm and 191.39 seconds
for the two training loops including epoch validation/export (not full startup).

| Final epoch10 metric | Control68 | Extended73 | Original baseline |
| --- | ---: | ---: | ---: |
| Real IoU | 0.32773 | 0.32442 | 0.06065 |
| Human-only real IoU | 0.34530 | 0.34495 | 0.06429 |
| Real visible FP fraction | 0.013643 | 0.012584 | 0.018086 |
| Real empty covered masks | 3/15 | 2/15 | 6/15 |
| Synthetic IoU | 0.94796 | 0.95082 | 0.97469 |
| Synthetic missed fraction | 0.036899 | 0.038027 | 0.016476 |
| Synthetic visible FP fraction | 0.002350 | 0.001725 | 0.001333 |
| Glare IoU (one validation case) | 0 | 0 | 0 |
| Mannequin IoU (one case) | 0.16104 | 0.13247 | 0.02029 |

Extended training reduces some false positives and one empty real mask, but does
not improve aggregate real overlap or transfer to the one glare case. Both lose
synthetic retention. Extended's best synthetic-IoU epoch exceeds baseline on
that metric alone (0.974870), but still fails the full gate. No thresholds changed.
Five added images and one seed support only this bounded comparison, not a claim
that broader annotation coverage cannot help.

## Independent verification

All 8,500 saved real/synthetic masks were recounted. Aggregate metrics and every
epoch's selection decision agree with the logs. Inventory and returned tensor
archive hashes match. Exact initial tensors equal the pinned parent checkpoint.
All 638 shared replay cases regenerate byte-for-byte locally. All 20 saved
checkpoints preserve the generator tensors. Both final checkpoints reproduce
all 25 real validation masks exactly on local CPU.

Final control SHA256:
`771c4347ee738d0be4d9a9ec146a8f6940c7339fcad9d530409770dc4f6f17d8`.
Final extended SHA256:
`9c251c89574156bfdb9e8594b2ba4355191f79a0d0bc89b20209fcaf75feddff`.

Evidence: `outputs/coverage_validation/results.json`, `checkpoint_audit.json`
and `preview.png`. Ten-row preview inspected: both arms show holes/fragmentation
on surgical masks and miss most patterned-mask area. The preview is a fixed
first-ten validation sample, not the full validation set. Human/mannequin/glare
metrics in the table are VM-reported subsets; aggregate masks were independently
recounted, but subset metrics were not yet separately recounted by the checker.
The original baseline report matches earlier recorded values; this turn did not
rerun all baseline/synthetic checkpoint inference. No promotion relies on those
remaining checks because all candidates already fail.

## Next bounded diagnostic

Inspect training-only predictions for the five added images under parent, control
and extended checkpoints. Determine whether the intervention learned those
examples at all before asking for more training or more labels. This can use
local inference with zero optimization. Do not use this training fit as selection
or completion-quality evidence. No new VM training now; application and generator
baseline unchanged. Hidden-face completion remains an estimate.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM run: `~/forensic-dgp/coverage_vm_bundle/outputs/coverage_training_vm/`.
