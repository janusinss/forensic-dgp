# Fixed additional-fitting pilot — 30 September 2026

Hypothesis: additional fitting from the pretrained detector improves synthetic
mask precision/recall while preserving real covering detection. The current
state has training replay IoU0.84827 and validation0.84378, below original parent
0.96904/0.97469. Real validation IoU is0.82283. This is an unproven intervention;
`FACE_OCCLUSION_REPLAY_RESULTS.md` records the supporting diagnostic.

## Fixed recipe

| Item | Specification |
| --- | --- |
| Start | Verified pretrained epoch10, SHA256 `cdd1752bce8e4a087ce2aac5af73ba81b6316ac9118e47f12acd7a24fcb62669`;210 completed model updates |
| Budget/order |20 additional epochs x21 batches x8 images =420 fresh optimizer steps; two independent copies of the original ten-epoch extended schedule |
| Data | Same73 real labels,638 cached replay cases, source images/splits/masks and four-way2/2/2/2 batch balance; no new data or augmentation |
| Optimization | Fresh AdamW; encoder LR1e-5, decoder/head LR1e-4, weight decay1e-4, clip1, seed42; all encoder/decoder/head parameters trainable, original frozen head/BatchNorm statistics preserved |
| Loss/checks | Same BCE+softDice+0.25 hard-visible penalty and weight1 synthetic teacher KL; candidate checks at additional epochs1,5,10,20 |

This is **weight continuation with an optimizer reset**. The original pilot did
not save optimizer moments. It is not exact optimizer-state resumption and cannot
isolate the effect of additional updates from that reset. There is no new random
initialization arm, rate/loss sweep, architecture change or adaptive budget.

Candidate global epochs are11,15,20,30; cumulative model updates are231,315,420,
630. Fresh optimizer updates are21,105,210,420. Logs distinguish both counters.
The frozen original completion parent supplies the same teacher and selection
baseline. Initial source metrics must reproduce their existing logged values
before optimization. Selecting a checkpoint requires the original real/synthetic
gates, and further improvement over any previously eligible real IoU in this run.
The starting checkpoint itself is not eligible because synthetic retention fails.

## Evidence and stopping rule

Stop exactly after420 additional steps, or preserve the failed run if finiteness/
integrity checks fail. No automatic retry, deletion, resume or additional epoch.
CPU/Windows execution refuses before file/model/optimizer work. Preflight checks
existing and new inventories, source/cache/protocol hashes and a single mixed
CUDA batch with zero additional updates. Actual optimization is VM-only.

Export exact source bytes as`initial.pth`, source and common-parent metrics/masks,
all candidate states/masks, training-fit and human/mannequin/glare scores, every
batch index/loss/norm, runtime and peak CUDA memory. Preserve frozen source state
and original parent/generator. Save final AdamW moments in`final_optimizer.pth`,
bound by hash to the final epoch30 state and tagged with420 fresh/630 cumulative
updates. Intermediate/best optimizer snapshots are not saved.

Four candidate states plus the initial source total approximately287MB before
compression. Final optimizer moments add approximately115MB; an eligible best
model copy adds another57MB. Runtime remains to be measured; the previous420-step
comparison reported112.13 seconds, excluding hashing/setup/archive compression.
Success here does not automatically authorize deployment or prove hidden identity.

## Paths and remaining requirements

Local:`C:\xampp\htdocs\YEAR 4\Testing\`.
VM:`~/forensic-dgp/coverage_vm_bundle/`.
Local source:
`outputs/downloaded_face_occlusion/outputs/face_occlusion_pilot_vm/pretrained/epoch_10.pth`.
VM source:`outputs/face_occlusion_pilot_vm/pretrained/epoch_10.pth`.
VM output:`outputs/face_occlusion_continuation_vm/`.
Return:`face-occlusion-continuation-results.tar.gz` at the VM workspace root,
downloaded to the Windows root's`outputs\` folder.

After return: verify exact420 new updates/source bytes, all four checkpoint masks,
frozen state and final optimizer binding, then independently recount selection.
Only an eligible candidate advances to reviewed end-to-end single-image completion
and restoration. Strong lens glare is still unresolved and must be checked
explicitly. External FFHQ pretraining overlap remains unknown and the development
set has been reused; final generalization is unproven. Keep the original generator,
Phase3 restoration and application behavior until measured/visual evidence
satisfies all requirements. If this pilot fails, record that outcome and inspect
its learning/precision evidence before another intervention. The goal remains open.
