# Direct occlusion initialization pilot — 30 September 2026

The pretrained encoder/decoder substantially improves real covering detection
within this development benchmark. Neither initialization meets the existing
synthetic retention safeguards. No checkpoint or application change is promoted.
Strong lens glare remains missed in the one real validation glare case.

| Checkpoint | Real IoU | Real missed fraction | Clear real false-positive cases | Synthetic IoU |
| --- | ---: | ---: | ---: | ---: |
| Original completion parent | 0.06065 | 0.93160 | 1/10 | 0.97469 |
| Random initialization, epoch 10 | 0.59269 | 0.36181 | 10/10 | 0.40058 |
| Pretrained initialization, epoch 5 | 0.83405 | 0.11128 | 1/10 | 0.78597 |
| Pretrained initialization, epoch 10 | 0.82283 | 0.13168 | 0/10 | 0.84378 |

Pretrained epoch 10 real visible false-positive fraction is0.00783 versus
parent0.01809; empty covered cases fall6/15 to1/15. Human-only IoU is0.82201;
the separate mannequin case is0.83106. Real training IoU is0.85042 with no empty
covered masks or clear-case false positives. The corresponding random control's
training IoU is0.58583 with26/26 clear-case false positives. This comparison tests
the full initialization, including frozen BatchNorm statistics; it does not
isolate encoder weights from normalization or establish final population accuracy.

Pretrained epoch 10 synthetic missed fraction is0.10060 versus parent0.01648;
visible false-positive fraction is0.00969 versus0.00133; clear false-positive
cases are2/80 versus0/80. Synthetic empty covered cases remain1/320. The real
gate passes at pretrained epochs5 and10, but every candidate fails synthetic
retention. Both arms' selected-epoch lists are empty; no `best_detector.pth` exists.
Epoch5 has the higher real IoU, while epoch10 has the higher synthetic IoU;
neither is an overall qualifying checkpoint.

## Artifact and independent verification

Returned archive:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\face-occlusion-results.tar.gz`.
VM source:
`~/forensic-dgp/coverage_vm_bundle/face-occlusion-results.tar.gz`.
Size320,055,468 bytes; SHA256:
`6a27d1685717e941f691235e3ee4489f3ecd304712ce39fce046b0663567b3c7`.
The L4 run reports112.13 seconds through training/evaluation/mask export,
1,110,456,320 peak allocated CUDA bytes and1,210,056,704 reserved bytes.
That interval excludes setup, input hashing and archive compression. It is a
420-update pilot using73 real labels and638 cached replay inputs, not a full
80,000-image training run.

`scripts/audit_face_occlusion_results.py` verifies the original sent archive and
every returned provenance file, immutable local source/data inventory, original
protocol and exact candidate/file membership. Both arms'210 logged updates match
the fixed ten-epoch schedule. Candidate mean losses match their step logs.
All3,413 masks (425 parent plus six sets of498) are independently recounted,
including human/mannequin/glare subsets and unchanged selection decisions.
Four audit tests cover schedule corruption, nonfinite gradients, false
eligibility and unsafe archive aliases/paths. Their checks pass.

CPU checkpoint-to-mask reproduction completed for all3,413 masks; evidence is
in `outputs/face_occlusion_validation/reproduction.json`. Every real validation
mask matches exactly, as do all pretrained synthetic validation masks and the
original parent's425 masks. Five pixels differ across random epoch5 synthetic
and training predictions, random epoch10 synthetic predictions and pretrained
epoch10 training predictions. Their cause was not independently established;
CPU/VM bitwise identity is not claimed. The differences do not change any
selection outcome. Both initial heads are independently verified equal. All
six checkpoints preserve92 frozen reference-head/BatchNorm tensors exactly;
each changes60 encoder,30 decoder and2 new-head tensors. Checkpoint metadata
matches the audited run, and all tensors/load/forward outputs are finite.
No local optimizer updates occurred. The six versioned checkpoints remain under
`outputs/downloaded_face_occlusion/outputs/face_occlusion_pilot_vm/`.

The ten-row diagnostic `outputs/face_occlusion_validation/preview.png` uses the
first ten validation records and shows input, target, original parent, random10,
pretrained5 and pretrained10 masks. Visual inspection confirms much fuller mask
coverage and fewer false clear-face regions from pretrained initialization.
Some straps/edges remain omitted and some boundaries differ from reviewed labels.
This is a detector mask grid; no end-to-end completion improvement is claimed.

Remote parent invariance is checked by the executed runner and reported true;
the remote parent tensors were not returned for independent comparison. Original
local parent hash remains pinned. Intermediate optimizer states are not archived.
FaceExtraction FFHQ pretraining overlap remains unresolved; these reused
development cases cannot establish independent final accuracy. The glare subset
has one validation case and does not establish broader glare behavior.

## Next action

Retain this pretrained architecture as a development candidate. Audit synthetic
errors by covering type/degradation and compare fit on the638 cached training
replay inputs. That distinguishes
insufficient replay learning from transfer failures before specifying a bounded
VM follow-up. Preserve all existing gates and the original restoration/generator
application baseline. A qualifying detector still requires a reviewed ten-row
end-to-end single-image restoration/completion result; hidden regions remain
plausible estimates. The overall goal is open.
