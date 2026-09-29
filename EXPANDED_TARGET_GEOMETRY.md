# Training target geometry diagnostic — 30 September 2026

Executed `venv/Scripts/python.exe scripts/audit_expanded_target_geometry.py` locally
without a model, optimizer or label changes. This regenerated all 704 irregular
training cases (352 clean, 352 degraded) from the hashed expanded manifest. The
dataset loader verifies original training split membership, excluded hashes and
source content. Fixed placement was used; irregular geometry is shared between
placement arms. No validation/test cases were inspected by this diagnostic.

Evidence: `outputs/expanded_target_geometry/results.json` and `preview.png`.
The six preview rows are the first degraded irregular cases in manifest order,
not selected by model performance. All six were visually inspected.

| Diagnostic | Clean irregular | Degraded irregular |
|---|---:|---:|
| Cases | 352 | 352 |
| Target-derived 64x64 roundtrip IoU | 0.98305 | 0.99235 |
| Added border / total target pixels | 0% | 46.8334% |

Roundtrip: area downsample binary target to 64x64, bilinear upsample to 256x256
with align_corners=False, threshold at 0.5. This is a label-derived diagnostic;
it is neither a deployable prediction nor a proven head-capacity bound. It shows
that a simple coarse representation can preserve these target shapes much better
than the learned head's degraded-irregular training IoU (0.90155 / 0.88405).
It does not establish that the frozen image features encode the needed information.

The 17x17 target dilation adds broad borders around the visible occluder and fills
some gaps between strokes. In the inspected examples the border extends beyond
the obvious covering into surrounding face pixels. This observation is not proof
that the conservative target policy is wrong: blurred contamination and downstream
completion requirements still need evaluation. Do not remove dilation or weaken
the original retention benchmark based on this target-only result.

Next: use existing VM cached features and final heads for inference-only core,
added-border and outside-target error counts, with predicted-mask overlays for the
same six training examples. Aggregate counts alone cannot localize the errors.
That evidence is required before choosing a border-aware loss, representation or
capacity experiment. Actual training remains VM-only. Baselines stay unchanged.

Local root: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM cached features: `~/forensic-dgp/expanded_feature_bundle/outputs/expanded_feature_training/`.
