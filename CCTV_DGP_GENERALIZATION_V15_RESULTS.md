# V15 fresh-image generalization result — 4 October 2026

**The ten-face prior conditioner is not an improvement on new faces. Do not
adopt it or continue training from its fitted checkpoint.** Fresh-image processing
reproduces all50 cached training outputs, but the unchanged104-reference/520-case
development validation comparison reveals severe overfitting. All five fixed
ten-row grids were reviewed and show distorted or unrelated features, including
changed eyes, mouths, glasses, expressions and repeated facial fragments.

Windows `C:\xampp\htdocs\YEAR 4\Testing\` ↔ VM `~/forensic-dgp/`.
This report's intended VM counterpart is `~/forensic-dgp/CCTV_DGP_GENERALIZATION_V15_RESULTS.md`
after document sync. Local return `outputs\cctv_dgp_generalization_return_v15\`
↔ VM `~/forensic-dgp/cctv_dgp_generalization_vm_v15/outputs/generalization_v15/`.

## Fixed development comparison

All arms use the same RGB256 inputs, exact observation support, targets and
fixed-alignment recognizer. The starting/trained prior arms use a frozen w0
decoder without AdaIN/statistic transfer, chosen from training-only evidence
before evaluation. This modified prior arm is not official default CodeFormer.
The trained arm is the existing V14 r2 update1000 head; no training occurred.

| Arm | Degraded PSNR dB | Degraded SSIM | Degraded fixed ArcFace | Clear PSNR dB | Clear fixed ArcFace |
|---|---:|---:|---:|---:|---:|
| Retained DGP V2 | **16.027** | **0.6193** | **0.33011** | **31.132** | **0.95701** |
| Starting prior, no statistics | 14.055 | 0.4944 | 0.19348 | 24.071 | 0.59986 |
| Our fitted conditioner, no statistics | 13.987 | 0.4445 | 0.07662 | 14.081 | 0.18170 |

Degraded416 cases and clear104 cases are reported separately. PSNR uses pooled
MSE, not an average of image PSNRs. The fitted head loses2.039dB degraded PSNR
against retained DGP and fails the unchanged diagnostic preservation guards
against both comparison models. Embedding similarity also regresses in every
degraded source/profile group. A sharp face-shaped output is insufficient when
it changes the observed person's facial structure.

The V14 training improvement13.522→24.272dB therefore cannot be presented as
model generalization. The fresh/cached parity success rules out an implementation
mismatch in this tested inference path; ten-face fitting alone was an invalid
assumption for restoration of a new person's photograph. Possible pretraining
overlap, lower-resolution Asian targets and prior development use still limit
interpretation; see `CCTV_FACE_PRIOR_DATA_LIMITS.md`.

## Executed and independently verified

The L4 runner took212.12 seconds. Inference, independent VM audit and export
took296.54 seconds (4 minutes57 seconds); peak allocated VRAM1,002,894,336 bytes.
Frozen states matched, with570 DGP/encoder/classifier/conditioner forwards,
1,090 generator and2,184 recognizer forwards. Zero optimizers, backward calls
or updates; no teacher model, native24, reserved32, selection or app promotion.

The292,456,028-byte return was downloaded through gcloud CLI and safely extracted.
Independent local audit took55.49 seconds, verified1,560 PNG metrics,2,080 input/
output embedding cosines,150 declared raw previews,50 fresh/cached raw/PNG parity
cases and250 original grid cells. It made zero neural calls or training updates
and does not replay the recognizer or CUDA models. All741 source/data/head assets
and116 parent assets were reverified on the VM after completion. Visual review
covered all five profiles using the predeclared ten references, five per source.

| Evidence | Windows path relative to workspace |
|---|---|
| Verified return | `outputs\cctv-dgp-generalization-v15-results.tar.gz` |
| Independent audit | `outputs\cctv_dgp_generalization_return_v15\local_independent_audit.json` |
| Visual review | `outputs\cctv_dgp_generalization_review_v15.json` |
| VM source proof | `outputs\cctv_dgp_generalization_v15_vm_sourceproof.json` |
| Restored stopped state | `outputs\cctv_dgp_generalization_v15_vm_stop.json` |

Protocol SHA256 `6c0e1de5bcb56b755b82a3289adb4905c591f49908f02365ad6ad5fbd151f9da`.
Execution archive SHA256 `5684c7d4a64f8913893cf179d02faefe9a96b35845f0831a5f9cba4c61fc3a19`.
Result archive SHA256 `5a2c6de809b09335bd69425f7b5d9795013c16ee11f19260cdb48145cf493459`.
Results JSON SHA256 `04456a809fb57c9c77dc2d5334938fa38813aa19cc6380f6f21c44b4b4eb5af4`.

## Next approach

Prepare a separately versioned broader **code-learning diagnostic**, resetting
our conditioning head to the starting prior rather than resuming the ten-face
checkpoint. Use the existing781 training references with balanced source/profile
exposure and keep the104 development-validation references separate. Remove
the demonstrated problematic statistics objective, bound updates/time/VRAM and
stop on bad timing, nonfinite values or clear evidence that fitting fails.
Every actual training/backward call remains on the L4 VM.

More code-fitting data is a hypothesis, not a guaranteed repair. Codebook
generation also changes clear facial appearance even before our conditioning;
a DGP-preserving output path needs separate evaluation before this component
can become the main restorer. Keep the existing DGP baseline, previous failures,
strict guards and original app inference. Do not relax preservation criteria or
adopt the prior to make a failed experiment appear successful. Useful held-out
and native development outputs, DGP-led app integration and independent final
review remain unfinished; the Goal remains active.
