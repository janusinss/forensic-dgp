# Reviewed covering completion: trained R1

R1 is our trained adaptation of S-Lab CodeFormer **inpainting**, separate from
Conditioned SFT V2 restoration. The restoration model, preparation, weights,
inference and output quantization are unchanged by this integration.

The guided application uses restoration, a manually selected covering area, and R1
completion in that order. R1 receives the exact reviewed restoration-result
RGB256 canvas and a binary removal mask. It uses the retained 512-code inpainting
architecture, 512 internal inference, fidelity 1 and AdaIN off, then the admitted
256 delivery and round-to-nearest PNG. It copies every pixel outside reviewed
support exactly from the supplied restored canvas. No extra completion margin,
global post-completion restoration or training is applied.

| Separate inference asset | Bytes | SHA-256 |
|---|---:|---|
| `codeformer_completion_r1.pth` | 370764381 | `7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0` |
| `covering_detector_segmenter_v1.pth` | 2681549 | `ccc22d524073368bc4f5d6b6d2136c90d0e2a117083e4c75fa98d42d7bfff6cf` |

The R1 asset contains 497 frozen inference tensors. Training optimizer/RNG states
and research archives are excluded from deployment. Its original inpainting
checkpoint is SHA `b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797`;
official source revision is `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`.
Retain [S-Lab License 1.0](third_party/codeformer_completion/LICENSE),
[notice](third_party/codeformer_completion/NOTICE.md), and the
[40-source training attribution](third_party/codeformer_completion/TRAINING_ATTRIBUTION.json).

R1 reduced delivered-PNG masked-region L1 error by 6.185334% against original
CodeFormer on 32 adaptive training-monitor cases. Six individual cases worsened.
This is a synthetic paired training-monitor result, not independent final
accuracy, native CCTV hidden-face truth or recovered identity. Pretraining
identity overlap is unknown. No Zamboanga-specific performance claim is made.

Known defects include invented eyewear/expression, asymmetric eyes, false color
on grayscale inputs and scarf/hair/hand boundary or generated texture defects.
The later V2/V3/V4/predictor experiments did not establish a better accepted
replacement; R1 remains experimental. Integration tests and exact visible-pixel
copying do not qualify useful completion across all covering families.

The fixed nine-case integration check preserves1,521,399 outside-mask RGB bytes
across seven covering estimates and two empty controls. All nine automatic
proposals in that degraded-photo set were empty, so the seven covering masks
required assistance. The grayscale hair example still has visible blue false
color. These failures remain recorded; no automatic-detection or full-family
quality claim follows from the working editor, model loader or downloads.


The 28 October 2026 integration audit reported 1,212 saved-output checks, ZIP
CRC and manifest checks, raw-to-PNG replay, provenance and worker receipts.
All planned comparison sheets were reviewed. The full historical integration
report, including installation observations and failed transfer evidence, remains
[available in Git history](https://github.com/janusinss/forensic-dgp/blob/d1fc262942754f44c188b4f318ebde970d3010a1/INTEGRATION_STATUS.md).
These functional and preservation checks do not establish model quality.

The guided editor no longer requests automatic proposals. The separate compact
detector remains available through the underlying API and is extracted from the retained trained epoch-2
segmenter, without its old completion generator. It thresholds at 0.5 and adds
one reviewable 3-pixel proposal margin, clipped to captured image support.
It can miss coverings and mark visible regions incorrectly. Every proposal
requires review; clear glasses, ordinary hair and uncovered faces should retain
an empty corrected mask. Empty automatic predictions are not a face-quality or
uncovered-face decision. Records distinguish unchanged reviewed proposals from
manual corrections and manually supplied masks.

Insufficient-information refusal combines user review with a frontal-crop
coverage heuristic. It is not a calibrated face recognizer or universal blur
threshold. Request a clearer/less-covered crop when meaningful facial structure
is unavailable. Hidden output is one plausible estimate, not verified identity.

Inference runs entirely on local Windows CPU workers. The measured development
machine is an Intel i5-11260H with 15.69 GiB RAM; initial R1 requests took about
18–20 seconds including worker startup. The 2.5 GiB available-memory admission
check, 6 GiB worker commitment cap and 120-second stage limit do not certify a
minimum device or guarantee arbitrary-hardware timing. Model weights are
downloaded only during installation; inference requires no cloud service.

Full-family usefulness, independent blinded human final review and minimum-device
certification remain incomplete. Preserve original, restoration, reviewed mask,
model provenance and result when interpreting any estimate.
