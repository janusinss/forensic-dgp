# CCTV DGP perceptual comparison: audited return, 4 October 2026

No V3 epoch qualified. Retain V2 identity epoch2 for research; no new checkpoint
or application default is approved. The full DGP-first Goal remains active.

Windows workspace: `C:\xampp\htdocs\YEAR 4\Testing\`.
VM workspace: `~/forensic-dgp/cctv_dgp_vm_bundle/` on the existing NVIDIA L4.
This report's intended VM copy is `~/forensic-dgp/CCTV_DGP_PERCEPTUAL_RESULTS.md`
after explicit document transfer. The frozen V3 runbook and manifests stay intact.

## Return and independent audit

Local archive:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-perceptual-v3-results.tar.gz`
plus its LF `.sha256`; VM source:
`~/forensic-dgp/cctv_dgp_vm_bundle/cctv-dgp-perceptual-v3-results.tar.gz`.
Archive size: 347,331,047 bytes. SHA256:
`f5efd077c7ef8254cbb395dd6bc4ff4b9b66f6472d3f661de9ca33dadc4a1674`.

The extracted Windows root is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_perceptual_return_v3\`.
Canonical outputs beneath it are `outputs\cctv_dgp_perceptual_v3\`, corresponding
to VM `~/forensic-dgp/cctv_dgp_vm_bundle/outputs/cctv_dgp_perceptual_v3/`.
Returned `results.json` SHA256:
`cdf603f35a62398c48c11f967da993792c8fad5cb79c42db7e61bf74363068d2`.

The independent local audit passed: 2,750 exported PNG metrics, 50 raw float
previews/compositions, 3,300 embedding cosines, 452 update records, 52 pinned
recognizer preview forwards, and four checkpoints each changing293 parameter
tensors. Frozen normalization buffers and teacher tensors remained unchanged.
The complete L4 run used452 optimizer updates in332.78 seconds against a
5,400-second limit. No local restoration forwards, backward calls or optimizer
updates were used by the returned-output audit.

Local receipt:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_perceptual_return_v3\local_independent_audit.json`.
SHA256: `2010cb66ea1571972367775bb1dbb1100e8d4c595e253ecb2fe9c1416ce461b1`.
The received VM receipt remains unchanged. The local receipt is bundled separately
for the next diagnostic; it does not overwrite the VM receipt.

## Paired results and guard failures

The fixed comparison has110 validation references:51 Asian-source and59 FFHQ-source,
with440 degraded and110 clear cases. These are photographs with synthetic camera
degradation, not native CCTV. PSNR is derived from mean observed-support MSE;
ArcFace is fixed-grid embedding similarity, not identification accuracy.

| Stage | Degraded PSNR | Degraded SSIM | Degraded MAE | Degraded ArcFace | Clear PSNR | Qualified |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Retained V2 start | 16.2167 | 0.6559 | 0.10800 | 0.32686 | 32.3735 | Research baseline |
| Postactivation epoch1 | 16.6068 | 0.6625 | 0.10338 | 0.32825 | 33.5621 | No |
| Postactivation epoch2 | 17.1393 | 0.6635 | 0.09882 | 0.32561 | 33.4443 | No |
| Calibrated preactivation epoch1 | 16.7089 | 0.6653 | 0.10238 | 0.32039 | 33.4065 | No |
| Calibrated preactivation epoch2 | 17.3187 | 0.6671 | 0.09718 | 0.31471 | 32.9941 | No |

The unchanged guard requires at least0.1dB aggregate degraded PSNR improvement
while preserving MSE, SSIM and fixed ArcFace in every aggregate/source/profile
group. Postactivation epoch1 improves aggregate identity similarity, but Asian
low-light similarity declines0.00647 and both source blur MSE values regress;
Asian motion MSE also regresses. Postactivation epoch2 lowers aggregate similarity
by0.00125 and introduces further blur/motion and Asian identity regressions.
Preactivation epoch1 andepoch2 lower aggregate similarity by0.00647 and0.01215;
epoch2's Asian-source similarity drops0.01470 and FFHQ-source similarity0.00995.
Average brightness/pixel improvement does not remove these structure safeguards.

| Source degraded group | Retained PSNR / ArcFace | Post epoch2 | Pre epoch2 |
| --- | --- | --- | --- |
| Asian dataset | 15.9959 / 0.41642 | 16.9398 / 0.41130 | 17.1033 / 0.40172 |
| FFHQ dataset | 16.4171 / 0.24944 | 17.3194 / 0.25153 | 17.5139 / 0.23949 |

Both V3 `best.pth` files selectepoch0. Their tensor-state fingerprint equals the
retained V2 start (`d89f4e52777b5260e94a3e533aa21a0204cbd79b578137881fc6a4c67f1bb0c3`);
their serialized file SHA256 is
`2349624c2416870a3748debcd4cb1076a1f1c09af5fd90f7445821fd0ba48444`.
Serialization bytes differ from the original V2 checkpoint, while tensor values
are identical. All550 V3 baseline PNGs also exactly match V2 identityepoch2.
Do not call these fallback files newly improved trained models.

## Development visual review and next experiment

The assistant inspected allfive fixed10-row grids: baseline and each of the four
trained epochs. Each grid covers two fixed references with five degradation
profiles. Clear controls remain coherent, while blur and compound results retain
soft smears and indistinct eye/mouth detail. Low-light faces become brighter;
the stronger preactivation change also lightens a clear control. The preview
does not demonstrate useful recovered structure or a preactivation advantage.
It is a small paired diagnostic, not broad native/population validation.

Separate local ledgers preserve the immutable return:
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_perceptual_v3_review\analysis.json`
and `visual_review.json` (optional VM copies under
`~/forensic-dgp/outputs/cctv_dgp_perceptual_v3_review/` after explicit transfer).
Analysis SHA256: `b95af85a7b3e40480c5608a5210afa439a4ad8da540be7e1ed33f17c08ce0fcb`.
Visual ledger SHA256: `809e8417fc6a323cf7b6b5138afd56b4484226477c2fb0981efe6287c8f23f52`.

There is no newly eligible V3 candidate for native forwarding; the earlier V2
native comparison remains authoritative. No V3 native forwards were run and
the32 reserved native crops remain untouched. Do not repeat either failed V3
recipe unchanged or relax the guards to accept aggregate PSNR.

Next: run the separate VM-only zero-update objective diagnostic described in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_OBJECTIVE_DIAGNOSTIC_V4.md`, bundled at
`~/forensic-dgp/cctv_dgp_vm_bundle/CCTV_DGP_OBJECTIVE_DIAGNOSTIC_V4.md` after transfer.
It measures weighted loss-gradient directions on40 training-only cases with a
10-minute cap. Its return informs a changed finite training proposal; it does
not itself train, select a checkpoint or prove useful outputs. DGP-led app
integration, useful native restoration, covering-family verification, Playwright
workflow verification and independent final assessment remain incomplete.
