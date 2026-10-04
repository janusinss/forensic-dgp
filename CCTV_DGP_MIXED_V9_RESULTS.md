# V9 mixed-source restoration results — 4 October 2026

**Decision:** completed and independently audited; no trained snapshot passed.
`best.pth` retains the starting V2 baseline, selected epoch 0. The original Phase3
production checkpoint and main-app defaults have not changed. The active Goal
still requires a useful DGP-led CCTV restoration workflow.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`; VM root: `~/forensic-dgp/`.
This report's intended VM path is `~/forensic-dgp/CCTV_DGP_MIXED_V9_RESULTS.md`
after document sync; no sync or git commit is claimed.

## Executed finite experiment

781 reviewed training references (391 HQ FFHQ / 390 Asian replay), all five
clear/degradation profiles, twenty epochs, batch 10, **7,820 optimizer updates**
and **78,200 exposures**. The unchanged 104 validation references provide 520
cases. The trainer finished in **1,356.35 seconds (22.61 minutes)**;
training, VM audit and export took **1,475.78 seconds (24.60 minutes)**.
The fixed caps were 2,400 / 3,600 seconds. CUDA preflight had one autograd call
and zero optimizer updates. PyTorch `2.9.1+cu129` / torchvision `0.24.1+cu129`
remained unchanged on the NVIDIA L4.

Recipe and source review are preserved in
`C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_MIXED_V9.md` ↔ intended
`~/forensic-dgp/CCTV_DGP_MIXED_V9.md` after sync. The executed root is
`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_mixed_vm_v9_r2\` ↔
`~/forensic-dgp/cctv_dgp_mixed_vm_v9_r2/`.

## Paired photographic-proxy results

PSNR here is derived from mean exported-PNG MSE over the 416 degraded cases;
it is not native CCTV PSNR or evidence of recovered identity. ArcFace uses the
same fixed observed-region affine and frozen recognizer across every snapshot.

| Snapshot | Degraded PSNR dB | SSIM | ArcFace similarity | Selected |
| --- | ---: | ---: | ---: | --- |
| Starting V2 | 16.0267 | 0.61930 | 0.33011 | Retained baseline |
| Epoch 2 | 19.6881 | 0.63973 | 0.29692 | No |
| Epoch 5 | 20.2649 | 0.64410 | 0.30494 | No |
| Epoch 10 | 20.7510 | 0.64809 | 0.30747 | No |
| Epoch 20 | 21.1986 | 0.65281 | 0.30431 | No |

At epoch20, pixel MSE and SSIM improve in every source/profile group, including
blur. However, ArcFace falls in **all eight degraded source/profile groups**.
Asian blur PSNR gains 0.9258 dB while similarity falls 0.06694; HQ FFHQ blur
gains 0.7764 dB while similarity falls 0.02356. Asian degraded similarity falls
0.03893 and FFHQ degraded similarity falls 0.01317. Clear preservation improves:
PSNR 31.1323→43.3459 dB, SSIM 0.92512→0.99479.

The unchanged strict guard rejects all four trained snapshots. A pixel gain
cannot override appearance safeguards. The retained serialized `best.pth` SHA256
is `646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`;
its tensors equal the V2 starting state, not an improved trained model.

## Original-cell visual inspection

All five 10-row grids (baseline, epochs2/5/10/20) were inspected at their original
256×256 cells. The grids repeat **two references** across five profiles; they
are not ten different people or an independent human review.

Clear faces are better retained in trained snapshots. Degraded eyes, noses and
mouths remain soft; motion/block residuals and warm/pink/green shifts remain on
some cases. The low-resolution Asian reference supplies no new HQ detail.
These observations do not demonstrate a useful released CCTV restorer. No
trained V9 native forwards, reserved-native use or app promotion occurred.

## Returned evidence and VM closure

Return: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-dgp-mixed-v9-results.tar.gz`
↔ `~/forensic-dgp/cctv_dgp_mixed_vm_v9_r2/cctv-dgp-mixed-v9-results.tar.gz`.
Bytes **319,681,838**; SHA256
`e12a68b42e36aa1b98d1eedd6aba303a2b625a8acb5a4e53815248eb5dc02a8c`.
Frozen protocol SHA256
`6cd9ac8d5f3bf27be773979c2f628e33f5d3cf09748452eeab91a65b05b01c70`.

Local extraction: `C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv_dgp_mixed_return_v9\`.
Its `outputs/cctv_dgp_mixed_v9/` corresponds to the executed VM output directory.
The **113.84-second independent local audit** rebuilt 2,600 PNG metrics, 50 raw
float previews, 520 input metrics, 3,120 embedding cosines, all 7,820 traces and
78,200 exposures. It checked four changed checkpoints (158 changed parameter
tensors each), loss-weight arithmetic, 3,905 reported calibration rows and20
independent raw-MSE checks. It did not replay CUDA gradients or recognizer
forwards. No local model forward, backward or optimizer update occurred in
this return audit.

Local receipts: `local_independent_audit.json`, `vm_source_bindings.json`,
`paired_diagnosis_v9.json` and `assistant_preview_review_v9.json` beneath the
extraction root. VM audit: executed output `independent_audit_vm.json`.
Local stop receipt: `outputs/cctv_dgp_mixed_v9_vm_stop_receipt.json` under the
Windows root; it is not claimed transferred to the VM.

After confirming no GPU jobs or experiment tmux session remained, the assistant
restored the VM to its prior stopped state. Cloud reports **TERMINATED**, stop
timestamp `2026-10-04T05:06:53.104-07:00`. Reserved32 remain untouched.

## Next decision

The user authorizes a pretrained face-generating prior with our own trained
conditioning layers **if reviewed output improves**. First freeze and run a
separate inference-only feasibility comparison on existing inputs, explicitly
declaring the pretrained baseline and any diagnostic DGP conditioning. Inspect
structure, detail and source-specific limitations before another finite VM
training recipe. Do not relabel CodeFormer as our trained DGP, deploy a failed
V9 snapshot, loosen the historic guard or repeat this recipe unchanged.
Research rationale: `C:\xampp\htdocs\YEAR 4\Testing\CCTV_DGP_ARCHITECTURE_REVIEW.md`
↔ intended `~/forensic-dgp/CCTV_DGP_ARCHITECTURE_REVIEW.md` after sync.
