# V29: training capacity passes; development restoration does not qualify

The human-run L4 pilot completes all800 updates. The original frozen return
checker passes837 files,38,394,279 saved gradient values, four checkpoints,
all50 paired TRAIN outputs and500 forward-only CPU replay calls. Original
checkpoint, encoder, inactive head4, evaluation buffers and recognizer remain.
Return SHA256: `ecf88626b31dee6b0ceb65f6a2a76e534cea8079183e0efd143c6bcb16fb5ae2`;
294,561,124 bytes. Fit93.172s, worker125.980s, terminal126.799s;
peak allocated CUDA1,249,038,336 bytes. No local training or derivatives.

Final800 TRAIN landmark high-frequency error improves20.0551%, with31.5290%
and17.8315% gains in the two photographic source folders. All17 TRAIN
preservation groups pass. RGB-mean-only improvement accounts for0.07936%
of degraded pixel-MSE gain, below the unchanged20% maximum. This fixes the
demonstrated V28 capacity limitation; V28's three failures and its mean-control
five failures remain rejected. The original all14-tensor failure remains.

All50 TRAIN faces were actually viewed in ten sheets. Some coarse eyes, nose,
mouth and outline are clearer; glasses, gaze and expression still need care.
Training capacity permits a development check and does not establish utility.

The separately frozen local inference module strictly loads the two fingerprinted
models, freezes their state and applies the same support-wise RGB delta centering
as training. Candidate weights alone do not implement this path. Seven inference
contract checks pass. All50 individual-input replay cases match VM raw outputs
within2.355e-6 and PNGs within one channel byte; outside-support pixels are exact.
This module is a benchmark candidate; the existing app has not adopted it.

The fixed24 ChokePoint C1 development crops are unpaired native evidence.
Resizing, retained Phase3, original DGP, CodeFormer w1, V29 and declared Auto
use identical256 inputs. Original DGP's fresh raw values exactly match its old
cache. All144 saved cells and outputs are audited; every24 face is actually
reviewed in six sheets. V29 preserves broad face arrangement but adds no
convincing useful structural clarity over resizing across this cohort.
The previously useful `choke_dev_02_t033` remains too diffuse around finer eyes
and lips. Model softness does not reclassify a usable input as information-poor.
CodeFormer remains a visibly clearer pretrained comparison, without exact-hidden
identity claims. Capture country is unspecified in acquired ChokePoint metadata.

The separate104-reference photographic DEV cohort retains all520 synthetic
cases and five existing profiles. All520 receive independent saved-output
verification: raw projection, delivered PNGs, frozen vectors, metrics and17
source/profile groups. Ten fixed preview sheets, all50 faces, are actually
viewed. Total runtime524.852s; independent checker69.412s.

| Paired degraded photographic DEV source | Cases | V29 landmark structure gain vs original DGP |
| --- | ---: | ---: |
| All sources |416|1.3516%|
| `dataset/asian_faces` |204|−3.4051%|
| `dataset/thumbnails128x128` |212|2.3437%|

All17 groups lose frozen ArcFace similarity. Four additional group/metric
failures concern Asian-folder clear MSE/SSIM, compound SSIM and motion MSE:
21 total. Overall degraded ArcFace drops0.3301111 to0.2809771; this is an
embedding preservation diagnostic, not identification accuracy. Degraded
MSE/SSIM improve overall, but those improvements do not override these failures.
Mean-only fraction0.17679% remains small. Several viewed outputs sharpen edges
while narrowing eyes or altering mouth shape and expression. Reject app promotion.

The hypothesis for the next finite pilot is inadequate identity/appearance
coverage: V29 repeatedly fits just10 references. Broaden only the training
coverage to the already approved781 TRAIN references /3905 existing cases,
keeping800 updates, original initialization, the same mean-centered path,
selected12 tensors, losses, optimizer and numeric preservation gates. The
existing50-case pre-optimizer proof and its loss normalizers stay fixed.
This tests a hypothesis; it does not establish coverage as the unique cause.
The104 DEV references remain outside optimization. No failed historical
checkpoint, optimizer or unchanged recipe is resumed.

Source folders are provenance categories, not inferred ethnicity. Historical
person/pretrained overlap remains unknown. No Zamboanga sample or performance
claim exists. The separate58 reserved native cases /45 namespaced identities
remain unopened. Paired PSNR/SSIM stay separate from unpaired native review.
All seven covering families, useful restoration, full app verification and
independent final review remain required; the overall goal remains active.

Evidence: `outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json`,
`outputs/cctv_dgp_mean_centered_decoder_v29_visual_review/visual_review.json`,
`outputs/cctv_dgp_v29_single_input_parity_v1/results.json`,
`outputs/cctv_dgp_v29_native_development_v1/saved_output_audit.json`,
`outputs/cctv_dgp_v29_native_development_v1/visual_review.json`,
`outputs/cctv_dgp_v29_paired_development_v1/saved_output_audit.json`,
`outputs/cctv_dgp_v29_paired_development_v1/visual_review.json`.
