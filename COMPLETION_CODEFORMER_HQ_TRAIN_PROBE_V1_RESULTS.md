# CodeFormer completion: paired TRAIN baseline and next learning step

25 October 2026. The selected pretrained inpainting model remains the completion baseline; no weights were changed. The local page continues at http://127.0.0.1:54734/ while running.

The fixed calibration completed30 requests on six genuine high-resolution TRAIN sources:24 nonempty white-hole estimates and six exact empty controls. Four prospectively fixed geometries cover lower face, both eyes, one lens/eye and a cheek object. The worker finished in140.105 seconds; the observed outer duration was141.861 seconds. These are procedural TRAIN measurements, not independent DEV/FINAL or real hidden-face accuracy.

The independent saved-array audit reproduces all24 raw512-to256 compositions and all30 PNGs exactly, verifies5,375,448 outside-mask source bytes, all six empty bypasses and all96 preview cells. Paired masked MAE is0.0537915 on[0,1] RGB, with maximum0.106325; boundary-gradient error mean is0.0222224. There is no pass threshold invented from these values. No real covering photo receives hidden-region PSNR/SSIM.

Root directly inspected all six result pages at native256 cells. Small cheek holes are usually plausible; lower-face holes often change mouth expression, and eye/lens holes show altered gaze, smoother anatomy or discontinuities with retained glasses. The clean reference makes those differences visible. A plausible estimate cannot establish a hidden expression or exact identity. Existing unpaired seven-family failures remain binding; the procedural calibration does not replace them.

Evidence: outputs/completion_codeformer_hq_train_probe_v1/protocol.json, results.json, independent_saved_output_audit.json and root_visual_review.json. Six output sheets are in that folder's preview/. The original preliminary arithmetic audit is retained: one-thread Torch resize differed slightly from the four-thread worker. Exact NumPy FP32 row-pair bilinear arithmetic matches every saved output without changing tolerance or regenerating images.

## Resource-bound repair

The original supervisor measured the Windows virtual-environment forwarding process instead of the real interpreter. Its tiny RAM figures do not describe model memory, and killing only that launcher did not prove termination of the whole worker tree. The original source/receipts stay retained with that limitation; this completed probe was not rerun.

The application now starts each Windows worker suspended, assigns it to a private Job Object and only then resumes it. Timeout, request shutdown and supervisor failure terminate the owned descendants. Independent harmless process tests cover the actual venv redirector, child/grandchild, assignment refusal and supervisor crash; the request thread alone handles lifecycle calls. No model source, weights, mask policy or restoration path changed.

The guarded browser check completed the same cloth-mask request in10.919s and empty bypass in0.461s. Its nonempty PNG is byte-identical to the retained pre-repair baseline; downloaded PNG/ZIP bytes verify. The job peak private committed memory was1,817,030,656bytes, including the real model worker and forwarding process. This is not physical RSS or a certified minimum-RAM requirement. See outputs/completion_codeformer_manual_v1/job_guard_functionality_v1.json and server_review_r2.json. Fresh-install portability remains pending; the updated14-file inventory/preflight is functional.

## Why the next manual diagnostic is different

The retained391 admitted cleanTRAIN references contain genuine1024 originals and exact256 targets; no new download is needed. Original53 validation references remain separate, with no new VAL/FINAL pixels used. Source terms and unknown pretraining/subject overlap are retained in outputs/completion_codeformer_quality_triage_v1/training_data_readiness.json.

CodeFormer's generic restoration configuration uses a different code vocabulary/fusion layout from the selected inpainting checkpoint. The current inference adapter freezes parameters, and the original network detaches discrete codes and encoder-to-fusion features. RGB loss alone must not be described as learning through the code predictor. The static architecture contract is saved separately.

A new manual-L4, zero-update diagnostic isolates CE/latent gradients for code prediction and masked RGB/boundary gradients for the existing fusion blocks. It uses four first-sorted TRAIN sources with the same two fixed lower-face/eye masks,16 gradient queries and24 typed forward slots, maximum600 seconds. It preserves original model/checkpoint hashes and every missing/zero gradient finding. No optimizer is created. Its clean latent/code teacher comes from the same frozen inpainting encoder/codebook; it is not an independent clean prior or proof of useful learning.

Only an audited returned diagnostic can justify the next finite fine-tuning pilot. Training remains manual on the authorized L4; no historical run is relaunched. Full covering-family quality, user satisfaction, independent final human review, automatic proposals and accepted fresh Windows deployment remain outstanding. Goal ACTIVE.
