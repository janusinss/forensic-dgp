# Filter-fusion V1: trained research model ready for comparison; quality stopped

2026-10-19: The VM committed 50 Adam updates across 200 microbatches. The [exact inference-only model](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_filter_fusion_training_v1_model/codeformer_filter_fusion_v1_u50.pth>) contains exactly 515 trained-state tensors. All 72 existing fusion tensors changed; 443 other tensors stayed exact to R2. Snapshot 0 matches R2, and snapshot 50/stopped retain the same model/Adam/sampler/RNG values. These are optimizer updates, not 50 epochs. The finite 151-update experiment stopped at 50; it did not complete 151.

This is our fine-tuned adaptation of CodeFormer, initialized from retained R2. Untouched pretrained CodeFormer remains the benchmark. `params_ema` is a loader compatibility key; no new EMA or architecture is claimed. Model file SHA256: `bc42ad4c55cef0c8e1e3df71e1ae50f6643611e178279fb67b5ffe8bbb511d61`. Canonical 515-tensor digest: `eff91cc6832bdf7ee27cd9f749a8f64a64a2351ab1443b14971b543ae2d86f35`.

## Audited result and limits

The strict import covers 9,081 files. The independent CPU state audit passes; the separate saved-pixel audit recomputed 2,390 comparison RAW/PNG pairs plus 48 TRAIN-guard pairs. All 63 planned comparison/refusal sheets were personally reviewed by root. The saved training stop remains `New/worsened preservation or R2 dark retention failure; retain actual stop`.

| Quality requirement | Result |
|---|---|
| preservation pass | FAIL |
| parent dark retention pass | pass |
| R2 dark retention pass | FAIL |
| additional interim stop pass | FAIL |

An audit passing proves evidence consistency; it does not reverse a failed quality requirement. The 48 native CCTV observations remain unpaired with no clean-reference scores. The 406 historical DEV slots and 72 mixed filter probes (36 TRAIN/36 DEV) remain separate. Reserved FINAL was not used. User wild1–wild5 have no clean originals and remain visual stress examples. No recovered identity, ethnicity or Zamboanga performance claim is established.

## Local research test

Open [http://127.0.0.1:55093/restoration-filter-test](http://127.0.0.1:55093/restoration-filter-test) and follow [CCTV_CODEFORMER_FILTER_FUSION_LOCAL_TEST_GUIDE.md](CCTV_CODEFORMER_FILTER_FUSION_LOCAL_TEST_GUIDE.md). The downloaded 256×256 outputs and comparison bundle were checked on the actual preview route. Layouts 375/768/1280 passed with no unexpected browser errors or horizontal overflow. This proves the functional preview; it is not independent FINAL review, main Auto qualification or user acceptance. The main application default was not changed. A rejected research checkpoint remains available only by explicit selection.

Original checkpoints, immutable training state/Adam/RNG/splits/provenance/failures and earlier storage recovery paths remain retained. Actual new training stays manual and finite on the L4/tmux workflow. Do not rerun or extend this failed recipe unchanged. Completion/detection model development remains deferred until restoration acceptance. The overall goal remains active.

Evidence: [state audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_filter_fusion_return_review_20261019_v1/state_v1/state_audit_v1.json>), [saved-pixel audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_filter_fusion_return_review_20261019_v1/pixels_v1/audit.json>), [root visual review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_filter_fusion_return_review_20261019_v1/root_visual_review_v1.json>), [research export](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_filter_fusion_training_v1_model/model_export_v1.json>). Browser closure and exact artifact pins are retained in the publication receipt.
