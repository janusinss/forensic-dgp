# U96 restoration workflow and improvement priority

21 October 2026. The user selected U96 after the fixed R2/U96/S96 comparison: “therefore swap R2 to u96 then give me the updated workflow”. U96 replaces R2 as the preferred development restorer and starting reference for future improvement. This supersedes the earlier R2-first preference. It does not erase failed gates or establish final forensic qualification.

## Exact selected model

U96 is our trained adaptation of CodeFormer: the unchanged R2 base plus the U adapter at 96 committed adapter updates. It is not 96 epochs, a newly trained checkpoint today or a standalone 213 KB face generator. Both files are required:

| Component | File | SHA256 |
|---|---|---|
| Frozen R2 base | [codeformer_role_preserving_r2.pth](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth>) | `d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896` |
| U96 adapter | [codeformer_spatial_u_u96.pth](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_spatial_continuation_v1_models/codeformer_spatial_u_u96.pth>) | `958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0` |

Keep both files unchanged. Retain R2 as a permanent comparison and rollback option, S96 as a comparison, untouched pretrained CodeFormer as the declared external benchmark, and all original DGP checkpoints, source splits and failure evidence. The exact resumable U checkpoint is retained separately in the model export manifest; the adapter inference export contains no optimizer or RNG state.

U96 has lower delivered PNG error and higher average SSIM than R2/S96 on the fixed 36 paired DEV observations from four photographs. U96 reduces average MAE versus R2 by 5.6209%; this is not an identity-accuracy or historical structure-gate percentage. Native ChokePoint and QMUL remain separate unpaired observations, with no aligned clean target. Visual differences are modest and strong localized blur/mosaic defects remain. [Objective evidence](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_R2_U96_S96_OBJECTIVE_V1_REVIEW.md>).

## User workflow

1. **Upload the original crop.** Use one frontal or mildly turned face; keep the uploaded bytes unchanged.
2. **Inspect usable facial structure.** Request a clearer or less-covered crop if the visible face cannot meaningfully support restoration. A plausible generated face does not decide admission. The current test pages require manual input review; a validated automatic threshold is still pending.
3. **Review optional surroundings preparation.** For localized blur, protect the complete face and its existing soft edges, preview the outside-only blur and explicitly confirm. Bypass this step for ordinary restoration or globally soft inputs. Every preview starts from the original. The existing page offers manual rectangle protection and a fixed approximate strength; automatic proposals, separate editable Protect/Apply brushes and adaptive matching remain planned.
4. **Restore with U96.** Use U96 by default to deliver 256 × 256 output. The normal test page runs one selected model; its optional four-model comparison remains available. The surroundings page deliberately runs the same selected U96 model twice, on original and reviewed prepared inputs, so its preprocessing effect stays separate from model learning. R2 and untouched CodeFormer remain selectable overrides.
5. **Review and download.** Inspect visible eyes, brows, nose, mouth and jaw beside the original. Download PNG or the available original/protection/prepared-input/result/RAW/provenance bundle. Input protection prevents added blur in that reviewed input area; it does not guarantee an unchanged generated face or recovered identity.

The selected default applies to the active experimental restoration and surroundings pages. Full production integration and its automatic selection/override qualification remain a later milestone. Coverings stay in the current restoration outputs. Completion development remains deferred until restoration is satisfactory. The later separate completion component may estimate only reviewed hidden regions; an uncovered face does not need it.

## Model improvement workflow

1. Freeze U96 as the current incumbent and preserve exact R2/untouched-CodeFormer comparisons. Future initialization must reproduce the complete U96 base-plus-adapter RAW/PNG output; copying only R2 would lose the selected adapter.
2. Audit a concrete cause of missing visible detail using term-audited clean TRAIN targets, recoverable degradations, clear/normal controls and source-separated native DEV crops. Keep reserved FINAL identities out of tuning. Unknown wild-image originals are unpaired stress evidence, never invented clean targets.
3. Prepare a distinct finite L4 experiment only when its data, trainable path, objective, learning rate, sampler, optimizer/resume state, update/epoch caps, timing, resource limits and stop rules are justified. Keep preprocessing identical across checkpoint comparisons. More epochs and an unchanged failed recipe are not automatic next steps.
4. Run actual training manually on the user's current L4 VM through verified transfers and tmux. Independently audit returned state, RAW/PNG outputs, preservation, native source groups and every planned visual sheet before claiming an improvement.
5. Replace the preferred development version only after useful visible improvement, preservation evidence and user review. Keep U96 and R2 permanently. Independent final review, meaningful regression checks and the full application flow remain required before thesis completion.

The older R2-bound source prototype remains unchanged and untrained; it has not silently become a U96-bound model. No new training pilot is launched by this selection. Local inference and audits are permitted, with actual training under the manual finite VM workflow.

## Updated goal

Develop and verify our trained CodeFormer-based U96 adaptation as the primary candidate for useful 256 × 256 restoration of degraded CCTV face crops for the thesis “Forensic Deep Generative Prior Face Reconstruction for Degraded CCTV Video in Zamboanga City”. Preserve the existing application design and visible appearance. Improve restoration first; keep completion separate and defer its further development until the user is satisfied with restoration. Prioritize usable facial structure over invented sharpness and request clearer or less-covered crops when information is insufficient.

Continue the five milestones: freeze public native CCTV DEV and separate labeled FINAL identities/provenance/terms/resolution/input-only criteria; compare resize and retained original DGP/R2/U96/declared pretrained references on identical inputs; diagnose limitations and prepare justified finite manual VM pilots with original checkpoints/splits/failures preserved; integrate the selected restoration model with automatic selection and override; verify reviewed development and independent final outputs, regressions and bundled inline Playwright flow. Update PROJECT_HANDOFF.md at milestones.

Keep native CCTV without an aligned clean reference unpaired and separate from synthetic paired PSNR/SSIM. Prioritize Asian capture sources when available, report sources separately and make no ethnicity or Zamboanga-performance claim. There are no real Zamboanga CCTV samples yet. Surroundings blur is an optional reviewed input experiment, not a general promise to copy arbitrary noise across an image; automatic and assisted preparation must be reported separately.

Later completion retains the agreed covering families: masks, sunglasses, strong lens glare, hands, obstructing hair, scarves and objects. Show its automatic removal proposal for optional correction before generation; preserve clear glasses, non-obstructing hair and visible appearance, with a small documented margin. Request a less-covered image when too little face remains. Return one plausible estimate beside original and mask with PNG and optional original/mask/result bundle; never claim exact hidden identity. Report automatic and assisted completion separately.

All actual training runs manually on the existing NVIDIA L4 g2-standard-4 VM at ~/forensic-dgp, or a later user-identified replacement, with verified transfers and finite timing/stop rules. No automatic historical pilots or unchanged failed recipes. Existing maintenance authorization remains inventory-first and hash-bound, preserving research assets and active work. The deadline is around/after 20 November 2026. The goal is complete only when reviewed restoration, later covering scope and the full local workflow meet the agreed standard; selecting U96 or completing training alone is insufficient.
