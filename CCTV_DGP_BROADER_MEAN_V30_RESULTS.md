# V30 audited early structure failure - 7 October 2026

**Audit status:** the full independent R1 audit completed successfully in
1,053.58 seconds. The VM's failed early structure gate remains unchanged.

V30 stopped after **50 of 800 updates**. Its paired synthetic TRAIN feature-error
reduction was **0.805717%**, below the unchanged **1%** early requirement. The
stop is intentional. No update800 result exists, and the stopped checkpoint is
not qualified for the local application. Preserve V30 as failed evidence.

The returned archive has SHA256
`19294b086020378df2c9157de6954c27ae3d3e2008d375cf91bb4cc15c43af69`
and 1,251,065,768 bytes. The independent R1 audit checks all 27,622 files,
both complete 3,905-case snapshots, observed-region support, PNG arithmetic,
saved embeddings, metrics, mean controls, unchanged partitions, provenance and
CPU replay on the 50 fixed previews at both states. Raw outputs and delivered
PNG outputs remain separate. Export `complete: true` means packaging completed;
the archive's failed training gate remains binding.

[Independent audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json>)

## What the stopped run establishes

All 17 fixed MSE/SSIM/ArcFace preservation groups pass at update50. Feature-error
reductions by photographic source group are:

| Photographic TRAIN group | Degraded feature-error reduction |
|---|---:|
| dataset/asian_faces/degraded | 1.072549% |
| dataset/thumbnails128x128/degraded | 0.742743% |
| All degraded TRAIN cases | 0.805717% |

These are paired synthetic degradations of photographs, not native CCTV
observations. Source-group labels are reported separately and do not establish
ethnicity or Zamboanga performance. Preservation of a similarity metric does
not establish recovered identity or useful visible structure.

The worker used 971.5583 seconds including cached inference and snapshots;
the fit receipt records 451.8949 seconds. Peak allocated VRAM was
8,908,186,624 bytes, within the frozen 20 GiB limit. Time and memory passed;
the early structure requirement failed. The stopped run cannot establish what
an unexecuted 800-update run would do.

The first 50 updates exposed 250 unique cases from 218 TRAIN references:
47 clear and 203 degraded cases. Batches contain five shuffled cases; they do
not each contain five profiles of one person. Only one of the fixed 50 preview
cases was exposed: `v9_tr_asian_00196_motion_lr48`. The degraded feature-error
reduction was 0.766211% on exposed cases and 0.808382% on cases not yet exposed.
Unexposed TRAIN cases are not independent held-out evaluation.

## Completed visual review

All 10 original-resolution sheets, 50 rows and 250 exact image cells were
actually viewed. Columns compare the input, unchanged DGP, V29 update50,
stopped V30 update50 and paired TRAIN target. No resizing or display enhancement
was applied to these cells. Every row was checked across eyes, nose, mouth,
face outline and overall visible appearance.

The V30 changes are generally small contrast or boundary adjustments. Broad
visible appearance remains present, but degraded eyes, nose and mouth contours
remain soft. Severe blur and compound degradation still lack convincing
whole-face clarity. Clear glasses, non-obstructing hair, broad pose, expression
and other visible appearance remain important. V29 update50 often has stronger
coarse boundaries on these previews; its previously rejected development and
native results remain binding. Neither preview comparison qualifies a model.

[All 50 observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_broader_mean_v30_failure_review_v1/visual_review.json>)
[Comparison preparation and source hashes](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_broader_mean_v30_failure_review_v1/preparation.json>)

## Separate local checker repair

**Location:** the original local return checker's final early-receipt equality
comparison. **Cause:** CPU and VM aggregate floating-point arithmetic differed
by 1.1102230246251565e-16. **Fix:** a distinct R1 checker accepts at most 1e-12
receipt arithmetic difference, while checking the exact 1% threshold and the
failed decisions from both reported and recomputed values. The original checker
source, failed logs and the arithmetic probe remain preserved. The full audit
was rerun; no model, training recipe, split, gate or output was changed.

[Original checker failure and arithmetic evidence](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_broader_mean_v30_audit_execution_v1/receipt_roundoff_diagnostic.json>)
[R1 audit execution](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_broader_mean_v30_audit_execution_r1/combined.log>)

## Next diagnostic before another training recipe

The user selected **Apply the best approach** after the required V28-V30
architecture discussion. The next distinct diagnostic measures **280 gradient
queries and zero optimizer updates**. It compares the original and stopped50
decoder on the first 50 actually exposed cases and 50 unexposed TRAIN cases
matched by source, degradation profile and one-to-one reference repetition.
Unexposed references exclude all
218 references touched by the stopped run. Selection uses the frozen schedule
and metadata, without selecting favorable outputs or gradients.

The measurements can assess cancellation between sampled learning directions
and opposition between improvement and preservation terms. They cannot
reconstruct historical AdamW steps, prove a unique failure cause or establish
finite-step or real CCTV usefulness. The existing decoder, seven loss weights,
normalizers and failed V30 gate remain fixed. No new training recipe or
checkpoint is created. The new packet requires the user's manual VM execution
and independent audit of the return before selecting any later finite pilot.

The worker limit is 600 seconds, with a 900-second external limit plus 30-second
grace. Export has a 300-second limit and a separate 330-second external limit
plus 30-second grace. Require 4 GiB free. Keep peak allocated VRAM below 20 GiB
and the uncompressed return below 1.5 GiB. Expected measurement time is 2-8
minutes plus export. Windows rejects the gradient run before model construction;
local return replay uses inference only.

[Five exact manual VM steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V30_SAMPLING_GRADIENT_V1_VM.md>)

The existing DGP-led local app remains unchanged. No new native development,
independent final review or reserved-final faces were opened. Useful native
restoration, automatic and assisted evidence across all seven covering families,
an independent final review and the full bundled inline Playwright application
flow remain required. Goal active/incomplete.
