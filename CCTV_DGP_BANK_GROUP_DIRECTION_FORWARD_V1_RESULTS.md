# Group-direction finite comparison — 12 October 2026

**All four finite proposals are rejected.** The manual forward diagnostic
completed, but it did not produce a qualified model or justify more epochs.
The best raw structure gain was **0.862655%**, below **1%**, with **37 raw and
37 PNG preservation failures**. Counts denote group–metric checks, not people.

| Additive step | Raw structure gain | PNG structure gain | Raw preservation failures | PNG preservation failures | Decision |
|---:|---:|---:|---:|---:|---|
| 0.005 | 0.424805% | 0.452323% | 24 | 25 | Rejected |
| 0.01 | 0.713231% | 0.735769% | 29 | 30 | Rejected |
| 0.02 | 0.862655% | 0.877397% | 37 | 37 | Rejected |
| 0.04 | -0.456246% | -0.440927% | 49 | 49 | Rejected |

At step .02, source-specific raw/PNG gains were **0.556856%/0.556644%** for
`dataset/asian_faces` and **0.939342%/0.957872%** for
`dataset/thumbnails128x128`. Both raw and PNG gains must reach1% for each source.
The largest .04 step regressed both sources. All four failed the retained
MSE/SSIM/ArcFace preservation requirements; the brightness limit passed.
These are paired photographic TRAIN measurements. Dataset labels do not imply
ethnicity, native CCTV performance or Zamboanga performance.

Location: finite model outputs from the all32-group direction.
Cause demonstrated: favorable local derivatives did not translate into finite
preservation and useful detail at .005/.01/.02/.04. A particular layer or code
bug is not isolated. Fix being investigated: measure much smaller signed
displacements against the saved derivatives before choosing another optimizer
or training schedule. The human answered the diagnostic question with
“apply best approach”; no scientific gate was weakened.

## Verified execution and image evidence

The manual run completed in **53.61 seconds**. It used100
B and160 recognizer forwards and four disposable parameter proposals, with
zero gradients, backwards calls, optimizer updates or epochs. No replacement
checkpoint was saved. Peak allocated CUDA memory was
1.106 GiB. Its saved run reports initial model
restoration and unchanged frozen original/recognizer states.

The 296,873,108-byte archive SHA256 is
`c0d796bc1c154d8e11044b6044b559b8ed23518dd4ffbd430ef2eae0184f209b`. All596 returned bindings,5,852 parent
bindings,100 exact control receipts,400 raw arrays/400 exact floor PNGs,
800 metric rows,17 source/profile groups and20 sheets/800 exact RGB cells
passed saved-evidence audits. Four disposable state hashes and their selected
parameter bytes were independently reconstructed from saved CPU values;
no local model was displaced and no new neural derivative was queried.
This is independent saved-output checking, not an independent neural rerun
or an independent human final review.

All20 sheets were visually inspected. Eyes, lips, teeth, ordinary glasses and
hair remain softened relative to clear inputs. Degraded outputs do not show
a convincing useful whole-face detail gain. Step .04 introduces bright
arched nose ridges and lip outlines in several faces. Side padding remains
fixed where present. This is restoration evidence; no covering completion
family was tested in these sheets.

[Full raw/PNG saved-output audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_direction_forward_v1_analysis/independent_audit.json>).
[All20 visual observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_direction_forward_v1_analysis/visual_review.json>).
[Representative exact comparison sheet](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_direction_forward_v1_analysis/sheets/b391.png>).

## Selected diagnosis before more training

All32 saved directional dots are negative, yet some finite losses increase.
For example, `dataset/asian_faces/clear/ArcFace_loss` predicts about4.37%
loss reduction at .005, while its independently measured loss increases.
The signed test will check whether the predictions become accurate near the
initial model or whether a derivative/forward-path discrepancy remains.
Current results cannot decide between these causes. Raw SSIM loss also has
a small gradient-versus-auditor baseline difference; it remains recorded.

The new **2,839,068-byte packet is verified and UNRUN**.
It uses **−.001,−.0001,+.0001,+.001**, the unchanged all32-group direction,
initialB, same20 references/100 inputs, frozen prior/original/recognizer,
fixed affines and raw-to-PNG policy. Magnitudes are5× and50× smaller than
the previous smallest .005. Each proposal resets to the initial state.

The future audit compares each pair using
`(loss(+h)−loss(−h))/(2h)` against the saved raw directional derivative.
It also records one-sided slopes, curvature, baseline-loss differences and
actual float32 parameter changes/rounding. PNG quantization, visible outputs
and preservation gates are reviewed separately; PNG is not treated as a
differentiable gradient test. No arbitrary derivative tolerance selects a
model. Negative steps diagnose the path only. Small positive steps are
expected to show less than1% structure gain; that does not relax the1%
qualification requirement or turn this diagnostic into a trained improvement.

Preparation checked5 archive members,4 assets,5,852 parents,100 retained
alignment grids,Python3.10 syntax,Windows pre-neural rejection,Bash syntax,
100 saved controls/200 metric rows,six preservation boundary cases,three
source-gate fixtures,64 analytical signed pairs and negative-step exclusion.
A local unsealed preparation key-name stop is retained under
`outputs/cctv_dgp_bank_group_step_check_v1_preparation_stop`; it occurred
before sealing,upload or any neural/VM call. The sealed packet is unchanged.

Require **4 GiB free after install**. Estimate **2–5 minutes** for verification,
measurement and export, based on the previous same-size53.61s measurement.
Worker/measurement1200/900s; outer1230s; export300/outer330s plus30s kill
grace. VRAM20GiB/RSS12GiB/output768MiB/reserve1GiB. Export additionally
checks space before archiving and retains the1GiB reserve. Current VM power
and free space are not claimed from older cleanup snapshots.

[Exact five manual transfer/tmux/download steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_VM.md>).
[New sealed transfer audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_bank_group_step_check_v1_transfer_audit.json>).

## App and full goal remain separate

Fresh bundled inline Playwright checks covered375/768/1280px initial layouts,
four actual generation requests (Off,On,Auto and a tiny assisted region),
editable-mask review and eight actual PNG/ZIP downloads. Saved artifact audits
verified metadata/pixel hashes,On/Auto identical raw arrays,exact Off-visible
pixels and separate empty automatic/52-pixel assisted masks. Eight screenshots
were inspected. This functional square photographic TRAIN fixture does not
qualify native/non-square behavior,negative API guards,responsive result states,
all seven covering families or useful model quality. Those checks remain pending.

[Bounded app-flow closure](<C:/xampp/htdocs/YEAR 4/Testing/scratch/app_flow_20261012_v1/closure.json>).

Goal ACTIVE. The original SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`
remains the permanent app baseline. Its DeblurGAN-v2-compatible origin and the
replacement's own conditioning/reconstruction path with externally pretrained
frozen StyleGAN2 prior remain declared. No own-trained prior claim or app
promotion is made. All five milestones and seven covering families remain
required, including independent native DEV/final review,primary DGP Auto/override,
preserved visible appearance,clearer/less-covered requests,editable automatic
area,separate automatic/assisted results,original/mask/one plausible estimate
and PNG/optional bundle downloads. Preserve clear glasses and ordinary hair;
make no exact hidden-identity claim. Final identities remain outside tuning;
unpaired CCTV and paired synthetic PSNR/SSIM remain separate.

Training/gradients/model displacements remain human-manual L4 tmux work.
Agent VM access remains inventory-first,hash-bound maintenance only.
No new VM neural job was launched by the agent. Checkpoints,splits,caches,
full states,provenance and failures remain retained. The credit/migration
notification policy and November2026 deadline remain in force.
