# CodeFormer normal-tone training V1: trained model exported, preservation failed

17 October 2026. **50 actual training updates completed. A genuine trained model exists. It is an unqualified research candidate because both preservation guards failed.**

[Exported model and provenance](outputs/codeformer_normal_tone_training_v1_candidate_unqualified/README.md) · [All 34 comparison sheets](outputs/codeformer_normal_tone_training_v1_return_review/pixels_v1/visual_sheets/INDEX.md) · [Independent export check](outputs/codeformer_normal_tone_training_v1_return_review/independent_candidate_export_v1.md)

## Actual model and retained evidence

The manual NVIDIA L4 run committed 50 Adam updates using 200 microbatches, with 200 backward calls and no pending microbatch or optimizer action. These are updates, not 50 epochs. All 350 trainable tensors changed; all 165 frozen tensors stayed exact. The initial 515 tensors match untouched CodeFormer. Update-50 and stopped checkpoints agree in model, optimizer, RNG, sampler and schedule. No actual training was performed locally during this audit.

The inference-weight export is `outputs/codeformer_normal_tone_training_v1_candidate_unqualified/codeformer_u0050_unqualified.pth`: **376,609,465 bytes**, SHA-256 **`5f2460d332764421fc998c00e0dda1e0316545e7e3fcd484fcc57773b0212d7a`**. A separate reviewer freshly hashed the source and export before CPU `weights_only` deserialization and compared all 515 tensor names, shapes, dtypes and value bytes. Every exported tensor equals the actual trained state. The `params_ema` container key supports existing loaders; this run did not use EMA. The exported file omits optimizer and RNG, which remain in the full retained research checkpoints.

This is our finite fine-tuned adaptation of the existing CodeFormer architecture and official pretrained initializer, not an architecture trained from scratch. Original CodeFormer, DGP checkpoints, previous trained states, caches, splits and failures remain intact. The main app has not switched to this candidate. Weight compatibility and checkpoint existence do not establish acceptance.

The returned archive is **2,849,701,103 bytes**, SHA-256 `c746c7d7edbca0685032f34e84e60299140d0922db2e687d2e13ec59c6cbb86b`; protocol `d0c0b7a5be4577803bc156b611386c3be10068a3e78a47f9348125242f0c79d1`. Independent transport verified all 6,304 regular members / 3,138,623,526 payload bytes, four companions, all 26 static source files, gzip CRC, TAR framing and complete EOF. All three full checkpoints are retained. No returned executable was run locally.

## Location, cause and required change

**Location:** the final update-50 snapshot in the worker's preservation assessment. The worker saved the complete trained state and both guard receipts before stopping with `PilotStop: Declared RAW/PNG clear-preservation guard failed; retained snapshot`.

**Cause:** the delivered model changed normal and clear images too much. Removing the normal coarse loss's common-brightness penalty did not preserve normal-image appearance. Normal TRAIN mean luminance error rose from approximately +0.000429 in updates 1–10 to +0.075903 in updates 41–50. Ordinary normal coarse error rose from 0.01544 to 0.07642 while the centered term only rose from 0.01535 to 0.02181. Dark full-coarse error improved concurrently, from 0.16312 to 0.08612. These observations support shared tone drift; they do not prove that every error has one cause.

The two tiny formula-gradient prerequisites passed their prospectively declared bounds. All 200 logged coarse values match the independently recomputed reporting terms; the saved objectives, accumulation divisor, sampler and states are consistent with the sealed source. No demonstrated formula, optimizer-divisor, transfer or counter bug explains away the quality failure. Old/new encoder and token-predictor state differences are much smaller than the fusion difference; merely freezing those routes is not a demonstrated remedy.

**Required change:** reject this centered recipe. Preserve the genuine checkpoint as failed research evidence. Do not append epochs, resume its optimizer, relaunch the historical guide unchanged, weaken limits or promote it into the app. The next intervention must directly address normal-image tone while retaining useful dark-image restoration and visible structure.

## Independently recomputed quality

The unchanged limits are mean increase **0.01** and worst-case increase **0.03**, on the declared RAW and delivered-PNG domains. All 61 clear and 53 normal-blur case bindings and scalars were recomputed from pixels; all saved decisions match.

| Guard | RAW mean / worst increase | PNG mean / worst increase | Cases exceeding 0.03, RAW / PNG |
|---|---:|---:|---:|
| Clear MAE | 0.054132484 / 0.098379989 | 0.052267492 / 0.096976026 | 60/61 / 60/61 |
| Blur absolute luminance bias | 0.097033943 / 0.129509466 | 0.093964652 / 0.125605475 | 53/53 / 53/53 |
| Blur coarse RGB MAE | 0.074209974 / 0.108650048 | 0.072242187 / 0.106662770 | 53/53 / 53/53 |

The previous historical full-coarse AdaIN-50 candidate had three clear failures, so this is a substantial preservation regression. Those earlier failures remain failures too.

Across 53 FFHQ validation cases per profile, PNG MAE versus untouched CodeFormer worsened on **all 53 cases in each of clear, light, blur and JPEG**. Group mean changes were +0.051845091, +0.051597617, +0.042313957 and +0.050862716 respectively. Lowlight improved on 52/53, with mean change −0.073313422; compound improved on 53/53, with mean change −0.093409987. Compared with historical AdaIN-50, lowlight improved on 46/53 and compound on 51/53. These paired synthetic measurements do not establish native CCTV correctness or cancel the normal-image failures.

All 812 new RAW/PNG pairs passed exact source hashes, floor conversion and protected-padding checks. All 406 new update-0 outputs exactly equal untouched historical controls. The 40 older paired comparison cases remain separately grouped. All 48 native CCTV crops remain unpaired, with clean-reference metrics null. Dataset labels do not establish ethnicity or local Zamboanga performance. Snapshot native-512 arrays were not returned, so this audit does not claim independent native-512-to-RAW replay.

Root personally inspected all **34 original-resolution sheets, 406 case rows and 2,030 exact 256×256 cells**. Many dark outputs are smoother and easier to inspect than untouched CodeFormer; some are visually pleasing. Clear and mild inputs still change tone and texture unnecessarily. Extremely block-like, occluded, glared or strongly turned native inputs still produce absent, misplaced or invented facial features. Fine generated texture is not recovered visible structure. These inputs require clearer suitable crops. This assistant review is neither human user acceptance nor the reserved independent final review.

## Feasibility result and next design boundary

A separate finite CPU saved-pixel experiment checked 500 fixed common-RGB offsets in [−0.25, +0.25], for both historical AdaIN-50 and new normal-tone-50 across the 61 clear and 53 normal-blur cases: 114,000 evaluations. One **reference-chosen offset per case** produces a joint witness satisfying the existing RAW/PNG case and mean guards for both parents. Historical AdaIN-50 has greater remaining margin, smaller residual error and less clipping. No corrected images or new model predictions were generated for this numerical experiment.

This is an oracle capacity result: clean targets and original comparison outputs were available when selecting offsets. It is **not** an input-only restoration method, a learned controller, native-512 evidence, a retroactive pass for either model, dark-benefit preservation or a facial-geometry fix. Clipping is material: mean clipped scalar fractions are about 3.2–3.5% for the historical parent and 5.9–6.4% for the new parent; some new cases reach about 41.4%. Contrast and edges can change. Float32 rounding can also change differences slightly without clipping. Exact protected pixels refer to delivered PNG padding.

The next design review therefore targets an explicitly trained, input-conditioned tone component over the retained full-coarse adaptation, separating that correction from the face-restoration path. It must predict corrections from the observed crop alone; use no clean reference, role label or identity at inference; keep dark benefits; pass the known dark/normal overlap stress; preserve appearance and every existing guard. It cannot be presented as new geometry learning. A joint loader/state would be required if such a component qualifies. The current recommendation is **review only**: no new runnable packet is sealed, no controller has been trained, and the historical training guide must not be reused. Actual next training remains the user's manual finite L4/tmux workflow after source, data, budget, numerical and independent transfer review.

Measured current worker time was 679.734 seconds; supervised run 698.365 seconds, preflight 8.811 seconds and supervised export 231.480 seconds. Peak sampled RSS was 3,174,875,136 bytes, CUDA reserved 8,470,396,928 bytes, and minimum sampled worker-root free space 41,936,031,744 bytes. No timeout, storage stop or resource watchdog caused this failure. The local oracle completed in 883.776 seconds under its 900-second cap, with zero neural, gradient or optimizer calls.

## Goal and evidence

Restoration remains the immediate priority: useful 256×256 results for one cropped frontal or mildly turned face, preserved appearance and existing app design, input-only insufficiency handling, Auto/override/PNG flow, user acceptance and independent final evaluation. Native sources, provenance, terms, overlap limits and reserved identities remain separate from paired synthetic evidence. No exact identity, ethnicity or Zamboanga claim follows. Separate detection and plausible completion for the seven covering families remain deferred until restoration acceptance. Storage offload was completed separately; no cleanup or deletion was repeated here. The overall goal remains incomplete.

Evidence: [transport](outputs/codeformer_normal_tone_training_v1_return_review/independent_transport_v1.md), [state audit](outputs/codeformer_normal_tone_training_v1_return_review/state_v1/state_audit_v1.md), [pixel findings](outputs/codeformer_normal_tone_training_v1_return_review/pixels_v1/SCIENTIFIC_READBACK.md), [root visual record](outputs/codeformer_normal_tone_training_v1_return_review/root_visual_review_v1.json), [oracle limits](outputs/codeformer_normal_tone_training_v1_return_review/pixels_v1/tone_feasibility_v1/SCIENTIFIC_READBACK.md), [next design review](outputs/codeformer_normal_tone_training_v1_return_review/state_v1/next_design_recommendation_v1.md).
