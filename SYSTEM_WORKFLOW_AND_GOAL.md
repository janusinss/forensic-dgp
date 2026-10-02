# Single-image restoration and covering removal: execution Goal

Confirmed with the user on 2 October 2026 after three workflow question rounds.
This document records intended behavior and completion criteria, not evidence
that the application or models already meet them.

| Location | Windows local workspace | Linux training VM |
| --- | --- | --- |
| Repository | `C:\xampp\htdocs\YEAR 4\Testing\` | `~/forensic-dgp/` |
| This specification | `C:\xampp\htdocs\YEAR 4\Testing\SYSTEM_WORKFLOW_AND_GOAL.md` | `~/forensic-dgp/SYSTEM_WORKFLOW_AND_GOAL.md` after transfer |
| Ongoing status | `C:\xampp\htdocs\YEAR 4\Testing\PROJECT_HANDOFF.md` | `~/forensic-dgp/PROJECT_HANDOFF.md` after transfer |
| Product scope | `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_OUTPUT_SCOPE.md` | `~/forensic-dgp/PRACTICAL_OUTPUT_SCOPE.md` after transfer |
| Evidence | `C:\xampp\htdocs\YEAR 4\Testing\outputs\` | `~/forensic-dgp/outputs/` or the named experiment bundle |

## Objective

Deliver a useful local workflow in the existing main application for school staff
and thesis researchers. From one already cropped frontal or mildly turned face,
preview the automatically detected removal area, allow manual correction, and
generate one plausible face estimate. Target masks, sunglasses, strong lens glare,
hands, hair obstructing facial features, scarves and other objects over the face.
Preserve ordinary clear glasses, non-obstructing hair and the appearance of visible
features. Permit a small, documented surrounding skin margin for complete removal.
Request a less-covered image when too little facial evidence remains.

Select blur/noise restoration automatically using the input, with a user override.
Show the original, removal area and one final output. Provide a final-image download
and an optional bundle containing the original, removal mask and result. Prioritize
useful output over speed. Hidden features are estimates; no reference photograph
of the same person is required and exact hidden anatomy is not an acceptance test.

## Execution boundaries

1. Run inference and the application locally. Actual training runs only on the
   user's existing NVIDIA L4 / g2-standard-4 VM (4 vCPUs, 16 GB RAM). There is no
   fixed GPU-hour limit in the user's answer; every experiment still needs a
   concrete update/epoch budget, timing and an evidence-based stop rule.
2. Start with existing data. Add suitable public research datasets when necessary;
   record sources, terms, duplicates and split membership before use. Asian-source
   inclusion does not establish Philippine population coverage.
3. Preserve the main application's design. First-version input is a cropped face,
   with one output. Full-scene detection, multi-face/video processing and full
   profiles are outside this first-version workflow.
4. Preserve old experiment files, hashes, data splits and gate outcomes. Version
   the new practical review protocol before evaluation. A deliberate removal
   margin is separate from the raw occlusion label and detector score.
5. Use bundled inline Playwright for browser verification. Keep milestone notes
   in `PROJECT_HANDOFF.md`. There is no configured SSH connection in this chat;
   prepare exact VM commands and verified transfer files when training is needed,
   then audit the returned results before selecting a checkpoint.

## Five sequential milestones

### 1. Freeze a practical review gallery

Audit existing native examples by covering family and pose without changing their
labels or splits. Select a fixed development/review gallery with clear/degraded
inputs and uncovered/transparent-glasses controls. Include every requested covering
family before declaring whole-scope readiness. Record missing families explicitly;
prepare additional data if needed. Keep previously trained or inspected cases
marked as such; they cannot become a pristine holdout by renaming them.

Freeze case IDs, source hashes, pose, covering type, allowed removal margin and the
five review criteria below before generating new comparison outputs. Save a source
contact sheet and machine-readable coverage report. Do not infer hidden-face
ground truth from a covered photograph or invent paired clean targets.

### 2. Identify the cause of poor output

On the same frozen cases, compare current automatic masks with reviewed/corrected
removal masks through the pretrained completion baseline. Compare restoration
off/on only on declared degraded cases and controls. Inspect covering remnants,
plausible facial structure, visible appearance and seams. Report automatic versus
assisted completion separately. Save masks and outputs so every conclusion has
inspectable evidence. Select changes that address the observed failure: detection,
mask construction, alignment, completion or visible restoration.

### 3. Apply justified improvements

Implement needed processing fixes first. Make restoration routing depend only on
the uploaded input and user override. Mask expansion must follow a frozen policy
and remain distinct from raw detector output. Do not apply visible restoration to
clear inputs indiscriminately or assume it improves generated regions.

When inference fixes do not address an observed model failure, prepare a bounded
VM pilot with fixed splits, one-batch preflight, baseline comparison, visible-face
regression checks and explicit checkpoint-selection rules. Train only on the VM.
Use reviewed real masks as detector supervision; completion targets must come
from clean images with valid synthetic occlusions or genuine paired data. Do not
train a completion model to reproduce a real covering as hidden facial truth.
Benchmark pretrained completion before committing to generator retraining.

The existing native-lens pilot archive is an immutable candidate, not a command
to run automatically: assess its relevance after the practical baseline comparison.
Do not repeat an unchanged failed experiment. Export results with an LF checksum;
independently verify source, budget, metrics and images on return.

### 4. Integrate the validated workflow in the existing application

Connect covering detection and completion to the main app. Preserve the current
design while adding review/correction of the removal area before generation,
automatic restoration with an override, one output and the requested downloads.
Provide a clear request for a less-covered image when facial evidence is inadequate.
Confirm the local runtime works: installed RTX 3050 hardware currently coexists
with CPU-only project PyTorch, so CUDA inference is not yet configured. Measure
actual local processing time and use a compatible runtime rather than claiming
the GPU is being used merely because it exists.

### 5. Verify usefulness and record readiness

Run meaningful processing/API regression checks and Playwright at 375, 768 and
1280 pixels, including the upload, mask correction, restoration override, generate
and download flow. Inspect a ten-row preview grid and the complete fixed gallery.
Record useful/failed cases per covering family and restoration setting, including
manual corrections. Keep the Phase 3 restoration baseline unless a new candidate
has measured benefit and no material visible-appearance regression.

Update the handoff with chosen artifacts, hashes, reproduction commands, runtime,
remaining limitations and the next step. Mark the Goal complete only when the
agreed existing-app workflow works locally and reviewed outputs demonstrate its
usefulness across the fixed requested-family gallery. A training run finishing,
a lower aggregate mask error or a checkpoint named `best.pth` does not meet this
completion condition by itself.

## Practical output review, frozen before inference

1. The covering is substantially removed within the intended facial area.
2. Generated features are plausible; exact hidden appearance is not required.
3. Visible features retain their appearance outside the documented removal margin.
4. The estimated region joins the surrounding face without conspicuous covering
   remnants, blur patches or seams.
5. Clear faces and ordinary transparent glasses avoid unnecessary generation;
   detection failures permit manual correction and near-total coverage requests
   a less-covered image.

Report these judgements per case and covering family. Record whether the result
used an automatic or manually corrected mask. Review of already trained/inspected
examples is developmental evidence, not an unbiased accuracy estimate. Preserve
earlier numerical gate failures and report new regression metrics alongside the
practical review. Do not label the system broadly ready when a requested family
has no reviewed examples or has only failing outputs.

## Starting state and immediate action

`app.py` provides the existing Phase 3 restoration app; `completion_web.py`
provides a separate experimental mask/completion page. Pretrained CodeFormer
inpainting and Phase 3 weights exist locally. They have not yet been integrated
into the confirmed combined workflow. Prior native mask experiments do not prove
all-covering output quality. The supported reviewed dataset has 115 records
(83 train / 25 validation / 7 previously inspected test); 68 training records lack
a covering-family tag. Untagged does not mean the covering is absent.

The descriptive 83-source training inventory and source contact sheets are complete
without label/split changes. A ten-source native comparison is frozen and audited;
see `PRACTICAL_NATIVE_OUTPUT_RESULTS.md`. Reviewed masks help several mouth/hand
cases, but dark sunglasses and white glare still fail. Full-family/degraded
coverage and the agreed main-application integration remain incomplete.

Generic Places2 LaMa and face-specific AOT-GAN comparisons are now audited in
`PRACTICAL_LAMA_RESULTS.md` and `PRACTICAL_AOT_RESULTS.md`. Neither replaces the
CodeFormer baseline. Reviewed-mask results still have difficult eyewear/glare and
mask-edge remnants; approximate labels do not guarantee a complete removal area.

Immediate action: source-review full covering footprints and freeze revised
operator proposals in a new version before a targeted CodeFormer/AOT comparison.
Preserve old labels and results. Prepare missing-family public data and compare
visible restoration on declared degraded inputs. No local training.
