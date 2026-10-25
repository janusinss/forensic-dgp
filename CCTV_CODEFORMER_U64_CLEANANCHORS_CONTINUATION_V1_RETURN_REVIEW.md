# Trained-u64 continuation: verified quality stop at update 8

23 October 2026. **The run trained successfully for eight new Adam updates, then
failed the agreed preservation requirements. It is not an accepted successor.**
The original trained u64 stays preserved; U96 remains the experimental app
incumbent. The goal's automatic verified-winner rule did not trigger.

## What improved and what failed

These are identical paired synthetic cases from eight repeatedly reused adaptive
TRAIN-monitor photographs. They are neither unseen FINAL identities nor native
CCTV evidence. The structure proxy is reduction in RGB adjacent-difference error
against the photographic target, relative to frozen U96. It is not a percentage
of identity recovered or a direct percentage of perceived facial improvement.

| Delivered PNG endpoint | Retained trained u64 | New update 8 |
| --- | ---: | ---: |
| Global-blur adjacent-error gain vs U96 | 1.627216% | 2.317733% |
| Localized central-128 adjacent-error gain vs U96 | 1.210135% | 1.375401% |
| Clear target MAE | 0.03172407 | 0.03567976 |
| Dark target MAE | 0.09335070 | 0.08594180 |

Clear PNG target error worsened **12.4691%**, global-blur error
**10.6817%**, and localized-center error
**8.5327%** relative to initial u64. Dark target
error improved **7.9366%**. RAW confirms the same
tradeoff. The fixed central128 region of clear images also worsened 8.8064% in
PNG, so this is not solely an error in the surrounding background.

At update 8 the additional clear-target improvement check failed in both RAW
and PNG. Original legacy clear behavior checks also failed in both views:
`tr_ffhq_52422__clear` and `tr_ffhq_60978__clear` exceeded their original
case limits, and the group means exceeded their original limits. Legacy dark
behavior means were 0.01005609 RAW and 0.01004413 PNG against the unchanged 0.01
limit. Ordinary low-light illumination movement means were 0.01403042 RAW and
0.01402780 PNG, also above 0.01. Four blur gain endpoints are above 1%, but that
does not override these failures. The new update16 bank was never attempted.

The update4 phase passed its scheduled checks, but clear MAE already worsened
2.8309% PNG. Its phase did not yet require clear improvement. No earlier
checkpoint is selected after the stop; the protocol did not permit that.

## What the independent audits verified

Archive **1,155,880,499 bytes**, SHA256
`1badafe044aae58c66d86a1a2bf52c9d66aa7e4b2888e9ccb3ca8850fa523104`.
Protocol SHA256
`4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419`.
The download/export completed; worker exit 2 is the recorded quality stop.

- Transport audit independently verified all 1,413 members / 1,275,245,874
  payload bytes, compressed hash, single-gzip CRC/ISIZE, complete zero tar tail,
  exact static/source/data bindings and all 540 ordered events. There were 204
  known forwards, 32 backwards and eight commits, with no unknown reservations.
- Saved-pixel audit passed 27,025 checks: every actual RAW/PNG, 32 photographic
  objectives, four exact historical initial-u64 witnesses, four same-candidate
  repeats, and all three complete 56-case banks. Delivered PNGs match their
  floor-quantized RAW. Saved metric/gate scalars match exactly. CPU objective
  arithmetic differs from CUDA by at most 4.823e-8, below the declared arithmetic
  readback bound; this adds zero tolerance to any quality threshold.
- State audit independently verified complete initial and failed-current 575
  states, all nine checkpoints 0–8, eight pre/post-clipping gradient pairs,
  saved Adam arithmetic, RNG/sampler/counters and partial failure state. All 36
  heads changed at every update; all 539 frozen R2/U96 values remain exact.
  The initial trained-u64 head canonical hash is unchanged. Maximum parameter
  recurrence error is 7.451e-9. Local Torch was used only for ten bounded CPU
  `weights_only=True` saved-checkpoint decodes, with zero model/optimizer
  constructions, forwards, backwards, optimizer steps or VM connections.
- Root personally inspected all 31 planned sheets, covering all 204 generated
  outputs once. All 88 unattempted slots are explicit. Independent visual
  binding checks verify exact saved RGB cells. The visible changes are modest,
  often involving tone and smoothness; no conspicuous overall improvement
  sufficient to overcome the preservation failures is established. This is not
  independent human FINAL review.

Worker stop time was approximately 210.2 seconds, within the finite caps. All
441 sampled resource observations respected their limits. Audit-reader repairs
for static reference naming, operation ordering, closing-state/cursor binding
and the Windows venv child launcher are retained with original source/fixtures
and independent rereviews. They changed no training recipe, output or quality
threshold. The stopped pilot is immutable and must not be resumed or rerun.

## Diagnosis and next work

[Bounded saved-array diagnosis](outputs/codeformer_u64_cleananchors_continuation_v1_return_pixels/bounded_tone_diagnosis_v1.md)
finds all eight clear cases worse in MAE despite better adjacent-error scores in
RAW and PNG at both updates4 and8. A spatially constant RGB offset is invisible
to the adjacent-difference metric. Tone movement accounts for about 53.6% of
clear-output change energy, with material spatial changes remaining. Seven
clear cases brighten; one darkens and still worsens. Another improves mean
luminance bias while overall error worsens. A universal brightness-only cause
or unique gradient conflict is not proved; applying a blanket correction is
not justified by these data.

The next design review must preserve initial-u64 clear/blur photographic
fidelity and original guards while retaining the dark/blur benefit. Evaluate
the learned response by degradation condition before justifying one distinct,
finite manual-VM pilot. Do not merely add epochs, increase LR, weaken gates,
select an earlier stopped state, or apply oracle clean-target corrections.
No next pilot is selected or launched by this return audit.

Optional reviewed Protect/Apply/bypass stays separate. Automatic matching and
later completion development remain deferred. Native/wild remain unpaired,
broader inherited DEV failures remain binding, reserved FINAL remains unopened
and execution-unfrozen with independent reviewers pending. App/defaults,
original checkpoints, splits and all failures remain preserved. Full goal
ACTIVE/incomplete.
