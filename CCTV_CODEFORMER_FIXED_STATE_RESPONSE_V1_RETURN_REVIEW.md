# Fixed-state response return review — 17 October 2026

**Measurement completed and independently audited; no training updates or new checkpoint. The tone-controller model remains unqualified.**

The archive is 516,081,377 bytes, SHA-256
`e60258356e148ae426dcc8863794ad62377559f858a2b5dbf75c3e0eee0d50ad`.
Its 2,536 regular files, four companions, physical archive boundaries, 153 frozen
files and declared dependencies passed readback. Root independently hashed the
whole archive. The worker recorded 656 neural forwards, no gradients, backwards,
optimizers, updates or new checkpoints. Frozen initial/final state witnesses
match. Worker time was 82.068 seconds; resource and supervised phase limits held.

## Actual fixed-final results

These are recorded FP32 behavioral mean-luminance errors, not PSNR/SSIM against
clean targets. The independent CPU64 array reductions agree within the disclosed
6.667e-8 mean-readout difference; the original readings remain retained.

| TRAIN cohort | Inputs | Initial mean MSE | Final mean MSE | Inputs with worse MSE |
|---|---:|---:|---:|---:|
| Fitted normal | 100 | 0.001052076389 | 0.000338663753 | 26 |
| Fitted dark | 100 | approximately 0 | 0.000509992052 | 100 |
| Unused-reference normal | 32 | 0.001427812125 | 0.000547091189 | 8 |
| Unused-reference dark | 32 | 0 | 0.000495880117 | 32 |

**All 264 final offsets are negative.** The mean dark shift is -0.0221823 on
fitted inputs and -0.0218936 on unused references. All 100 fitted dark cases
already worsen: this is not solely a failure to transfer to new photographs.
Even the role-balanced fitted MSE improves by 19.34% while every dark case worsens.
A lower combined loss does not establish selective restoration improvement.

## Images and diagnosis

The independent audit checked all 528 RAW/PNG responses, 128 new parent RAW
arrays, 400 reused fitted parent arrays, source/input/features/target joins,
FP32 correction arithmetic, floor-to-PNG delivery and all 264 zero-head controls.
Root personally viewed all 24 planned sheets: 264 inputs and 1,320 panels.
[All comparison sheets](outputs/codeformer_fixed_state_response_v1_return_review/pixels_v1/visual_sheets/INDEX.md).
The displayed final changes are brightness shifts across faces and backgrounds;
softened facial details remain. The sheets contain no aligned clean target and
support no clean-reference quality, independent FINAL or native CCTV claim.
All support masks are full; covering/protected-pixel checks are vacuous here.

The scalar head cannot directly change facial structure. Source review also
shows the earlier restorer anchored untouched-model behavior only on B-normal,
while other clean-label losses continued on normal inputs. Centering the normal
brightness loss and then adding a scalar head both failed in retained returns.
These observations motivate a new objective design; they do not isolate a unique
cause or prove that the next recipe will succeed.

## Next actual-model work

Prepare a new finite direct-CodeFormer training pilot from the retained
full-coarse research parent. Preserve both A-normal and B-normal with full
spatial RGB and token behavior targets from untouched CodeFormer. Keep full
dark clean-reference supervision and explicit frozen-parent retention. Train
restoration weights with fresh Adam; omit the failed tone head. This is a new
prospective objective, not a continuation of its optimizer or a gate relaxation.

The proposed first pilot is capped at 50 committed restoration updates and
200 microbatches, with retained checkpoints and complete 406-case DEV RAW/PNG
comparisons. It is **being prepared, not sealed or launched** at this publication.
Its exact commands will be published only after source, transfer and finite
resource review. Actual training remains the user's manual L4/tmux action.

Preserve original CodeFormer, original DGP, caches, splits, provenance/terms,
failure archives and genuine failed trained models. App promotion, user-visible
acceptance, independent FINAL review and full app verification remain pending.
Native CCTV is unpaired and source-separated; no ethnicity, Zamboanga performance
or exact hidden-identity claim follows. Separate covering/completion development
stays deferred until restoration acceptance. Storage offload is already complete.

## Evidence

- [Transport](outputs/codeformer_fixed_state_response_v1_return_review/independent_transport_v1.json)
- [Independent saved-pixel audit](outputs/codeformer_fixed_state_response_v1_return_review/pixels_v1/audit.json)
- [Independent state/chronology audit](outputs/codeformer_fixed_state_response_v1_return_review/state_v1/audit.json)
- [Root visual review](outputs/codeformer_fixed_state_response_v1_return_review/root_visual_review_v1.json)
- [Source-based mechanism review](outputs/codeformer_fixed_state_response_v1_return_review/state_v1/mechanism_review_v1.md)
