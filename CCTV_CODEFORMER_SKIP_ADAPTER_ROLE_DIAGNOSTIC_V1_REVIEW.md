# Skip-adapter correction diagnosis: completed, improvement not accepted

20 October 2026. The user reported **“They look similar”** after comparing skip-feature update55 with R2. The supplied wild3 and wild4 screenshots retain blocks across facial features. This is a failed visible improvement, consistent with the earlier return review; the model remains experimental. The photographs have no supplied unblurred references and are unpaired stress examples, not paired accuracy measurements.

The measured follow-up is complete. It finds small feature corrections dominated by spatially constant channel offsets. It does not create a new model, demonstrate image brightening as the cause, or establish that missing mosaic detail can be recovered. The coarsely mosaicked wild4 example should request a clearer input under the agreed scope.

## Actual bounded execution

The original V1 preflight stopped before Torch or neural execution because available host RAM was below its 8 GiB admission requirement. Its source is retained and its planned output directory was not created. A separately named, independently reviewed R1 implementation reduced live state and measurement allocation before execution; it did not weaken model-quality requirements.

The actual R1 worker exited 0. It processed six fixed historical TRAIN photographs in six roles: clear, moderate Gaussian blur, strong blur/downsample/JPEG, linear-light darkening, local mosaic and global mosaic. There were **36 encoder-prefix evaluations and 144 small adapter evaluations**, with **zero full CodeFormer/transformer/quantizer/decoder calls, backward calls, gradient queries or optimizer updates**. The worker took 157.46 seconds; the supervising process took 173.56 seconds. The outer cap was 900 seconds.

The existing R2 encoder was evaluated only through block14 to obtain its four SFT input sites. Its retained prefix and all24 update55 adapter tensors remained exact. Twenty-four retained clear R2 feature arrays supplied fixed comparison targets and normalization scales. No clean target is supplied at deployment.

Six photographs repeated across roles are not36 independent identities. The four previously input-adjudicated readable strong cases remain separate from two borderline stress cases. The12 new mosaic observations are **unadjudicated feature stress tests**, not new TRAIN admission. No native CCTV or reserved FINAL pixels entered this diagnosis.

## Independently checked findings

The independent auditor checked **432 saved arrays, 24 cached clean targets and144 site measurements**. Full-resolution FP64 moment maps and channel means support the scalar and DC/spatial decompositions. Pooled16×16 features do not reconstruct full activations. Source-captured maximum absolute wanted correction is retained but cannot be recomputed exactly from this compression; it is not independently validated as an exact maximum.

The following is the relative reduction in the mean fixed-clean-RMS-normalized feature L1, aggregated equally over the four sites and six photographs. It is **not** a face-quality score, the earlier DGP structure gate, a PNG preservation result or an independent final evaluation.

| Diagnostic role | Relative feature L1 reduction |
| --- | ---: |
| Moderate blur | 0.4004% |
| Strong blur, all six | 0.2401% |
| Strong blur, four predeclared readable cases | 0.2397% |
| Local mosaic | 0.2031% |
| Global mosaic | 0.1113% |

Across individual role/site groups, approximately76–97% of the learned correction energy is spatially constant within each feature channel. Spatial direction alignment with the wanted clean-feature correction is close to zero. Local mosaic at the finest256-resolution site has **0.716% worse feature L1**. This supports weak learned spatial correction as the next design question; it does not prove that adapter capacity or learning rate is the cause.

Feature DC offsets must not be equated with image brightness. R2's useful brightening remains desired. Clear feature distance is measured against R2 clear features, while actual clear training targeted untouched CodeFormer RGB behavior. Dark feature distance here is also measured against clear features, while dark training targeted R2 dark RGB behavior. Neither diagnostic comparison substitutes for the actual delivered clear/dark preservation checks, which remain failed in the returned experiment.

## Resource evidence and its limit

The worker's post-load self-samples reached592.29 MiB RSS, with at least3.114 GiB host memory available at those samples. The supervisor reported only4,255,744 bytes of worker RSS, inconsistent with the worker's self-measurement. **Whole-process-lifetime peak memory and enforcement of the configured2 GiB RSS ceiling are not independently established.** The exact Windows/sandbox measurement cause is unresolved. Constructor-time memory was not captured by the credible self-samples. Array integrity and the actual terminal exit0 remain separate from this resource limitation.

The closed output namespace contains453 files and447,617,166 bytes. The measured final free disk was43,484,151,808 bytes. The configured output cap was512 MiB with a4 GiB continuing disk reserve. These observations are not future runtime or memory guarantees.

## Design boundary

The prior clean-SFT oracle showed substantial conditional headroom when unavailable clean encoder features were substituted. The present result shows that the trained adapter has learned very little of that spatial correction. Simply adding unchanged epochs, increasing learning rate, globally disabling SFT, or removing feature offsets after training has not been justified as a solution.

Independent arithmetic using the saved full-resolution moments and channel means shows that removing the existing correction's spatially constant part would discard most of its already small feature-MSE gains. The wanted blur correction is predominantly spatial, but that does not make every channel offset harmful. This is an analytic feature-space intervention, not new neural inference or a generated-image result. The [projection receipt](outputs/codeformer_skip_feature_adapter_return_review_v1/post_user_diagnosis_v1/new_design_review_v1/posthoc_zeroDC_algebra_independent_v1.json) retains every case and site.

Root selected a controlled prerequisite to a larger joint-loss pilot: compare two fresh identity-initialized adapters with **feature-only training**, changing only whether feature error is spatially centered per channel. Keep the same existing clean-feature RMS normalization, architecture, frozen R2 base, optimizer and learning-rate schedule. Use only six already admitted moderate-blur and four predeclared readable strong-blur TRAIN observations, rather than admitting the new mosaics or borderline examples. The prospective cap is24 updates per branch, three microbatches per update:48 updates and144 backwards total, not48 epochs. This short fitting screen cannot establish a capacity limit, unseen performance or a qualified restorer.

The centered loss has a DC nullspace, so channel-mean drift must be reported and unchanged clear/dark RAW/PNG guards remain binding. Both branches and failures must be retained. No branch is selected automatically from DEV. A separately normalized spatial/DC plus image objective remains a later proposal; it is not bundled into this centering comparison. A favorable feature fit still needs visible improvement and all existing delivered-output, development and final requirements.

The prospective [paired-objective design](outputs/codeformer_skip_spatial_objective_design_v1/DESIGN.md) must be converted into an independently checked source/transfer packet with explicit operation and resource bounds before the user's manual L4 launch. **The design is not an upload-ready packet; training is unrun.** Previously failed gates, original/R2 checkpoints and all return evidence remain intact.

CodeFormer explicitly exposes a fidelity/quality trade-off through its feature transformation path; changing that control is not evidence of our learned improvement. The official [CodeFormer repository](https://github.com/sczhou/CodeFormer) and [paper](https://arxiv.org/abs/2206.11253) were rechecked. Any proposed separately normalized spatial objective is our experimental adaptation, not a result established by that paper.

## Evidence and current status

- [User feedback](outputs/codeformer_skip_feature_adapter_return_review_v1/post_user_diagnosis_v1/user_feedback_v1.json) and [retained screenshot ledger](outputs/codeformer_skip_feature_adapter_return_review_v1/post_user_diagnosis_v1/user_screenshot_capture_v1.json).
- [Actual R1 result](outputs/codeformer_skip_encoder_adapter_role_diagnostic_v1_r1/result.json) and [supervisor exit](outputs/codeformer_skip_encoder_adapter_role_diagnostic_v1_r1/supervisor_final.json).
- [Independent saved-array audit](outputs/codeformer_skip_feature_adapter_return_review_v1/post_user_diagnosis_v1/encoder_adapter_roles_r1_result_audit_v1/audit_v1.json) and [interpretation/resource correction](outputs/codeformer_skip_feature_adapter_return_review_v1/post_user_diagnosis_v1/encoder_adapter_roles_r1_result_audit_v1/interpretation_and_resource_limits_v1.json).
- [Previous return/model/app evidence](CCTV_CODEFORMER_SKIP_FEATURE_ADAPTER_V1_RETURN.md), [clean-SFT oracle](CCTV_CODEFORMER_CLEAN_SKIP_ORACLE_V1_REVIEW.md) and [global fidelity-path comparison](CCTV_CODEFORMER_FILTER_PATH_ISOLATION_V1_REVIEW.md).

The experimental comparison page and main application defaults are unchanged. Restoration acceptance, automatic input sufficiency/routing qualification and independent final review remain open. Completion development remains deferred until restoration acceptance. Training remains manual on the L4 VM. No cleanup, transfer, VM execution or new training occurred here; the19 October storage milestone remains closed. The full goal is active and incomplete.
