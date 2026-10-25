# Skip-feature adapter V1: trained export ready for experimental testing

20 October 2026. The downloaded return was independently checked and its exact saved **55-update** model exported for local comparison. This is an actual trained CodeFormer adaptation with four learned feature adapters. It remains **unqualified**: the run stopped because it did not preserve R2 dark-image performance, and inherited preservation failures against untouched CodeFormer remain.

[Open the running local test page](http://127.0.0.1:52139/restoration-skip-test). The measured host server is recorded in `scratch/codeformer_skip_research_server_host_v1.json`. Select **Compare all three models** to compare untouched CodeFormer, R2 and **New skip-feature update55** on identical prepared input. Model selection is explicit on this research page; the main application default and automatic selection policy were not changed.

## Model and lineage

- Inference model: `outputs/codeformer_skip_feature_adapter_v1_model/codeformer_skip_feature_adapter_v1_u55.pth`, 376,824,649 bytes, SHA256 `1a086299da7c263bbdda896f63be6e912a5ba76678f07ae76217898b8e941704`.
- Source checkpoint: `outputs/codeformer_skip_feature_adapter_v1_return/outputs/skip_feature_adapter_v1/snapshot_0055/checkpoint.pt`, SHA256 `f947643b80b6871dc507a1f7cb1f873f56317cc1fbd14bb2407f09ebc231dcbb`. The original training state, optimizer, RNG and failure evidence remain retained.
- Base: all515 R2 tensors unchanged. All24 new adapter tensors changed;51,328 adapter parameters. The539 exported inference tensors exactly match the retained checkpoint. CodeFormer architecture/prior is credited to S-Lab; these are our trained adapters, not an architecture trained from scratch.
- Training committed55 Adam updates/220microbatches, including two optimized Gaussian2.4 blur exposures per55 TRAIN photographs. These are **not55 epochs**. Planned110 updates were not completed. No training or VM execution occurred during this local audit/export/app turn.

## What the return establishes

The archive SHA256 `f445f5ffe3a97215046cb8e265eece2dd2d36961ac0b607948a52a29a5809c76` matched its companions and exact canonical9265-member import. State/operation and saved-pixel audits passed integrity checks with no metric/gate mismatch. Root personally reviewed all69 planned sheets:60comparisons,3auxiliary,3guard and3input-only refusal sheets. Integrity passing does not override quality failures.

478 comparison inputs include358 historical paired synthetic DEV,48 source-separated native/unpaired CCTV,36 mixed-filter TRAIN and36 mixed-filter DEV. Each of five saved output banks was checked, totaling2,390RAW/PNG pairs; auxiliary,guard,teacher and feature evidence was checked separately. Reserved FINAL identities were not used. Native CCTV has no aligned clean reference; no ethnicity or Zamboanga performance claim follows.

| Saved PNG measure | Initial R2 | Adapter55 | Interpretation |
|---|---:|---:|---|
| Paired DEV lowlight MAE |0.08087236|0.08109319|R2 dark retention fails|
| Paired DEV compound MAE |0.10688842|0.10716730|R2 dark retention fails|
| Clear preservation mean MAE delta vs untouched CodeFormer |0.01413092|0.01388585|Improved slightly; still above0.01|

The four predeclared readable strong auxiliary TRAIN examples have1.0446% lower PNG MAE but only0.1342% lower adjacent RGB error. This is not the old DGP1%structure gate, independent evaluation, or a visible strong-blur solution. Root review found the new outputs close to R2; local blur and mosaic defects persist. Strong blur/mosaic/line patterns were probes, not the optimized blur recipe. More unchanged epochs are not authorized by these results.

## Local functional verification

Two bounded CPU forwards established exact native512/RAW256/PNG256 parity between the new local wrapper and an independent functional adapter implementation, with all4sites executed and all539state values unchanged. The240s script limit was also supervised by a300s outer process timeout; the actual run took48.57s. This is local graph parity, not CPU/CUDA bit parity or model usefulness.

Bundled inline Playwright verified375/768/1280px layouts before and after results, with no horizontal overflow or unexpected console/page exceptions. It verified one selected-model run, an all-three comparison and a rectangular `test_images/wild2.png` stress input, totaling5CPU forwards. Five PNGs and3ZIPdownloads passed independent payload/hash/quantization/geometry checks; both RAW and PNG off-support pixels matched prepared input exactly. Selected and all-model skip outputs matched exactly. Measured generation times were9.74s,21.06s and5.88s on this host; these are observations, not latency guarantees.

Five refusal/invalid-input cases returned422 with zero neural calls. Input review on this page is **operator-assisted**, not a validated automatic insufficiency threshold. It does not implement automatic model selection or qualify the main app. Coverings remain; completion/removal training is deferred until restoration acceptance.

The original browser attempt could not find its sandbox browser; the next host-browser check could not reach the sandbox listener. Host server receipt resolves the actual reachable URL. A flow-harness assumption about choosing the identical file caused an assertion after the first completed comparison; that result was retained, and only the remaining cases ran. The failed harness receipt remains. No completed inference was repeated. Total local forwards for parity plus app verification:7; gradient/optimizer/training/VM calls:0.

## Continue from evidence

1. Test the new experimental model beside R2 and untouched CodeFormer on the same readable cropped faces.
2. Record visible structure and appearance differences; ask for clearer input where structure is insufficient. A sharp plausible estimate does not prove recovered identity.
3. Before another changed finite manual VM pilot, diagnose fixed initial/55 per-site corrections across moderate,strong,clear and dark TRAIN roles. Do not extend this failed recipe unchanged.
4. Keep restoration acceptance, independent final review, main-app Auto/override integration and later covering-family qualification open. The complete goal is not achieved.

To restart this optional page later, use PowerShell from the project root and a fresh receipt filename:

```powershell
$skipReceipt = 'scratch/skip-test-server-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.json'
& '.\venv\Scripts\python.exe' -B serve_codeformer_skip_research_comparison_v1.py --receipt $skipReceipt
```

Use the URL printed by the launcher. This command runs local inference only.

## Evidence

[Import](outputs/codeformer_skip_feature_adapter_return_review_v1/return_import_v1.json), [state/operations](outputs/codeformer_skip_feature_adapter_return_review_v1/state_v1/state_audit_v1.json), [pixels](outputs/codeformer_skip_feature_adapter_return_review_v1/pixels_v1/audit.json), [root69-sheet review](outputs/codeformer_skip_feature_adapter_return_review_v1/root_visual_review_v1.json), [independent results review](outputs/codeformer_skip_feature_adapter_return_review_v1/independent_review_v1/INDEPENDENT_RESULTS_REVIEW.md), [export](outputs/codeformer_skip_feature_adapter_v1_model/model_export_v1.json), [local parity](outputs/codeformer_skip_feature_adapter_return_review_v1/local_parity_v1/parity_v1.json), [independent downloads](outputs/codeformer_skip_feature_adapter_return_review_v1/app_runtime_v1/independent_download_payload_audit_v1.json), [root runtime review](outputs/codeformer_skip_feature_adapter_return_review_v1/app_runtime_v1/root_runtime_visual_review_v1.json), [Playwright flow](scratch/codeformer-skip-app-v1/flow-resume.json).

The19October Drive offload remains a separate completed storage milestone. This return/model publication deleted or transferred no local or VM files. It makes no additional reclaimed-space claim.
