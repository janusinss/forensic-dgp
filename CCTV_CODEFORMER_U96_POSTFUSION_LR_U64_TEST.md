# Test U96 against the new post-fusion u64 model

Open [the local comparison page](http://127.0.0.1:59807/restoration-postfusion-test). This is an experimental comparison. U96 remains the current development model.

1. Upload one already cropped frontal or mildly turned face.
2. Inspect the original input and choose the appropriate structure-review answer.
3. Select **Compare both models** when visible facial structure is readable.
4. Compare the visible eyes, brows, nose, mouth and jaw in both results against the input.
5. Download either PNG or the original/RAW/processing ZIP.

Each request sends the same prepared input to both frozen models and produces 256×256 outputs. There is no added blur, noise, sharpening, display brightening, completion or covering removal in this page. Too-degraded, too-covered and out-of-scope selections are refused before model execution. The input review is manual; it is not a validated automatic information threshold.

The new u64 passed its small, adaptive TRAIN screen: PNG structure-error reduction was 1.627216% for global Gaussian blur and 1.210135% in the central region for localized Gaussian blur, with the unchanged preservation checks passing. These are relative reductions in the declared paired error metric, not percentages of recovered identity or proof that faces look that much better. Broader development evidence and user/independent final acceptance remain separate requirements.

## Exact model composition

The complete candidate uses the retained R2 CodeFormer base, retained U96 adapter and the new terminal u64 post-fusion heads. It is our trained adaptation of CodeFormer by S-Lab. All 575 parameter tensors are frozen for local inference.

| Component | Exact local artifact |
| --- | --- |
| Retained R2 base | `outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights/codeformer_r2.pth` |
| Retained U96 adapter | `outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights/codeformer_u96_adapter.pth` |
| New trained u64 heads | `outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_inference_export/terminal_u0064_heads.npz` |

The head file is the unchanged 685,818-byte payload from the audited checkpoint. File SHA256: `77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17`. Canonical tensor SHA256: `6629d8414321b0d2ba3a9cc476e415d31911fd4bd1fa121a1a8db9097ce235be`.

[Audited return](CCTV_CODEFORMER_U96_POSTFUSION_SMOOTHANCHORS_LR_V1_RETURN_REVIEW.md) and [bound test manifest](outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_page_preparation/experimental_test_manifest_v1.json) retain provenance and qualification limits. Native CCTV without an aligned clean reference remains unpaired evidence; no exact identity or Zamboanga-performance claim follows.

## Start a fresh local server if the page is closed

Use Windows PowerShell from the project root. This runs inference only; it never launches VM training. Choose an unused receipt filename if the example already exists.

```powershell
cd 'C:\xampp\htdocs\YEAR 4\Testing'
& '.\venv\Scripts\python.exe' -B serve_codeformer_postfusion_research_comparison_v1.py --manifest outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_page_preparation/experimental_test_manifest_v1.json --manifest-sha 8c63f9f1ce080e1397df8c7708f2ed2c36601e47edce8d07a9b8036546dd4a68 --receipt scratch/codeformer_postfusion_manual_test_v1.json
```

Open the URL printed by that command. The port is assigned at startup; do not assume the previously assigned port is still running. Existing application files, model defaults, original checkpoints and prior failed experiments remain preserved.
