# Test our trained R2 restoration model locally

**The model is created. It completed 50 new training updates and remains
experimental: dark-image groups improved, while preservation checks failed.**
[Read the actual model comparison and limits](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_CODEFORMER_ROLE_PRESERVING_TRAINING_V1_R2_RETURN_REVIEW.md>).

1. Open the measured local research page: [http://127.0.0.1:52104/restoration-test](http://127.0.0.1:52104/restoration-test).
2. Upload one cropped face as PNG/JPEG and mark whether its visible structure is readable.
3. Select **Compare restoration** to view untouched CodeFormer beside **Our trained R2 · experimental**.

For a block-like, unreadable, strongly profiled or heavily covered face, choose
the corresponding input review and supply a clearer or less-covered crop.
This operator check is not a validated automatic threshold. The test restores
visible information; it does not remove masks or complete hidden regions.

Use **Download PNG** under either result or **Download comparison bundle**.
You can select either untouched CodeFormer or R2 alone to override the comparison
selection. This is an opt-in research route; the main application's default
restoration selection has not been replaced. Sharpening/display brightening is
not added by this test. Outputs are 256×256.

The measured URL belongs to the currently tested local server. After a restart,
its port can change. To launch the same research page from **Windows PowerShell**,
use a new receipt filename; the launcher prints the actual URL:

```powershell
Set-Location -LiteralPath 'C:\xampp\htdocs\YEAR 4\Testing'
& '.\venv\Scripts\python.exe' -B '.\serve_codeformer_research_comparison_v1.py' --receipt '.\scratch\codeformer_r2_user_server_v1.json'
```

Keep that terminal running while using the page. An existing receipt is retained;
select a new filename for another launch. This command starts local inference,
not training. The test route is also available at `/restoration-test` when the
updated existing application is running.

Exact model: [codeformer_role_preserving_r2.pth](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth>). SHA-256:`d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896`.
The 515 FP32 tensors exactly match the retained final training checkpoint.
`params_ema` is a loader compatibility key; no new EMA averaging was performed.
User acceptance, independent FINAL evaluation, main Auto qualification and later
covering/completion development remain pending. Do not rerun the historical R2
training packet or append epochs from this guide.
