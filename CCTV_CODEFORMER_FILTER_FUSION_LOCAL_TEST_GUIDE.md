# Test the experimental filter-fusion restoration model

The model is created for explicit research testing. It completed 50 updates and failed preservation/R2-dark retention. Keep this result separate from an accepted improvement. [Audited result and limits](CCTV_CODEFORMER_FILTER_FUSION_TRAINING_V1_RETURN_REVIEW.md).

1. Open [http://127.0.0.1:55093/restoration-filter-test](http://127.0.0.1:55093/restoration-filter-test).
2. Upload one cropped frontal or mildly turned face.
3. Review whether usable facial structure remains before generating a comparison.
4. Compare the explicitly selected experimental output with untouched CodeFormer and retained R2.
5. Download the 256×256 PNG or comparison bundle.

Supply a clearer or less-covered input when the face is block-like, unreadable, too strongly turned or heavily covered. The input review is not a calibrated automatic sufficiency threshold. This restoration test preserves visible appearance; covering detection and hidden-region completion remain separate and deferred.

Exact model: [exact inference-only model](<C:/xampp/htdocs/YEAR 4/Testing/outputs/codeformer_filter_fusion_training_v1_model/codeformer_filter_fusion_v1_u50.pth>). SHA256: `bc42ad4c55cef0c8e1e3df71e1ae50f6643611e178279fb67b5ffe8bbb511d61`. The original training checkpoint/Adam/RNG state and failed gates remain preserved. Useful-looking sharpness cannot verify missing identity details. This opt-in route does not replace the main application default. The measured local URL can change after a server restart; use the current launcher receipt rather than guessing a port. No new training command or recipe is authorized by this guide.
