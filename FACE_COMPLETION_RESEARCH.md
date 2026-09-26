# Single-image face restoration and completion: research and implementation plan

Researched 25 September 2026. This extends the earlier restoration-only scope. The original research/design is preserved below; see the implementation update for current status. Recommendations are project-specific hypotheses unless explicitly attributed to a source.

**Implementation update:** a separate custom completion/segmentation baseline, deterministic synthetic pairs, region metrics, benchmark export/scoring and an experimental correction UI are now implemented. See [COMPLETION_TRAINING.md](COMPLETION_TRAINING.md) for exact capabilities, local mechanical verification and outstanding pretrained comparisons. No full completion training or quality claim follows from this update; the research plan below describes the broader target.

## 1. Agreed objective and decision

Use one uploaded image. Improve degraded visible facial regions while preserving their appearance and structure, and generate a plausible estimate in regions covered by a mask, hand, glasses or another object. An uncovered photograph of the person is not required. Clear visible areas should need little or no modification. Hidden features are estimates, not verified recovery of the person's actual features.

**Recommended next step: build a region-controlled completion benchmark before another long training run.** Compare an existing pretrained face-inpainting model against a general inpainting model using identical images and explicit occlusion maps. Then choose the completion component from measured output quality. Preserve the current restoration model as a baseline; adding more Phase 5 epochs alone will not implement this task.

A modular first implementation is preferable here because it lets us measure mistakes in occlusion detection, visible-region restoration and hidden-region synthesis separately. This is an engineering recommendation, not evidence that a modular pipeline universally beats an end-to-end model.

## 2. What research supports

| Method / primary evidence | Relevant finding | Application and limitation |
|---|---|---|
| [Partial convolutions, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.html) | Explicit validity masks prevent treating hole-fill values as ordinary observed pixels. | Pass an occlusion map to a completion model; the existing RGB-only restorer does not distinguish a surgical mask from real facial appearance. This paper is not specific to degraded CCTV faces. |
| [Gated convolutions, ICCV 2019](https://arxiv.org/abs/1806.03589) | Learned feature selection supports free-form holes. | A compact gated U-Net is a useful custom trainable baseline, but training one from scratch is not automatically better than pretrained face priors. |
| [CodeFormer official inpainting script](https://raw.githubusercontent.com/sczhou/CodeFormer/master/inference_inpainting.py) | A separate inpainting checkpoint is supplied; the script uses aligned 512×512 faces and composites the prediction only inside white holes. | First face-specific pretrained candidate to benchmark. Its white-pixel hole convention is not real-world occluder detection. An adapter must maintain an explicit mask rather than infer holes from bright image pixels. |
| [LaMa official implementation, WACV 2022](https://github.com/advimman/lama) | Fourier convolutions and large training masks support large-hole completion. | General-purpose comparison baseline. Its results on general scenes do not establish faithful masked-face reconstruction. |
| [BrushNet official implementation, ECCV 2024](https://github.com/TencentARC/BrushNet) | A diffusion branch conditions synthesis on masked-image information; random and segmentation-mask variants are available. | Later generative comparison if initial models lack plausibility/diversity. Face quality, environment compatibility, GPU memory and latency require measurement here. |

Additional principles:

- [DiffBIR, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/07690.pdf) separates degradation removal from information generation. This motivates testing separate restoration/completion stages, without establishing their best order for occluded faces.
- [RePaint, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Lugmayr_RePaint_Inpainting_Using_Denoising_Diffusion_Probabilistic_Models_CVPR_2022_paper.html) provides a pretrained diffusion approach to inpainting. Its known-pixel conditioning must not be assumed to clean noisy known pixels.
- [Pluralistic Image Completion, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zheng_Pluralistic_Image_Completion_CVPR_2019_paper.html) demonstrates multiple plausible completions. A single missing mouth need not have one uniquely inferable solution.
- [LPIPS, CVPR 2018](https://richzhang.github.io/PerceptualSimilarity/) measures perceptual similarity. It complements pixel errors; it is not an identity-accuracy score.
- [MaskTheFace, authors' repository](https://github.com/aqeelanwar/MaskTheFace) provides synthetic face-mask generation. It is a data-generation reference, not proof of performance on all real coverings.

No surveyed paper establishes the best method for this project's Philippine school images. Published benchmark superiority is not a substitute for the proposed comparison.

## 3. Proposed processing pipeline

```text
One uploaded image
    -> face location / alignment with failure handling
    -> occlusion estimate, with editable region preview
    -> visible-region restoration when needed
    -> completion using visible context plus explicit occlusion map
    -> controlled composition and inverse alignment
    -> output plus a view identifying generated regions
```

### Occlusion map is a first-class input

Use `M=1` for regions to synthesize and `M=0` for observed regions. Keep this convention consistent in data, models, metrics and UI. Model adapters must translate external conventions explicitly.

Begin experiments with known synthetic masks and manually verified real masks to establish completion capability. Then evaluate an automatic occlusion-segmentation model. A small segmentation network trained with synthetic occlusion labels and corrected real examples is a candidate, not yet a selected architecture. Give the user a correction brush on the same uploaded image when automatic regions are wrong; this does not require a second photo.

Do not use only a landmark-defined rectangle as final segmentation: it can remove visible cheeks, miss straps and fail for hands or profile faces. Do not classify every dark region, beard or skin-colored object as an occluder. Separate opaque coverage from translucent glasses; useful visible information can remain through glass.

Use input-derived alignment at inference. Training target landmarks can generate synthetic masks and training supervision, but must not silently supply alignment or occlusion information to the deployed pipeline. Report ideal-mask and end-to-end results separately. Severe occlusion can invalidate face detection and landmarks; offer manual crop/region correction rather than silently returning an unrelated face.

### Preserve what was actually visible

Let `V` be the observed-region image after optional conservative restoration, `G` the completion prediction, and `A` an explicit synthesis/blending map. Compose:

`output = (1 - A) * V + A * G`

Away from the documented boundary band, the completion stage cannot change visible pixels. That preserves `V`, not necessarily the original degraded pixels. Keep an identity/bypass option for clear input so visible pixels need not be unnecessarily restored. Validate any quality-based automatic routing rather than assuming it works.

Blur spreads an occluder's appearance beyond a sharp geometric boundary. Test a small, measured mask expansion and transition band at each resolution. Record that expanded region as generated too. Expanding too far sacrifices visible evidence; expanding too little leaves mask remnants. Keep original images and maps with the result.

Compare these orders: restoration then completion, completion then restoration with generation restricted afterward, and joint mask-conditioned restoration/completion. The current restorer was not trained on occluded faces, so treating its output as trustworthy context without testing is unsafe scientifically.

## 4. Data and simulation plan

Use uncovered reference faces to construct paired examples. Retain the clean target only for training/evaluation, never as an extra inference input.

1. Split source identities and remove duplicates before generating variants. Where identity labels are unavailable, document that limitation; an image-only split is weaker.
2. Render a covering onto the reference face using a known geometry/alpha map. Include face masks, irregular shapes, hand/object overlays and eye-region coverings with varied pose and size.
3. Apply camera degradation to the covered image: blur, noise, low resolution, compression and lighting. Covering-before-degradation approximates image capture and avoids unrealistically sharp pasted masks on blurry faces.
4. Retain the original reference, geometric coverage, expanded affected-region map, seed, source and degradation settings in a manifest.
5. Include clear covered, degraded covered, clear uncovered and degraded uncovered examples. The model must also learn when not to generate missing content.

Use real reference detail: FFHQ128 enlarged to 256 or 512 is not high-resolution ground truth. Prefer appropriately licensed genuine reference images at least as detailed as the training resolution. [FFHQ documentation](https://github.com/NVlabs/ffhq-dataset) distinguishes thumbnails from 1024px images and describes restrictions on facial-recognition development. Review both data and pretrained-weight terms against the intended use; this plan does not resolve those permissions.

Keep dataset-source reporting but do not treat source as ethnicity. A representative, appropriately sourced Filipino test set is still needed. Retain the old Phase 4 split for historical restoration regression only; its possible earlier-training exposure makes it inadequate as the sole completion test set.

Synthetic masking offers exact ground truth but is not sufficient. Evaluate real mask/hand/glasses images separately. An uncovered photo taken at a different moment is an identity reference, not an exactly aligned pixel target for PSNR. When no true paired target exists, report visual assessments and failure coverage without inventing reference metrics.

## 5. Training approach after the benchmark

Prefer adapting a selected pretrained completion prior where its code, training support and weight permissions allow. A released inference checkpoint does not imply an easy or compatible fine-tuning pipeline. Preserve its environment separately from the working DGP environment.

For a custom baseline, use an RGB-plus-occlusion-map encoder-decoder with gated or partial convolutions. Reusing the existing model requires explicit checkpoint migration: changing its input from three to four channels breaks naive strict weight loading, and its residual connection currently carries the covering into the output. Keep the existing restorer intact and version the new model separately.

Proposed loss components, to tune by validation rather than adopt as established constants:

| Component | Purpose / caveat |
|---|---|
| Area-normalized visible-region reconstruction | Preserve visible structure and remove known synthetic degradation; prevent large regions from dominating solely by area |
| Area-normalized hidden-region reconstruction | Learn paired completion; strong pixel-only optimization can favor averaged appearances |
| Perceptual and boundary consistency | Encourage coherent facial texture and smooth transition; do not interpret plausibility as recovered truth |
| Frozen identity supervision | Experimental consistency term only, evaluated with and without it; use an independent evaluator to avoid optimizing the reported score alone |
| Generative objective appropriate to the architecture | Retain a pretrained model's supported objective or test an adversarial/diffusion objective separately; do not stack every loss by default |

Start with a mechanics smoke test, then a small overfit test to prove the network uses the mask and can remove coverings, then a bounded pilot on the full distribution. Begin fine-tuning at the architecture's supported resolution. A compact 256px baseline and a native-512px pretrained model can be compared, but report their different input/resolution conditions; upsampling does not add evidence.

EMA, reproducible manifests and resumable state from Phase 5 remain useful. The old hyperparameters and PSNR-only or whole-face selection rules must not be copied blindly. Fully observed images and empty masks need well-defined zero-area loss handling; fully hidden faces contain no visible identity evidence and should be treated as unsupported completion cases rather than ordinary successes.

## 6. Benchmark and selection protocol

Suggested first comparison: 200 fixed synthetic test faces with several fixed coverings/degradation levels, plus 30–50 manually reviewed real examples where available. These are pilot sizes, not statistical guarantees. Split development and final testing before tuning; group repeated variants by source person/image in analysis.

| Experiment | Question |
|---|---|
| Existing Phase 3 restoration only | What fails without dedicated completion? |
| CodeFormer inpainting with verified masks | Does a pretrained face prior already give a useful completion baseline? |
| LaMa with the same masks | Does a general inpainting model preserve context or handle shapes better? |
| Best candidate plus optional restoration, testing order | Does deblurring help or distort the context used for completion? |
| Custom mask-conditioned adaptation or diffusion challenger | Does added training/complexity improve measured tradeoffs enough to justify it? |

Measure five groups, stratified by coverage, pose, degradation, face size and data source:

1. **Visible-region fidelity:** region-normalized error against clean synthetic targets, visible landmark displacement where valid, and change to clear inputs. Whole-image averages can hide damaged eyes.
2. **Completion quality:** hole-region MAE/PSNR and documented perceptual comparisons, plus blinded human assessment of coherence. LPIPS on a hole-containing crop is not a strictly hole-only metric; report crop/mask handling. Do not calculate ordinary PSNR on blacked-out images and call it region-normalized.
3. **Identity consistency:** same fixed eligible pairs for paired comparisons, coverage/failure rate, and a recognizer independent of training supervision. Reference-guided scores on synthetic data do not prove hidden-feature truth in real images.
4. **Occlusion and boundary accuracy:** segmentation IoU/boundary errors, leftover covering, invented changes outside the map, and detection/alignment failures. Count failed cases, not just successful outputs.
5. **Operational cost:** GPU peak memory, preparation time and median/p95 inference latency on the actual VM. No T4 throughput or training-duration claim is established yet.

Freeze candidate-selection rules on development data. Choose a model only if it improves completion while meeting visible-region preservation and failure-rate criteria against baseline, then evaluate once on the held-out set. Predefine tolerances from baseline variability; use paired confidence intervals grouped by person/image. PSNR, perceptual quality and identity can disagree, so report the tradeoff rather than hide it inside an arbitrary score.

If a stochastic model generates multiple options, fix seeds and the number of candidates. Report average performance and variability; do not select the closest-to-ground-truth candidate at test time and present that as deployable performance. Current application variants are sharpened/denoised versions, not genuinely different missing-face hypotheses. Candidate diversity is not a calibrated confidence interval.

## 7. Repository changes proposed, not yet implemented

| Area | Proposed work |
|---|---|
| Dataset | A separate occlusion-pair dataset returning input, target, masks and metadata; preserve the old restoration dataset for regression |
| Model interface | Completion adapters with explicit mask conventions, resolution/normalization checks and checkpoint provenance |
| Training | Separate completion experiment entry point, region-aware losses, reproducible synthetic coverings, versioned full-state resume |
| Evaluation | Fixed completion manifest, per-region and per-stratum reports, ideal-mask versus predicted-mask comparison |
| Application | Occlusion preview/correction, restoration bypass for clear areas, constrained blending, generated-region display and truthful candidate descriptions |

Important implementation tests include empty-mask behavior, exact composition outside the blend region, correct map alignment after crop/resize/paste-back, no target information in inference, stable synthetic-pair replay, and checkpoint resume equivalence. A white shirt or bright skin highlight must not become a hole because an external script uses white-pixel detection.

## 8. Execution order and stop conditions

1. Implement the evaluation manifest and synthetic coverings; inspect examples before training.
2. Benchmark pretrained CodeFormer inpainting and LaMa with verified masks in isolated environments; preserve outputs and profile the VM.
3. Select a candidate and test the restoration/completion order with controlled composition.
4. Add automatic occlusion detection and user correction; quantify the gap from verified-mask results.
5. Fine-tune only after locating the remaining error source; rerun independent evaluation before replacing the application baseline.

Do not launch the old two-epoch Phase 5 script expecting mask removal. It remains a restoration experiment, useful as a comparison if already underway. This research does not prove any candidate improves our images yet. The outcome of the first benchmark decides whether to adapt a pretrained face model, train a compact custom component, or justify the added cost of diffusion.
