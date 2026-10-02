# Face-specific AOT-GAN comparison — 2 October 2026

The author-provided CelebA-HQ AOT-GAN generator produces useful lower-face and
eye estimates on several reviewed cases. It provides facial anatomy where the
generic Places2 LaMa checkpoint produced blur, but it does **not** resolve the
dark-eyewear, white-glare or mask/clear-glasses boundary failures. It is not
selected as a replacement for the CodeFormer baseline.

| Location | Windows local | VM counterpart after transfer |
| --- | --- | --- |
| Report | `C:\xampp\htdocs\YEAR 4\Testing\PRACTICAL_AOT_RESULTS.md` | `~/forensic-dgp/PRACTICAL_AOT_RESULTS.md` |
| Frozen protocol | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_aot_protocol_v1.json` | `~/forensic-dgp/outputs/practical_aot_protocol_v1.json` |
| Outputs | `C:\xampp\htdocs\YEAR 4\Testing\outputs\practical_aot_outputs_v1\` | `~/forensic-dgp/outputs/practical_aot_outputs_v1/` |
| Weights | `C:\xampp\htdocs\YEAR 4\Testing\outputs\aot_celeba_pretrained_v1\G0000000.pt` | `~/forensic-dgp/outputs/aot_celeba_pretrained_v1/G0000000.pt` |

These are local inference/review artifacts. No training, VM transfer, commit or
push occurred. Main-application behavior remains unchanged and the Goal is active.

## Source and runtime

The [official repository](https://github.com/researchmm/AOT-GAN-for-Inpainting)
was pinned at `2cd1afd8fdfabb101c678f6062d14bc7d302509e`. Its exact architecture,
common helpers and Apache-2.0 license are retained under `third_party/aot_gan/`.
No architecture changes or old training-environment installation were required.

The [author-linked CelebA-HQ folder](https://drive.google.com/drive/folders/1Zks5Hyb9WAEpupbTdBqsCafmb25yqsGJ)
supplies `G0000000.pt`, public file ID `1T7Xkv09pvf6gy-R2Cn0RN5V-Pq0vSFPw`.
The 60,829,150-byte file downloaded in 6.9 seconds. Its observed SHA256 is
`5cfbf8e545e75adc1be66682741d22b4e5288f734971cc896b795e7e137f42f9`.
This is a pinned acquisition fingerprint, **not a publisher-supplied checksum**.
The legacy-format tensor checkpoint loads with `weights_only=True`; all 108
tensor states match the eight-block generator strictly and are finite.

Eight blocks and rates `[1,2,4,8]` match the checkpoint/source configuration.
The benchmark uses 512-pixel inference, following the repository default.
Exact original training resolution/history are not verified from this tensor-only
checkpoint; `provenance_erratum.json` clarifies the earlier `training_size` field
without mutating frozen provenance. `aot_completion.py` uses official normalized
white fill **plus the explicit mask**, isolation from covered RGB during resizing,
and exact original compositing outside the removal region. Four adapter tests pass.

## Fixed output comparison

The ten native cases and reviewed removal masks remain unchanged from the frozen
`practical_gallery_v2` protocol. Twenty already audited CodeFormer/LaMa baseline
outputs are reused; no baseline regeneration, detector routing, new alignment or
restoration is introduced. There is no uncovered reference or hidden-face MAE.

Ten requests comprise eight nonempty generator forwards and two empty control
bypasses. CPU elapsed **29.4 seconds**; zero inference failures, detector forwards
or optimizer updates. Recorded before/after model-state digests match. An
independent PIL/NumPy audit verifies all 10 outputs, 20 cached baselines, frozen
assets/source/weight hashes, and zero changed pixels outside each reviewed mask.

Assistant visual inspection of both five-row pages finds four useful covered
estimates, one partial estimate and three needing fixes; both clear controls are
exactly preserved. This is developmental triage of previously inspected training
sources, not independent expert scoring or an unbiased accuracy estimate.

| Case | AOT-GAN reviewed-output finding |
| --- | --- |
| Cloth mask | Useful lower-face estimate; softer join/detail than CodeFormer |
| Pink mask | Useful lower-face estimate; soft/waxy texture and boundary residue |
| Mask with clear glasses | Mouth estimated, but bright remnant near nose and uneven join remain |
| Dark sunglasses | Eye-like content, but dark bridge/upper frame and diffuse artifacts remain |
| Sunglasses | Useful estimated eyes with a faint boundary |
| White glare | Strong white reflection remains |
| Mirrored eyewear | Estimated eyes; residual circular frame/rim outside current proposal |
| Hand over mask | Useful lower face; softer/patchier texture than CodeFormer |
| Uncovered control | Exact unchanged empty-mask bypass |
| Ordinary clear-glasses control | Exact unchanged empty-mask bypass |

| Evidence | SHA256 |
| --- | --- |
| Frozen protocol | `a53c4b8e3a27f30d9ccf0c49c7e04454f6303d72b3c6c08c5977ee168fb7d09b` |
| `results.json` | `bf77388894dd27b68d7f8065a237b6d4a21855770eded8c88e1b79e5ce32a278` |
| `independent_verification.json` | `412222ebb436be618a2ada7b92ae1e4ac5bd30a014620b25dc593f33a1bd25af` |
| `preview.png` | `870eb3c05eee9848ddab0ea806dcf4ce358a9786632ed9364f4f79ca28252b70` |

## Next action before VM training

Audit covering footprints using the source images, particularly complete opaque
eyewear rims/bridges, clear-frame glare boundaries and the mask near the nose.
Prepare revised operator removal proposals in a **new version**; keep old labels,
proposals, protocol hashes and output findings immutable. Do not assume that a
reviewed approximate label is a complete covering footprint. Do not remove
ordinary clear frames as part of glare correction.

Freeze the revised policy before new outputs. Compare CodeFormer and AOT-GAN on
the targeted cases with the same revised masks; inspect whether remnants are
outside the active mask or newly synthesized inside it. This distinguishes
mask-construction limits from completion limits without another unchanged model
or detector run. Only the demonstrated remaining limitation should determine a
bounded VM experiment. Historical detector gate failures remain unchanged.

Degraded restoration, standalone-hand/hair/scarf/object coverage, meaningful
facial-visibility rejection and integration in the existing main UI remain pending.
The current pilot does not establish full-scope readiness. Actual training remains
on the NVIDIA L4 VM; local work here is inference, preparation and verification.
