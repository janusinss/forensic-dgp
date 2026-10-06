# V16 r2: training verified; preservation failed

5 October2026. All3128 updates are independently audited after correcting the
stale20-versus30 cache timing assertion in a separate auditor. The original
pilot, failed VM auditor, checkpoints, protocol and failure export remain
unchanged. **Do not adopt or repeat this recipe unchanged.** Lower pixel error
does not override changed visible facial features.

Windows `C:\xampp\htdocs\YEAR 4\Testing\` ↔ VM `~/forensic-dgp/`.
Intended report counterpart: `~/forensic-dgp/CCTV_DGP_BROADER_CODES_V16_R2_RESULTS.md`
after separate transfer. No report upload or assistant cloud operation occurred.
Return: `outputs/cctv_dgp_broader_codes_failure_return_v16_r2/` ↔
`~/forensic-dgp/cctv_dgp_broader_codes_vm_v16_r2/`.
Separate local audit: `outputs/cctv_dgp_broader_codes_v16_r2_audit_recovery_1/`.

## Executed and verified

The839,574,170-byte archive hash, sidecar and downloaded receipt match the
user's pasted expectation. Safe import59.08s preserved the original failure,
verified25 manifest-bound assets and unchanged parent/data/baseline dependencies,
and checked3128 trace records/31280 exposures. Seven focused regressions passed
before the separate audit correction. Actual full output audit73.05s passes
within240s; bounded recovery including wrapper73.50s.

Coverage:1710 PNG metrics/cosines,300 raw/PNG previews,150 training-code probes,
100 fresh-image parity cases,781 training-only teacher arrays,4425 cache bindings,
3128 update records and600 original grid cells. Frozen model states and recorded
counts match. This is serialized arithmetic/provenance verification, not a CUDA
gradient/recognizer replay. The9GB cache stays on the VM. Zero local neural,
backward or optimizer calls occurred during import/audits. The original supervisor
failure stays a failure; the new full-audit receipt does not imply useful output.

VM cache285.74s, fit/evaluation769.38s, recorded total1055.12s (17m35s).
Exactly3128 optimizer updates,3129 backwards including one zero-update preflight,
31280 exposures. Cache estimate372.84s under900s; fit estimate968.40s under1200s.
Peak allocated VRAM2,199,736,320 bytes. Required finite/frozen-state evidence passes.

## Paired photographic development comparison

The same104 references/520 synthetic camera cases are held out from **our
code-head training**:416 degraded and104 clear. These are photographic proxies,
not native CCTV or Zamboanga evidence. Prior/teacher pretraining overlap remains
unverified. Native24 and reserved32 were not used. See `CCTV_FACE_PRIOR_DATA_LIMITS.md`.

All arms share256 inputs, observation support, targets and fixed alignment.
Starting/learned code arms use the declared frozen renderer with statistics
omitted and fidelityw0; this differs from official default CodeFormer. Raw
decoder floats are retained separately from delivered PNGs. Delivery uses the
declared512-to256 resize, clamp/quantization and unsupported-padding preservation;
no contrast enhancement or output-driven display processing was applied.

| Arm | Degraded PSNR dB | Degraded SSIM | Degraded fixed ArcFace | Clear PSNR dB | Clear fixed ArcFace |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained DGP V2 |16.027 |0.6193 |0.33011 |31.132 |0.95701 |
| Starting prior, no statistics |14.055 |0.4944 |0.19348 |24.071 |0.59986 |
| Our epoch4 component |17.312 |0.5666 |0.19041 |23.492 |0.58247 |
| Our epoch8 component |17.402 |0.5788 |0.19280 |24.023 |0.56713 |
| Exact input |14.482 |0.5755 |0.33262 |Exact reference |1.00000 |

PSNR comes from pooled MSE. Fixed-affine ArcFace embedding similarity is not
recognition accuracy or identity proof. Epoch8 gains1.375dB degraded PSNR over
retained DGP, while SSIM and embedding similarity regress. Both epochs4/8 fail
unchanged preservation guards against DGP and the starting prior. Epoch8 has34
failed group/metric comparisons against DGP and14 against the starting prior.
Embedding similarity is lower in all eight degraded source/profile groups.
Blur and motion pixel error also regress in both sources; low-light/compound
pixel gains do not establish faithful visible features.

Source folders describe photographic provenance, not ethnicity or capture location.

| Source | Degraded cases | DGP → epoch8 PSNR dB | DGP → epoch8 SSIM | DGP → epoch8 fixed ArcFace |
| --- | ---: | ---: | ---: | ---: |
| `dataset/asian_faces` |204 |15.996 →17.606 |0.7043 →0.6409 |0.41642 →0.19746 |
| `dataset/thumbnails128x128` (FFHQ) |212 |16.057 →17.214 |0.5375 →0.5191 |0.24706 →0.18831 |

Fixed training-preview token CE7.12595 →5.14334 →4.85662 at epochs0/4/8;
accuracy5.02% →7.66% →10.86%. Epoch4 CE improvement27.82% passes its predeclared
≥1% fitting stop. This demonstrates a learned component contribution, while
showing that token fitting does not ensure coherent structure preservation.
No best.pth, checkpoint selection or application promotion.

## Original-cell review and next diagnostic

**Follow-up closed:** `CCTV_DGP_FIDELITY_SPOTCHECK_V17_RESULTS.md` records the
executed 12-case local training-preview control. Original encoder connections
with w1/no statistics improve clear previews but worsen degraded preservation
and leave fragments. Zero training updates; no adoption. The proposed control
below is historical and must not be repeated unchanged.

All ten predeclared sheets were viewed at original256 cells: five training and
five validation sheets, ten rows each, five references per source per role.
Six columns show camera, retained DGP, starting prior, own epoch4, own epoch8
and clean proxy. This is assistant development review, not final human review.

Some clear faces look coherent, but others have black/blue eye patches, changed
or missing clear glasses, altered expression and a nose fragment. Blur,
low-light and compound rows contain displaced eyes/mouths, invented frame or
facial-hair texture and large fragments. Motion includes coherent examples
alongside visible-feature changes and seams. Similar defects occur in training
previews. Retained DGP is softer but better preserves visible structure here.

Before more training, investigate explicit DGP/visible-appearance preservation.
V16 omits both observed statistics and every original-image fidelity connection.
V13's statistics controls fixedw0; they did not isolatew1 connections. A separate
bounded training-only inference control can compare unchanged initial/final codes
with original encoder connections, keeping statistics omitted. This is a new
processing hypothesis, not proof of a useful DGP-led output or permission to
relax failed guards. Do not add epochs or repeat this failed recipe unchanged.

| Evidence | Workspace-relative path |
| --- | --- |
| Original export/sidecar/receipt | `outputs/cctv-dgp-broader-codes-v16-r2-failure.tar.gz`, sidecar, `failure_export_v16_r2.json` |
| Full audit / bound recovery | `outputs/cctv_dgp_broader_codes_v16_r2_audit_recovery_1/local_full_audit.json`, `recovery_completion.json` |
| Verified metrics/counters/checkpoint pins | `outputs/cctv_dgp_broader_codes_v16_r2_verified_summary.json` |
| Assistant gallery review | `outputs/cctv_dgp_broader_codes_review_v16_r2.json` |
| Original checkpoints/logs/raws/grids | `outputs/cctv_dgp_broader_codes_failure_return_v16_r2/` |

Protocol SHA256 `4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0`.
Return archive SHA256 `3555eb65b37bcf4cbc24ff36a31f5a80a6dab02051fe3e2c9e7dd2cad026ed65`.
Results SHA256 `ccb9c94416a08672a37700fdf3c460170a7f7b9f1beecae88205e71820f2a20b`.
Original auditor SHA256 `0ec24b386ccfc210f4efe126d48d1e3e4064c38574bf158f139f310b7d6472fd`.
Corrected auditor SHA256 `a76ae49f66b36a232d39593bff7329a0631f99cd5a0f79f4f4df72fe230adb21`.

Useful native outputs, independent final review, DGP-led app integration,
covering-family readiness and full app Playwright remain incomplete. No local
training, assistant cloud operation, application promotion or Goal completion.
