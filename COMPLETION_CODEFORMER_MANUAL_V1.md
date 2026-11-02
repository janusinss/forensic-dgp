# Manual reviewed-mask CodeFormer completion prototype

25 October 2026. Status: experimental local prototype; focused local app flow and independent saved-output checks passed. The user chose CodeFormer completion as the preferred starting point after reviewing the saved comparison. This is not a newly trained model, full-family qualification or a release in IMPLEMENTATION.

## Open or launch

Current running prototype URL: **http://127.0.0.1:54734/**. The port is temporary; a new launch chooses an available loopback port and prints its actual URL. The current URL/PID is recorded in scratch/completion-codeformer-runtime-v1.json.

From C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION:

```powershell
& 'C:\xampp\htdocs\YEAR 4\Testing\venv\Scripts\python.exe' -B completion_codeformer_web_v1.py --admission-sha 188e54fd998ad97e9bef84607b8e92643f555565bce5a5c64c4a7a759f2c61ab --worker-sha 9230025591e6fd4ef5cbd254e826009f938dea9408c70c859bc54d0dadb1b861 --guard-sha 1277b0fde51c35337a0391d4089b193aa837f21c7e00ecc9ee4ed7ecd5f687c0
```

The first hash binds the reviewed source-admission receipt; it is **not** the web-server source hash. The second binds scripts/completion_codeformer_worker_v1.py. The third binds completion_process_guard_v1.py, which owns the complete worker process tree. Startup/source checks refuse missing or changed admitted files. If the reviewed source changes during a repair, repeat source review and use its new admission hash; do not bypass the check.

The current manual prototype uses the verified original asset root C:\xampp\htdocs\YEAR 4\Testing. That root supplies the retained CodeFormer inpainting weights and the original Python environment; this is not yet a fresh-install deployment package. Finished deployment will configure verified assets explicitly for local Windows, including CPU-only devices.

## Use the page

1. Upload one already cropped frontal or mildly turned face (PNG/JPEG, maximum8MiB). Inspect the prepared256 canvas; retain the original upload as evidence.
2. Paint the covering area orange. Use Erase, Undo, Clear or a256Ã—256 opaque binary mask PNG to correct it. White mask pixels allow generation; black pixels stay preserved. Keep clear glasses, non-obstructing hair and visible features outside removal support.
3. Confirm the removal area and that enough facial structure remains. Any image or mask edit resets confirmation. An empty reviewed mask bypasses the model; insufficient support asks for a clearer or less-covered crop.
4. Generate one estimate. Review facial anatomy and mask boundaries beside the input and mask. Hidden features are plausible estimates, not verified hidden identity.
5. Download the PNG and, when needed, the original/mask/result evidence bundle. Each completed request gets a new immutable generation ID; later runs do not overwrite earlier evidence.

## What is fixed in this test

- Model: **CodeFormer inpainting**, checkpoints/codeformer_inpainting.pth, SHA256 b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797. It is a different checkpoint/function from CodeFormer face restoration and its U96/u64 adaptations.
- No restoration, added blur, automatic covering prediction or training occurs in this prototype. Manual and future automatic mask results remain separate.
- Pixels outside the reviewed removal support on the prepared256 input are copied exactly by completion. Native upload preparation is a separate geometry step; copying prepared pixels does not prove hidden-region accuracy or useful anatomy.
- Worker inference is local CPU-only with a120-second process cap. Windows workers start suspended and join a private Job Object before resuming; timeout/shutdown terminates and awaits that owned tree, including redirector descendants. This boundary is independently tested and has no unguarded fallback. The agreed2-minute per-face target on other Windows hardware still requires measurements; this is not an arbitrary-device runtime guarantee.
- MI-GAN and converted MAT outputs remain comparison evidence. Their speed, successful exports or preserved outside-mask bytes do not qualify their generated regions. Existing failures and original checkpoints remain retained.

## Evidence and next gate

Bundled inline Playwright verified upload, binary paint/undo/import, confirmation resets, generation, empty bypass, insufficient-visibility refusal and PNG/ZIP downloads at375/768/1280px without horizontal overflow. The saved closure is scratch/completion-codeformer-playwright-r2.json; its retained initial audit contains a mistaken negative test whose dot-segment URL normalized to the legitimate root page. The corrected path checks passed without repeating neural inference.

One existing cloth-mask example took12.283 seconds including worker startup on the declared i5-11260H/15.69GiB reference computer, using CPU4threads. Empty-mask bypass took0.618 seconds with zero model calls. Independent saved-output review verified337,623 unchanged outside-mask source bytes across the two requests, raw-to-PNG arithmetic and downloaded evidence. The nonempty raw completion exactly matches the historical CodeFormer baseline; byte differences in its PNG are only declared rounding (at most1 per RGB channel), not a new model improvement. See outputs/completion_codeformer_manual_v1/independent_functionality_audit.json.

The latest guarded browser flow is scratch/completion-codeformer-job-guard-playwright-v1.json. Its nonempty request2ca6b33aca9d4c5ba1c908d8 completed in10.919seconds with one neural call; empty request072c5734d6714f31807225bd took0.461seconds with zero calls. The nonempty supervisor records returncode0, zero active owned processes, peak job private/commit1,817,030,656bytes and peak-process private/commit1,814,564,864bytes. These are owned Job Object commitment measurements, not RSS, physical RAM or a certified minimum-RAM requirement. Timing variation from the earlier12.283-second request does not establish model improvement. The earlier HQ supervisor followed only a launcher; do not use its memory or termination records as actual-worker evidence. The guard repair passed11 server tests and11 guard tests; see outputs/completion_codeformer_manual_v1/server_review_r2.json.

The retained CodeFormer sources carry S-Lab License1.0, permitting non-commercial use/redistribution under its conditions; preserve third_party/codeformer/LICENSE and attribution with any deployment. This is a declared pretrained third-party component. Pretraining identity overlap is unknown; the existing development photos are unpaired and do not support hidden-region PSNR/SSIM or true-identity claims. Useful manual completion and user satisfaction must precede automatic-detector improvement and later selective integration into IMPLEMENTATION. Restoration development remains isolated in RESTORATION. The full completion goal remains active.
