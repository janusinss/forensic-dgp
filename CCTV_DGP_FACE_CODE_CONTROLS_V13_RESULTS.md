# V13 rendering controls — completed 4 October 2026

**Both code prediction and rendering statistics need repair.** Clean target codes
produce coherent face layouts through the frozen renderer. V12's predicted codes
still produce displaced features, invented glasses and changed mouths. Original
degraded-image AdaIN statistics also transfer dark/noisy texture into otherwise
coherent oracle faces. Removing those statistics with wrong predicted codes
does not solve the problem and worsens several outputs.

This is an inference-only diagnostic on the same ten training references, five
from each source, with five degradation profiles: 50 cases. Clean-label arms
consume ground-truth teacher codes and are explicitly **oracle controls**.
They are not trained restoration results, exact identity recovery or deployable
upgrades. No validation, native development or reserved image was used. The
teacher/no-stat arm has only ten unique renders reused across profiles.

Windows root: `C:\xampp\htdocs\YEAR 4\Testing\`; Linux root: `~/forensic-dgp/`.
Intended document counterpart: `~/forensic-dgp/CCTV_DGP_FACE_CODE_CONTROLS_V13_RESULTS.md`
after explicit sync. No document/git publication is claimed.

## Executed control and audit

Four new arms compare initial and update300 predicted codes without input
statistics, and clean teacher codes with versus without observed-input
statistics. The cached learned original-stat w0 output, input and clean target
appear beside them in each seven-column grid. Fidelity is fixed at w0; this
does **not** isolate or evaluate w1 skip connections.

The VM performed 162 generator forwards: 150 case-specific renders, ten cached
teacher/no-stat renders and two cached-render parity checks. Both parity checks
have exactly zero maximum float deviation. There were zero encoder,
transformer-head, conditioner, DGP, teacher or recognizer forwards, zero backward
calls and zero optimizer updates. All frozen prior tensors remained identical.

Loading/neural execution took **28.16 seconds**; full rendering/grid/export took
**52.21 seconds**. Peak allocated VRAM was **795,228,672 bytes**. The original
PyTorch `2.9.1+cu129` runtime was retained. Execution stayed within the frozen
240-second neural budget and 360-second external process timeout.

The **10.98-second** independent local audit checked 200 PNG compositions,
160 distinct raw renders and all 350 original grid cells. It rebuilt saved-pixel
metrics and verified recorded counts, states, pins, parity and timing. It made
zero neural, backward or optimizer calls. It does not replay the renderer or
measure identity. All five original-cell grids were visually reviewed.

## Pixel results and their limits

PSNR below is calculated from each arm's mean MSE, rather than averaging image
PSNR. Fifty cases are dependent views of ten training faces; oracle/no-stat
repetition is not fifty independent reconstructions. These photometric metrics
cannot select a structurally faithful learned model.

| Arm | Mean MSE | PSNR from mean MSE | Mean SSIM | Mean MAE |
| --- | ---: | ---: | ---: | ---: |
| Initial predicted codes, no statistics | 0.036251 | 14.4069 | 0.56436 | 0.12695 |
| Learned update300 codes, no statistics | 0.032318 | 14.9056 | 0.57116 | 0.13522 |
| Clean teacher codes, observed statistics | 0.027954 | 15.5356 | 0.69562 | 0.10816 |
| Clean teacher codes, no statistics — oracle | 0.002850 | 25.4514 | 0.80546 | 0.03212 |

The oracle shows useful renderer capacity, with remaining changes to eyes,
lips and expression relative to the clean target. It does not establish exact
hidden identity. Observed-stat oracle outputs preserve coherent geometry but
carry unwanted brightness, gray tone and noisy texture, especially for low-light
and compound profiles. Initial/learned no-stat outputs remain structurally wrong
and several become more stylized or develop stronger contrast/color errors.

No checkpoint, `best.pth`, selection, held-out evaluation or app promotion was
created. Existing production weights and the current app route remain unchanged.
Independent final review, native CCTV usefulness and DGP-led integration are
still pending.

## Evidence and execution pins

| Evidence | Windows below the root | Linux VM |
| --- | --- | --- |
| Frozen execution preparation | `outputs\cctv_dgp_face_code_render_controls_vm_v13\` | `~/forensic-dgp/cctv_dgp_face_code_render_controls_vm_v13/` |
| Result archive and SHA file | `outputs\cctv-dgp-face-code-controls-v13-results.tar.gz` and `.sha256` | Execution root, same basenames |
| Audited result directory | `outputs\cctv_dgp_face_code_render_controls_return_v13\outputs\cctv_dgp_face_code_render_controls_v13\` | Execution `outputs/cctv_dgp_face_code_render_controls_v13/` |
| Independent audit and original grids | Return root `local_independent_audit.json`; result `grids\` | Grids in execution result; local audit not synced |
| Review / stop receipts | `outputs\cctv_dgp_face_code_controls_review_v13.json`; `outputs\cctv_dgp_face_code_controls_v13_vm_stop.json` | Intended same relative paths after explicit sync |

Protocol: `093cf67b630de4ed74d0c2fa62fc928a69c2c2464e5b9ecfd23d22a8a0a603ac`.
Execution archive: 19,887 bytes,
`90f7a25bf834c7e8ffc8be2e0271d4fd945c9e83834ef3c18d24a5a51f4b7df5`.
Result archive: **152,609,077 bytes**,
`8315b26f79fff3f1e63440e289d56fcfb7edf277db4d3517e23138993089058e`.
Result JSON: `367ffef001155a4a551baeb1a03576a515f78f75f3366218a13f97a7788e5be6`.
Local audit: `e126b7e95e53af8b5049a55db3f90993bac63026c5c8cfadd4dd032d069c1f3e`.
Frozen prior before/after:
`5c13ca19e6bddeca7425aad535e4d9aa99adb7b509a064300c6d1cc374166deb`.

Return files and receipts are collected. GPU, tmux and project-process checks
found no competing work. Cloud confirms the VM was restored to **TERMINATED**,
last stop `2026-10-04T06:55:04.228-07:00`.

## Next distinct repair

Prepare our own direct code-prediction layer and a separately supervised clean
rendering-statistics layer, rather than repeating the V12 feature-residual loss.
Keep the declared pretrained renderer and retained DGP frozen. First establish
starting-output parity, VM-only gradient/VRAM bounds and train-cohort capacity
with a finite pilot. Clear-face preservation and structural review remain
required; lower code loss alone is insufficient. Only useful train-cohort outputs
justify a separate broader held-out experiment. Do not use oracle results to
adopt weights or relax previous gates.
