# Local Windows completion installation plan V1

25 October 2026. DESIGN/PREFLIGHT ONLY. The current CodeFormer reviewed-mask prototype is functional on the reference laptop. It is not yet a self-contained fresh-clone package, a qualified covering-family model or an IMPLEMENTATION release. No existing admitted source, restoration file, dependency environment or implementation branch was changed for this plan.

## What is already verified

The latest root browser audit with the Windows Job Object guard records one nonempty CPU request in **10.919 seconds**, one CodeFormer call and an unchanged empty-mask bypass in **0.461 seconds**. The guard repair passed11 server tests and11 guard tests. Earlier12.283-second generation, refusal, real PNG/ZIP downloads and375/768/1280-pixel layouts remain retained historical functional checks. Timing variation does not establish a faster or better model; the worker, model weights and preprocessing are unchanged. These checks establish functionality on one device. They do not establish plausible anatomy across all seven covering families, minimum hardware, user acceptance or independent final quality review.

Reference device: Windows, Intel i5-11260H, 6 cores/12 threads, 15.69 GiB RAM. Python 3.13.5 64-bit is the measured interpreter. The worker uses CPU, four Torch intra-op threads, one inter-op thread and a parent-enforced 120-second process cap. Other devices remain unmeasured. The actual guarded request recorded peak Job Object private/commit usage of1,817,030,656bytes for the owned job and1,814,564,864bytes for its peak process, including the real worker descendants. This is not physical RAM/RSS, a minimum-RAM requirement or a device guarantee. Temporary disk requirements, installed environment bytes, physical peak RAM and minimum RAM remain unknown. The older HQ supervisor sampled only a launcher; its resource and termination records remain invalid for the real worker, and the new guard does not retroactively validate them.

## Concrete inventory and executable preflight

Run this from the FACE_MASK_COMPLETION worktree with the existing research interpreter:

```powershell
& 'C:\xampp\htdocs\YEAR 4\Testing\venv\Scripts\python.exe' -B .\completion_portable_preflight_v1.py
```

This reads and hashes the 13 pinned source/notice files, exact inpainting checkpoint and current source-admission receipt; it reads package metadata without importing Torch or application modules. It does not install, download, change assets or run neural inference. Exit 0 means the current reference inventory/runtime matches. `portable_ready` remains false. Add `--require-portable` to return exit 2 while packaging blockers remain. `--root` and `--weights` select read-only inventory locations while retaining fixed expected hashes. `--output` creates one new report, refuses overwrite, and requires an existing parent directory.

The retained report is `outputs/completion_portable_preflight_v1/report_r2.json`; its current result is `integrity_pass=true`, `current_reference_preflight_pass=true`, `portable_ready=false`. Current source closure is **14 files / 138,555 bytes**. The checkpoint is **370,786,611 bytes**. Their sum is **370,925,166 bytes** before installer, evidence, dependencies or generated outputs. This is not an installed-size estimate. Negative checks reject changed source, missing checkpoint and parent traversal without importing Torch.

## Source closure to carry forward after acceptance

Keep these completion-owned runtime files and relative layout:

- `completion_codeformer_web_v1.py`
- `completion_process_guard_v1.py` (stdlib-only Windows Job Object/owned POSIX session lifecycle guard)
- `scripts/completion_codeformer_worker_v1.py`
- `templates/completion_codeformer_v1.html`
- `pretrained_completion.py`, `completion.py`, `face_workflow.py`
- `third_party/__init__.py`
- `third_party/codeformer/__init__.py`, `codeformer_arch.py`, `vqgan_arch.py`, `registry.py`, `LICENSE`, `NOTICE.md`

This is the conservative closure for the current lazy-import completion route. Do not copy the research repository, datasets, restoration weights, training caches or historical outputs into a deployment package. OpenCV is currently required because shared completion/face-workflow helpers import `cv2`; extracting smaller pure helpers would change admitted source and needs separate review. The package will also need newly reviewed admission/configuration, installer/launcher and model-manifest entries; those are not implemented by this inventory.

## Runtime dependencies and asset identity

The measured completion route has four direct external packages:

```text
torch==2.13.0
numpy==2.5.2
Pillow==12.3.0
opencv-python==5.0.0.93
```

Use the official CPU Torch wheel index when implementing installation: `https://download.pytorch.org/whl/cpu`. Distribution metadata alone does not establish wheel flavor. The current worker receipt establishes that this request ran on CPU. The existing IMPLEMENTATION installer supports 64-bit Python 3.12/3.13; completion has only been measured with 3.13.5 here. Treat 3.12 as pending separate fresh-install verification.

Required asset: **CodeFormer inpainting**, not `codeformer.pth`, `codeformer_r2.pth`, U96 or u64 restoration assets.

- Official asset: [codeformer_inpainting.pth, release v0.1.0](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth).
- Bytes: `370786611`.
- SHA-256: `b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797`.
- Existing location: `C:\xampp\htdocs\YEAR 4\Testing\checkpoints\codeformer_inpainting.pth`.
- Architecture source revision: `b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b`; local `NOTICE.md` records relative-import/logger adaptations.

Retain the [S-Lab License 1.0](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/LICENSE), copyright, conditions, disclaimer and upstream notices when redistributing source or binaries. Its permission is for non-commercial use; commercial use requires contacting contributors. A thesis experiment does not by itself establish permission for every later deployment. Keep original release provenance for the weights; no broader weight permission is asserted here.

## Exact blockers to repair in a separately reviewed packaging change

1. The server's `ASSET_ROOT`/`CHECKPOINT` points into original Testing. The worker's `DEFAULT_CHECKPOINT` does too. Resolve the accepted asset from a fixed package-relative model manifest instead; a device must not need the old directory. Keep exact size/hash verification before use.
2. The source admission pins absolute source/import and checkpoint paths. Do not alter its old receipt or disable its checks. Preserve it, then generate a new reviewed package admission bound to the new source fingerprints, relative member names and expected import origins for that checkout. Runtime verification must reject paths outside the package root and changed sources.
3. The current IMPLEMENTATION model manifest lists restoration R2/U96/u64 only. Add a separately named inpainting asset after acceptance, with its own official URL, fixed filename, bytes, hash and terms. Download into a temporary file, verify before installation and preserve mismatching existing files.
4. Reuse the inspected IMPLEMENTATION installation pattern deliberately: a checkout-specific short environment under `%LOCALAPPDATA%\DGP\<checkout-hash>`, computed/validated environment configuration, CPU Torch first, pinned dependencies, no administrator requirement and no deletion of existing venv/model files. The launcher must use the computed checkout environment, wait for actual `/health`, report its actual loopback URL and never kill unrelated occupied-port processes.
5. Keep model work in a disposable CPU worker with its120-second cap, exact outside-mask PNG audit, immutable generation IDs and original/mask/result evidence. Carry the reviewed process guard into the package and bind its SHA at startup. On Windows, assign the suspended worker to its private Job Object before resuming; timeout/shutdown must terminate and await the owned tree, including venv redirector descendants. There is no unguarded fallback. A timeout or load failure must return an error rather than publish an unaudited result.

## Acceptance sequence before IMPLEMENTATION

First review the fixed development examples across masks, sunglasses, strong glare, hands, obstructing hair, scarves/objects and uncovered/clear-glasses controls. Keep manually assisted results separate from future automatic proposals. Judge hidden-region plausibility, boundaries and exact outside-mask pixels; generation is an estimate, not verified identity. Ask for clearer/less-covered input when visible support is insufficient.

After user acceptance, make the small packaging/path changes, preserve the old admission and establish a new one. Test a fresh Windows checkout outside both existing Testing paths, with CPU only and no research asset root. Measure install/environment/model sizes, peak RAM and CPU time. Exercise nonempty completion, empty bypass, insufficient support, malformed masks/uploads, timeout/error flow, repeated immutable generations and PNG/ZIP contents. Re-run full application/restoration regressions and responsive browser flow before selectively porting accepted sources/assets to IMPLEMENTATION. Automatic removal-area detection remains a later, separately reviewed stage.

No installer was created or run by this plan; no training or inference ran for this inventory. The completion goal remains active.

## Guard refresh evidence

Current server SHA-256: `be708101d0ded43a1ea0a1a2072c2b23fc6310d43a6f15e62ea9dded766daa6e`. Guard SHA-256: `1277b0fde51c35337a0391d4089b193aa837f21c7e00ecc9ee4ed7ecd5f687c0`. The independent server R2 review is `outputs/completion_codeformer_manual_v1/server_review_r2.json`, SHA-256 `68922364a949b9847b6dc2934bd5f66077cc0d3422a0aaf00fad202a2d91d8ac`. Actual browser flow is `scratch/completion-codeformer-job-guard-playwright-v1.json`; its nonempty request2ca6b33aca9d4c5ba1c908d8 records returncode0 and zero active owned processes in its supervisor. Original report.json and pre-guard source/doc copies remain retained in outputs/completion_portable_preflight_v1.
