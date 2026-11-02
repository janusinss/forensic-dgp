# Current status: R1 transfer verified; awaiting manual VM execution

The source-only status below is historical. The repaired R1 packet is staged,
independently admitted and archived; all28 transfer members and checksum bytes
verify. No model/gradient/training execution has occurred for this diagnostic.
The earlier unrun draft is retained and must not be uploaded or launched.
Use [the exact five-step R1 guide](COMPLETION_CODEFORMER_GRADIENT_V1_R1_VM.md).
Protocol e1e4b9fd8c4bf71db806deca03a66e187e9726cafff65ff799acc25aa432387e;
archive c17dea799cc29f560b567acf138b0ee966cad4421976321152112d630c0e61ff.

The R1 supervisor adds owned process-group cleanup and a terminal receipt on
Ctrl+C, SIGTERM or launch failure. All gradient/objective functions are unchanged
from the unrun draft. Independent fake-child checks verify those stop paths.
The exporter always distinguishes archive success from diagnostic qualification.

---

# CodeFormer completion gradient diagnostic V1

25 October 2026. SOURCE DESIGN PREPARED; NOT STAGED, TRANSFERRED OR RUN. This is a finite diagnostic before proposing fine-tuning. It performs **zero optimizer updates**, saves no trained checkpoint, changes no original checkpoint and makes no model-quality claim.

The completed HQ TRAIN baseline probe remains independent retained evidence: 30 requests, 24 neural calls, six exact empty controls, six visual pages. Its source selection is reused prospectively here; selecting four references is by ascending zero-padded reference ID, not output quality. This diagnostic does not reuse reserved final evaluation identities or original VAL-role examples.

## Concrete question

Can the selected original CodeFormer inpainting architecture expose finite, nonzero connected gradients through its intended code-prediction path and existing fusion blocks on the same masked HQ TRAIN examples? Existing CodeFormerCompletion uses inference_mode, freezes parameters and detaches selected codes; it must not be used as a trainable wrapper. This worker imports the exact underlying architecture directly and preserves its forward implementation.

A pass permits designing a separate finite training proposal; it does not establish that gradients improve anatomy, preserve appearance, generalize to CCTV or satisfy the completion goal. A failure identifies disconnected, zero or non-finite groups and retains each group/tensor finding; never relax it or launch training automatically.

## Frozen selection and data

The parent readiness receipt records391 source-reviewed TRAIN-role FFHQ1024 originals with genuine256 targets, preserving53 original VAL-role references separately. Readiness SHA-256: `87dabe0b2c81350d91f9c0a00666c44dd8b514d1ac23d83c2b571f4cef80f0d6`.

The six-source HQ probe protocol SHA-256 is `38fcd82360e547e32189d6887a3102182ec21a56692e728a632ce955311aaa0a`. Take only its first four sources: `tr_ffhq_00084`, `tr_ffhq_00178`, `tr_ffhq_00323`, `tr_ffhq_00616`. Use each source's existing, hash-pinned **lower_face** and **eye_band**256 binary masks, giving exactly eight TRAIN diagnostic cases. Mask geometry is held fixed.

Copy the exact genuine1024 original bytes, not an upscaled128 thumbnail. On the VM, resize each clean1024 RGB PNG directly to512 with Pillow LANCZOS. Resize its256 mask to512 with nearest neighbor. The masked hole is white RGB1, all observed pixels remain the clean target outside that support, and model input is normalized to[-1,1]. This is synthetic procedural missing support on TRAIN faces. It is not a real occlusion dataset or native CCTV evaluation. Clean-target access is legitimate training supervision here, not deployment behavior.

Per-image Flickr attribution/license records and the readiness/probe/source-admission receipts are included in prospective packet evidence. Unknown official pretraining overlap and unestablished subject-identity disjointness remain explicit. Neither Asian ethnicity nor Zamboanga performance is inferred.

## Architecture API actually inspected

Original `codeformer_inpainting.pth`:370,786,611bytes; SHA-256 `b0d1b868c3dacf75d637bbcc0d4b3dd4ce2a064a338dfb4d35c740d3b4ae8797`. The model is instantiated with dimension512, vocabulary512, eight attention heads, nine transformer layers and existing fusion resolutions32/64/128. State loading uses `torch.load(..., weights_only=True)` and strict `params_ema` loading. No restoration model is imported or called.

The inspected `VectorQuantizer.forward(clean_latent)` returns `(quantized, codebook_loss, stats)`; nearest code indices are `stats['min_encoding_indices']`. Reshape those256 indices to[B,256]. `net(masked, code_only=True)` returns logits[B,256,512] and latent[B,256,16,16]. No softmax is applied before cross entropy. `net(masked,w=1,detach_16=True,adain=False)` returns raw RGB plus logits/latent; its chosen quantized feature and encoder-to-fusion features are detached in the unchanged architecture.

## Two separately isolated gradient paths

1. **Code path:** under no_grad, cache each clean reference's original encoder latent and nearest quantizer codes. Freeze the quantizer, generator and fusion modules. Enable only encoder, position embedding, feature embedding, transformer and logit-prediction parameters. On a white-hole input, query gradients of `CE(masked_logits, clean_nearest_codes) + MSE(masked_latent, clean_latent)` once per case. CE uses all256 tokens; this is a connectivity/objective diagnostic, not a tuned final loss.
2. **Fusion path:** freeze everything except the three existing fusion blocks. Perform a separate original forward with w1/detached quantized codes/adaIN false. Query gradients of mask-only RGB L1 plus inner-boundary RGB L1 once per case. Losses use un-clamped `(raw+1)/2` so clipping does not hide a gradient. The inner boundary is `mask - erode(mask)` using a seven-pixel max-pool kernel on the inverse mask. The source is copied exactly outside the mask only for the retained estimate image, never added as a source of supervision outside the declared loss support.

Each gradient query uses `torch.autograd.grad`, does not accumulate `.grad`, and never calls an optimizer. Record every intended tensor's norm and SHA, missing/zero tensor names, and per-group finite/nonzero/connected status. The required diagnostic condition is every intended group finite/nonzero and no disconnected intended tensor in every case; individual exactly-zero tensors remain visible in receipts. Original complete model-state SHA before/after and original checkpoint SHA after must match. Eval mode and fixed seed240 prevent stateful training updates; FP32/no autocast/no TF32 avoids precision changes masquerading as path behavior.

**Teacher limitation:** these clean latent/code labels come from the **same frozen inpainting encoder/codebook**. They are not an independent high-quality face prior and cannot prove that the original code vocabulary represents the true hidden region. A gradient-path pass may still be followed by a failed learning experiment.

## Finite execution and receipts

Run only manually on the authorized GCE g2-standard-4 with one NVIDIA L4, at `~/forensic-dgp/cctv_completion_codeformer_gradient_v1_vm`. The script refuses another OS/path, machine type, missing CUDA, a different GPU or a competing GPU workload before its first Torch import. No local neural/gradient run is authorized by preparation.

-600-second external process cap, including child initialization. Timeout kills only the diagnostic-owned child process group and records exit124.
-Exactly24 declared forward slots:4 clean encoder calls +4 clean quantizer calls +8 code-only full forwards +8 RGB full forwards. Reused nested architecture operations are not additional top-level slots; typed counts are retained.
-Exactly16 gradient queries,8 cases,4 sources, zero optimizer updates/epochs.
-128MiB output ceiling, checked between operations; memory errors, invalid losses/gradients, changed sources/checkpoint, or timeout stop and retain evidence.
-No historical recipe auto-runs, training state updates, model promotion or IMPLEMENTATION merge.

Each case retains component losses, per-tensor gradient fingerprints/norms, group connectivity, elapsed time, clean/white-hole/estimate512 PNGs and compressed logits/latent/raw RGB/mask arrays. These512 artifacts diagnose internal model behavior; they are not a new256 app acceptance result. `results.json` separates diagnostic completion from `gradient_path_pass` and `model_qualified=false`; `failure.json`, `diagnostic.log` and `supervisor.json` retain failures/caps. The root-owned exporter and shell wrapper are separate required packet sources. The wrapper must retain the diagnostic exit status and attempt evidence export after success, failed gradient-path qualification or timeout; export completion must not imply training or quality success. No archive is created during source preparation.

## Source-only preparation and later manual invocation

New files:

- `scripts/cctv_completion_codeformer_gradient_v1_vm.py`
- `scripts/prepare_completion_codeformer_gradient_v1.py`
- `tests/test_completion_gradient_prepare_v1.py` (tiny synthetic builder fixtures only).
- This design document.

The preparer also requires the separately reviewed root-owned `scripts/export_completion_codeformer_gradient_v1.py` and `scripts/run_completion_codeformer_gradient_v1.sh` before staging. Both become exact pinned members of the frozen packet.

The preparer has **not been executed**. After root source review only, this exact command stages a NEW packet directory and fixed protocol; it copies the original checkpoint, exact source files/notices, four originals, eight masks and evidence. It creates no archive and refuses an existing namespace:

```powershell
& 'C:\xampp\htdocs\YEAR 4\Testing\venv\Scripts\python.exe' -B .\scripts\prepare_completion_codeformer_gradient_v1.py --stage --probe-protocol-sha 38fcd82360e547e32189d6887a3102182ec21a56692e728a632ce955311aaa0a
```

The staged entry can then be checked locally with `--verify-transfer` (hashes only, no Torch/model/gradient calls). Root must independently review/freeze it before producing a transfer archive and exact gcloud/tmux commands. The eventual VM entry is:

```bash
python -B -u scripts/cctv_completion_codeformer_gradient_v1_vm.py --root . --protocol-sha <NEW_FROZEN_PROTOCOL_SHA> --verify-transfer
python -B -u scripts/cctv_completion_codeformer_gradient_v1_vm.py --root . --protocol-sha <NEW_FROZEN_PROTOCOL_SHA> --run
```

The placeholder has no valid value until packet preparation/review. These are design entrypoints, not authorization to launch a VM run now. Actual gradients must run under the user's manual VM workflow. Keep the CodeFormer [S-Lab license](https://github.com/sczhou/CodeFormer/blob/b33cc7d639d6545bfcccc7e0bc6ae51f24e79c2b/LICENSE), source notices and [official inpainting asset provenance](https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer_inpainting.pth) with the packet. No blanket deployment license is asserted.

## Source checks performed locally

Both CLI help paths and source compilation pass without Torch imports. Three tiny packet-builder fixtures pass: a complete staged closure includes source admission/provenance and the required exporter/shell members; a missing exporter and changed admission each fail before creating a packet. Fixtures use tiny synthetic bytes, not real photos/checkpoints or model execution. No actual packet, archive, GPU computation, gradient query or optimizer update was produced by these checks.
