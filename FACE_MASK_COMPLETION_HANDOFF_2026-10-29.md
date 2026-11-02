# FACE_MASK_COMPLETION and IMPLEMENTATION continuation handoff

Verified 29 October 2026 (Asia/Manila). This is a source, Git-ref, and artifact-hash audit. No new training, inference, installation, or browser verification was run for this update.

## Current answer

| Function | Current model | Status |
|---|---|---|
| IMPLEMENTATION restoration | Conditioned SFT V2, `p2_u0224` | Sole supported restoration model; experimental |
| Reviewed covering completion | Our trained CodeFormer inpainting R1 | Selected experimental completion model; integrated into IMPLEMENTATION |
| Covering proposal | Compact trained epoch-2 segmenter, portable `covering_detector_segmenter_v1.pth` | Available through the API; current guided editor uses manual selection |

Conditioned SFT V2 restoration and the failed completion coverage V2 experiment are different models/experiments. Historical notes saying IMPLEMENTATION defaults to U96 are superseded: U96 and terminal-u64 components now form part of the V2 restoration backbone.

Current guided flow: preparation/restoration -> manual covering selection and review -> R1 completion -> result/evidence downloads. Completion copies every pixel outside reviewed support exactly from the restoration-result canvas. This guarantee does not compare those pixels with the original upload. Empty reviewed masks bypass completion. There is no restoration pass after completion.

## Verified code locations and publication

| Checkout | Branch/ref | Verified revision |
|---|---|---|
| `C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION` | FACE_MASK_COMPLETION | `7de6643` plus substantial local work |
| `C:\Users\janus\.codex\worktrees\implementation-one-shot\Testing` | IMPLEMENTATION | `eb94632d233fac0f81cf653cec5d0af8bcb7ed7b` |
| GitHub `janusinss/forensic-dgp` | refs/heads/IMPLEMENTATION | `eb94632d233fac0f81cf653cec5d0af8bcb7ed7b` |
| Original repository `C:\xampp\htdocs\YEAR 4\Testing` | main | `570a8f7bc2ee0001e63ad8d7adc002d7c2e60cad` |

Live `git ls-remote` verified IMPLEMENTATION and main. It returned no remote FACE_MASK_COMPLETION branch. The current IMPLEMENTATION commit is dated 29 October 2026, 15:12:35 +08:00. The 28 October R1 integration publication at `6059fc32acbe04800241fc5be15f07549d30b0b9` is an ancestor. Later app changes include the switch to manual covering selection and editor refinements; both model manifests and inference runtimes remain unchanged from that integration.

Before this handoff was added, the completion checkout had 2 modified tracked files (`.gitignore`, `PROJECT_HANDOFF.md`) and 489 untracked files. Its committed branch alone does not contain the current research work. The current IMPLEMENTATION checkout's working-tree cleanliness was not established by this audit. Its models directory and installed-environment record were absent when inspected.

The ordinary research worktrees share Git metadata with the original Testing repository. Copying a linked worktree directory alone does not create an independent repository. Preserve local changes and required ignored assets when moving to a different computer or repository.

## Restoration model composition

These four files are loaded together as Conditioned SFT V2; they are not four selectable restoration models.

| File in `models/implementation/` | Bytes | SHA-256 |
|---|---:|---|
| `codeformer_r2.pth` | 376609465 | `d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896` |
| `codeformer_u96_adapter.pth` | 212827 | `958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0` |
| `terminal_u0064_heads.npz` | 685818 | `77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17` |
| `v2_p2_u0224_heads.npz` | 3831362 | `1aee2b840397925049a2c821edd7395c3a9e4b48ce1552e06c5ffd627a33e31f` |

The 575-tensor R2/U96/u64 backbone plus 212 V2 tensors makes 787 frozen tensors. Restoration uses local CPU FP32 inference at 512 pixels, fidelity 1 with AdaIN, followed by conditioned 256-pixel delivery and floor PNG quantization. Off bypasses restoration inference. The installer obtains the first three files from `implementation-models-v1`; the V2 heads are also bundled under `implementation/assets/`.

Authoritative current sources: IMPLEMENTATION `MODEL_CARD.md`, `implementation/models.json`, and `implementation/runtime.py` (only model ID `v2` is accepted).

## Completion model and preserved research state

| Deployment asset in `models/completion/` | Bytes | SHA-256 |
|---|---:|---|
| `codeformer_completion_r1.pth` | 370764381 | `7ae2486739de0f3efa99cf721a0fcb755f8f56cf69e295412f74190d2fda37e0` |
| `covering_detector_segmenter_v1.pth` | 2681549 | `ccc22d524073368bc4f5d6b6d2136c90d0e2a117083e4c75fa98d42d7bfff6cf` |

These are installed from the `completion-r1-models-v1` release. R1 has 497 inference tensors, uses internal 512-pixel inference, fidelity 1, AdaIN off, reviewed-mask-only composition at 256 pixels, and round-to-nearest PNG quantization. Restoration and completion have distinct inference and delivery policies.

The research copy of R1 has a different filename but identical bytes. Its current verified location is:

`C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION\outputs\completion_codeformer_finetune_v1_r1_candidate\codeformer_completion_finetune_r1.pth`

The separate full research recovery checkpoint is:

`C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION\outputs\completion_codeformer_finetune_v1_r1_return\cctv_completion_codeformer_finetune_v1_r1_results\outputs\terminal_training_state.pth`

Its verified size is 701952165 bytes and SHA-256 is `0a6b4ad907c6ead86d8f26d3446666387493cdce81c6adc1ed78a8677d50fadd`. Retain this Adam/RNG/schedule state and the candidate manifest; the inference-only deployment file cannot replace it for research continuation.

The initial R1 run completed 32 paired cycles with code and fusion Adam steps. Its delivered-PNG masked-region L1 error decreased 6.185334% against original CodeFormer on 32 adaptive synthetic TRAIN-monitor cases; 26 improved and 6 worsened. This is a development result, not independent final accuracy or true hidden-face recovery.

## Later completion experiments

| Experiment | Audited outcome | Decision |
|---|---|---|
| Coverage V2 | Quality gates failed; terminal serialization timed out, leaving an invalid partial checkpoint | Retain as failed evidence; no model adoption or unchanged rerun |
| Fusion-only V3 | Valid recovery state; gains versus R1 below required threshold with subgroup regressions | R1 retained |
| Delivered-objective V4 | Valid trained state; failed quality gates, including challenge regression | R1 retained |
| Predictor CE V1 | 32 new predictor updates completed correctly; approximately 0.31% clear / 0.45% challenge improvement versus R1 failed the 1% gates | R1 retained |
| Same-codebook probe V1 | Completed zero-update clean-context oracle diagnostic; informed predictor/spatial research | Diagnostic evidence, no new trained model |

The latest prepared successor is **spatial-code intervention V1**: PREPARED, UNRUN, ZERO TRAINING UPDATES. All 497 R1 tensors stay frozen. On the same 48 paired adaptive TRAIN cases, it compares clean-context code substitution in mask-touched cells against substitution in visible-context cells. It plans 108 forwards with no optimizer, backward pass, or new learned checkpoint. Clean-context oracle information is unavailable in real app use.

The next research step is the user's manual L4 transfer/run from `COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_VM.md`, followed by independent state/code/pixel and all-case visual review. No return evidence for this diagnostic was found in the inspected local evidence. Verify whether a return has arrived before launching anything. Do not automatically run the VM, repeat failed recipes, append epochs, rescue partial checkpoints, or promote an oracle output.

The execution archive remains at:

`C:\xampp\htdocs\YEAR 4\Testing\outputs\cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz`

Verified size: 13610328 bytes. Verified SHA-256: `24f480add44e4fe8176a68a3b67321b11b4baee0ebb43d0a3790fefc434549a5`.

An identical verified copy is retained inside the completion checkout at `outputs/completion_codeformer_spatial_code_intervention_v1_packet/draft_004/cctv-completion-codeformer-spatial-code-intervention-v1-execution.tar.gz`.

Protocol SHA-256: `b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123`. The small archive depends on exact original/R1 weights and 196 prior controls already on the authorized VM; it is not a self-contained research migration bundle.

## Working application versus unfinished research

The 28 October publication evidence records a successful fresh-clone public-asset installation, real CPU restoration/completion, browser workflow and PNG/ZIP downloads. Those historical checks were not rerun for this status update. Evidence is currently under this completion worktree's `outputs/completion_r1_implementation_publication_20261028_v1/`.

All nine automatic proposals in the retained degraded-photo integration set were empty; seven covering cases needed assisted masks. The current guided editor uses manual covering selection. Known R1 issues include invented eyewear/expression, asymmetry, grayscale false color, and scarf/hair/hand artifacts. Useful output across every covering family, independent blinded human FINAL review, representative CCTV/Zamboanga evidence, and minimum-device certification remain incomplete. The full research goal remains active. Hidden regions are plausible estimates.

## Continue in another workspace

For application/UI work, create a fresh standalone checkout in the intended destination directory:

```powershell
git clone --depth 1 --single-branch --branch IMPLEMENTATION https://github.com/janusinss/forensic-dgp.git
Set-Location forensic-dgp
.\install.bat
.\start.bat
```

The Windows installer configures the CPU environment and verifies model sizes/hashes. Inference is local; installation downloads the assets. Read that checkout's `README.MD`, `MODEL_CARD.md`, and `COMPLETION_MODEL_CARD.md` first. Verify its HEAD because the remote can advance beyond this snapshot.

For completion research in another chat on this computer, open `C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION`. Preserve the dirty source work, ignored `outputs/`, checkpoints, raw arrays/PNGs/masks, review sheets, protocols, failure evidence, original split membership and attribution. Do not switch a branch used by another chat or merge the application-only branch wholesale into research. Recreate an environment for a different OS rather than copying a Windows virtual environment.

Suggested continuation prompt:

> Continue FACE_MASK_COMPLETION research from FACE_MASK_COMPLETION_HANDOFF_2026-10-29.md. Inspect Git status and actual artifact paths first. Trained CodeFormer completion R1 remains selected. IMPLEMENTATION uses Conditioned SFT V2 p2_u0224 restoration plus manual reviewed-mask R1 completion. The spatial-code intervention V1 diagnostic is prepared but locally unrun; verify whether returned evidence exists. Preserve research/failure artifacts and exact outside-support pixels. Actual VM execution stays manual. Read the latest entries of PROJECT_HANDOFF.md, the completion spec, and spatial-code preparation/VM guides before deciding the next research action.

## Reading order and drift corrections

1. Read this dated handoff.
2. Read the first 53 lines of `PROJECT_HANDOFF.md` for the 28 October integration evidence.
3. Read `COMPLETION_CODEFORMER_SPATIAL_CODE_INTERVENTION_V1_PREPARATION.md` for the latest research preparation.
4. Read IMPLEMENTATION's current model cards for current UI/model behavior.
5. Inspect the relevant result/evidence directories before changing a model.

`PROJECT_HANDOFF.md` is newest-first and contains historical entries that contradict later decisions. Its 28 October published ref and guided automatic-proposal description have since advanced to the 29 October revision and manual selection. Older documents' statements that fresh installation or selective IMPLEMENTATION publication is still incomplete are superseded by the 28 October publication record. Some older paths still point to original Testing/outputs; the R1 research checkpoint, recovery state, and publication evidence were verified in FACE_MASK_COMPLETION/outputs during this audit.
