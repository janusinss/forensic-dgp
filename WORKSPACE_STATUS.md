# Ordinary worktree: RESTORATION

Both folders are ordinary Git worktrees, usable without Codex from an editor, terminal or another coding tool.
They share the original Testing repository Git metadata, so keep that repository in place.
Codex cannot attach this existing chat to an unmanaged folder through its attachment tool. Open the normal completion folder as your project before continuing model work.
The previous managed completion checkout is a preserved detached copy for the still-active session, not the canonical development folder.

# Branch workspace assignments

Source isolation setup completed on 25 October 2026. These are local branches
with separate worktrees; changes transfer only through intentional Git actions.

| Branch | Worktree folder |
|---|---|
| RESTORATION | C:\xampp\htdocs\YEAR 4\Testing_RESTORATION |
| FACE_MASK_COMPLETION | C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION |

This worktree is for RESTORATION. Use the matching folder for its development chat. Opening the original Testing folder in both chats still
shares files. A branch name alone does not provide physical separation.

All 3,536 original source files were checked byte for byte in both checkouts.
The new branches preserve exact bytes through .gitattributes and worktree-local
Git line-ending settings. Only handoff metadata and this status note were added
after the shared source snapshot. Existing model code and checkpoint bytes are
unchanged. The original main checkout and index remain preserved.

Ignored runtime data, model assets, archives and the virtual environment remain
in C:\xampp\htdocs\YEAR 4\Testing. They were not duplicated or automatically
connected. Verify and configure asset paths before inference; the new code
worktrees are not claims of ready-to-run app installations. Use a separate
environment for dependency changes and separate directories for new outputs.

Mask detection proposes the editable removal area. Completion generates only
inside the reviewed area. Restoration remains a distinct model. Both branches
initially retain the complete source tree; shared integration files require
review when changes are eventually merged. No checkpoint is promoted by this
Git setup. All actual training remains manual on the authorized L4 VM.

# Conditioned SFT Learning V1 Status (25 October 2026)

- **Feasibility Proof (`outputs/codeformer_conditioned_sft_learning_v1_feasibility/feasibility_receipt_v1.json`)**: `STAGE1_SAVED_ARRAY_FEASIBILITY_PROVEN` — `single_pass_input_anchored_B8_beta0p005` passes `56/56` inherited + `40/40` monitor gates at `u8` and `u24/u48` (`-48,923` RGB8 code improvement on clear monitors).
- **Sealed Manual L4 VM Packet (`outputs/cctv-codeformer-conditioned-sft-learning-v1-execution.tar.gz`)**: `107,283,842` bytes (`690` PAX members, SHA256 `4e3d07432ea1951a60050bb5aba5dfbbf30f6ad7b2cdc5e54232f0fa62b7af64`), Protocol SHA256 `f54c37e8b6b05473b1530395549b31f89189445f642ecfc49b1280ba8442d7f0` (`691` static files).
- **Returned L4 GPU Audit (`outputs/codeformer_conditioned_sft_learning_v1_return_audit_v1/`)**: `cctv-codeformer-conditioned-sft-learning-v1-results.tar.gz` (`2,360,962,311` bytes, `3,825` members, `712` full forwards, `18` checkpoints, `16` Adam commits) verified in [CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V1_RETURN_REVIEW.md](CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V1_RETURN_REVIEW.md). Arm `B` solved the `clear` PNG regression (`-0.000273`) and delivered `5.3x`–`15.9x` larger degraded-role MAE gains; coarse `g_dark` exposure drift sign alignment passes `96/96` gates (`inh8`, `inh48`, `sup8`, `sup48`).
- **Active Goal V3 (`Conditioned SFT V2` — All 5 Stages Complete, Live NVIDIA L4 GPU Return Audited & Verified)**:
  - **Stage 1 Headroom Proof**: [CCTV_CODEFORMER_CONDITIONED_SFT_FEASIBILITY_V2_REVIEW.md](CCTV_CODEFORMER_CONDITIONED_SFT_FEASIBILITY_V2_REVIEW.md) + `outputs/codeformer_conditioned_sft_learning_v2_feasibility/v2_feasibility_receipt_v1.json` (`+0.073` to `+0.141` `SSIM7` headroom via Exposure-Matched SFT).
  - **Stage 3 Sealed Manual L4 VM Packet**: [CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V2_VM.md](CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V2_VM.md) + `outputs/cctv-codeformer-conditioned-sft-learning-v2-execution.tar.gz` (`124,224,129` bytes, `849` PAX members, SHA256 `f373cf3b1eb217a409381ff133c47061f4bf06a4db0bb99b34bc78af3eb01d1d`; Protocol SHA256 `84e4976079dd0d29ccb7408254a500298160b08479ce739163ed7dc47d90e5a0`; zero-CPU-cache on-the-fly batched GPU extraction; `INDEPENDENT_V2_PACKET_AUDIT_PASSED`).
  - **Stage 4 Live NVIDIA L4 Return Audit (`outputs/codeformer_conditioned_sft_learning_v2_return_audit_v1/`)**: [CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V2_RETURN_REVIEW.md](CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V2_RETURN_REVIEW.md) + `return_audit_v2.json` (`cctv-codeformer-conditioned-sft-learning-v2-results.tar.gz`, `158,864,004` bytes, `1,020` PAX members, SHA256 `8b9632c4f85b46deeb061e63cdefb4318aef48da06adbf07dead81e77b006c22`, `2731.7s`). Winning checkpoint `p2_u0224` passes `56/56` forensic retention checks (`inherited_56_pass: true`), cuts 4-role degraded `MAE` by `-37.4%` (`0.04452 -> 0.02785`), raises `SSIM7` by `+0.02062` (`0.83276 -> 0.85338`), raises paired degraded `Central Laplacian Sharpness` (`0.01897 -> 0.01985`), and boosts native CCTV (`ChokePoint` & `QMUL-SurvFace`) sharpness by `+22.1%` to `+47.1%`.
  - **Stage 5 Interactive Browser Workbench (`serve_conditioned_sft_review_v2.py`)**: Verified via inline Playwright (`375px`, `768px`, `1280px`) with `p2_u0224` auto-loaded, Stage 4 Return Audit tab, and live upload/preset comparison.
