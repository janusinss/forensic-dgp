# Branch workspace assignments

Source isolation setup completed on 25 October 2026. These are local branches
with separate worktrees; changes transfer only through intentional Git actions.

| Branch | Worktree folder |
|---|---|
| RESTORATION | C:\Users\janus\.codex\worktrees\restoration\Testing |
| FACE_MASK_COMPLETION | C:\Users\janus\.codex\worktrees\face-mask-completion\Testing |

This chat continues in FACE_MASK_COMPLETION. Use the RESTORATION folder in the
restoration chat. Opening the original Testing folder in both chats still
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
