# Restoration and covering development branches

Decision: 25 October 2026. The user selected two branches: `RESTORATION` and
`FACE_MASK_COMPLETION`. Restoration continues in a separate chat. This chat's
next model-development focus is covering detection and completion. This later
decision supersedes the earlier instruction to defer all completion development
until restoration is satisfactory. It does not qualify any existing model.

## Physical separation

A separate chat alone does not isolate files. Both chats opening
`C:\xampp\htdocs\YEAR 4\Testing` edit the same files and see the same checked-out
branch. Use a separate Git worktree folder for each branch. Never switch the
branch of a folder another chat is actively editing.

Each worktree has its own source files and index. Git history and object storage
are shared. A source edit in one worktree does not appear in the other until an
intentional commit transfer or merge. Branches do not restrict which files Git
allows an editor to change; the ownership rules below still apply.

## Ownership

| Branch | Primary development ownership |
|---|---|
| `RESTORATION` | CodeFormer restoration controllers/adapters, U96/u64 research, restoration-specific launchers, templates, JS, tests and new restoration outputs. |
| `FACE_MASK_COMPLETION` | Covering proposals, reviewed removal masks, completion/inpainting, covering-specific training/data, editor behavior, tests and new covering outputs. |

Mask detection and completion stay together because they exchange the removal
area. They remain distinct model functions: the detector proposes where to
generate; the completion network estimates hidden pixels. Neither is a new
version of the restoration model.

## Shared integration boundaries

1. Treat `app.py`, `face_workflow.py`, `dgp_face_workflow_v3.py`, `models.py`,
   `third_party/`, dependency files and image geometry/normalization as integration
   boundaries. Their copies can diverge between branches; an eventual merge
   requires review and appropriate regression checks for both workflows.
2. Preserve existing trained checkpoints, source manifests, splits, provenance,
   failures and comparisons. Use new filenames/directories for candidates; do
   not overwrite original u64, U96/R2, detector or inpainting weights.
3. Do not share a writable `outputs/` directory between new worktrees. Keep new
   results in the producing worktree. Existing ignored datasets, weights and
   archives in the original Testing folder remain retained assets; Git branches
   and worktrees do not copy or back them up.
4. Resolve required external asset paths explicitly before running an app or
   experiment. Read-only reuse of verified assets is permitted. Do not use a
   whole-folder junction or shared writable environment as a shortcut to claim
   independence. Dependency upgrades need a branch-specific environment.
5. Model training remains manual on the authorized L4 VM, with finite stops,
   verified transfers and independently audited returns. Creating branches or
   worktrees does not run training, qualify a model, publish code or push Git.

## Current code map

- Restoration controller: `codeformer_restoration_workflow_v1.py`, delivery:
  `codeformer_app_delivery_v1.py`, web routes:
  `codeformer_restoration_web_v1.py`.
- Existing covering route: `face_workflow_web.py` →
  `dgp_face_workflow_v3.py` → `face_workflow.py`. This retained integration still
  references the older DGP restoration adapter; it is not proof that U96 is
  already integrated with completion.
- Detector baseline: `completion_inference.py` loads the segmenter in
  `completion.py`; default weights:
  `outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth`.
- Completion baseline: `pretrained_completion.py`; default inpainting weights:
  `checkpoints/codeformer_inpainting.pth`. The retained main workflow uses this
  separate pretrained component, not the restoration checkpoint.
- Both the detector and compact completion architecture also have definitions
  in `completion.py`. Changing that module can affect both covering functions.
  The inpainting adapter imports its mask/compositing helpers too.

Presence of these files was checked on 25 October 2026. No fresh inference or
quality acceptance follows from that inventory. The retained covering readiness
review reports automatic missed areas and generated anatomy/join defects; its
historical outcomes remain binding until a distinct comparison is reviewed.
