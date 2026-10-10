# V41: audited failure and applied-optimizer-step design review

V41 stopped correctly at50 of800 updates. Its independently recomputed delivered
TRAIN structure gain is **0.0516710943%**, below the unchanged **1%** requirement.
The FFHQ compound-degradation group also fails ArcFace preservation: mean
0.0995538561 becomes0.0995263895, a drop0.0000274666 exceeding1e-6.
V41 is retained as a failed treatment. Do not resume it, rerun unchanged, weaken
the gates or adopt it into the app.

The human-returned archive is1,178,179,150bytes, SHA256
01a6e9198e731d81895c0b64cde21f45ed7eb01be2bb48b4c78a9186726c94dc.
The full independent audit passes in919.003seconds. It verifies27,551 exact
members, all50 saved parameter/moment chains and350 component-gradient queries,
7,810 delivered PNGs plus7,810 mean-only PNGs, saved embeddings and100 frozen
CPU preview replays. CPU raw replay differs by at most2.3246e-6 and PNG replay by
at most one byte, inside the prospective allowances. Parameter chain/snapshot
tensors remain exact; independent AdamW arithmetic error is at most9.1565e-8
against the predeclared3e-7 allowance. Saved gradients are not independently
differentiated locally. Original DGP, reference decoder and recognizer stay frozen.

The VM run takes1,042.201seconds, about17.37minutes; export takes62.274seconds.
It has50 optimizer updates,350 gradient queries and no800-update result. Successful
export means the evidence was packaged. It does not establish useful restoration.

| Paired photographic TRAIN subset | Degraded cases | Structure gain |
| --- | ---: | ---: |
| All TRAIN | 3,124 | 0.051671% |
| Exposed by50 updates | 200 | 0.052245% |
| Not yet exposed | 2,924 | 0.051623% |
| Fixed50 previews | 40 | 0.100762% |
| Previously selected optimized50 previews | 40 | 0.052653% |

Both source gains are positive: Asian-dataset label0.102795%, FFHQ label0.039605%.
Mean-only brightness explains0.606389% of the pixel-MSE gain, below the20% cap.
Those successes do not override the structure and preservation failures.
Source labels do not establish ethnicity or Zamboanga performance. These are
synthetic paired photographs, not native CCTV or independent evaluation.

The independently audited saved-array analysis compares all3,905 initial PNGs
with V40: pixel parity is exact. All100 metadata-selected TRAIN crops and500
unprocessed256-pixel comparison cells are actually viewed on20 sheets. Eye,
nose, mouth, outline and visible appearance remain in scope, including clear
glasses, ordinary hair and visible objects. V41 stays close to the retained DGP
and V40; convincing useful added central-feature definition is not established.
This implementing-assistant inspection is not an independent final review.

The per-update evidence reveals a limitation of projecting gradients before
AdamW. At update10, PCGrad's negative direction itself has a positive landmark
directional derivative. At nine other updates—12,22,27,37,38,39,42,44,45—the
projected direction is nonascending, but the actual parameter displacement has
a positive landmark derivative. Removing weight-decay displacement preserves
all nine findings. Observed-detail loss has nine analogous mismatches. These
are first-order observations, not proof of finite loss increases or a unique
cause of the failed face output. Loss values from different reference batches
must not be treated as a repeated-case loss curve.

AdamW combines momentum with coordinate-dependent variance scaling, so its
parameter step differs from the negative incoming gradient.
[PyTorch's algorithm](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html).
GEM motivates a closest-direction projection subject to nonincrease constraints;
its evidence concerns continual-learning tasks, not CCTV faces.
[Original GEM paper, Section3](https://arxiv.org/pdf/1706.08840).
Our inference is that the next diagnostic should evaluate **actual parameter
proposals**, rather than assuming gradient surgery protects delivered features.
This is a proposed adaptation, not a GEM reproduction or established remedy.

One fixed saved-array convex calculation projects each actual V41 displacement
onto all seven local nonincrease halfspaces. The dual is solved using
[SciPy NNLS](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html)
with a fixed21-iteration maximum, no sweep. Independent KKT checks pass on all50
saved arrays; projected norms retain98.1893%–100% of the original step, mean
99.7269%. These coefficients are arithmetic diagnostics only. No proposed
parameters are assigned to a neural network or new checkpoint. Finite model
losses, delivered-image preservation and improved structure remain untested.
Inactive hinges, curvature and shared-parameter effects still prevent a finite
preservation guarantee.

The prepared exporter retained V40's archive prefix; that packaging error was
missed in our V41 packet verification. The original auditor rejects it before
extraction. Its source, log and receipt remain. A separate R1 auditor accepts
only that exact immutable export prefix and imports into the separate V41
folder. It passes11 archive-boundary regressions. Inverse-source verification
confirms that only the prefix handling and receipt filename change; training,
image, step-evidence, safety limits and qualification gates remain unchanged.

V38,V40,V41 now trigger the workspace circuit breaker. The invalid assumption
is that projecting the incoming gradients suffices to protect the delivered
face through an adaptive finite update. The required diagnostic question is
pending: test actual-step preservation before more training, or review the
spatial reconstruction/paired-target path first. No new training recipe is
modified or prepared while that review is pending. A prospective next diagnostic
must be finite, preserve all recorded failures and use the manual VM workflow.

All14 DGP-primary app bindings remain exact. Original/stopped checkpoints,
research caches/local backup, splits, provenance and failure records remain.
Native CCTV stays unpaired, synthetic PSNR/SSIM separate, reserved45 identities/
58 crops unopened. No Zamboanga samples or hidden-identity claim exist. Useful
native restoration, all seven automatic/assisted covering families, independent
final review and qualified app flow remain outstanding. The overall goal is incomplete.

[Independent return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_pcgrad_fit_v41_independent_audit_r1.json>)
[Independent saved-array and sheet audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_pcgrad_fit_v41_analysis/independent_analysis_audit.json>)
[All100 visual observations](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_pcgrad_fit_v41_analysis/visual_review.json>)
