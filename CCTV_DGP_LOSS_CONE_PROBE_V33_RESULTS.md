# V33 finite output probe: verified result and next diagnostic

The human-run V33 probe completed, but it does not qualify a training recipe
or an app checkpoint. All original-state constrained proposals fail at least
one retained preservation check. Stopped-state proposals improve detail in
small TRAIN subsets, but the unexposed matched subset still fails identity
preservation. Do not resume the stopped V32 checkpoint or repeat V33 unchanged.

The 1,166,123,827-byte return matches SHA256
20a196d6515bc46d2e53c74d1ae4182c28f39812649e32bf44a2c684b56b7c40.
The independent R2 audit verifies 5,914 files, all 1,400 raw/PNG measurements,
the fresh AdamW moments/displacements, independent primal projections and
280 frozen CPU replay outputs. Maximum raw replay error is 0.000002414,
maximum PNG difference one byte, vector error 0.000001349 and loss-component
error 0.000003875, all within the original pinned tolerances. The CPU audit
took 569.79 seconds; its supervisor took 571.18 seconds. Local optimizer,
gradient and backward calls are zero. Original checkpoints and app bindings
remain intact.

The first checker stopped on exact group-dictionary float equality. Independent
readback found differences at about 0.00000000000000017. Its source and failure
are retained. R2 requires exact aggregation from the saved case rows, uses the
same pre-existing case tolerance for fresh group arithmetic, and independently
rechecks identical group failure keys, source-gain signs and brightness decisions.
No model recipe, output threshold or replay tolerance was relaxed.

The VM performed eight reset first-step AdamW proposals in 192.75 seconds,
with zero committed trajectory updates, new gradient queries, epochs or new
checkpoints. Export took 78.06 seconds. Its complete flag describes this finite
probe and packaging; it is not an 800-update training result.

| State and TRAIN subset | Cone scale 1 detail gain against that state | Failed group/metric checks against original |
| --- | ---: | ---: |
| Original, exposed | 0.34505% | 15 |
| Original, unexposed | 0.53813% | 15 |
| Stopped 50, exposed | 0.37288% | 0 |
| Stopped 50, unexposed | 0.44015% | 2 |

The stopped exposed subset reaches 1.25714% gain against original after that
single probe; the stopped unexposed subset reaches 1.42734%. These subset values
cannot replace the full 3,905-case requirement at exactly 50 updates. V32's
0.970672% full-corpus failed early gate remains binding.

Before V33, the stopped unexposed subset already failed ArcFace preservation
on dataset/asian_faces/blur_lr24 and dataset/thumbnails128x128/compound_lr24.
Cone scale 1 repairs the blur-group failure but introduces a clear-group failure:
dataset/asian_faces/clear cosine declines from 0.977832592 to 0.976057148.
The compound-group failure remains, 0.089385797 to 0.083053163. At scale 1/8,
the two earlier group failures remain. A passing exposed average therefore
cannot establish preservation across sources and degradation profiles.

At the original state, the clear-anchor and three preservation-hinge gradients
are exactly zero, as already proved in the V32 gradient return. Thus cone scale 1
equals the unconstrained restoration proposal there: it has no active preservation
direction to constrain. At stopped 50, first-order projected loss derivatives
can satisfy the linear constraints while actual finite hinge values increase.
This directly rejects treating a gradient projection as a finite-output guarantee.
It does not uniquely explain every earlier architecture failure.

Gradient Episodic Memory explicitly relies on local linearity and representative
memory for its gradient-angle approximation. That explains the limit of using
a projected derivative alone, rather than proving our face outputs are safe.
[GEM, NeurIPS 2017](https://proceedings.neurips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf).

Recent epsilon-constraint work formulates stability as a constraint and changes
the update direction when a memory-loss bound is violated. Its continual-learning
results are not CCTV evidence; its tolerance strategy is not permission to relax
our preservation limits. It motivates measuring the actual protected functions
before choosing a new optimizer design.
[GEC, NeurIPS 2025](https://proceedings.nips.cc/paper_files/paper/2025/hash/b3c2854d9e94282a373d8fa58b567b27-Abstract-Conference.html).

Four metadata-selected pages were actually reviewed at original 256px cell
resolution: first reference per source, all five profiles, both matched TRAIN
cohorts, 20 cases and 160 input/target/comparison cells. Clear DGP outputs soften
visible detail; blur, low light and compound rows do not gain convincingly useful
eyes, nose, mouth or other visible structure. This review covers those 20 cases,
not all 1,400 outputs or an independent final assessment. All 1,400 measurements
were audited separately. Clear glasses remain visible rather than being removed.

The justified next measurement is V34: original-state, non-hinged gradients of
raw MSE, one minus raw SSIM, and one minus fixed raw ArcFace for each individual
case in the same two 50-case TRAIN subsets. The original batch context and 23
selected own-DGP tensors remain. All 17 group derivatives can then be assembled
independently from saved case gradients, exposing cancellation that a single
average can hide. This changes diagnostic measurement, not the training losses
or qualification thresholds. There are exactly 300 gradient queries, zero
optimizer/parameter updates, no stopped-state continuation and no new checkpoint.
It must run manually on the existing L4 VM. Only audited returned directions
can justify a separate finite-output probe; no follow-on training is automatic.

These are paired synthetic degradations of photographic TRAIN references.
Unexposed means not optimized in V32's first 50 updates; it is not DEV, held-out
evaluation or final evidence. Source labels do not establish ethnicity, CCTV
capture performance or Zamboanga performance. No new native or reserved-final
images were opened. Public native CCTV development and its separate labeled
final identities remain frozen. Native/unpaired and synthetic/paired claims stay
separate. All seven covering families, distinct automatic/assisted usefulness,
independent final review and useful native DGP restoration remain outstanding.

[Independent return audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_loss_cone_probe_v33_independent_audit_r2.json>)
[All measured comparisons](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_loss_cone_probe_v33_analysis_v1/analysis.json>)
[Actual development visual review](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_loss_cone_probe_v33_analysis_v1/visual_review.json>)
