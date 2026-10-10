# V40 learning-signal review — 8 October 2026

The independently audited V40 return fails the retained 1% early structure
requirement after50 updates: 0.0106079234% delivered-PNG feature gain. All17
preservation groups pass. The fixed50 previews remain soft without convincing
new facial definition. All visible facial features remain required. The original
packet, initial/stopped weights, all failures and application bindings remain.

Saved-state arithmetic now independently verifies every one of17,952 values:
all57 tensors changed, and the stopped checkpoint exactly equals snapshot50.
The float64 weight-change norm is0.21028403907940274. On the same initial50
cohort, negative initial-component gradient alignment with that finite change
is0.0204007062 for landmark detail,0.0300057107 for observed detail and
0.0598935766 for pixel error. Initial preservation gradients are exactly zero.
These endpoint tangent projections establish movement, not useful learning or
a unique cause. Per-update component losses and optimizer moments were not saved.

AdamW uses per-step momentum/variance and separate weight decay. Therefore two
saved weight endpoints cannot reconstruct this optimizer's actual trajectory;
this is an inference from the documented update rule, not a measured cause of
V40's failure. [PyTorch2.9 AdamW](https://docs.pytorch.org/docs/2.9/generated/torch.optim.AdamW.html)

A distinct diagnostic compares the original initialization and stopped50 state
on two50-case TRAIN cohorts. One is the existing fixed preview set, unused in
the first50 updates. The other selects the firstfive optimized references from
each source in the frozen first50 paired batches. Each includes the same clear,
blur/low-resolution, low-light, motion and compound profiles. Selection uses
schedule and source/profile metadata only; no output is ranked or relabeled.
These are previously exposed photographic TRAIN cases, not held-out evaluation.
Source labels do not infer ethnicity, native CCTV or Zamboanga performance.

The architecture, original/fixed reference decoder and recognizer stay exact.
Only the permitted VM guard root and class name differ from V40. The same seven
component definitions, weights, fixed initial50 normalizers and all numerical
gates remain. Both endpoint states are immutable inputs; loading a saved state
is not an optimizer update. There are280 first-order component queries, zero
optimizer construction/steps, zero `.backward()` calls, no epochs, saved new
checkpoint, loss sweep, learning-rate sweep, resumption or automatic follow-on.

`torch.autograd.grad` returns selected derivatives without accumulating `.grad`;
`create_graph=False` avoids a higher-order derivative graph. The VM checks all
states and empty `.grad` fields after each batch. Those derivative queries run
only on the existing L4 VM, never locally.
[PyTorch2.9 autograd.grad](https://docs.pytorch.org/docs/2.9/generated/torch.autograd.grad.html)

The saved component norms, pairwise inner products, negative total-objective
directional derivatives,57 tensor partitions and endpoint-gradient projections
onto the actual50-update weight change distinguish measured improvement and
preservation signals. They do not predict finite optimizer steps or qualify
structure. A zero stopped-state component is recorded, not forcibly rejected
as disconnected; the older V39 connectivity requirement remains preserved.

The self-contained packet needs only the existing VM venv. It has a600s worker,
630s external limit plus30s kill grace,300s export/330s external plus30s grace,
20GiB allocated-VRAM limit,2GiB minimum free disk and512MiB uncompressed return.
Its fresh initial/stopped fixed50 raw replay must remain within the retained
1e-5 compatibility tolerance. Initial output/zero delta/preservation parity
stays exact; the historical normalization receipt stays exact. A failure is
packaged and retained. The agent neither connects nor launches this diagnostic.

The prospective independent checker verifies exact packet/return/source hashes,
metadata selection, all280 saved queries and every aggregate/partition/direction.
It audits all200 raw compositions,300 exact PNG compositions and40 metadata
selected frozen CPU endpoint replay cases with prospective raw1e-5/PNG1-byte/
vector1e-4/scalar5e-5 tolerances. These replay/readback tolerances do not alter
the1%-at50/10%-at800, source nonregression, mean-only or all17 preservation gates.
No returned Python is executed, and no local derivatives or training occur.

Only an audited diagnostic can justify a distinct future finite learning recipe.
It is not the next training pilot. Development qualification and useful native
structure still fail; reserved45 identities/58 crops remain unopened. Paired
synthetic PSNR/SSIM stay separate from unpaired native CCTV. Pretrained restorers
remain declared comparisons. All seven covering families, separate automatic
and assisted completion quality, clear-glasses/ordinary-hair preservation,
insufficient-information requests, independent final review and the qualified
DGP-led local flow remain outstanding. Original models, research caches/local
backup, splits, provenance and previous gates remain. Goal active/incomplete.

[V40 audited stop](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_SPATIAL_FIT_V40_RESULTS.md>)
[Saved-weight analysis](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v40_saved_learning_analysis/results.json>)
[Independent saved-weight check](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v40_saved_learning_analysis/independent_analysis_audit.json>)
[Five manual steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_V40_LEARNING_SIGNAL_V1_VM.md>)
