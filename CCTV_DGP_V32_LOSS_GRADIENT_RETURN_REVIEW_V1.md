# Audited V32 loss diagnostic and finite V33 design — 7 October 2026

The downloaded diagnostic succeeded. Independent audit verifies756 returned
files, all301,298,844 saved gradient-array values and200 frozen CPU restoration
replays. Maximum raw output difference is0.000002563, PNG difference is one byte,
and recognizer-vector difference is0.0000005253, within the frozen tolerances.
The VM diagnostic took45.58seconds, made280 gradient queries and zero optimizer
updates, and created no checkpoint. Local audit took282.12seconds with no local
gradients, backwards or optimizer calls. Export completion is packaging evidence.

The721,569,318-byte archive SHA256 is
529acd5750c64e7320260c09f55cdd3a115c2f465afd02abc4101009155143e3.
The original V32 r2 failure remains:50/800 updates,0.970672% delivered structure
gain against the unchanged1% requirement. No800-update result or resume exists.

Two50-case TRAIN cohorts are exactly the previously frozen source/profile sets.
Exposed references were optimized in the first50 V32 batches. The other cohort
contains the fixed preflight/normalization previews, unoptimized by50. Each uses
five references from each photographic source and all five profiles. These are
paired synthetic photographic TRAIN observations. They are not unseen evaluation,
native CCTV or Zamboanga evidence; source labels do not identify ethnicity.

| Stopped50 cohort | Preservation/restoration gradient norm ratio | Gradient cosine | Facial-detail derivative along negative seven-loss sum |
|---|---:|---:|---:|
| Exposed |7.5746|−0.24866|+0.00160696|
| Unoptimized fixed previews |4.2689|−0.25694|+0.05886022|

Positive derivatives mean that the **ordinary raw summed-gradient direction**
locally increases this facial-detail term at the stopped state. This is a measured
gradient conflict, not a reconstruction of unsaved AdamW moments, a curvature
measurement, or proof of a unique failure cause. Preservation gradients protect
actual regressions; the earlier audit found raw identity penalties and a clear
control with3.802615% higher raw pixel error. Removing protection is unjustified.

On paired five-case batches,3/10 exposed and4/10 unexposed stopped summed directions
increase facial-detail loss. Five and four respectively increase at least one
restoration term. The aggregate observation therefore does not imply that every
batch behaves alike or that the proposed correction will generalize.

Two mathematical approaches were retained separately. Projecting the complete
seven-loss sum can put a restoration derivative at zero, including facial detail
in the unexposed aggregate. Projecting the first three unchanged restoration
gradients while constraining all seven nonzero gradients yields descent in all
three restoration terms in both stopped aggregates. Their facial-detail
derivatives become−0.121224 and−0.247386. Protection dot products satisfy the
declared numerical bounds. No model parameters were assigned by these analyses.

All44 saved matrices and both proposals passed88 independent unrestricted-primal
SLSQP checks. Maximum direction discrepancy is0.00000009879. Five analytic
fixtures, three invalid-input checks and24 random primal checks also pass.
This establishes arithmetic consistency; finite face improvement is unproven.

Primary research supports investigation of conflicting optimization directions.
[PCGrad](https://proceedings.neurips.cc/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)
examines conflict, magnitude disparity and curvature. Our data measure the first
two at saved endpoints; they do not establish the paper's curvature conditions
or transfer its reported performance. Its pairwise surgery does not guarantee
all seven finite appearance constraints.

[MGDA](https://proceedings.neurips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
seeks common descent through minimum-norm combinations. Including initially
inactive zero-gradient protection terms can produce a zero minimum-norm proposal
here. [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) motivates measuring
gradient magnitudes; changing weights would require separate justification and
would not itself guarantee facial preservation. Neither method was installed.

[GEM](https://proceedings.neurips.cc/paper/2017/hash/f87522788a2be2d171666752f97ddebb-Abstract.html)
uses nearest-direction projection under gradient inequalities. Our own adaptation
constrains the seven existing same-output losses and projects an actual AdamW
displacement. This is an inference from its mathematical formulation, not a
continual-learning experiment or claim of transferred face-restoration quality.
The NumPy dual implementation is ours; no author code was copied or executed.
The source record identifies the primary paper sections and these limits.

The next manually executed experiment is **V33 finite displacement testing**.
It reuses the four audited aggregate gradient matrices, making zero new gradient
queries. Four state/cohort combinations each receive two fresh, disposable AdamW
proposal steps: the unchanged seven-loss sum and the first three restoration
terms. The original learning rate, clipping and decay stay fixed. Eight total
proposal steps are counted honestly; no continuing trajectory, epoch or checkpoint
is created. These are parameter changes and must run on the existing L4 VM.

The restoration AdamW displacement is projected **after** the adaptive step.
Four fixed scales1,1/2,1/4,1/8 are evaluated alongside both unprojected controls.
The restoration-only control is diagnostic and cannot qualify a training recipe.
Each trial starts from the same original or stopped copy; no result-dependent
scale search, seed change or historical continuation is allowed. All seven raw
loss values, per-case findings, raw outputs and delivered PNG metrics are saved.
Initially zero hinge gradients and float32 rounding can still permit finite
regressions. Actual outputs, rather than linear inequalities alone, decide them.

The probe emits1,200 trial outputs plus200 before outputs. The17 delivered
preservation groups use the existing MSE/SSIM/ArcFace tolerances against the
original. Source and brightness guards are reported. The1% requirement at50 and
10% at800 remain applicable to a later justified training pilot; a single finite
probe cannot qualify capacity or the app. Visible eyes, nose, mouth, outline and
overall appearance still require review, with clearer input requested when the
observed structure is insufficient.

The42,130-byte five-file packet passes414 dependency bindings, Python3.10 syntax,
pre-neural Windows update rejection, seven unsafe import boundaries, eight
independent displacement projections and an actual read-only Bash parse. The
NumPy loss reviewer reproduces the first six VM loss components on200 saved cases,
with maximum error0.00000016393. No new neural or optimizer work occurred in these
packet checks. A prospective return audit checks all1,400 output compositions and
metrics, eight actual AdamW moment/displacement proofs, four independent primal
projections and280 predeclared CPU neural replays.

Prepublication failures remain: the first preparation rejected a missing closing
list bracket before copying a worker or freezing a protocol. A raw-coordinate
independent SLSQP check later exceeded the original tiny discrepancy bound on one
fixture. The separate R1 reviewer uses orthonormal primal coordinates and tighter
solver convergence, passing the unchanged bound with maximum error2.783e−18 on
eight hypothetical displacement fixtures. A NumPy-Boolean serialization error in
its initial diagnostic wrapper is also retained. The final VM packet, protocol,
losses, cone implementation and output tolerances are unchanged by reviewer R1.

Require6GiB free and an idle L4/g2-standard-4. Worker900seconds, external930seconds
plus30second grace, export300seconds/external330seconds plus30second grace,20GiB
allocated VRAM and2GiB uncompressed return bounds are enforced. Estimated probe
time is3–10minutes plus1–4minutes export; the new VM probe has not been timed.
Failure evidence is exported without permitting a rerun or automatic follow-on.

The manual guide contains five exact upload/install/tmux/launch/download steps,
including one remote source per PuTTY download. Protocol SHA256:
ba8d1f87cae38cac8e1b3b893378a185c6126c25151dadbcfa0ebb373a51adee.
Archive SHA256:
282931a3b265c834a5736ac7f70c0816e08b74a15948d4c94645679a309eeace.
The separate R1 return reviewer supersedes the initial mathematical reviewer;
both versions are retained. No agent VM launch or upload occurred.

Native development and reserved final pixels remain unopened during this stage.
Original checkpoints, splits, research caches, gate failures, actual Windows
backup, own-DGP app checkpoint, automatic selection, override and design remain.
The converted MAT baseline remains unqualified. Useful native restoration,
automatic/assisted quality for all seven covering families and independent final
review remain outstanding. The goal is active and incomplete.

[Five manual V33 steps](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_LOSS_CONE_PROBE_V33_VM.md>)
[Independent returned-diagnostic audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v32_loss_gradient_v1_independent_audit.json>)
[Primary research record](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v32_restoration_cone_v1/primary_research.json>)
[V33 reviewer R1](<C:/xampp/htdocs/YEAR 4/Testing/scripts/audit_cctv_dgp_loss_cone_probe_v33_return_r1.py>)
