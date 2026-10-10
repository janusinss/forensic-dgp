# Original-feature diagnostic return: verified, no model qualified

9 October 2026. The manually returned diagnostic finishes in214.59 seconds
with30 derivative queries, nine reset trials, zero optimizer updates and zero
epochs. All nine trials fail the unchanged preservation requirements. The
current application checkpoint remains646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b.
The full restoration and seven-covering-family goal remains active/incomplete.

The strongest decoder-only trial exceeds the1% structure requirement in both
small TRAIN cohorts: raw1.32597%/1.17942%, delivered PNG1.34283%/1.17117%.
However, its raw/PNG preservation failure counts are23/23 and18/18. This
demonstrates a measurable structure direction in the original reconstruction
path; it does not solve V42 or qualify useful CCTV restoration. V42 measured
the full3905-case training corpus with an added decoder, whereas these results
cover two50-case photographic TRAIN cohorts and reset parameter displacements.
The percentages are not a same-corpus before/after comparison.

| Trial | Gradient cohort raw / PNG gain (%) | Cross-cohort raw / PNG gain (%) | Preservation failures, gradient raw / PNG; cross raw / PNG |
|---|---:|---:|---:|
| decoder,1e-5 |0.01414 /0.02850 |0.01331 /0.01097 |15 /7;9 /14 |
| decoder,1e-4 |0.13965 /0.14787 |0.12805 /0.15065 |16 /14;9 /11 |
| decoder,1e-3 |1.32597 /1.34283 |1.17942 /1.17117 |23 /23;18 /18 |
| feature,1e-5 |0.11507 /0.12947 |0.11824 /0.12136 |25 /24;38 /37 |
| feature,1e-4 |-0.23944 /-0.22868 |-0.55442 /-0.56330 |49 /49;50 /50 |
| feature,1e-3 |-2.84079 /-2.82342 |-3.97647 /-3.96461 |52 /52;52 /52 |
| joint,1e-5 |0.12989 /0.14276 |0.13093 /0.12566 |24 /24;37 /36 |
| joint,1e-4 |-0.21133 /-0.19328 |-0.57500 /-0.55513 |49 /49;51 /51 |
| joint,1e-3 |-2.80958 /-2.79209 |-3.96076 /-3.95577 |52 /52;52 /52 |

Exact source/profile failures and all36 decisions are retained in
outputs/cctv_dgp_original_feature_probe_v1_return_review_v1/partition_analysis.json.
Counts refer to metric/group failure records, not numbers of incorrectly
identified people. The fixed recognition metric is a preservation proxy, not
identity accuracy or proof of recovered identity.

All158 selected reachable original tensors have connected, nonzero derivatives
for each of the three losses in all ten batches. The two inactive head4 kernels
produce zero output and remain frozen, as do the unused backbone tail and all
normalization buffers. The initial, original, recognizer and restored candidate
states match the pinned states. Each trial resets to the same current checkpoint.
No resumable trained model or optimizer state was created.

Saved-vector arithmetic reveals different interference across partitions:

| Partition | Structure gradient L2 | RGB gradient L2 | Identity gradient L2 | Structure/RGB cosine | Structure/identity cosine |
|---|---:|---:|---:|---:|---:|
| Original decoder |0.00071359 |0.01996561 |0.53084685 |+0.39790 |-0.10041 |
| Original feature/FPN |0.00254905 |0.06280976 |1.80048474 |-0.11426 |+0.59890 |
| Joint |0.00264704 |0.06590669 |1.87711046 |-0.07236 |+0.54553 |

At the initial point, decoder structure descent predicts mean identity loss
regression. Feature structure descent predicts mean RGB regression, despite a
connected feature path. Larger pure feature/joint steps then worsen finite
structure and appearance. This rejects the assumption that merely increasing
trainable layers or repeating the same structure direction is sufficient.

A different direction is justified for a finite diagnostic: feature identity
descent has positive alignment with both feature structure and RGB gradients.
Combining it with decoder structure descent can neutralize the adverse mean
identity derivative. This is a first-order hypothesis; finite images and every
source/profile requirement still need testing. Prior V33 decoder-only cone
trials and V41 added-decoder PCGrad remain failed and are not repeated.
Research on gradient magnitudes and interference supports measuring these
effects, without guaranteeing preservation for this model:
[GradNorm, ICML2018](https://proceedings.mlr.press/v80/chen18a.html) and
[Gradient Surgery, NeurIPS2020](https://arxiv.org/abs/2001.06782).

All60 exact256-pixel sheets and1000 unique model outputs were visually reviewed
by the primary assistant as a development review. An independent pixel checker
verified1800 gallery cells against their saved sources. Decoder trials remain
close to the soft baseline, without a convincing useful clarity gain. Strong
feature/joint trials wash out clear facial detail, glasses, hair and boundaries
and introduce colored edge artifacts. No visually poor trial is adopted.
This is not an independent final reviewer or a native CCTV quality review.
Per-reference observations are in the later visual_review.json; creation-time
pending flags in the immutable run/audit receipts are preserved.

The downloaded1051691000-byte archive matches SHA256
dbba13dac96fe0bcdf58c03083f6bdfc1e939383b3996b6039ba5c8e93b0f290,
its sidecar and export receipt. The independent R2 audit verifies4042 regular
members, all1000 raw/PNG/mean-only metrics, all30 saved gradient vectors and
nine displacement formulas, exact stored-row groups and all36 failure decisions.
Fresh CPU inference/recognizer replay covers200 outputs. Maximum errors are
2.6747584e-6 raw, one PNG byte and3.4458935e-7 embedding, within the prospective
arithmetic replay tolerances. Worker355.18s, supervised357.61s. No derivatives
or optimizer updates ran locally.

The original R1 audit failed on derived brightness-ratio arithmetic. Pixel MSE
recomputation differs by at most4.16334e-17; a denominator clamped at1e-12
amplifies this to1.0408341e-5 in an already-failing derived ratio. R1's source,
traceback and import are retained. Separate R2 uses an explicit propagation
bound only for this ratio and verifies exact scientific decisions. All other
comparisons and all scientific thresholds remain unchanged. Evidence audit
success does not make any trial pass quality.

Peak allocated VM VRAM is1.464GiB. Export59.43s. Source labels retain their
audited photographic provenance and overlap limits; they imply no ethnicity.
Neither cohort is newly unexposed/final. No native CCTV, DEV/final pixels,
completion generation, new actual training or app selection enters this
diagnostic. Useful native development review, separate final review, full app
flow and every automatic/assisted covering family remain required.

Evidence: outputs/cctv_dgp_original_feature_probe_v1_vm_return/;
outputs/cctv_dgp_original_feature_probe_v1_independent_audit_r2.json;
outputs/cctv_dgp_original_feature_probe_v1_return_review_v1/.
