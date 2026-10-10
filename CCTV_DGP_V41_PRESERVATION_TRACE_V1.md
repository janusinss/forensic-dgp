# V41 preservation failure: coverage and finite-proposal review

The failed FFHQ compound-degradation group contains 391 paired photographic
TRAIN cases. An independent saved-output trace verifies all 1,173 embedding
arrays, 782 delivered PNGs, 391 support masks and 20 exact comparison cells.
Its 2,373 source bindings match the previously audited return and packet.
This changes what the next diagnostic must measure; it does not qualify V41.

| Optimization exposure by update50 | Cases | Mean ArcFace score change | Cases with a score decrease |
| --- | ---: | ---: | ---: |
| Whole failed group | 391 | -0.0000274666 | 189 |
| Already exposed | 27 | +0.0001794937 | 10 |
| Not yet exposed | 364 | -0.0000428180 | 179 |

The 364 unexposed cases contribute a summed change of -0.0155857676; the
27 exposed cases contribute +0.0048463289. Their combined average reproduces
the original group failure exactly. This is an attribution of the recorded
score change, not a unique explanation of its neural cause. Positive average
performance on the currently optimized references did not preserve this group.
The unchanged group allowance remains 1e-6; individual decreases are descriptive
counts, not a replacement acceptance rule.

The training source computes ArcFace regression on the current five-case batch,
using the unquantized prediction and retained baseline in one recognizer call.
The delivered gate uses saved PNG embeddings over the full 3,905-case TRAIN set.
These are different coverage and representation scopes. At50 updates, only
250 cases have been optimized. The V41 applied-step analysis also found nine
actual AdamW steps with a positive landmark directional derivative despite
the incoming PCGrad direction being nonascending. A projection of those seven
current-batch gradients therefore needs finite checks beyond that batch.
Neither observation proves that one optimizer or quantization mechanism alone
caused the failure.

The delivered PNG changes affect 38.6715% of observed channel entries in this
group; the largest absolute channel change is12 byte levels. This cannot be
described as an exclusively sub-byte difference. Only five of the391 cases
retain raw float stages. None of the five largest score decreases has a saved
raw stage, so this trace cannot separate raw from PNG loss on those cases.
The original raw-retention limit and failure remain; no missing array is
reconstructed or presented as recorded VM evidence.

The five largest score decreases are selected after the outcome solely for
TRAIN failure diagnosis, with ties resolved by case id. All20 unscaled cells
are actually viewed: input, retained DGP, stopped V41 and paired clean target.
The stopped outputs remain soft and lack convincing added eye, nose and mouth
definition. The ranking does not establish exact identity changes, a new
evaluation set or useful native CCTV performance. All original input-only
decisions and splits remain unchanged.

## Minimum requirements for the pending design discussion

The recommended next diagnostic should replay finite parameter proposals
before approving another learning trajectory. These are review requirements,
not a prepared VM packet or permission to run a historical recipe.

1. Use the recorded V41 before-state and gradients at updates10,12,22,27,37,
   38,39,42,44,45. These are explicitly selected TRAIN failure probes: the
   original projected-direction ascent plus the nine applied-step mismatches.
   Preserve every original parameter endpoint, moment, checkpoint and failure.
2. Compare exactly three transient proposals: zero displacement, recorded V41
   displacement and the already reviewed fixed saved-array cone displacement.
   Do not construct a new optimizer, commit a trajectory, tune a damping sweep
   or store a new trained checkpoint in this diagnostic.
3. Check each current five-case batch and both unchanged preselected50-case
   TRAIN cohorts. This bounds the review at10 states ×3 proposals ×105 slots
   =3,150 forward slots before any later full-corpus verification. Retain raw
   floats and exact delivered PNGs for every slot. The post-outcome worst-five
   sheet must not replace the preselected cohorts.
4. Report finite raw component losses, delivered structure and preservation
   separately. Existing 1%-at50/10%-at800, all17 full-corpus preservation groups,
   source gains and mean-only limits remain mandatory for any later pilot.
   A small diagnostic passing would not establish those full-corpus gates.
5. Freeze measured timing, forward/export/storage bounds and stop rules before
   making a runnable packet. Execution remains manual on the existing NVIDIA
   L4/g2-standard-4 at ~/forensic-dgp, followed by an independent returned-output
   audit and visual review. No local gradients or training are authorized.

The circuit-breaker design question remains pending. The invalid assumption is
that projecting incoming gradients protects delivered features through an
adaptive finite update and on unexposed references. AGENTS.md requires stopping
code modifications after three unsuccessful attempts and asking one diagnostic
question; no new training or app code is modified here. The separate inference
probe proposed above also remains unimplemented until that discussion.

The first trace recorder stopped before reading face outputs because the packet
protocol was sought in the newer return-only milestone. Its source, log and
failure receipt remain. A distinct R1 resolves those bindings from the exact
prior prepared-packet milestone; inverse-source checking verifies that the
trace calculations and all thresholds are unchanged. Its independent audit
passes in6.751seconds, with no neural, gradient, optimizer or VM calls.

Original checkpoints, research caches/local backup, splits, provenance, failures
and the DGP-led app remain preserved. This is paired synthetic TRAIN evidence;
public native CCTV remains unpaired, reserved final identities stay unopened,
source names do not infer ethnicity, and no Zamboanga performance is established.
Useful native restoration, all seven automatic/assisted covering families,
independent final review and qualified full app flow remain outstanding.

[Independent trace audit](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v41_preservation_trace_v1/independent_trace_audit.json>)
[Exact diagnostic comparison](<C:/xampp/htdocs/YEAR 4/Testing/outputs/cctv_dgp_v41_preservation_trace_v1/failed_group_worst5.png>)
[Prior optimizer analysis and primary research](<C:/xampp/htdocs/YEAR 4/Testing/CCTV_DGP_POST_V41_OPTIMIZER_REVIEW.md>)
