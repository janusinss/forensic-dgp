# Head4 capacity V1 return — 10 October 2026

The manual L4 pilot completed all 50 planned optimizer updates, then failed its
unchanged structure/preservation requirement. The downloaded archive and the
full independent audit pass; the experimental model's quality requirement fails.
Location: `scripts/cctv_dgp_head4_capacity_vm_v1.py:210` in the returned VM source.
Cause: insufficient measured structural gain, two small blur-group pixel-error
regressions, and excessive brightness-only contribution. The assertion preserves
the failed candidate and optimizer state and prevents continuation or promotion.

| Saved output | Structure gain | Required gain | Brightness share of pixel-error gain | Maximum share |
| --- | ---: | ---: | ---: | ---: |
| Raw float32 | 0.00608187% | 1% | 53.2186% | 20% |
| Delivered PNG | 0.00578294% | 1% | 54.6191% | 20% |

Structure gain is the relative reduction in landmark-region high-frequency
error on the degraded paired TRAIN cases. It is not identity accuracy, native
CCTV quality or a percentage of recovered hidden information. Brightness share
is the fixed counterfactual test: how much of the measured pixel-error reduction
is explained by shifting the baseline's mean RGB brightness alone.

Both raw and PNG output increase MSE slightly in the `blur_lr24` groups of the
`dataset/asian_faces` TRAIN source and FFHQ-derived TRAIN source. Raw MSE
changes from 0.005974067877 to 0.005974858028 and from 0.009668438246 to 0.009668655779.
These are small numerical regressions; they do not establish conspicuous visual
damage. The existing group-preservation requirement still rejects them. Dataset
directory labels do not establish ethnicity, capture geography or Zamboanga CCTV
performance.

All three repaired trainable pieces receive positive finite gradients in the six
preflight queries and actually change during fitting. Reconnecting a previously
inactive path was successful; useful finite-step image learning was not proved.
Only three tensors/147,456 elements were trained; all other original weights and
stored normalization remained frozen. The pilot uses 50 updates/781 batches, about
6.402% of one epoch, with zero complete additional epochs. It does not determine
whether longer training, a different objective, learning rate or a different
spatial path would resolve the quality failures. More epochs alone are therefore
not justified by this returned evidence, and this failed state must not resume.

The run took 21.03 minutes and reached its planned evaluation.
It did not stop from storage, CUDA, an external deadline or a transfer problem.
The export's `complete:true` means the archive was written successfully.

The 3,586,556,555-byte archive has SHA256
`da008028d730db86b7d5dfe92a718474bfee1fdb45e63fb48c5782757800fd3d`. The unchanged independent checker verifies all archive members,
all 7,810 saved baseline/candidate raw-and-PNG case records across two 3905-case
snapshots, exact gate decisions, six gradient-query receipts, all 50 step receipts, frozen
state, full optimizer/scheduler/RNG state and two fresh CPU inference replays.
Local auditing used zero gradient queries or optimizer updates. Both planned
diagnostic visual sheets were inspected, comparing all five profiles of one
reference from each TRAIN source (10 comparisons). These are the largest blur
MSE increases within the saved previews, not a representative population sample.
Baseline and update50 look almost identical; heavily degraded eyes and mouths
remain poorly defined. There is no convincing additional whole-face clarity in
this small diagnostic set. Comprehensive preview and native development review
remain pending. The failure, candidate, full stopped state and original
checkpoints remain. Native development and final identities were not used in
this TRAIN capacity pilot. The app/model selection is unchanged.

Evidence: `outputs/cctv_dgp_head4_capacity_v1_independent_audit.json`,
`outputs/cctv_dgp_head4_capacity_v1_return_audit/diagnosis.json`,
`outputs/cctv_dgp_head4_capacity_v1_return_audit/visual_samples/visual_review.json`,
and retained `outputs/cctv_dgp_head4_capacity_vm_v1_return/outputs/early_gate.json`.
The full restoration, completion and application goal remains incomplete.
