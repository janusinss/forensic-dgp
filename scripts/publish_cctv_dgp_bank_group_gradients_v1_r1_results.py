"""Publish audited measurement, bounded next probe and unchanged full goal."""
from datetime import datetime, timezone
from import_cctv_dgp_bank_group_gradients_v1_r1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import prepend_expected, new, link


def main():
    receipt=ANALYSIS/'publication.json'
    assert not receipt.exists()
    base='cctv_dgp_bank_group_direction_forward_v1'
    evidence=[ANALYSIS/'import.json',ANALYSIS/'independent_audit.json',
        ANALYSIS/'common_direction/direction_review.json',ANALYSIS/'common_direction/independent_audit.json',
        OUT/(base+'_preparation.json'),OUT/(base+'_transfer_audit.json'),
        OUT/(base+'_return_checker_preparation.json'),OUT/(base+'_source_gate_audit.json'),OUT/(base+'_bash_syntax.json')]
    assert all(read(f)['complete'] for f in evidence)
    imported,audit,direction,direction_audit,prep,transfer,checker,source_gates,bash=[read(f) for f in evidence]
    packet=OUT/prep['name'];p=read(packet/'protocol.json');pin=sha(packet/'protocol.json')
    assert pin==prep['protocol_sha256']==transfer['protocol_sha256']==checker['protocol_sha256']==source_gates['protocol_sha256']
    assert not (packet/'outputs').exists()
    assert transfer['future_auditor_sha256']==source_gates['future_auditor_sha256']
    assert bash['exit_code']==0 and bash['checked_shell_sha256']==sha(packet/'scripts/run_forward.sh')
    assert all(sha(packet/n)==h for n,h in p['assets_sha256'].items())
    assert sha(OUT/(prep['stem']+'-execution.tar.gz'))==prep['archive_sha256']
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    assert audit['counts_recomputed']['gradient_queries']==320 and audit['counts_recomputed']['optimizer_updates']==0
    assert audit['all100_raw_controls_match'] and audit['all32_aggregate_vector_norm_dot_mean_and_loss_bindings_checked']
    counts=direction['old_directions_initially_regressing_group_counts']
    report_name='CCTV_DGP_BANK_GROUP_GRADIENTS_V1_R1_RESULTS.md'
    guide_name='CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_VM.md'
    report=f'''# Source/profile R1 measurements and next finite comparison — 12 October 2026

**The manual diagnostic completed and its saved evidence passed independent
audits. It measured learning conflicts; it did not train or improve a checkpoint.**
Runtime: **{audit['VM_seconds']:.2f} seconds**. Scope: 20 paired photographic
TRAIN references, 100 cases, 32 distinct source/profile loss groups, 320 gradient
queries, 40 B and 20 fixed-recognizer forwards. There were zero backwards calls,
optimizer updates, parameter displacements or epochs. Peak allocated CUDA
memory was {audit['VM_peak_VRAM']/1024**3:.3f} GiB and peak RSS
{audit['VM_peak_RSS']/1024**3:.3f} GiB. The diagnostic reported unchanged initial
model/buffer, original-restorer and recognizer states. No new checkpoint exists.
Those state claims are checked against saved receipts; no local neural replay
or new local gradient query was performed.

The **{imported['archive_bytes']:,} byte** archive, SHA256
`{imported['archive_sha256']}`, has {imported['returned_file_bindings']} verified
file bindings. All 5,852 parent bindings, 100 byte-identical raw controls, 32
aggregate-vector norms/dots/loss means, 320 query-scope bindings and four saved
sample vectors passed checks. Two clear-MSE sample derivatives also agreed
with prior measured normalized derivatives after conversion to raw loss units.
All320 neural derivatives were not independently recomputed. No new delivered
PNG quality or visual result is inferred from the frozen control arrays.

{link('Independent measurement audit','outputs/cctv_dgp_bank_group_gradients_v1_r1_analysis/independent_audit.json')}.

## Demonstrated limitation

At this initial model state, the old ordinary direction predicts increasing
loss in **{counts['ordinary']} of32 groups**; its two-reference projected version
does so in **{counts['constrained']} of32**. The conflicts include low-light and
compound-degradation groups. For example, the projected direction's raw-MSE dot
is +3.5166 for `dataset/asian_faces/lowlight_lr32`, and +13.1256 for
`dataset/thumbnails128x128/compound_lr24`. Positive dots mean an initial loss
increase for the old additive direction. These are mathematical predictions;
the earlier finite R2 outputs already independently failed preservation.

The old design constrained a small reference average. That average did not
represent every source/profile. More epochs following the same direction are
not justified by this diagnostic. Dataset labels describe source partitions;
they establish neither ethnicity nor performance on Zamboanga CCTV.

## Changed design and its limits

A minimum-L2 direction was derived from all32 saved group vectors. It targets
fractional structure descent for each source and a positive preservation margin
for every one of30 source/profile MSE, SSIM-loss and ArcFace-loss groups. Raw
losses and unit-gradient normalization are declared separately. A dual NNLS
solve, KKT checks, a separate zero-initialized primal SLSQP solve and exact
float32 direction/dot reconstruction passed. The smallest normalized-Gram
eigenvalue is {direction['smallest_Gram_eigenvalue']:.8g}; the independent primal
direction discrepancy is {direction_audit['zero_initialized_independent_primal_direction_L2_error']:.8g}.

Research on [multi-objective learning](https://proceedings.neurips.cc/paper_files/paper/2018/hash/432aca3a1e345e339f35a30c8f65edce-Abstract.html)
motivates checking competing task gradients. Our inequality-constrained
direction is a project-specific design, not that paper's implementation or a
convergence guarantee. All derived gains are local first-order predictions.
Finite network behavior, quantization and visible quality still need checking.

{link('Independent saved-array direction audit','outputs/cctv_dgp_bank_group_gradients_v1_r1_analysis/common_direction/independent_audit.json')}.

## One prepared manual forward comparison

| Additive step | Minimum predicted structure gain per source | Minimum predicted preservation-loss reduction per group |
|---:|---:|---:|
| 0.005 | 0.5% | 0.05% |
| 0.01 | 1% | 0.1% |
| 0.02 | 2% | 0.2% |
| 0.04 | 4% | 0.4% |

These are predictions and step coefficients, not completed epochs. The direction
L2 norm is {direction['direction_norm']:.8f}; the largest prospective
displacement is {direction['maximum_prospective_displacement_L2']:.8f},
{100*direction['maximum_prospective_displacement_relative_to_parameter_L2']:.5f}%
of the selected initial-parameter L2 norm. No model was displaced locally.

The fresh **{prep['archive_bytes']:,} byte** packet is verified and **UNRUN**.
It uses the same20 references/100 inputs, exact initial controls, affines,
frozen original/prior/recognizer and PNG policy. It changes the derived
direction and its four step coefficients; original used packets and failures
remain intact. Every proposal resets to initial state. No optimizer, autograd,
training epoch or saved replacement checkpoint is planned.

It plans 100 B/160 recognizer forwards, 100 exact controls, 400 raw outputs,
400 PNGs, saved raw/PNG embeddings and20 eight-column review sheets. The retained
all17-group MSE/SSIM/ArcFace preservation gates and brightness limit remain
unchanged. A new additional requirement prevents one sampled source's structure
gain from hiding a weak source: each source must reach1% on raw and PNG outputs.
This strengthens the probe; it does not relax any historical failed gate.
Saved-gate fixtures confirm that a passing average cannot hide a failed source.

Estimate **2–5 minutes** from the previous same-size54-second forward run plus
verification/export; this exact new direction is unmeasured. Require **4 GiB**
free after installation; recheck at launch. Worker1200s/measurement900s,
external1230s; export300s/external330s plus30s kill grace. VRAM20GiB/RSS12GiB,
output768MiB/reserve1GiB. Explicit startup storage failures are retained before
any neural import. Scientific failure, nonfinite state, mismatched controls or
resource limits retain a stop; no automatic retraining or historical relaunch.

{link('Exact five manual steps',guide_name)}.
{link('Sealed transfer audit','outputs/'+base+'_transfer_audit.json')}.

## Goal status and next decision

Goal ACTIVE. Neither diagnostic completion nor this direction qualifies more
epochs or a new incumbent. The next manual return must pass saved-output audits
and every planned visual sheet review before a finite training design is
considered. Full TRAIN, audited native CCTV DEV and independent reserved final
identity review remain separate requirements. Synthetic paired PSNR/SSIM are
not unpaired CCTV evidence.

The permanent original checkpoint SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`
is retained. Its DeblurGAN-v2-compatible origin and the replacement's own
condition/reconstruction path with externally pretrained frozen StyleGAN2 prior
remain declared. Pretrained restoration models remain comparison baselines.
No random-weight replacement, app promotion or exact hidden-identity claim.

All five milestones, preserved app design, DGP primary/Auto/override, insufficient
input handling, independent final/app review and all seven covering families
remain required. Masks, sunglasses, strong glare, hands, obstructing hair,
scarves and objects need separate automatic and assisted results; clear glasses,
ordinary hair and visible appearance must be preserved. Preview editable removal
area, original/mask/one plausible estimate, PNG and optional bundle downloads
remain required. The fresh three-width browser layout audit is partial evidence;
the complete model/mask/download flow and quality scope are not newly qualified.

All actual gradients, parameter proposals and training remain human-manual L4
tmux work. Agent VM access remains authorized inventory-first, hash-bound
maintenance only. No agent launched a new VM neural job. Credits/migration
notifications and the November2026 deadline stay as previously agreed.
'''
    new(ROOT/report_name,report)
    guide=f'''# Four source/profile direction probes — manual L4 steps

**Prepared; not launched by the agent.** This is a forward-only comparison of
four disposable parameter proposals, not training or additional epochs.
Upload **{prep['archive_bytes']:,} bytes**. Require **4 GiB free after install**.
Estimate **2–5 minutes**, unmeasured for this new direction. Worker/measurement
1200/900s, outer1230s; export300s/outer330s plus30s kill grace.
VRAM20GiB/RSS12GiB/output768MiB/reserve1GiB. No optimizer, gradients or checkpoints.
Preserve every prior used run and failed gate. Do not bypass a stop or rerun
this immutable packet after it produces outputs/logs.

1. Upload from **Windows Google Cloud SDK Shell**:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{prep['stem']}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{prep['stem']}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:
```bash
cd ~ &&
sha256sum -c {prep['stem']}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_dgp_bank_photometric_guard_v1_vm &&
test ! -e ~/forensic-dgp/{prep['name']} &&
tar -xzf {prep['stem']}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:
```bash
tmux new-session -A -s dgp_group_direction_v1
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/{prep['name']} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{base}_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_forward.sh {pin}
```

The preflight reports available and required bytes. Detach with **Ctrl+B, D**;
reattach with `tmux attach -t dgp_group_direction_v1`. The exporter runs after
success or a retained diagnostic stop. Export completion does not establish
image quality or authorize more epochs.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prep['stem']}-results.tar.gz" "{prep['stem']}-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prep['stem']}-results.tar.gz.sha256" "{prep['stem']}-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prep['stem']}-export.json" "{prep['stem']}-export.json"
```

Each download has one remote source, compatible with Windows PuTTY.
Return all three files for independent raw/PNG, source-specific gate and all20
sheet review. A newly inspected result is required before training is proposed.

Protocol SHA256: `{pin}`.
Execution archive SHA256: `{prep['archive_sha256']}`.
{link('Audited reason for this changed probe',report_name)}.
'''
    new(ROOT/guide_name,guide)
    prefix=f'''# Latest verified status — 12 October 2026: source/profile R1 completed; new direction UNRUN

The manual R1 diagnostic completed in{audit['VM_seconds']:.2f}s. All89 return
bindings/5,852 parent hashes,100 exact raw controls and32 group derivative
norm/dot/loss bindings passed saved-evidence audits. It measured320 gradients
with40 B/20 recognizer forwards, zero updates/displacements/epochs and no new
checkpoint. No new PNG or native/final quality qualification is claimed.

The previous ordinary/projected directions initially worsen{counts['ordinary']}/{counts['constrained']}
of32 measured groups. Their earlier finite preservation failures remain. A new
minimum-L2 direction uses all20 references and32 distinct source/profile
groups. Independent saved-array/KKT/zero-start primal checks passed; these
are first-order predictions and provide no finite image-quality result.

A fresh {prep['archive_bytes']:,}B four-step forward packet is prepared and UNRUN.
Steps .005/.01/.02/.04; zero optimizers/gradients/epochs/checkpoints. Same100
inputs/controls, raw/PNG separation, frozen weights/affines/palette and all
retained17-group preservation/brightness gates. Additional1% structure gain
is required for each sampled source on raw and PNG outputs. Exact source-gate
fixtures,200 saved control metric rows, alignment/Python/Bash/transfer checks pass.
Require4GiB free after install. Estimate2–5min; worker/measurement1200/900s,
outer1230s; export300/330s plus30s grace; VRAM20GiB/RSS12GiB/output768MiB/reserve1GiB.
No automatic launch, failure bypass or promotion. Audit every returned output
and all20 visual sheets before considering any training continuation.

{link('R1 result and design limits',report_name)}.
{link('Five exact new manual VM steps',guide_name)}.

Goal ACTIVE. Original baseline SHA256646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b
and original checkpoints/splits/caches/provenance/failures/app design stay
preserved. All five milestones, independent DEV/final and full app review,
DGP primary Auto/override, visible preservation and clearer/less-covered input
handling remain required. All seven completion families, editable automatic
mask, separate automatic/assisted results, original/mask/one estimate and PNG/
optional bundle remain required. Clear glasses/ordinary hair/visible appearance
stay preserved. Paired photographic TRAIN evidence stays separate from unpaired
native CCTV and reserved final identities; no ethnicity/Zamboanga/hidden-identity
claim. Training/gradients/model displacements remain manual L4 tmux work;
agent VM access remains hash-bound maintenance only. Historical text is retained.

'''
    previous=read(OUT/'cctv_dgp_bank_group_gradients_v1_analysis/publication.json')
    names=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']
    expected={name:previous['documents'][name]['after_sha256'] for name in names}
    expected['CCTV_DGP_BANK_GROUP_GRADIENTS_V1_VM.md']=previous['documents']['CCTV_DGP_BANK_GROUP_GRADIENTS_V1_VM.md']['after_sha256']
    expected['CCTV_DGP_BANK_GROUP_GRADIENTS_V1_R1_VM.md']=previous['new_documents_sha256']['CCTV_DGP_BANK_GROUP_GRADIENTS_V1_R1_VM.md']
    assert all(sha(ROOT/n)==h for n,h in expected.items())
    backups=ANALYSIS/'publication_before';backups.mkdir()
    records={}
    for name,h in expected.items():
        block=prefix if name in names else f'''# Superseded used measurement packet — do not relaunch

R1's manual measurement is complete and independently audited. Its original
V1 storage failure and subsequent successful R1 are retained. No optimizer
update or checkpoint improvement occurred. Use the changed forward-only guide.
{link('Audited R1 measurements',report_name)}.
{link('Fresh prepared/unrun four-step comparison',guide_name)}.
Original commands below are retained historical evidence.

'''
        records[name]=prepend_expected(ROOT/name,block,backups/name,h)
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in evidence},documents=records,
        new_documents_sha256={report_name:sha(ROOT/report_name),guide_name:sha(ROOT/guide_name)},
        measurement_pass_not_training_success=True,manual_forward_probe_prepared_unrun=True,
        scientific_thresholds_and_original_checkpoints_preserved=True,
        all_five_milestones_and_seven_families_preserved=True,local_neural_gradient_optimizer_or_model_displacement_calls=0,
        new_actual_training_or_neural_VM_job_launched_by_agent=False,more_epochs_qualified=False,
        model_qualified=False,goal_status='active',goal_complete=False))
    print(dict(complete=True,documents=len(records),new_reports_and_guides=2,manual_VM_probe_unrun=True,goal_status='active'))


if __name__=='__main__':
    main()
