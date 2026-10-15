"""Publish reviewed return and new manual measurement; preserve historical text."""
from datetime import datetime, timezone
from pathlib import Path
from import_cctv_dgp_bank_saved_direction_forward_v1_r2_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import prepend_expected, new, link


def main():
    receipt = ANALYSIS/'publication.json'
    assert not receipt.exists()
    paths = [ANALYSIS/'import.json',ANALYSIS/'timing_audit_repair.json',ANALYSIS/'independent_audit.json',
             ANALYSIS/'finite_metrics.json',ANALYSIS/'visual_review.json',ANALYSIS/'finite_direction_review.json',
             OUT/'cctv_dgp_bank_group_gradients_v1_preparation.json',OUT/'cctv_dgp_bank_group_gradients_v1_transfer_audit.json',
             OUT/'cctv_dgp_bank_group_gradients_v1_bash_syntax.json']
    assert all(read(p)['complete'] for p in paths)
    imported, timing, audit, finite, visual, review, prep, transfer, bash = [read(p) for p in paths]
    assert sha(paths[3])==audit['finite_metrics_sha256'] and sha(paths[2])==visual['independent_audit_sha256']
    assert visual['all20_sheets_inspected'] and visual['all400_candidate_PNGs_inspected']
    assert review['bigger_proposals_cross1percent_mean_structure_but_source_and_preservation_fail']
    assert prep['protocol_sha256']==transfer['protocol_sha256']
    packet = OUT/prep['name'];p = read(packet/'protocol.json')
    assert sha(packet/'protocol.json')==prep['protocol_sha256'] and not (packet/'outputs').exists()
    assert bash['exit_code']==0 and bash['checked_shell_sha256']==sha(packet/'scripts/run_groups.sh')
    assert all(sha(packet/n)==h for n,h in p['assets_sha256'].items())
    assert sha(ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_return.py')==transfer['future_auditor_sha256']
    protected = read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    report_name = 'CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R2_RESULTS.md'
    guide_name = 'CCTV_DGP_BANK_GROUP_GRADIENTS_V1_VM.md'
    table = '\n'.join('| '+name+' | '+f"{100*x['gates']['raw']['relative_feature_gain']:.4f}%"+' | '+
        f"{100*x['gates']['png']['relative_feature_gain']:.4f}%"+' | '+str(len(x['gates']['raw']['preservation_failures']))+' | '+
        str(len(x['gates']['png']['preservation_failures']))+' | Rejected |' for name,x in finite['proposals'].items())
    report = f'''# Saved-direction finite forward R2 results — 11 October 2026

**Execution completed, but none of the four proposals qualifies.** The larger
changes exceed the 1% mean structure requirement and still regress preservation.
The smaller changes remain below 1% and also regress preservation. More epochs
and app adoption remain unqualified. The original checkpoint is preserved.

The human manual L4 diagnostic completed in **{audit['VM_seconds']:.2f} seconds**,
using **100 B forwards, 160 recognizer forwards, zero gradient queries,
zero optimizer updates and zero epochs**. Four temporary parameter proposals
were evaluated from the pinned initial B state. The model was restored between
proposals and at completion; original and recognizer state hashes were retained.
No new checkpoint was saved. Peak allocated VRAM was {audit['VM_peak_VRAM']/1024**3:.3f} GiB.
This measured peak is allocated Torch memory, not total device usage.

Return SHA256: `{imported['archive_sha256']}` ({imported['archive_bytes']:,} bytes).
All **596 returned file bindings**, **5,852 parent hashes**, **100 control-case
hashes**, four proposed-state/tensor hashes, **400 raw output cases**, **400
delivered PNGs**, **800 raw/PNG metric rows** and **17 source/profile groups**
were checked. Saved embedding calculations were audited; all embeddings were
not independently regenerated through a new recognizer forward. The audit used
saved arrays and parameter bytes; it did not run local neural gradients,
optimizers or model-parameter displacements.

| Proposal | Raw degraded structure gain | PNG gain | Raw preservation failures | PNG failures | Decision |
|---|---:|---:|---:|---:|---|
{table}

Failure counts are **group/metric checks**, not numbers of distinct faces.
The threshold remains 1%; MSE must not increase beyond 1e-12, SSIM and fixed
observed ArcFace must not decrease beyond 1e-6 in any retained group. Both
source-level degraded structure changes must be nonnegative. Brightness-only
gain fractions were zero for all four proposals; colour changes can still occur
and remain subject to appearance preservation and visual review.

At the larger rate, ordinary source-level raw/PNG structure changes were
**−0.2820%/−0.2770%** for `dataset/asian_faces`, versus **+1.4585%/+1.4858%**
for `dataset/thumbnails128x128`. Constrained changes were
**−0.3425%/−0.3302%** versus **+1.4544%/+1.4705%**. These names are source
labels. They do not establish ethnicity, native capture location, or Zamboanga
performance. The 100 cases are paired photographic TRAIN inputs across five
synthetic profiles, not native CCTV or reserved final identities.

All **20 exact comparison sheets and 800 cells** were inspected at original
resolution. Clear glasses, visible hair, hands and fabric remain present;
the sheets do not test covering removal or hidden-region completion. Eyes,
mouths, teeth and hair detail remain soft. Larger ordinary proposals often
change skin and background colour. No convincing whole-face clarity gain
supports promotion from this review. This assistant review is not the independent
final reviewer required by the goal.

{link('All20 bound comparison sheets','outputs/cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/sheets')}.
{link('Complete visual notes','outputs/cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/visual_review.json')}.
{link('Independent raw/PNG audit','outputs/cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/independent_audit.json')}.
{link('All800 saved metric rows','outputs/cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/finite_metrics.json')}.

## Cause and next diagnostic

The invalid assumption was that **two-reference, first-order aggregate
constraints would preserve every source/profile group after a finite change**.
The saved direction used only two of 20 TRAIN references. SSIM and ArcFace
constraints averaged five profiles, leaving separate profile behaviour
unprotected. The detail constraint and evaluation both use the four degraded
profiles: a clear/degraded detail-scope mismatch was checked and rejected.

The baseline losses agree in original metric units. For the sampled
`tr_asian_00048` at constrained rate1e-5, the predicted degraded-MSE improvement
was **0.0010%**, while the finite result worsened it by **0.0193%**. This is a
counterexample even on a derivation reference. It does not prove which individual
nonlinear layer causes the discrepancy. The other18 references also lack
constraints, and source/profile averaging leaves additional regressions visible.

The next bounded packet **measures** MSE, SSIM loss and ArcFace loss for each
source/profile combination, plus the two source/degraded feature groups, over
all20 frozen references. It saves 32 aggregate gradient vectors from 320 queries
and four sampled clear-profile vectors. The supplied previous directions are
used only for dot-product diagnosis. It performs **zero parameter displacements,
optimizer updates or training epochs**. Returned evidence must be audited before
a new direction is derived. Any later finite proposal needs a bounded line
search and actual raw/PNG gates; no preserved requirement is relaxed.

The original [GEM paper](https://papers.nips.cc/paper_files/paper/2017/file/f87522788a2be2d171666752f97ddebb-Paper.pdf)
explicitly bases gradient constraints on local linearity and representative
samples. [Boyd and Vandenberghe's descent-method notes](https://web.stanford.edu/~boyd/cvxbook/bv_cvxslides.pdf)
describe testing finite step lengths through backtracking. These inform the
diagnostic design. Their results do not establish convergence, preservation or
quality for this nonconvex face model.

{link('Saved first-order versus finite comparison','outputs/cctv_dgp_bank_saved_direction_forward_v1_r2_analysis/finite_direction_review.json')}.
{link('New five-step manual diagnostic',guide_name)}.

## Local checker repairs

The prepared return checker incorrectly equated two independent clock reads.
The actual projection timestamp followed the second logged timestamp by
**{timing['separate_clock_delta_seconds']*1e6:.3f} microseconds**. Its repaired
check verifies recorded timestamp order and the unchanged 900-second cap.
Both invalid clock-order fixtures were rejected; metric/gate/support/PNG/state
hash functions remained AST-identical. The VM return and original checker were
preserved; no VM rerun occurred.

An additional diagnostic checker compared normalized SSIM loss with a tolerance
intended for original metric units. The original saved visual receipt and checker
remain preserved. The corrected check compares original metric units: maximum
sampled SSIM arithmetic discrepancy **5.023e-7**, below 1e-6. This numerical
cross-implementation check did not change scientific quality gates or the
rejected preservation result.

## Goal remains active

Permanent baseline checkpoint SHA256:
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`.
The current restorer remains a fine-tuned DeblurGAN-v2-compatible baseline.
Replacement B uses an externally trained frozen StyleGAN2 prior; its prior is
not our trained model, and this diagnostic does not qualify replacement B.
Original checkpoints, splits, caches, failed runs and existing app design are
preserved. No new incumbent or agent VM connection/compute occurred.

All five milestones, native/final/app validation and all seven automatic versus
assisted covering families remain required. Clear glasses, non-obstructing hair
and visible appearance must be preserved. Insufficient usable structure needs
a clearer or less-covered crop. Completion must show the original, automatic
editable removal mask and one plausible estimate, with PNG and optional bundle
downloads; it must not claim exact hidden identity. No local training, automatic
historical pilot, ethnicity or Zamboanga-performance claim is permitted.
'''
    pin = prep['protocol_sha256'];stem=prep['stem'];name=prep['name']
    guide = f'''# Source/profile gradient diagnostic: five manual steps

**New diagnostic, prepared and independently transfer-checked; actual VM run
unrun.** Measures the preservation groups missed by the two-reference direction.
All20 photographic TRAIN references and five synthetic profiles are retained.
No native/final inputs, optimizer updates, parameter displacements or epochs.
The previous R2 return remains rejected and immutable; do not rerun it unchanged.

Upload **{prep['archive_bytes']:,} bytes** (19.7 MiB). Require **4 GiB free after
installation** and an idle existing L4. Estimate **3–8 minutes** for measurement
and export; this new run has not been timed. Hard caps: worker/measurement/export
900/600/300 seconds, external930/330 seconds with30-second kill grace, VRAM20GiB,
host RSS12GiB, outputs1.25GiB and disk reserve1GiB. Approximate uncompressed
gradient/sample/control arrays are948MiB; the resulting return size is unmeasured.

{link('Audited R2 result and reason for this diagnostic',report_name)}.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{stem}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {stem}-execution.tar.gz.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test ! -e ~/forensic-dgp/{name} &&
tar -xzf {stem}-execution.tar.gz -C ~/forensic-dgp
```

The checksum file is verified LF-only. Archive SHA256:
`{prep['archive_sha256']}`. A missing hash-bound parent stops the transfer
check; it does not silently reacquire data, overwrite a checkpoint or launch
historical training. A failed checksum stops extraction.

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_group_gradients_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{name} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_bank_group_gradients_v1_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_groups.sh {pin}
```

Expect the transfer receipt to show32 groups and320 planned queries, with zero
actual neural/gradient calls in that transfer check. The manual worker then
measures20 batches. Completion prints `Diagnostic exit code:0` and
`Export exit code:0`; successful export alone does not prove measurement success
or better image quality. Detach with **Ctrl+B**, release, then **D**; this keeps
tmux running. Leave the archive and source records untouched for audit.

5. Download from **Windows Google Cloud SDK Shell**, after export finishes:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz" "{stem}-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-results.tar.gz.sha256" "{stem}-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{stem}-export.json" "{stem}-export.json"
```

Each download uses **one remote source and an explicit local filename**, avoiding
the Windows PuTTY multiple-remote-source limitation. Hash/path import, all100
raw controls, all32 metric/gradient groups,320 scope bindings, four saved sample
vectors, original states, resource stops and gradient arithmetic must be checked
locally before a new training direction is chosen. Actual derivative execution
is measured on the VM; the saved-data audit does not independently rerun every
neural derivative. No automatic next pilot, more-epoch qualification or app
promotion follows this diagnostic.
'''
    new(ROOT/report_name,report);new(ROOT/guide_name,guide)
    prefix = f'''# Latest verified status — 11 October 2026: finite forward R2 completed; preservation failed

Manual saved-direction R2 completed in53.96s:100 B/160 recognizer forwards,
zero gradient queries/optimizer updates/epochs; four temporary proposals, no
new checkpoint. All596 return bindings/5,852 parent hashes,100 controls,
400 raw cases/400 delivered PNGs/800 metric rows,17 groups and20 exact
visual sheets passed the saved-evidence audit. Two local checker assumptions
(clock equality and normalized-loss tolerance units) were repaired with old
sources/receipts preserved; VM outputs and scientific gates were not changed.

Larger proposals achieve about1.1% mean degraded structure improvement but
fail preservation and regress the dataset/asian_faces source. Smaller proposals
remain below1% and also fail preservation. All20 reviewed faces remain soft in
degraded eye/mouth/teeth details; larger changes alter facial colour. None
qualifies a new incumbent, app adoption or additional training epochs. This is
paired photographic TRAIN evidence, not native/final/Zamboanga performance.

Cause: two-reference first-order constraints averaged SSIM/ArcFace over all
profiles and did not cover all20 reference/source/profile groups. A finite
counterexample occurs even on a derivation reference. Detail scopes agree.
The new manual source/profile gradient packet measures32 matched groups via
320 queries over the same20 references, without parameter displacements or
updates. Transfer/geometry/Python/Bash/saved-auditor checks passed. Its actual
VM execution remains unrun. Upload20,635,583B; require4GiB free after install;
estimated3–8min, hard worker/measurement/export900/600/300s, external930/330s
plus30s grace, VRAM20GiB/RSS12GiB/output1.25GiB/reserve1GiB. Independently
audit returned measurements before deriving another direction; any later finite
trial needs actual raw/PNG preservation checks and bounded step selection.

{link('Reviewed R2 result',report_name)}.
{link('Five exact new manual commands',guide_name)}.

Goal ACTIVE. Permanent baseline SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`
and existing app design are preserved. Allfive milestones, separate native and
reserved final identities, independent final/app review and allseven automatic
and assisted covering families remain required. Clear glasses, visible appearance
and non-obstructing hair remain protected; insufficient crops need clearer or
less-covered inputs. Completion keeps original/mask/one plausible estimate and
downloads, with no exact hidden-identity, ethnicity or Zamboanga claim.
Training/gradients/displacements remain human-manual L4 work. Agent VM access
is maintenance-only under the existing inventory/hash-bound authorization;
no agent VM connection, launch, local neural derivatives or optimizer ran here.
Historical text and failed records are retained below and superseded by this status.

---

'''
    earlier = read(OUT/'cctv_dgp_bank_saved_direction_forward_v1_r1_analysis/publication.json')
    expected = {n:r['after_sha256'] for n,r in earlier['documents'].items()}
    expected.update(earlier['new_documents_sha256'])
    folder = ANALYSIS/'publication_before';folder.mkdir()
    documents = {}
    canonical = ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']
    for name in canonical:
        documents[name]=prepend_expected(ROOT/name,prefix,folder/name,expected[name])
    short = f'''# Superseding verified R2 status — 11 October 2026

R2 is completed and rejected for preservation, not prepared/unrun. Original
commands below are historical and must not be rerun unchanged. Larger proposals
exceed1% mean structure while regressing a source and preservation; smaller
ones also fail. All400 raw/PNG cases and20 sheets were checked. No checkpoint
or more-epoch qualification. {link('Audited R2 result',report_name)}.
{link('Fresh manual measurement',guide_name)}. Prior text is retained verbatim.

---

'''
    for name in ['CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R2_VM.md','CCTV_DGP_BANK_SAVED_DIRECTION_FORWARD_V1_R1_RESULTS.md']:
        documents[name]=prepend_expected(ROOT/name,short,folder/name,expected[name])
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in paths},documents=documents,
        new_documents_sha256={n:sha(ROOT/n) for n in [report_name,guide_name]},
        result_audit_complete=True,visual_review_complete=True,larger_mean_structure_gain_not_preservation_qualification=True,
        scientific_gates_original_states_sources_splits_and_failures_preserved=True,
        new_manual_measurement_prepared_unrun=True,new_training_not_launched=True,more_epochs_qualified=False,
        all_five_milestones_seven_families_and_independent_final_app_requirements_preserved=True,
        local_neural_gradients_optimizer_or_parameter_displacements=0,agent_VM_connection_or_compute=False,
        model_qualified=False,goal_status='active',goal_complete=False))
    print(dict(complete=True,reviewed_outputs=400,documents_updated=len(documents),new_documents=2,new_manual_packet_unrun=True,goal_complete=False))


if __name__ == '__main__':
    main()
