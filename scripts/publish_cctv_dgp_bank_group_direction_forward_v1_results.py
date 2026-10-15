"""Publish rejected finite images and the verified UNRUN signed-step diagnosis."""
from datetime import datetime, timezone
from import_cctv_dgp_bank_group_direction_forward_v1_return import ROOT, OUT, RETURN, ANALYSIS, read, sha, write
from publish_cctv_dgp_bank_saved_direction_forward_v1_r1_failure import prepend_expected, new, link

BASE='cctv_dgp_bank_group_step_check_v1'


def main():
    receipt=ANALYSIS/'publication.json'
    assert not receipt.exists()
    evidence=[ANALYSIS/n for n in ['import.json','independent_audit.json','finite_metrics.json',
        'visual_review.json','finite_direction_review.json','review_binding_audit.json']]
    evidence += [OUT/(BASE+s) for s in ['_preparation.json','_transfer_audit.json',
        '_return_checker_preparation.json','_source_gate_audit.json','_signed_derivative_fixture_audit.json',
        '_bash_syntax.json']]
    evidence += [ROOT/'scratch/app_flow_20261012_v1'/n for n in ['download_content_audit.json','visual_review.json','closure.json']]
    assert all(read(path)['complete'] for path in evidence)
    imported=read(ANALYSIS/'import.json');audit=read(ANALYSIS/'independent_audit.json')
    finite=read(ANALYSIS/'finite_metrics.json');review=read(ANALYSIS/'finite_direction_review.json')
    visual=read(ANALYSIS/'visual_review.json');bound=read(ANALYSIS/'review_binding_audit.json')
    prep=read(OUT/(BASE+'_preparation.json'));transfer=read(OUT/(BASE+'_transfer_audit.json'))
    packet=OUT/prep['name'];p=read(packet/'protocol.json');pin=sha(packet/'protocol.json')
    assert not (packet/'outputs').exists() and not (OUT/(BASE+'_return')).exists()
    assert pin==prep['protocol_sha256']==transfer['protocol_sha256']
    assert transfer['actual_manual_VM_run_unrun'] and prep['manual_VM_run_unrun']
    assert read(OUT/(BASE+'_bash_syntax.json'))['checked_shell_sha256']==sha(packet/'scripts/run_forward.sh')
    assert read(OUT/(BASE+'_bash_syntax.json'))['exit_code']==0
    assert all(sha(packet/n)==h for n,h in p['assets_sha256'].items())
    assert all(sha(ROOT/n)==h for n,h in p['evidence_sha256'].items())
    assert sha(OUT/(prep['stem']+'-execution.tar.gz'))==prep['archive_sha256']
    assert visual['all20_sheets_and800_cells_inspected'] and bound['finite_prediction_comparisons']==128
    assert all(not v['all_preview_requirements_pass'] for v in finite['proposals'].values())
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    previous=read(OUT/'cctv_dgp_bank_group_gradients_v1_r1_analysis/publication.json')
    names=['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md',
           'CCTV_DGP_COMPARATIVE_IMPROVEMENT_PLAN_V2.md']
    old_guide='CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_VM.md'
    expected={n:previous['documents'][n]['after_sha256'] for n in names}
    expected[old_guide]=previous['new_documents_sha256'][old_guide]
    assert all(sha(ROOT/n)==h for n,h in expected.items()),'Preserve concurrent document edits'
    report_name='CCTV_DGP_BANK_GROUP_DIRECTION_FORWARD_V1_RESULTS.md'
    guide_name='CCTV_DGP_BANK_GROUP_STEP_CHECK_V1_VM.md'
    assert not (ROOT/report_name).exists() and not (ROOT/guide_name).exists()
    table='\n'.join('| '+r['proposal'].removeprefix('balanced_')+' | '+
        f"{r['raw_structure_gain_percent']:.6f}% | {r['PNG_structure_gain_percent']:.6f}% | "+
        f"{r['raw_preservation_failure_count']} | {r['PNG_preservation_failure_count']} | Rejected |"
        for r in review['summary'])
    report=f'''# Group-direction finite comparison — 12 October 2026

**All four finite proposals are rejected.** The manual forward diagnostic
completed, but it did not produce a qualified model or justify more epochs.
The best raw structure gain was **0.862655%**, below **1%**, with **37 raw and
37 PNG preservation failures**. Counts denote group–metric checks, not people.

| Additive step | Raw structure gain | PNG structure gain | Raw preservation failures | PNG preservation failures | Decision |
|---:|---:|---:|---:|---:|---|
{table}

At step .02, source-specific raw/PNG gains were **0.556856%/0.556644%** for
`dataset/asian_faces` and **0.939342%/0.957872%** for
`dataset/thumbnails128x128`. Both raw and PNG gains must reach1% for each source.
The largest .04 step regressed both sources. All four failed the retained
MSE/SSIM/ArcFace preservation requirements; the brightness limit passed.
These are paired photographic TRAIN measurements. Dataset labels do not imply
ethnicity, native CCTV performance or Zamboanga performance.

Location: finite model outputs from the all32-group direction.
Cause demonstrated: favorable local derivatives did not translate into finite
preservation and useful detail at .005/.01/.02/.04. A particular layer or code
bug is not isolated. Fix being investigated: measure much smaller signed
displacements against the saved derivatives before choosing another optimizer
or training schedule. The human answered the diagnostic question with
“apply best approach”; no scientific gate was weakened.

## Verified execution and image evidence

The manual run completed in **{audit['VM_seconds']:.2f} seconds**. It used100
B and160 recognizer forwards and four disposable parameter proposals, with
zero gradients, backwards calls, optimizer updates or epochs. No replacement
checkpoint was saved. Peak allocated CUDA memory was
{audit['VM_peak_VRAM']/1024**3:.3f} GiB. Its saved run reports initial model
restoration and unchanged frozen original/recognizer states.

The {imported['archive_bytes']:,}-byte archive SHA256 is
`{imported['archive_sha256']}`. All596 returned bindings,5,852 parent
bindings,100 exact control receipts,400 raw arrays/400 exact floor PNGs,
800 metric rows,17 source/profile groups and20 sheets/800 exact RGB cells
passed saved-evidence audits. Four disposable state hashes and their selected
parameter bytes were independently reconstructed from saved CPU values;
no local model was displaced and no new neural derivative was queried.
This is independent saved-output checking, not an independent neural rerun
or an independent human final review.

All20 sheets were visually inspected. Eyes, lips, teeth, ordinary glasses and
hair remain softened relative to clear inputs. Degraded outputs do not show
a convincing useful whole-face detail gain. Step .04 introduces bright
arched nose ridges and lip outlines in several faces. Side padding remains
fixed where present. This is restoration evidence; no covering completion
family was tested in these sheets.

{link('Full raw/PNG saved-output audit','outputs/cctv_dgp_bank_group_direction_forward_v1_analysis/independent_audit.json')}.
{link('All20 visual observations','outputs/cctv_dgp_bank_group_direction_forward_v1_analysis/visual_review.json')}.
{link('Representative exact comparison sheet','outputs/cctv_dgp_bank_group_direction_forward_v1_analysis/sheets/b391.png')}.

## Selected diagnosis before more training

All32 saved directional dots are negative, yet some finite losses increase.
For example, `dataset/asian_faces/clear/ArcFace_loss` predicts about4.37%
loss reduction at .005, while its independently measured loss increases.
The signed test will check whether the predictions become accurate near the
initial model or whether a derivative/forward-path discrepancy remains.
Current results cannot decide between these causes. Raw SSIM loss also has
a small gradient-versus-auditor baseline difference; it remains recorded.

The new **{prep['archive_bytes']:,}-byte packet is verified and UNRUN**.
It uses **−.001,−.0001,+.0001,+.001**, the unchanged all32-group direction,
initialB, same20 references/100 inputs, frozen prior/original/recognizer,
fixed affines and raw-to-PNG policy. Magnitudes are5× and50× smaller than
the previous smallest .005. Each proposal resets to the initial state.

The future audit compares each pair using
`(loss(+h)−loss(−h))/(2h)` against the saved raw directional derivative.
It also records one-sided slopes, curvature, baseline-loss differences and
actual float32 parameter changes/rounding. PNG quantization, visible outputs
and preservation gates are reviewed separately; PNG is not treated as a
differentiable gradient test. No arbitrary derivative tolerance selects a
model. Negative steps diagnose the path only. Small positive steps are
expected to show less than1% structure gain; that does not relax the1%
qualification requirement or turn this diagnostic into a trained improvement.

Preparation checked5 archive members,4 assets,5,852 parents,100 retained
alignment grids,Python3.10 syntax,Windows pre-neural rejection,Bash syntax,
100 saved controls/200 metric rows,six preservation boundary cases,three
source-gate fixtures,64 analytical signed pairs and negative-step exclusion.
A local unsealed preparation key-name stop is retained under
`outputs/cctv_dgp_bank_group_step_check_v1_preparation_stop`; it occurred
before sealing,upload or any neural/VM call. The sealed packet is unchanged.

Require **4 GiB free after install**. Estimate **2–5 minutes** for verification,
measurement and export, based on the previous same-size53.61s measurement.
Worker/measurement1200/900s; outer1230s; export300/outer330s plus30s kill
grace. VRAM20GiB/RSS12GiB/output768MiB/reserve1GiB. Export additionally
checks space before archiving and retains the1GiB reserve. Current VM power
and free space are not claimed from older cleanup snapshots.

{link('Exact five manual transfer/tmux/download steps',guide_name)}.
{link('New sealed transfer audit','outputs/'+BASE+'_transfer_audit.json')}.

## App and full goal remain separate

Fresh bundled inline Playwright checks covered375/768/1280px initial layouts,
four actual generation requests (Off,On,Auto and a tiny assisted region),
editable-mask review and eight actual PNG/ZIP downloads. Saved artifact audits
verified metadata/pixel hashes,On/Auto identical raw arrays,exact Off-visible
pixels and separate empty automatic/52-pixel assisted masks. Eight screenshots
were inspected. This functional square photographic TRAIN fixture does not
qualify native/non-square behavior,negative API guards,responsive result states,
all seven covering families or useful model quality. Those checks remain pending.

{link('Bounded app-flow closure','scratch/app_flow_20261012_v1/closure.json')}.

Goal ACTIVE. The original SHA256
`646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b`
remains the permanent app baseline. Its DeblurGAN-v2-compatible origin and the
replacement's own conditioning/reconstruction path with externally pretrained
frozen StyleGAN2 prior remain declared. No own-trained prior claim or app
promotion is made. All five milestones and seven covering families remain
required, including independent native DEV/final review,primary DGP Auto/override,
preserved visible appearance,clearer/less-covered requests,editable automatic
area,separate automatic/assisted results,original/mask/one plausible estimate
and PNG/optional bundle downloads. Preserve clear glasses and ordinary hair;
make no exact hidden-identity claim. Final identities remain outside tuning;
unpaired CCTV and paired synthetic PSNR/SSIM remain separate.

Training/gradients/model displacements remain human-manual L4 tmux work.
Agent VM access remains inventory-first,hash-bound maintenance only.
No new VM neural job was launched by the agent. Checkpoints,splits,caches,
full states,provenance and failures remain retained. The credit/migration
notification policy and November2026 deadline remain in force.
'''
    new(ROOT/report_name,report)
    guide=f'''# Signed small-step diagnosis — exact manual L4 commands

**Verified and UNRUN.** This checks why predicted improvement failed to preserve
the finite outputs. Four signed disposable proposals; zero optimizer updates,
gradients,epochs or new checkpoints. Original checkpoints and failed runs stay
retained. Require **4 GiB free after install**. Estimate **2–5 minutes**.
Worker/measurement1200/900s,outer1230s; export300/outer330s plus30s grace;
VRAM20GiB/RSS12GiB/output768MiB/reserve1GiB. Upload **{prep['archive_bytes']:,} bytes**.

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
tmux new-session -A -s dgp_group_step_check_v1
```

4. Launch **inside tmux**:
```bash
cd ~/forensic-dgp/{prep['name']} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{BASE}_vm.py --root . --parent-root ../cctv_dgp_bank_photometric_guard_v1_vm --protocol-sha {pin} --verify-transfer &&
bash scripts/run_forward.sh {pin}
```

Preflight reports available/required bytes. Detach with **Ctrl+B,D**;
reattach with `tmux attach -t dgp_group_step_check_v1`. Export runs after a
measurement success or retained stop. A complete export is not a quality pass.
Do not repeat this used packet or bypass a stop. If changing VMs, verify the
migration and new VM identity first; these commands target the existing L4 VM.

5. Download from **Windows Google Cloud SDK Shell** after export finishes:
```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prep['stem']}-results.tar.gz" "{prep['stem']}-results.tar.gz"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prep['stem']}-results.tar.gz.sha256" "{prep['stem']}-results.tar.gz.sha256"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{prep['stem']}-export.json" "{prep['stem']}-export.json"
```

Each download has one remote source for Windows PuTTY. Return all three files
for independent raw/PNG,32-group signed-derivative,rounding,preservation and
all20-sheet visual review. Negative steps are diagnosis only. The1% structure
and visible-preservation requirements remain unchanged; no more epochs are
automatically authorized or launched.

Protocol SHA256: `{pin}`.
Execution archive SHA256: `{prep['archive_sha256']}`.
The checksum file is verified LF text to avoid the previous CRLF filename error.
{link('Audited finite failure and diagnostic purpose',report_name)}.
'''
    new(ROOT/guide_name,guide)
    prefix=f'''# Latest verified status — 12 October 2026: finite proposals rejected; signed diagnosis UNRUN

The manual group-direction run completed in{audit['VM_seconds']:.2f}s with100 B/
160 recognizer forwards,four disposable proposals,zero gradients/updates/epochs
and no checkpoint. All596 returned bindings/5,852 parents,100 controls,
400 raw outputs/400 exact PNGs/800 metric rows and20 sheets/800 cells passed
saved-evidence checks. Every sheet was inspected; useful whole-face detail
gain was not demonstrated,and the largest step adds nose/lip contour artifacts.

All four proposals failed unchanged preservation and structure requirements.
Best raw/PNG structure gain:0.862655%/0.877397%,with37/37 group–metric failures;
both sampled sources also remain below1%. Favorable local derivatives did not
predict finite preservation at .005/.01/.02/.04. A specific bug/layer cause is
not isolated. Human diagnostic reply:“apply best approach”.

New signed diagnosis is verified and UNRUN:{prep['archive_bytes']:,}B,
steps−.001/−.0001/+.0001/+.001,unchanged direction/initialB/100 inputs/affines/
frozen components. Two signed radii compare actual raw slopes/curvature with
32 saved derivatives and record float32 rounding. Negative steps are diagnosis
only; PNG quantization is reviewed separately. This is not more training.
All17 preservation/brightness groups and per-source raw/PNG1% remain required.
Preparation/transfer/Python/Bash/100 saved controls/200 rows/signed analytical
fixtures pass. Require4GiB free after install; estimate2–5min; worker/measurement
1200/900s,outer1230s; export300/330s plus30s grace;VRAM20GiB/RSS12GiB/
output768MiB/reserve1GiB. Space is guarded during export. No automatic launch.

{link('Full rejected result and diagnosis',report_name)}.
{link('Five exact new manual VM steps',guide_name)}.

The bounded fresh app flow passed Auto/On/Off,one tiny assisted-region generation,
eight PNG/ZIP downloads and initial375/768/1280 layouts. This square TRAIN
fixture is functional evidence;native/non-square,negative API,result-responsive,
seven covering families and useful independent DEV/final review remain pending.

Goal ACTIVE. Permanent baseline SHA256
646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b,
original checkpoints/splits/caches/full states/provenance/failures/app design
remain retained. All five milestones and all seven covering families remain
required;DGP primary Auto/override,visible preservation,clearer/less-covered
requests,editable automatic mask,separate automatic/assisted results,
original/mask/one plausible estimate and PNG/optional bundle stay in scope.
Preserve clear glasses/ordinary hair. Paired TRAIN/synthetic evidence stays
separate from unpaired native CCTV and reserved final identities. No ethnicity,
Zamboanga or exact hidden-identity claim. Training/gradients/displacements remain
human-manual L4 tmux work;agent VM access remains hash-bound maintenance only.
More epochs and app adoption are unqualified. Historical bytes below are retained.

'''
    old_prefix=f'''# Completed and rejected — 12 October 2026

This packet completed its manual run;all four proposals failed preservation
and structure requirements. **Do not rerun the historical commands below.**
The original bytes are retained. {link('Verified failure report',report_name)}.
Use the fresh {link('signed diagnostic manual guide',guide_name)} for the next
bounded diagnosis. It is UNRUN and does not authorize additional epochs.

'''
    backup=ANALYSIS/'publication_before';backup.mkdir()
    documents={}
    for name in names+[old_guide]:
        documents[name]=prepend_expected(ROOT/name,old_prefix if name==old_guide else prefix,
                                          backup/name,expected[name])
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),publisher_sha256=sha(__file__),
        evidence_sha256={f.relative_to(ROOT).as_posix():sha(f) for f in evidence},documents=documents,
        new_documents_sha256={n:sha(ROOT/n) for n in [report_name,guide_name]},
        four_finite_quality_proposals_rejected=True,no_new_incumbent_or_extra_epochs_qualified=True,
        manual_signed_diagnostic_prepared_unrun=True,all_five_milestones_and_seven_families_preserved=True,
        original_checkpoint_app_and_failed_runs_retained=True,new_agent_VM_neural_job_launched=False,
        local_neural_gradient_optimizer_or_model_displacement_calls=0,model_qualified=False,goal_status='active',goal_complete=False))
    print(dict(complete=True,preserved_document_histories=5,new_documents=2,manual_signed_diagnostic_UNRUN=True))


if __name__=='__main__':main()
