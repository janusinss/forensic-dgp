"""Freeze a self-contained manual training stage; no neural imports or VM action."""
import ast
from datetime import datetime,timezone
from pathlib import Path
import random
import shutil
import tarfile
import time
from cctv_dgp_head4_capacity_contract_v1 import NAME,STEM,BUDGETS,WEIGHTS,NORMALIZERS,validate,read,write,sha

ROOT=Path(__file__).resolve().parents[1];PARENT=ROOT/'outputs/cctv_dgp_head4_reactivation_vm_v1'
BASE=ROOT/'outputs/cctv_dgp_mixed_vm_v9_r2';OUT=ROOT/'outputs'/NAME
PREP=ROOT/'outputs/cctv_dgp_head4_capacity_v1_preparation'
NEW={'cctv_dgp_head4_capacity_model_v1.py':'cctv_dgp_head4_capacity_model_v1.py',
    'cctv_dgp_head4_capacity_contract_v1.py':'cctv_dgp_head4_capacity_contract_v1.py',
    'cctv_dgp_head4_lossless_v1.py':'cctv_dgp_head4_lossless_v1.py',
    'cctv_dgp_head4_capacity_vm_v1.py':'scripts/cctv_dgp_head4_capacity_vm_v1.py',
    'supervise_cctv_dgp_head4_capacity_v1.py':'scripts/supervise_cctv_dgp_head4_capacity_v1.py'}


def guide(pin,digest,size):
    return f'''# Repaired DGP branch: first manual training stage

The downloaded gradient diagnostic is audited. This new packet trains only the
three repaired original-DGP pieces for **50 updates**, with all**3,905 TRAIN**
cases checked before/after. It has not run. It completes0 full epochs and cannot
qualify restoration; later additional epochs1/2/5 require reviewed evidence.

Require **14GiB free after installation**. Estimated stage**8–25 minutes** plus
export**2–10 minutes**; enforced worker2400s/external2430s, export600s/external630s,
fit300s, each snapshot900s, VRAM20GiB, return6GiB and free-disk reserve1GiB.
The larger export has not been timed. Stop on failed gates or insufficient space.
Original checkpoints, research data/caches and failures remain protected.

The last API check found the existing VM stopped. Start it manually if needed
from **Windows Google Cloud SDK Shell**; this starts billed VM runtime:

```bat
gcloud compute instances start forensic-dgp-thesis --project=forensic-dgp-thesis --zone=us-central1-a
```

Execution archive:{size:,} bytes; SHA256 `{digest}`.
Protocol SHA256:`{pin}`. Use only after the independent packet audit passes.
No automatic start/training, failed-run resume, historical rerun or app adoption.

1. Upload from **Windows Google Cloud SDK Shell**:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{STEM}-execution.tar.gz.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

2. Install in **VM SSH**:

```bash
cd ~ &&
sha256sum -c {STEM}-execution.tar.gz.sha256 &&
test -x ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp &&
df -h ~/forensic-dgp &&
python3 -c 'import shutil; from pathlib import Path; free=shutil.disk_usage(Path.home()/"forensic-dgp").free; print("Free GiB:",round(free/1024**3,2)); assert free>=14*1024**3,"Stop: need14GiB free after install"'
```

Stop if the disk check fails. No deletion is bundled. Authorized maintenance
requires fresh workload/disk inventory and exact hash-matched retained backups.

3. Open tmux:

```bash
tmux new-session -A -s dgp_head4_capacity_v1
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/cctv_dgp_head4_capacity_vm_v1.py --root . --protocol-sha {pin} --verify-transfer &&
python -B -u scripts/supervise_cctv_dgp_head4_capacity_v1.py --root . --protocol-sha {pin}
```

Detach: Ctrl+B, release, D. Reattach:
`tmux attach-session -t dgp_head4_capacity_v1`.
The supervisor prints progress and exports retained failures when possible.
Initial parity, six real gradient checks and measured storage reservation occur
before the optimizer. The1% structure and raw/PNG preservation gates stay fixed.
`early_capacity_pass:true` is a necessary TRAIN result only; native development,
visual review, independent final review and app qualification remain pending.
`complete:true` in the export receipt means the archive finished.

5. Download from **Windows Google Cloud SDK Shell**, one remote source per command:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

Return allthree files, including a failed run. The independent audit checks every
saved raw/PNG record and the full training state; the visual review follows.
Keep originals on the VM until the local full-state backup is verified.
This packet is self-contained apart from the existing environment. No reserved
final images or native CCTV are paired training targets. Allfive milestones and
allseven completion families remain required; the full goal remains incomplete.
'''


def main():
    start=time.monotonic();assert not OUT.exists() and not PREP.exists()
    parent=read(PARENT/'protocol.json');audit=read(ROOT/'outputs/cctv_dgp_head4_reactivation_v1_independent_audit_r2.json')
    assert audit['complete'] and audit['connected_route_pass'] and audit['optimizer_updates']==0
    parity_folder=ROOT/'outputs/cctv_dgp_head4_capacity_v1_parity';parity=read(parity_folder/'results.json')
    assert parity['complete'] and parity['exact_initial_parity_cases']==100 and parity['maximum_initial_error']==0
    for n,d in read(parity_folder/'plan.json')['source_bindings'].items():assert sha(ROOT/n)==d,n
    for n,d in parent['assets_sha256'].items():assert sha(PARENT/n)==d,n
    for n,d in parent['local_source_bindings'].items():assert sha(ROOT/n)==d,n
    coverage=read(ROOT/'outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json')
    assert coverage['complete'] and coverage['canonical_TRAIN_targets_verified']==781 and coverage['validation_or_reserved_pixels_decoded']==0
    old=read(ROOT/'outputs/cctv_dgp_profile_batches_vm_v31/protocol.json');mixed=read(BASE/'mixed_protocol_v9.json')
    assert sha(BASE/'mixed_protocol_v9.json')==old['mixed_data_protocol_sha256']
    corpus=old['mixed_TRAIN_assets_sha256'];assert len(corpus)==5467
    for n,d in corpus.items():assert sha(BASE/n)==d,n
    cases=old['case_rows'];assert len(cases)==3905
    refmap={r['id']:r for r in mixed['references']};refs=[refmap[cases[i]['source_person_or_reference']] for i in range(0,3905,5)]
    assert len({r['id'] for r in refs})==781 and all(r['role']=='train' for r in refs)
    order=list(range(781));random.Random(501050).shuffle(order)
    schedule=[list(range(i*5,i*5+5)) for i in order[:50]]
    indices={c['id']:i for i,c in enumerate(cases)}
    preflight=[list(range(indices[parent['cases'][j]['id']],indices[parent['cases'][j]['id']]+5)) for j in [0,50]]
    OUT.mkdir();PREP.mkdir();mapping={}
    def copy(origin,name):
        dest=OUT/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(origin,dest)
        mapping[name]=Path(origin).relative_to(ROOT).as_posix()
    for n in corpus:copy(BASE/n,n)
    root_files={'cctv_dgp_pilot.py','dgp_face_restoration.py','dgp_frozen_inference_v2.py','cctv_dgp_frozen_norm.py',
        'frozen_capacity_contract.py','frozen_raw_metrics.py','frozen_definitions.py','cctv_dgp_group_conflicts_v1_losses.py'}
    for n in parent['assets_sha256']:
        if n.startswith(('models/','weights/')) or n in root_files:copy(PARENT/n,n)
    for source,n in NEW.items():ast.parse((ROOT/'scripts'/source).read_text(),feature_version=(3,10));copy(ROOT/'scripts'/source,n)
    evidence=['CCTV_DGP_HEAD4_REACTIVATION_V1_RESULTS.md','CCTV_DGP_HEAD4_TRAINING_V1_DESIGN.md','CCTV_DGP_FULL_TRAINING_COVERAGE_V1_RESULTS.md',
        'CCTV_DGP_CURRENT_TRAINING_IMPROVEMENT_PLAN.md','outputs/cctv_dgp_head4_reactivation_v1_independent_audit_r2.json',
        'outputs/cctv_dgp_head4_capacity_v1_parity/plan.json','outputs/cctv_dgp_head4_capacity_v1_parity/results.json',
        'outputs/cctv_dgp_head4_capacity_v1_parity/runner_binding.json','outputs/cctv_dgp_head4_capacity_v1_parity/codec_proof.json',
        'outputs/cctv_dgp_head4_training_design_review_v1/decay_direction_review.json',
        'outputs/cctv_dgp_head4_training_design_review_v1/source_balanced_direction_review.json',
        'outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json',
        'outputs/cctv_dgp_head4_reactivation_v1_return_audit_failure/original_failure.json',
        'outputs/cctv_dgp_head4_reactivation_v1_return_audit_failure/r2_plan.json']
    for i,n in enumerate(evidence):copy(ROOT/n,'evidence/'+f'{i:02d}_'+Path(n).name)
    copy(BASE/'mixed_protocol_v9.json','evidence/mixed_protocol_v9.json');copy(PARENT/'protocol.json','evidence/gradient_protocol.json')
    p={'format':'own-DGP-repaired-head4-capacity-stage-v1','UTC':datetime.now(timezone.utc).isoformat(),
        'original_state':parent['original_state'],'original_checkpoint_sha256':parent['original_checkpoint_sha256'],
        'recognizer_state':parent['recognizer_state'],'initial_candidate_state':parity['before']['candidate'],
        'cases':cases,'references':refs,'canonical_target_RGB_sha256':{r['id']:mixed['canonical_target_rgb_sha256'][r['id']] for r in refs},
        'mixed_protocol_sha256':sha(BASE/'mixed_protocol_v9.json'),'mixed_TRAIN_assets_sha256':corpus,'source_mapping':mapping,
        'full_epoch_reference_order':order,'schedule':schedule,'preflight_batches':preflight,
        'preview_case_ids':[c['id'] for c in parent['cases']], 'independent_replay_ids':parent['independent_replay_case_ids'],
        'reconstruction_weights':WEIGHTS,'normalizers':NORMALIZERS,'regression_barrier_coefficient':5.,
        'optimizer':{'name':'Adam','lr':1e-5,'betas':[.9,.999],'eps':1e-8,'weight_decay':1e-5,'gradient_clip_L2':1.},
        'parameter_tensors':3,'parameter_elements':147456,'maximum_updates':50,'snapshots':[0,50],'updates_per_full_epoch':781,
        'budgets':BUDGETS,'scientific_thresholds':{'early_structure_gain':.01,'final_structure_gain':.1,
            'MSE_regression_tolerance':1e-12,'SSIM_ArcFace_regression_tolerance':1e-6,'brightness_fraction_maximum':.2},
        'raw_storage':'lossless float32 byte planes; candidate XOR to original; all3905 retained in both snapshots',
        'epoch_study_next':'Compare additional epochs1,2,5 with maximum5 only after audited capacity and useful native development gains; no automatic continuation',
        'manual_tmux_required':True,'stage50_does_not_qualify_epochs_or_model':True,'native_DEV_or_reserved_final_used':False,
        'automatic_follow_on':False,'app_promotion':False,'goal_complete':False,'failed_run_resume_allowed':False,
        'native_unpaired_evidence_stays_separate':True,'no_ethnicity_or_Zamboanga_performance_claim':True,
        'provenance_terms_exposure_overlap_inherited':True,'historical_full_subject_overlap_unconfirmed':True,
        'all_seven_completion_families_still_required':True,'original_failure_evidence_retained':True}
    p['assets_sha256']={f.relative_to(OUT).as_posix():sha(f) for f in sorted(OUT.rglob('*')) if f.is_file()}
    extra=evidence+['outputs/cctv_dgp_profile_batches_vm_v31/protocol.json','outputs/cctv_dgp_mixed_vm_v9_r2/mixed_protocol_v9.json',
        'scripts/prepare_cctv_dgp_head4_capacity_vm_v1.py','scripts/verify_cctv_dgp_head4_capacity_packet_v1.py',
        'scripts/audit_cctv_dgp_head4_capacity_return_v1.py','scripts/verify_cctv_dgp_head4_capacity_initialization_v1.py']+['scripts/'+n for n in NEW]
    p['local_sources']={**parent['local_source_bindings'],**{n:sha(ROOT/n) for n in extra}}
    for n in p['local_sources']:
        if n.endswith('.py'):ast.parse((ROOT/n).read_text(),feature_version=(3,10))
    validate(p);write(OUT/'protocol.json',p);pin=sha(OUT/'protocol.json')
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for f in sorted(OUT.rglob('*')):
            if f.is_file():tar.add(f,arcname=NAME+'/'+f.relative_to(OUT).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(digest+'  '+archive.name+'\n')
    manual=ROOT/'CCTV_DGP_HEAD4_CAPACITY_V1_VM.md'
    with manual.open('x',encoding='utf-8',newline='\n') as f:f.write(guide(pin,digest,archive.stat().st_size))
    write(PREP/'preparation.json',{'complete':True,'protocol_sha256':pin,'packet_sha256':digest,'packet_bytes':archive.stat().st_size,
        'assets':len(p['assets_sha256']),'local_source_bindings':len(p['local_sources']),'TRAIN_assets':5467,
        'manual_guide_sha256':sha(manual),'all_neural_or_training_calls':0,'VM_launched':False,
        'independent_packet_audit_pending':True,'goal_complete':False,'seconds':time.monotonic()-start})
    print({'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'bytes':archive.stat().st_size,'VM_launched':False},flush=True)


if __name__=='__main__':main()
