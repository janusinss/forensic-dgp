"""Prepare a distinct, manual finite image probe after the audited V34 diagnostic."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parents[1]
NAME='cctv_dgp_group_guard_probe_v35_r1_vm'
STEM='cctv-dgp-group-guard-probe-v35-r1'
BUNDLE=ROOT/'outputs'/NAME
PREP=ROOT/'outputs/cctv_dgp_group_guard_probe_v35_r1_preparation'
ANALYSIS=ROOT/'outputs/cctv_dgp_group_guard_grad_v34_analysis_v1_r1'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def text_file(path,text):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)


def write(path,value):text_file(path,json.dumps(value,indent=2,allow_nan=False)+'\n')


VERIFY='''def verify(root, pin):
    assert sha(root / 'protocol.json') == pin
    p = read(root / 'protocol.json')
    assert p['format'] == 'own-DGP-all-group-guard-finite-image-probe-v35'
    assert p['optimizer_updates'] == p['committed_trajectory_updates'] == p['new_gradient_queries'] == p['backwards'] == p['epochs'] == 0
    assert p['states'] == [0] and p['candidate_displacement_trials'] == 4
    assert p['variants'] == [{'name':name,'proposal':'projected_restoration','scale':scale} for name,scale in
        [('joint_1',1.),('joint_half',.5),('joint_quarter',.25),('joint_eighth',.125)]]
    assert p['before_outputs'] == 100 and p['trial_outputs'] == 400
    assert p['budgets']['worker_seconds'] == 900 and p['budgets']['external_seconds'] == 930
    check_files(root,p['assets_sha256'])
    return p


def dependencies(root,p):
    guard_root=root.parent/'cctv_dgp_group_guard_grad_v34_vm'
    check_files(guard_root,p['guard_return_sha256'])
    guard_p=read(guard_root/'protocol.json')
    assert sha(guard_root/'protocol.json')==p['guard_protocol_sha256']
    spec=importlib.util.spec_from_file_location('pinned_V34_metadata_only',guard_root/'scripts/cctv_dgp_group_guard_grad_v34_vm.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    basis,parent,active,closed,mixed,helper,probe=module.dependencies(root,guard_p)
    result=read(guard_root/'outputs/results.json')
    assert result['complete'] and result['gradient_queries']==300 and result['optimizer_updates']==result['parameter_updates']==0
    assert not (guard_root/'outputs/failure.json').exists()
    assert basis['cohorts']==p['cohorts'] and basis['parameter_layout']==p['parameter_layout'] and basis['terms']==p['terms']
    assert p['retained_capacity_gates']==guard_p['retained_capacity_gates']
    return root.parent/OLD,basis,parent,active,closed,mixed,helper


'''


TRIALS='''        shipped_theta,direction,guards,geometry=verify_geometry(root,p,sha,read)
        assert np.array_equal(flat(),shipped_theta)
        write(out/'geometry_verification.json',geometry)
        before_receipts,receipts={},[]
        for label,items in cohorts:
            folder=out/f'state0_{label}';folder.mkdir()
            np.save(folder/'theta_before.npy',shipped_theta,allow_pickle=False)
            before_receipts[label]=snapshot(folder/'before',items,0,label,'before')
        for variant in p['variants']:
            clock();candidate.net.load_state_dict(initial,strict=True)
            actual_theta=(shipped_theta.astype(np.float64)-variant['scale']*direction).astype(np.float32)
            assign(actual_theta);assert_frozen(initial)
            progress['candidate_displacement_trials']+=1
            actual_delta=shipped_theta.astype(np.float64)-flat().astype(np.float64)
            for label,items in cohorts:
                folder=out/f'state0_{label}'/variant['name']
                current=snapshot(folder,items,0,label,variant['name'])
                before=before_receipts[label]
                decision=compare_groups(before['groups'],current['groups'])
                comparison={'variant':variant,'preservation_against_original':decision,
                    'incremental_degraded_PNG_structure_gain':1-current['groups']['degraded']['landmark_high_frequency_MSE']/before['groups']['degraded']['landmark_high_frequency_MSE'],
                    'finite_raw_component_change':(np.asarray(current['raw_component_means'])-np.asarray(before['raw_component_means'])).tolist(),
                    'all108_linear_function_changes':(-(guards@actual_delta)).tolist(),
                    'float32_roundoff_norm':float(np.linalg.norm(actual_delta-variant['scale']*direction)),
                    'capacity_qualification':False,'app_promotion':False,'independent_audit_and_visual_review_required':True}
                write(folder/'comparison.json',comparison)
                receipts.append({'state':0,'cohort':label,'variant':variant['name'],'receipt_sha256':sha(folder/'receipt.json')})
            candidate.net.load_state_dict(initial,strict=True)
            assert np.array_equal(flat(),shipped_theta) and state_hash(candidate.net)==basis['original_DGP_state']
        clock();dependencies(root,p);assert_frozen(initial)
        assert progress['candidate_displacement_trials']==4 and progress['raw_outputs']==500
        assert {k:progress[k] for k in p['forward_call_limits']}==p['forward_call_limits']
        assert progress['optimizer_updates']==progress['new_gradient_queries']==progress['backwards']==progress['committed_trajectory_updates']==progress['epochs']==0
        write(out/'results.json',{'complete':True,'protocol_sha256':pin,'seconds':time.monotonic()-started,**progress,
            'receipts':receipts,'peak_allocated_VRAM_bytes':torch.cuda.max_memory_allocated(),
            'original_DGP_state':state_hash(original.net),'recognizer_state':state_hash(identity),
            'new_checkpoint_created':False,'all_trial_states_reset':True,'no_follow_on_training':True,
            'native_or_reserved_used':False,'app_promotion':False,'goal_complete':False})
        print(json.dumps({'complete':True,'disposable_trials':4,'optimizer_updates':0,'raw_outputs':500,'seconds':time.monotonic()-started}),flush=True)
    except BaseException as error:
        write(out/'failure.json',{'complete':False,'protocol_sha256':pin,'seconds':time.monotonic()-started,**progress,
            'cause':str(error),'traceback':traceback.format_exc(),'resume_permitted':False,
            'new_checkpoint_created':False,'app_promotion':False,'goal_complete':False})
        raise


'''


def make_worker():
    source_path=ROOT/'scripts/cctv_dgp_loss_cone_probe_v33_vm.py'
    original=source_path.read_text(encoding='utf-8')
    assert sha(source_path)=='1b295dc4ffa0266e7965e67943e537453e1e894bd3b5aed7ab96156d5aeb2c71'
    prefix=original[:original.index('def verify(root, pin):')]
    start=original.index('def run(root, p, pin):')
    end=original.index('        before_receipts, receipts, projections = {}, [], []')
    run=original[start:end]
    before="'optimizer_updates': 0, 'committed_trajectory_updates': 0, 'new_gradient_queries': 0,"
    assert run.count(before)==1
    run=run.replace(before,"'candidate_displacement_trials': 0, 'optimizer_updates': 0, 'committed_trajectory_updates': 0, 'new_gradient_queries': 0,")
    run=run.replace('        from cctv_dgp_loss_cone_v33 import project_vectors','        from cctv_dgp_group_guard_geometry_v35 import verify_geometry')
    lo=run.index("        stopped = torch.load(closed / 'outputs/update50/dgp_candidate_v32.pth'")
    hi=run.index('        refs = ',lo)
    run=run[:lo]+run[hi:]
    tail=original[original.index('def export(root, p, pin):'):]
    text=prefix+VERIFY+run+TRIALS+tail
    text=text.replace('cctv_dgp_loss_cone_probe_v33_vm',NAME).replace('cctv-dgp-loss-cone-probe-v33',STEM)
    text=text.replace('cctv_dgp_loss_cone_probe_v33_return/','cctv_dgp_group_guard_probe_v35_return/')
    text=text.replace('V33','V35')
    text=text.replace("'optimizer_updates': terminal['optimizer_updates'], 'committed_trajectory_updates': 0,",
        "'optimizer_updates': terminal['optimizer_updates'], 'candidate_displacement_trials':terminal['candidate_displacement_trials'], 'committed_trajectory_updates': 0,")
    assert 'torch.optim' not in text and 'optimizer.step' not in text and 'autograd.grad' not in text and 'stopped = torch.load' not in text
    ast.parse(text,feature_version=(3,10))
    return text,sha(source_path)


def guide(pin,digest):
    return f'''# V35 finite image probe: five manual steps

V34's independent audit passed:300 saved gradient queries, zero parameter
updates,230 files verified and280 frozen CPU outputs replayed. Original raw
outputs match V33 exactly. Its raw preservation derivatives expose12 harmful
checks per TRAIN subset under the old proposal, all in fixed ArcFace.

V35 tests a changed direction: project the mean of the two audited original
AdamW displacements against all102 non-hinged source/profile preservation
rows plus all6 nonzero existing restoration-loss rows. Independent group
reassembly and full-row KKT/primal checks pass. The zero-at-original4 hinges
are retained in output measurements. This mean displacement is not AdamW
applied to a mean gradient. Source labels establish neither ethnicity nor
CCTV/Zamboanga performance; both subsets are now used for TRAIN design.

**Four reset disposable directions at scales1,1/2,1/4,1/8;100 baseline plus
400 trial images. Zero optimizer updates, new gradient queries, committed
training updates, epochs or new checkpoints.** All trials start from the
original DGP. No V32 continuation or historical pilot runs. The original17
PNG group, source, brightness and1%-at50/10%-at800 gates remain. Linear
geometry cannot prove finite PNG preservation or visible structural usefulness.
Every failed check is exported. No automatic training or app replacement.

Require the existing running L4/g2-standard-4, idle GPU and6GiB free. Estimate
2-6minutes for the new probe,1-3minutes export; not yet timed. Worker900s,
external930s plus30s grace; export300s/external330s plus30s grace;20GiB
allocated VRAM;768MiB uncompressed return. No files are deleted.

Protocol SHA256:{pin}
Execution archive SHA256:{digest}

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
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -f ~/forensic-dgp/cctv_dgp_group_guard_grad_v34_vm/outputs/results.json &&
test ! -e ~/forensic-dgp/{NAME} &&
tar -xzf {STEM}-execution.tar.gz -C ~/forensic-dgp
```

3. Open **tmux**:

```bash
tmux new-session -A -s dgp_probe_v35
```

4. Launch **inside tmux**:

```bash
cd ~/forensic-dgp/{NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/{NAME}.py --root . --protocol-sha {pin} --verify-transfer &&
bash scripts/run_v35_probe.sh {pin}
```

Detach with **Ctrl+B**, release, then **D**. Step3 reconnects.
Expected probe:raw_outputs:500, candidate_displacement_trials:4,
optimizer_updates:0. Complete:true means diagnostic/export completion; it
does not qualify restoration or training. Download failures too. Do not
edit the frozen protocol, resume a trial, or run follow-on training.

5. Download from **Windows Google Cloud SDK Shell** after export:

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{STEM}-export.json" "."
```

The returned images require the prospective independent audit and a bounded
actual review before choosing any further pilot. Native CCTV, reserved final
identities, the app and covering-family failures remain separate and unchanged.
'''


def main():
    assert not BUNDLE.exists() and not PREP.exists()
    audit=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_independent_audit.json')
    a=read(ANALYSIS/'analysis.json');checked=read(ANALYSIS/'independent_analysis_audit.json')
    assert audit['complete'] and audit['diagnostic_complete'] and checked['complete']
    assert checked['analysis_sha256']==sha(ANALYSIS/'analysis.json') and a['joint_projection']['constraint_count']==108
    assert a['joint_projection']['projected_norm']>0 and not a['training_capacity_pass']
    old=read(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_vm/protocol.json')
    guard=read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_vm/protocol.json')
    BUNDLE.mkdir();(BUNDLE/'scripts').mkdir();PREP.mkdir()
    worker,source_hash=make_worker()
    worker_path=ROOT/'scripts'/f'{NAME}.py';text_file(worker_path,worker)
    text_file(BUNDLE/'scripts'/f'{NAME}.py',worker)
    shell=(ROOT/'outputs/cctv_dgp_loss_cone_probe_v33_vm/scripts/run_v33_probe.sh').read_text(encoding='utf-8')
    shell=shell.replace('V33','V35').replace('cctv_dgp_loss_cone_probe_v33_vm',NAME)
    text_file(BUNDLE/'scripts/run_v35_probe.sh',shell)
    for source,destination in [(ROOT/'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py','cctv_dgp_loss_cone_probe_v33_metrics.py'),
        (ROOT/'scripts/cctv_dgp_group_guard_geometry_v35.py','cctv_dgp_group_guard_geometry_v35.py')]:
        with (BUNDLE/destination).open('xb') as stream:stream.write(source.read_bytes())
    for name in ['theta_before.npy','projected_displacement.npy','mean_original_displacement.npy','projection.json']:
        with (BUNDLE/name).open('xb') as stream:stream.write((ANALYSIS/name).read_bytes())
    app={n:h for n,h in guard['local_basis_sha256'].items() if n in read(ROOT/'outputs/completion_input_footprints_comparison_v1_r1/protocol.json')['app_preservation_sha256']}
    assert len(app)==14
    local=[ROOT/'outputs/cctv_dgp_group_guard_grad_v34_independent_audit.json',ANALYSIS/'analysis.json',ANALYSIS/'independent_analysis_audit.json',
        ROOT/'scripts/audit_cctv_dgp_group_guard_probe_v35_r1_return.py',ROOT/'scripts/verify_cctv_dgp_v34_group_analysis_v1_r1.py']
    p={'format':'own-DGP-all-group-guard-finite-image-probe-v35','frozen_UTC':datetime.now(timezone.utc).isoformat(),
        'hypothesis':'Non-hinged group constraints can change the original-state direction while preserving useful structural descent; actual finite images must verify this.',
        'states':[0],'cohorts':guard['cohorts'],'parameter_layout':guard['parameter_layout'],'terms':old['terms'],
        'guard_metrics':guard['guard_metrics'],'constraint_labels':a['constraint_labels'],
        'guard_protocol_sha256':sha(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_vm/protocol.json'),
        'guard_return_sha256':read(ROOT/'outputs/cctv_dgp_group_guard_grad_v34_return_import.json')['files_sha256'],
        'local_basis_sha256':{**app,**{q.relative_to(ROOT).as_posix():sha(q) for q in local}},
        'candidate_displacement_trials':4,'optimizer_updates':0,'committed_trajectory_updates':0,'new_gradient_queries':0,'backwards':0,'epochs':0,
        'before_outputs':100,'trial_outputs':400,'variants':[{'name':name,'proposal':'projected_restoration','scale':scale}
            for name,scale in [('joint_1',1.),('joint_half',.5),('joint_quarter',.25),('joint_eighth',.125)]],
        'proposal_policy':a['proposal_policy'],'all108_nonzero_constraints_retained':True,'retained_capacity_gates':guard['retained_capacity_gates'],
        'same_VM_raw_tolerance':old['same_VM_raw_tolerance'],'CPU_raw_absolute_tolerance':old['CPU_raw_absolute_tolerance'],
        'CPU_PNG_byte_tolerance':old['CPU_PNG_byte_tolerance'],'CPU_vector_absolute_tolerance':old['CPU_vector_absolute_tolerance'],
        'CPU_component_value_tolerance':old['CPU_component_value_tolerance'],'CPU_replay_case_ids':old['CPU_replay_case_ids'],'CPU_replay_outputs':100,
        'forward_call_limits':{'reference_DGP_forwards':20,'candidate_DGP_forwards':100,'recognizer_forwards':320},
        'budgets':{'worker_seconds':900,'external_seconds':930,'kill_grace_seconds':30,'export_seconds':300,'external_export_seconds':330,
            'minimum_free_disk_bytes':6*1024**3,'peak_vram_bytes':20*1024**3,'export_uncompressed_bytes':768*1024**2,'return_files_maximum':2300,'local_audit_seconds':1200},
        'copied_inference_definition_source_sha256':source_hash,'TRAIN_only':True,'source_labels_not_ethnicity':True,
        'both_subsets_used_for_TRAIN_design':True,'VM_execution_started':False,'new_checkpoint_created':False,'native_or_reserved_used':False,'app_promotion':False,'goal_complete':False}
    p['assets_sha256']={q.relative_to(BUNDLE).as_posix():sha(q) for q in sorted(BUNDLE.rglob('*')) if q.is_file()}
    write(BUNDLE/'protocol.json',p);pin=sha(BUNDLE/'protocol.json')
    archive=ROOT/'outputs'/(STEM+'-execution.tar.gz')
    with tarfile.open(archive,'x:gz',compresslevel=3) as tar:
        for q in sorted(BUNDLE.rglob('*')):
            if q.is_file():tar.add(q,arcname=NAME+'/'+q.relative_to(BUNDLE).as_posix(),recursive=False)
    digest=sha(archive);text_file(Path(str(archive)+'.sha256'),digest+'  '+archive.name+'\n')
    text_file(ROOT/'CCTV_DGP_GROUP_GUARD_PROBE_V35_R1_VM.md',guide(pin,digest))
    write(PREP/'preparation_receipt.json',{'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'archive_bytes':archive.stat().st_size,
        'copied_source_sha256':source_hash,'assets':len(p['assets_sha256']),'VM_launches':0,'neural_calls':0,'gradient_calls':0,'optimizer_updates':0,
        'independent_packet_audit_pending':True,'VM_execution_started':False,'app_promotion':False,'goal_complete':False})
    print(json.dumps({'complete':True,'protocol_sha256':pin,'archive_sha256':digest,'bytes':archive.stat().st_size,'assets':len(p['assets_sha256'])}))


if __name__=='__main__':main()
