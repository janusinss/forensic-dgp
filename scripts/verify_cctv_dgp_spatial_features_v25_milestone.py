"""Independent saved evidence/history/runbook readback; zero model calls."""
import ast
import hashlib
import json
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_spatial_features_v25_preparation_milestone'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    start=time.monotonic();m=read(OUT/'milestone.json')
    for n,d in m['new_evidence_sha256'].items():
        p=(ROOT/n).resolve();assert p.is_relative_to(ROOT) and sha(p)==d,n
    prior_path=ROOT/'outputs/dgp_degraded_detail_v24_audit_and_architecture_milestone/milestone.json'
    assert sha(prior_path)==m['previous_milestone_sha256'];prior=read(prior_path)
    for n,d in prior['new_evidence_sha256'].items():assert sha(ROOT/m['previous434_original_locations'].get(n,n))==d,n
    histories=0
    for n,path in m['previous434_original_locations'].items():
        old=(ROOT/path).read_bytes();current=(ROOT/n).read_bytes();split=old.index(b'\n')+1
        assert current.startswith(old[:split]) and current.endswith(old[split:]),n
        prefix=current[split:len(current)-len(old[split:])].decode('utf-8')
        assert 'V25 transfer verified' in prefix and 'Goal active/incomplete' in prefix and 'original' in prefix
        histories+=1
    assert b'VM storage cleanup completed; future VM use preserved' in (ROOT/'PROJECT_HANDOFF.md').read_bytes()
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';assert sha(app_path)==m['app_record_sha256'];app=read(app_path)
    app_count=0
    for n,d in {**app['sources_sha256'],**app['evidence_sha256']}.items():assert sha(ROOT/n)==d,n;app_count+=1
    folder=ROOT/'outputs/cctv_dgp_spatial_features_v25_preparation'
    evidence=[read(folder/n) for n in ['preparation.json','independent_execution_audit.json','independent_transfer_preparation_audit.json']]
    assert all(e['complete'] for e in evidence)
    pin=m['protocol_sha256'];bundle=ROOT/'outputs/cctv_dgp_spatial_features_vm_v25';p=read(bundle/'protocol.json')
    assert sha(bundle/'protocol.json')==pin
    for key in ['cases','references','budgets','prospective_gates','fresh_DGP_parity_cases']:
        assert p[key]==read(ROOT/'outputs/cctv_dgp_degraded_detail_vm_v24/protocol.json')[key],key
    plan=read(folder/'return_audit_plan.json')
    assert plan['protocol_sha256']==pin and not plan['return_received'] and plan['planned_local_backward_calls']==plan['planned_local_optimizer_updates']==0
    for n,d in plan['source_bindings_sha256'].items():assert sha(ROOT/n)==d,n
    old_audit=ast.parse((ROOT/'scripts/audit_cctv_dgp_degraded_detail_v24.py').read_text())
    constants={n.targets[0].id:n.value.value for n in old_audit.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and isinstance(n.value,ast.Constant)}
    replay=plan['saved_feature_head_replay']
    assert replay['raw_maximum_tolerance']==constants['HEAD_RAW_TOLERANCE']
    assert replay['original_recognizer_vector_tolerance']==constants['RECOGNIZER_VECTOR_TOLERANCE']
    assert replay['original_recognizer_cosine_tolerance']==constants['RECOGNIZER_COSINE_TOLERANCE']
    assert replay['fixed_float64_metric_tolerance']==1e-9
    runbook=(ROOT/'CCTV_DGP_SPATIAL_FEATURES_V25_VM.md').read_text(encoding='utf-8')
    assert [int(v) for v in re.findall(r'^([1-5])\. ',runbook,re.M)]==list(range(1,6))
    commands=re.findall(r'^gcloud compute scp (.+)$',runbook,re.M);assert len(commands)==5
    for command in commands:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in command
        remote=re.findall(r'"(janusdominic0@forensic-dgp-thesis:[^"]+)"',command);assert len(remote)==1
    expected_archive='cctv-dgp-spatial-features-v25-execution.tar.gz'
    assert expected_archive in commands[0] and expected_archive+'.sha256' in commands[1]
    for command,suffix in zip(commands[2:],['results.tar.gz','results.tar.gz.sha256','export.json']):
        assert '/home/janusdominic0/cctv-dgp-spatial-features-v25-'+suffix in command and command.endswith('"."')
    assert runbook.count(pin)==3 and 'tmux new-session -A -s dgp_spatial_features_v25' in runbook
    assert 'test ! -e ~/forensic-dgp/cctv_dgp_spatial_features_vm_v25' in runbook
    assert 'source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate' in runbook
    script=(bundle/'scripts/run_v25.sh').read_text();assert 'scripts/cctv_dgp_spatial_features_v25_vm.py' in script
    assert '2100s' in script and '150s' in script and 'PIPESTATUS[0]' in script
    interface=read(ROOT/'outputs/cctv_dgp_spatial_features_v25_interface/results.json')
    assert interface['counts']=={'DGP_forwards':4,'head_forwards':5} and not interface['learned_capacity_or_quality_proven']
    assert m['goal_status']=='active' and not m['goal_complete'] and not m['capacity_or_usefulness_pass'] and m['VM_run_pending']
    result={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),
        'milestone_sha256':sha(OUT/'milestone.json'),'current_bindings_verified':len(m['new_evidence_sha256']),
        'previous434_bindings_verified':len(prior['new_evidence_sha256']),'app22_bindings_verified':app_count,
        'original_history_bodies_preserved':histories,'concurrent_maintenance_preserved':True,
        'all_original_capacity_gates_and_replay_tolerances_retained':True,'five_manual_steps_and_separate_scp_commands_verified':True,
        'VM_training_and_real_return_review_pending':True,'neural_calls':0,'backward_calls':0,'optimizer_updates':0,
        'VM_actions':False,'app_promotion':False,'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-start}
    with (OUT/'independent_readback.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
