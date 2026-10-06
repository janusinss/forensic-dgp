"""Independent milestone, failure, preserved history, packet and manual-command readback."""
import hashlib
import json
from pathlib import Path
import re
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_spatial_features_v25_audit_and_gradient_diagnostic_milestone'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    start=time.monotonic();m=read(OUT/'milestone.json')
    for name,digest in m['new_evidence_sha256'].items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and sha(p)==digest,name
    oldpath=ROOT/'outputs/dgp_spatial_features_v25_preparation_milestone/milestone.json'
    assert sha(oldpath)==m['previous_milestone_sha256'];old=read(oldpath)
    for name,digest in old['new_evidence_sha256'].items():assert sha(ROOT/m['previous513_original_locations'].get(name,name))==digest,name
    history=0
    for name,before in m['previous513_original_locations'].items():
        original=(ROOT/before).read_bytes();current=(ROOT/name).read_bytes();split=original.index(b'\n')+1
        assert current.startswith(original[:split]) and current.endswith(original[split:]),name
        prefix=current[split:len(current)-len(original[split:])].decode('utf-8')
        assert 'V25 audited failure' in prefix and 'zero-update fixed-state gradient diagnostic' in prefix
        assert '0.0282235213%' in prefix and 'Goal active/incomplete' in prefix and 'historical' in prefix
        history+=1
    assert b'VM storage cleanup completed; future VM use preserved' in (ROOT/'PROJECT_HANDOFF.md').read_bytes()
    apppath=ROOT/'outputs/dgp_app_v3_integration_record.json';assert sha(apppath)==m['app_record_sha256'];app=read(apppath);appcount=0
    for name,digest in {**app['sources_sha256'],**app['evidence_sha256']}.items():assert sha(ROOT/name)==digest,name;appcount+=1
    v25=read(ROOT/'outputs/cctv_dgp_spatial_features_v25_independent_audit.json')
    assert v25['complete'] and v25['VM_failure_present'] and not v25['VM_training_result_present'] and not v25['early_structure_stop']['pass']
    assert v25['early_structure_stop']['minimum']==.01 and m['early_structure_gain_percent']==100*v25['early_structure_stop']['relative_feature_error_gain']
    assert v25['source_assets_verified']==235 and v25['returned_files_verified']==628
    visual=read(ROOT/'outputs/cctv_dgp_spatial_features_v25_diagnostic/visual_review.json')
    assert visual['complete'] and visual['cases_reviewed']==50 and visual['exact_cells']==200
    preparation=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_preparation'
    packet=read(preparation/'preparation.json');audit=read(preparation/'independent_packet_audit.json');check=read(preparation/'independent_preparation_readback.json')
    assert all(x['complete'] for x in [packet,audit,check]) and check['unsafe_packet_source_matrix_tests_passed']==9
    pin=m['gradient_diagnostic_protocol_sha256'];bundle=ROOT/'outputs/cctv_dgp_v25_gradient_diagnostic_v1_vm'
    assert sha(bundle/'protocol.json')==pin==packet['protocol_sha256']==audit['protocol_sha256']==check['protocol_sha256']
    p=read(bundle/'protocol.json');assert p['optimizer_updates']==0 and not p['new_training_recipe'] and p['snapshots']==[0,50]
    assert p['component_gradient_calls']==140 and p['head_batches']==20 and p['recognizer_forwards']==120
    for name,digest in p['assets_sha256'].items():assert sha(bundle/name)==digest,name
    original=ROOT/'outputs/cctv_dgp_spatial_features_vm_v25';assert sha(original/'protocol.json')==p['closed_V25_protocol_sha256']
    for name,digest in read(original/'protocol.json')['assets_sha256'].items():assert sha(original/name)==digest,name
    assert p['head_states']['0']==v25['VM_initial_state'] and p['head_states']['50']==v25['stopped_head_state']
    runbook=(ROOT/'CCTV_DGP_V25_GRADIENT_DIAGNOSTIC_V1_VM.md').read_text(encoding='utf-8')
    assert [int(v) for v in re.findall(r'^([1-5])\. ',runbook,re.M)]==list(range(1,6))
    commands=re.findall(r'^gcloud compute scp (.+)$',runbook,re.M);assert len(commands)==5
    for command in commands:
        assert '--project=forensic-dgp-thesis --zone=us-central1-a' in command
        assert len(re.findall(r'"(janusdominic0@forensic-dgp-thesis:[^"]+)"',command))==1
    name='cctv-dgp-v25-gradient-diagnostic-v1'
    assert name+'-execution.tar.gz' in commands[0] and name+'-execution.tar.gz.sha256' in commands[1]
    for command,suffix in zip(commands[2:],['results.tar.gz','results.tar.gz.sha256','export.json']):
        assert '/home/janusdominic0/'+name+'-'+suffix in command and command.endswith('"."')
    assert runbook.count(pin)==3 and 'tmux new-session -A -s dgp_v25_gradient_diagnostic_v1' in runbook
    assert 'test ! -e ~/forensic-dgp/cctv_dgp_v25_gradient_diagnostic_v1_vm' in runbook
    assert 'source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate' in runbook
    assert '--verify-transfer' in runbook and 'bash scripts/run_gradient.sh '+pin in runbook
    assert 'test -d ~/forensic-dgp/cctv_dgp_spatial_features_vm_v25/outputs/frozen_DGP_features' in runbook
    assert not m['new_training_recipe'] and m['actual_L4_gradient_diagnostic_pending'] and m['goal_status']=='active' and not m['goal_complete']
    assert m['local_gradient_calls']==m['local_backward_calls']==m['local_optimizer_updates']==0
    result={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),
        'milestone_sha256':sha(OUT/'milestone.json'),'new_bindings_verified':len(m['new_evidence_sha256']),
        'previous513_bindings_verified':len(old['new_evidence_sha256']),'original_history_bodies_preserved':history,
        'app22_bindings_verified':appcount,'concurrent_maintenance_preserved':True,'V25_failure_and_quality_gates_retained':True,
        'all50_training_cases_visually_reviewed':True,'finite_diagnostic_packet_verified':True,
        'five_manual_steps_and_separate_PuTTY_downloads_verified':True,'actual_L4_gradient_diagnostic_pending':True,
        'neural_calls':0,'local_gradient_calls':0,'optimizer_updates':0,'VM_actions':False,'app_promotion':False,
        'goal_status':'active','goal_complete':False,'seconds':time.monotonic()-start}
    with (OUT/'independent_readback.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
