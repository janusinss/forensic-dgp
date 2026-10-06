"""Independent hash/doc/command readback. No model, optimizer, VM or browser calls."""
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shlex
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/dgp_detail_skip_v23_audit_and_degraded_detail_v24_milestone'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def safe(name):
    part=PurePosixPath(name)
    assert part.parts and not part.is_absolute() and '\\' not in name and ':' not in name
    assert all(p not in ('','.','..') for p in part.parts)
    file=ROOT.joinpath(*part.parts).resolve();assert file.is_relative_to(ROOT)
    return file


def main():
    started=time.monotonic();m=read(OUT/'milestone.json');plan=read(OUT/'plan.json')
    assert m['complete'] and m['goal_status']=='active' and not m['goal_complete']
    assert m['local_optimizer_updates']==m['local_backward_calls']==m['assistant_VM_cloud_actions']==0
    for key in ['model_changed_in_current_app','app_design_source_changed','historical_gates_waived',
                'native_reserved_covering_COFW_test_pixels_enter_new_pilot','historical_app_regressions_or_Playwright_rerun',
                'native_usefulness_qualified','covering_families_qualified','independent_final_review_complete',
                'V23_original_early_gate_pass','V23_final800_gate_evaluated','V23_visible_structure_gain',
                'V24_VM_gradient_training_capacity_quality_verified','canonical_app_parity_complete_for_new_head']:
        assert m[key] is False,key
    assert len(m['new_evidence_sha256'])==m['new_bindings']
    for name,digest in m['new_evidence_sha256'].items():assert sha(safe(name))==digest,name
    previous_path=ROOT/'outputs/dgp_detail_prior_v22_r1_audit_and_skip_v23_milestone/milestone.json'
    older_path=ROOT/'outputs/dgp_structure_and_detail_pilot_milestone_v21_v22/milestone.json'
    assert sha(previous_path)==m['previous_milestone_sha256'] and sha(older_path)==m['older_milestone_sha256']
    previous=read(previous_path);older=read(older_path)
    assert len(previous['new_evidence_sha256'])==m['previous644_bindings_verified']==644
    assert len(older['new_evidence_sha256'])==m['older560_bindings_verified']==560
    for p,mapping in [(previous,m['previous644_document_locations']),(older,m['older560_document_locations'])]:
        for name,digest in p['new_evidence_sha256'].items():assert sha(safe(mapping.get(name,name)))==digest,name
    app_path=ROOT/'outputs/dgp_app_v3_integration_record.json';assert sha(app_path)==m['app_record_sha256']
    app=read(app_path);bindings={**app['sources_sha256'],**app['evidence_sha256']}
    assert len(bindings)==m['app22_bindings_verified']==22
    for name,digest in bindings.items():assert sha(safe(name))==digest,name
    for name,digest in plan['before_docs_sha256'].items():
        file=OUT/'before_docs'/name;assert sha(file)==digest
        old=file.read_bytes();new=(ROOT/name).read_bytes();split=old.index(b'\n')+1
        assert new.startswith(old[:split]) and new.endswith(old[split:]),name
    reported=read(ROOT/'outputs/cctv_dgp_detail_skip_v23_reported_stop_v1/status_update_receipt.json')
    for name,digest in reported['bindings_sha256'].items():
        path=OUT/'before_docs'/name if name in plan['before_docs_sha256'] else safe(name)
        assert sha(path)==digest,'Prior reported-stop status changed: '+name
    audit=read(ROOT/'outputs/cctv_dgp_detail_skip_v23_independent_audit.json')
    visual=read(ROOT/'outputs/cctv_dgp_detail_skip_v23_diagnostic/visual_review.json')
    loss=read(ROOT/'outputs/cctv_dgp_detail_skip_v23_loss_audit_v1/independent_saved_loss_audit.json')
    assert audit['complete'] and audit['returned_files_verified']==370 and audit['source_assets_verified']==209
    assert audit['counts']=={'head_forwards':100,'recognizer_forwards':110,'raw_PNG_pairs':100,'metric_rows':100,'complete_snapshots':2}
    assert audit['training_progress_receipt']['updates']==50 and audit['training_progress_receipt']['backwards']==51
    assert not audit['early_structure_stop']['pass'] and not audit['necessary_capacity_pass'] and not audit['VM_training_result_present']
    assert m['V23_early_gain_percent']==audit['early_structure_stop']['relative_feature_error_gain']*100
    cases=read(ROOT/'outputs/cctv_dgp_detail_skip_vm_v23/protocol.json')['cases']
    seen=[]
    assert visual['complete'] and len(visual['sheets'])==10
    for sheet in visual['sheets']:
        assert sheet['all_five_rows_reviewed'] and len(sheet['cases'])==5
        assert sha(ROOT/'outputs/cctv_dgp_detail_skip_v23_diagnostic'/sheet['file'])==sheet['sha256'];seen+=sheet['cases']
    assert sorted(seen)==sorted(c['id'] for c in cases) and len(set(seen))==50
    assert loss['complete'] and loss['case_states_verified']==100 and loss['source_bindings_verified']==174
    assert loss['clear_contribution_fraction_of_net_objective_reduction']>1 and loss['degraded_contribution']<0
    assert m['V23_clear_contribution_fraction']==loss['clear_contribution_fraction_of_net_objective_reduction']
    bundle=ROOT/'outputs/cctv_dgp_degraded_detail_vm_v24';p=read(bundle/'protocol.json')
    assert sha(bundle/'protocol.json')==m['V24_protocol_sha256']
    assert sha(ROOT/'outputs/cctv-dgp-degraded-detail-v24-execution.tar.gz')==m['V24_archive_sha256']
    assert len(p['assets_sha256'])==221 and len(p['cases'])==50 and p['design']['updates']==800 and p['design']['epochs']==80
    for name,digest in p['assets_sha256'].items():
        file=(bundle/name).resolve();assert file.is_relative_to(bundle) and sha(file)==digest
    preparation=ROOT/'outputs/cctv_dgp_degraded_detail_v24_preparation'
    check=read(preparation/'independent_preparation_audit.json')
    shell=read(preparation/'shell_and_loss_contract_receipt.json');tests=read(preparation/'pure_objective_tests.json')
    assert check['complete'] and check['entire_runtime_AST_equal_except_allowed_loss_setup_flags_names'] and check['head_and_quality_gates_unchanged']
    assert check['regular_archive_members']==223 and check['fixed_filter_calls']=={'fixed_high_pass':100,'head_forward':0}
    assert check['maximum_baseline_row_difference']==0 and check['optimizer_updates']==check['backward_calls']==check['recognizer_DGP_forwards']==0
    assert shell['complete'] and shell['Bash_syntax_exit_code']==0 and tests['tests_passed']==8
    assert sha(ROOT/'tests/test_cctv_dgp_degraded_objective_v24.py')==shell['objective_tests_sha256']
    assert sha(ROOT/'scripts/cctv_dgp_degraded_objective_v24.py')==shell['helper_sha256']
    assert sha(ROOT/'scripts/verify_cctv_dgp_degraded_detail_v24_preparation.py')==check['checker_sha256']
    assert sha(preparation/'verification_plan_r1.json')==check['plan_sha256']
    book=(ROOT/'CCTV_DGP_DEGRADED_DETAIL_V24_VM.md').read_text(encoding='utf-8')
    blocks=re.findall(r'```(?:cmd|bash)\n(.*?)```',book,flags=re.S)
    assert len(blocks)==5 and all(re.search(r'^'+str(n)+r'\. ',book,re.M) for n in range(1,6))
    windows_cd='cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"'
    upload=blocks[0].strip().splitlines();downloads=blocks[4].strip().splitlines()
    assert upload[0]==downloads[0]==windows_cd and len(upload)==2 and len(downloads)==4
    prefix=['gcloud','compute','scp','--project=forensic-dgp-thesis','--zone=us-central1-a']
    up=shlex.split(upload[1]);assert up[:5]==prefix
    assert up[5:]==['cctv-dgp-degraded-detail-v24-execution.tar.gz','cctv-dgp-degraded-detail-v24-execution.tar.gz.sha256',
                      'janusdominic0@forensic-dgp-thesis:/home/janusdominic0/']
    for line,name in zip(downloads[1:],['cctv-dgp-degraded-detail-v24-results.tar.gz','cctv-dgp-degraded-detail-v24-results.tar.gz.sha256','cctv-dgp-degraded-detail-v24-export.json']):
        args=shlex.split(line);assert args[:5]==prefix and len(args)==7
        assert args[5]=='janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+name and args[6]=='.'
    assert 'sha256sum -c cctv-dgp-degraded-detail-v24-execution.tar.gz.sha256' in blocks[1]
    assert 'test ! -e ~/forensic-dgp/'+bundle.name in blocks[1] and 'tar -xzf cctv-dgp-degraded-detail-v24-execution.tar.gz -C ~/forensic-dgp' in blocks[1]
    assert blocks[2].strip()=='tmux new-session -A -s dgp_degraded_detail_v24'
    pin=m['V24_protocol_sha256']
    assert 'cd ~/forensic-dgp/'+bundle.name in blocks[3]
    assert 'python -B -u scripts/cctv_dgp_degraded_detail_v24.py --root . --protocol-sha '+pin+' --preflight' in blocks[3]
    assert 'bash scripts/run_v24.sh '+pin in blocks[3]
    source=(bundle/'scripts/cctv_dgp_degraded_detail_v24.py').read_text()
    assert "name='cctv-dgp-degraded-detail-v24-results.tar.gz'" in source and "home/'cctv-dgp-degraded-detail-v24-export.json'" in source
    for name in ['PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md']:
        text=(ROOT/name).read_text(encoding='utf-8')
        assert 'V23 failed run independently audited; V24 objective experiment prepared' in text
        assert 'Goal active' in text and 'CCTV_DGP_DEGRADED_DETAIL_V24_VM.md' in text
    for name in ['scripts/finalize_cctv_dgp_detail_v23_v24_milestone.py','scripts/verify_cctv_dgp_detail_v23_v24_milestone.py']:
        ast.parse((ROOT/name).read_text(),feature_version=(3,10))
    receipt={'complete':True,'date':'2026-10-06','checker_sha256':sha(Path(__file__)),'milestone_sha256':sha(OUT/'milestone.json'),
        'new_bindings_verified':m['new_bindings'],'previous644_bindings_verified':644,'older560_bindings_verified':560,
        'app22_bindings_verified':22,'history_bodies_exact':4,'prior_reported_stop_bindings_verified':len(reported['bindings_sha256']),
        'all50_case_visual_ledger_complete':True,'five_runbook_steps_and_separate_downloads_verified':True,
        'archive_protocol_and_export_names_verified':True,'quality_or_training_acceptance_not_implied':True,
        'seconds':time.monotonic()-started,'neural_or_training_or_VM_calls':0,'goal_complete':False}
    with (OUT/'independent_milestone_audit.json').open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
