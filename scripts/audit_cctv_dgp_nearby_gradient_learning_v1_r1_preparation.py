"""Independent archive, dependency, guard, finite-scope and hash-only CLI audit."""
import ast
from datetime import datetime,timezone
import hashlib
import json
import subprocess
import sys
import tarfile

from import_cctv_dgp_nearby_gradient_learning_v1_r1_return import ROOT,OUT,NAME,STEM,PIN,PARENT,RESPONSE,RETURN,ANALYSIS,member_names,read,sha,write


def main():
    packet=OUT/NAME;p=read(packet/'protocol.json');prepared=read(OUT/'cctv_dgp_nearby_gradient_learning_v1_r1_preparation.json')
    assert prepared['complete'] and prepared['manual_UNRUN'] and not prepared['VM_launch']
    assert PIN==prepared['protocol_sha256']==sha(packet/'protocol.json') and len(PIN)==64
    assert sha(ROOT/'scripts/import_cctv_dgp_nearby_gradient_learning_v1_r1_return.py')==prepared['immutable_importer_sha256']
    assert not RETURN.exists() and not ANALYSIS.exists() and not (packet/'outputs').exists() and not (packet/'review.log').exists()
    archive=OUT/(STEM+'-execution.tar.gz');assert sha(archive)==prepared['archive_sha256'] and archive.stat().st_size==prepared['archive_bytes']
    assert (OUT/(archive.name+'.sha256')).read_bytes()==(prepared['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    bindings={**p['assets_sha256'],'protocol.json':PIN}
    with tarfile.open(archive,'r:gz') as tar:
        members=member_names(tar.getmembers());assert set(members)==set(bindings)
        for n,m in members.items():
            assert sha(packet/n)==bindings[n] and (packet/n).stat().st_size==m.size
            with tar.extractfile(m) as f:assert hashlib.sha256(f.read()).hexdigest()==bindings[n]
    old=read(PARENT/'protocol.json');source=OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_vm'
    for folder,files in [(PARENT,p['parent_required_sha256']),(RESPONSE,p['response_required_sha256']),(source,p['source_required_sha256'])]:
        assert all(sha(folder/n)==h for n,h in files.items())
    for name,origin in p['source_origins'].items():
        path=ROOT/origin['source_path'];assert sha(path)==origin['source_sha256']
        if origin['exact_copy']:assert sha(packet/name)==sha(path)
        else:assert name=='conditioning_model.py' and (packet/name).read_text()==path.read_text().replace('cctv_dgp_raw_conditioning_learning_v1_r1_vm',NAME)
    assert p['case_ids']==old['preview_case_ids']
    assert p['reference_blocks']==read(source/'protocol.json')['reference_blocks']
    assert set(p['case_ids'])=={cid for b in p['reference_blocks'] for cid in b['case_ids']}
    blocks=p['reference_blocks'];assert len(blocks)==20 and len(p['case_ids'])==p['paired_TRAIN_cases']==100
    assert len({b['reference'] for b in blocks})==20 and all(len(b['case_ids'])==5 for b in blocks)
    assert sorted({b['source'] for b in blocks})==['dataset/asian_faces','dataset/thumbnails128x128']
    assert all(sum(b['source']==s for b in blocks)==10 for s in {b['source'] for b in blocks})
    assert p['sampling_sites']==['initial','plus02','minus02'] and len(p['raw_gradient_groups'])==32
    assert p['gradient_rounds']*len(blocks)*16==p['maximum_gradient_queries']==960
    visits=len(p['sampling_sites'])+p['maximum_trials_per_update']+1
    assert (visits+p['gradient_rounds'])*len(blocks)==p['model_forward_cap']==200
    assert (2*visits+p['gradient_rounds'])*len(blocks)==p['recognizer_forward_cap']==340
    assert p['maximum_accepted_updates']==p['direction_solves']==1 and p['maximum_trials_per_update']==3
    assert p['maximum_derivative_sampling_displacements']==2 and p['full_epochs']==p['native_pixels_used']==p['final_identity_pixels']==0
    assert p['conditioning_parameters']==2580864 and len(p['conditioning_names'])==12
    assert p['preserved_scientific_thresholds']==old['scientific_thresholds']
    assert p['preserved_scientific_thresholds']==dict(early_structure_gain=.01,final_structure_gain=.1,MSE_regression_tolerance=1e-12,SSIM_ArcFace_regression_tolerance=1e-6,brightness_fraction_maximum=.2)
    assert p['maximum_run_relative_parameter_distance']==.001 and p['maximum_rate']==.02 and p['backtrack_factors']==[1.,.5,.25]
    assert p['solver']['inequality_rows']==96 and p['solver']['no_ridge_or_loss_removal'] and p['solver']['feature_fraction']==1. and p['solver']['other_loss_fraction']==.1
    assert p['worker_seconds']==1500 and p['measurement_seconds']==1350 and p['gradient_round_seconds']==600 and p['solver_seconds']==60
    assert p['external_seconds']+p['export_external_seconds']+2*p['kill_grace_seconds']==2040
    assert p['minimum_free_bytes']==9*1024**3 and p['disk_reserve_bytes']==1024**3 and p['output_byte_cap']==3584*1024**2
    assert p['peak_VRAM_bytes']==20*1024**3 and p['peak_RSS_bytes']==12*1024**3
    assert not p['PNG_surrogate_used_for_optimization'] and not p['automatic_follow_on'] and not p['app_promotion'] and not p['failed_run_resume_allowed']
    assert p['original_checkpoint_sha256']=='646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    parsed=[]
    for f in packet.rglob('*.py'):
        ast.parse(f.read_text(encoding='utf-8'),feature_version=(3,10));parsed.append(f.relative_to(packet).as_posix())
    worker=(packet/'scripts/cctv_dgp_nearby_gradient_learning_v1_r1_vm.py').read_text()
    assert worker.count('direction,certificate=solve_direction(vectors,rows)')==1
    assert 'for site in p[\'sampling_sites\']:' in worker and 'current_snapshot=evaluate(site)' in worker
    assert 'np.array_equal(raw,expected_raw)' in worker and 'snapshot=evaluate(label)' in worker
    assert "(out/label).mkdir();save_array(out/label/'conditioning.npy',proposal)" in worker
    assert "load_values(initial);restore_rng(initial_RNG)" in worker and 'accepted_component_full_state.pth' in worker
    assert "snapshot=dict(label=label" in worker and 'exact_controls=exact_controls' in worker
    assert '.backward(' not in worker and '.step(' not in worker
    contract=(packet/'conditioning_contract.py').read_text()
    assert 'q @ z - target' in contract and 'len(vectors) == len(records) == 96' in contract and 'loss_removal=True' in contract
    assert '96 * np.finfo(np.float64).eps' in contract and 'np.all(-dots / losses >= fractions - 1e-6)' in contract
    shell=(packet/'scripts/run_learning.sh').read_bytes()
    assert b'\r' not in shell and b'1530s' in shell and b'450s' in shell and shell.count(b'--kill-after=30s')==2
    assert b'test ! -e review.log' in shell and b'test -n "${TMUX:-}"' in shell
    fixture=read(OUT/'cctv_dgp_nearby_gradient_learning_v1_saved_fixture_r1.json')
    assert fixture['complete'] and fixture['saved_cases']==100 and fixture['RAW_PNG_rows_replayed']==600
    assert fixture['preserved_PNG_failure_count']==5 and fixture['forged_failure_erasure_rejected'] and fixture['initial_and_rollback_exact_group_scores']
    assert fixture['return_auditor_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_nearby_gradient_learning_v1_return.py')
    repair=read(packet/'evidence/startup_root_review.json')
    assert repair['original_return_auditor_sha256']==fixture['return_auditor_sha256']
    assert repair['corrected_return_auditor_sha256']==sha(ROOT/'scripts/audit_cctv_dgp_nearby_gradient_learning_v1_r1_return.py')
    new=(ROOT/'scripts/audit_cctv_dgp_nearby_gradient_learning_v1_r1_return.py').read_text()
    assert new.replace('from import_cctv_dgp_nearby_gradient_learning_v1_r1_return import','from import_cctv_dgp_nearby_gradient_learning_v1_return import')==(ROOT/'scripts/audit_cctv_dgp_nearby_gradient_learning_v1_return.py').read_text()
    assert fixture['snapshot_auditor_sha256']==sha(ROOT/'scripts/cctv_dgp_nearby_snapshot_audit_v1.py')
    assert read(OUT/'cctv_dgp_nearby_gradient_learning_v1_source_paths_r1.json')['return_auditor_sha256']==fixture['return_auditor_sha256']
    assert all(sha(ROOT/n)==h for n,h in prepared['preparation_evidence_sha256'].items())
    regressions={}
    for suffix in ['contract','certificate','guard']:
        f=ROOT/(f'scripts/check_cctv_dgp_nearby_gradient_learning_v1_{suffix}.py' if suffix!='guard' else 'scripts/check_cctv_dgp_nearby_gradient_learning_v1_r1_guard.py')
        result=subprocess.run([sys.executable,'-B',str(f)],capture_output=True,text=True,timeout=45)
        assert result.returncode==0,result.stdout+result.stderr
        regressions[suffix]=dict(checker_sha256=sha(f),exit_code=result.returncode,output=result.stdout)
    result=subprocess.run([sys.executable,'-B',str(packet/'scripts/cctv_dgp_nearby_gradient_learning_v1_r1_vm.py'),
        '--root',str(packet),'--parent-root',str(PARENT),'--response-root',str(RESPONSE),'--protocol-sha',PIN,'--verify-transfer'],capture_output=True,text=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    receipt=ast.literal_eval(result.stdout.strip());assert isinstance(receipt,dict)
    assert receipt['complete'] and receipt['packet_assets']==len(p['assets_sha256']) and receipt['parent_assets']==len(p['parent_required_sha256'])
    assert receipt['cases']==100 and receipt['maximum_accepted_updates']==1 and receipt['maximum_gradient_queries']==960 and receipt['actual_neural_calls']==0
    protected=read(OUT/'cctv_dgp_recognizer_precision_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(OUT/'cctv_dgp_nearby_gradient_learning_v1_r1_transfer_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),protocol_sha256=PIN,archive_sha256=sha(archive),packet_bindings=len(bindings),dependency_bindings=prepared['dependency_bindings'],
        code_files_python310_parsed=parsed,maximum_forwards_derived=dict(restoration=200,recognizer=340),maximum_gradient_queries_derived=960,
        groups_per_site=32,all96_inequality_rows_retained=True,gradient_sites=3,proposal_cap=3,accepted_cap=1,
        measured_preparation_regressions=regressions,hash_only_worker_receipt=receipt,
        hash_only_free_space_observation_is_Windows_not_VM=True,actual_neural_or_training_calls=0,manual_UNRUN=True,
        Windows_manual_guard_passed=True,unsafe_archive_sets_rejected=14,saved_600_RAW_PNG_rows_replayed=True,
        original_gates_and_12_protected_files_unchanged=True,external_ceiling_seconds=2040,automatic_VM_launch=False,
        prospective_return_auditor_sha256=repair['corrected_return_auditor_sha256'],unchanged_saved_fixture_reused_with_exact_namespace_bridge=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,packet_bindings=len(bindings),maximum_gradient_queries=960,maximum_accepted_updates=1,actual_neural_calls=0,manual_UNRUN=True))


if __name__=='__main__':main()
