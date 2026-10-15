"""Independent closure of retained failure, storage proof and unrun R1 handoff."""
import ast
from datetime import datetime, timezone
import importlib.util
import re
from import_cctv_dgp_bank_group_gradients_v1_return_r1 import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    value=importlib.util.module_from_spec(spec);spec.loader.exec_module(value)
    return value


def main():
    final=ANALYSIS/'publication_audit.json';assert not final.exists()
    pub=read(ANALYSIS/'publication.json');assert pub['complete'] and pub['goal_status']=='active' and not pub['goal_complete']
    assert pub['fresh_manual_recovery_prepared_unrun'] and not pub['new_training_or_gradient_measurement_launched']
    assert pub['agent_VM_maintenance_only'] and pub['local_neural_gradient_optimizer_or_displacement_calls']==0
    for name,digest in pub['evidence_sha256'].items():
        assert sha(ROOT/name)==digest and read(ROOT/name)['complete']
    for name,row in pub['documents'].items():
        path=ROOT/name;backup=__import__('pathlib').Path(row['backup'])
        assert sha(path)==row['after_sha256'] and sha(backup)==row['before_sha256']
        assert path.read_bytes()[row['prefix_bytes']:]==backup.read_bytes()
        assert row['old_bytes_preserved_as_exact_suffix'] and row['expected_hash_checked_under_exclusive_lock']
        assert path.read_text(encoding='utf-8').count('```')%2==0
    assert len(pub['documents'])==5 and len(pub['new_documents_sha256'])==2
    for name,digest in pub['new_documents_sha256'].items():
        assert sha(ROOT/name)==digest and (ROOT/name).read_text(encoding='utf-8').count('```')%2==0
    failure=read(ANALYSIS/'failure_audit.json');imported=read(ANALYSIS/'import.json')
    assert failure['complete'] and failure['scope']=='pre-model storage preflight stop'
    assert failure['required_free_bytes']==4294967296 and failure['neural_gradient_optimizer_and_parameter_displacement_calls']==0
    assert sha(RETURN/'review.log')==failure['review_log_sha256'] and sha(RETURN/'protocol.json')==PIN
    assert sha(OUT/'cctv-dgp-bank-group-gradients-v1-results.tar.gz')==imported['archive_sha256']
    returned=read(RETURN/'export_manifest.json')
    assert not returned['complete_diagnostic'] and not returned['failure_present']
    assert all(sha(RETURN/n)==h for n,h in returned['files'].items())
    assert not (RETURN/'outputs').exists()
    old_p=read(RETURN/'protocol.json')
    assert all(sha(PARENT/n)==h for n,h in old_p['parent_required_sha256'].items())
    prepared=OUT/'cctv_dgp_bank_group_gradients_v1_vm'
    assert sha(prepared/'protocol.json')==PIN and all(sha(prepared/n)==h for n,h in old_p['assets_sha256'].items())
    repair=read(OUT/'cctv_dgp_bank_group_gradients_v1_local_audit_repair_r1.json')
    assert sha(ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_return.py')==repair['original_importer_sha256']
    assert sha(ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_return.py')==repair['original_auditor_sha256']
    assert sha(ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_return_r1.py')==repair['repaired_importer_sha256']
    assert sha(ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_return_r1.py')==repair['repaired_auditor_sha256']
    cleanup_dir=OUT/'cctv_dgp_bank_group_gradients_v1_storage_cleanup'
    cleanup=read(cleanup_dir/'independent_audit.json');plan=read(cleanup_dir/'plan.json')
    assert cleanup['complete'] and cleanup['exact_deleted_archives']==1 and cleanup['free_GiB_live']>=4
    assert sha(cleanup_dir/'plan.json')==cleanup['plan_sha256']
    assert sha(ROOT/plan['candidates'][0]['local_backup'])==cleanup['full_local_backup_sha256']
    assert cleanup['protected_metadata_and_packet_bytes_identical'] and cleanup['checkpoints_research_caches_splits_logs_and_gate_failures_retained']
    assert cleanup['model_gradient_or_training_calls']==0
    prep=read(OUT/'cctv_dgp_bank_group_gradients_v1_r1_preparation.json')
    transfer=read(OUT/'cctv_dgp_bank_group_gradients_v1_r1_transfer_audit.json')
    packet=OUT/prep['name'];p=read(packet/'protocol.json')
    assert prep['protocol_sha256']==sha(packet/'protocol.json')==transfer['protocol_sha256']
    assert prep['actual_manual_VM_diagnostic_unrun'] and not (packet/'outputs').exists()
    assert sha(OUT/(prep['stem']+'-execution.tar.gz'))==prep['archive_sha256']
    assert all(sha(packet/n)==h for n,h in p['assets_sha256'].items())
    assert p['minimum_free_bytes']==4*1024**3 and p['disk_reserve_bytes']==1024**3
    assert p['maximum_gradient_queries']==320 and len(p['gradient_groups'])==32
    assert p['optimizer_updates']==p['backwards']==p['parameter_displacements']==p['full_epochs']==0
    assert p['case_ids']==old_p['case_ids'] and p['parent_required_sha256']==old_p['parent_required_sha256']
    assert p['preserved_scientific_thresholds']==old_p['preserved_scientific_thresholds']
    importer=module(ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_r1_return.py','sealed_recovery_importer_closure')
    assert importer.NAME==prep['name'] and importer.STEM==prep['stem'] and importer.PIN==prep['protocol_sha256']
    assert importer.RETURN==OUT/'cctv_dgp_bank_group_gradients_v1_r1_return'
    assert importer.ANALYSIS==OUT/'cctv_dgp_bank_group_gradients_v1_r1_analysis'
    worker=(packet/'scripts/cctv_dgp_bank_group_gradients_v1_r1_vm.py').read_text(encoding='utf-8')
    assert worker.index('available = shutil.disk_usage(root).free')<worker.index('import torch')
    assert "available_free_bytes=available" in worker and "write(out/'failure.json'" in worker
    for file in [ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_r1_return.py',ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_r1_return.py']:
        ast.parse(file.read_text(encoding='utf-8'))
    assert sha(ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_r1_return.py')==transfer['future_auditor_sha256']
    assert sha(ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_r1_return.py')==prep['importer_sha256']
    bash=read(OUT/'cctv_dgp_bank_group_gradients_v1_r1_bash_syntax.json')
    assert bash['exit_code']==0 and bash['actual_script_not_executed']
    assert bash['checked_shell_sha256']==sha(packet/'scripts/run_groups.sh')
    guide=(ROOT/'CCTV_DGP_BANK_GROUP_GRADIENTS_V1_R1_VM.md').read_text(encoding='utf-8')
    assert re.findall(r'^([1-5])\. ',guide,re.M)==['1','2','3','4','5']
    blocks=re.findall(r'```(?:bat|bash)\n(.*?)```',guide,re.S)
    assert len(blocks)==5
    assert prep['protocol_sha256'] in blocks[3] and prep['name'] in blocks[1] and prep['name'] in blocks[3]
    assert 'tmux new-session -A -s dgp_group_gradients_v1_r1' in blocks[2]
    uploads=[line for line in blocks[0].splitlines() if line.startswith('gcloud compute scp')]
    downloads=[line for line in blocks[4].splitlines() if line.startswith('gcloud compute scp')]
    assert len(uploads)==2 and len(downloads)==3
    for line in uploads+downloads:
        assert '--project=forensic-dgp-thesis' in line and '--zone=us-central1-a' in line
        assert line.count('janusdominic0@forensic-dgp-thesis:')==1
    for line in downloads:
        remote,local=re.findall(r'"([^"]+)"',line)
        assert remote.rsplit('/',1)[-1]==local and local.startswith(prep['stem'])
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    report=(ROOT/'CCTV_DGP_BANK_GROUP_GRADIENTS_V1_RESULTS.md').read_text(encoding='utf-8')
    assert all(term in report for term in ['masks','sunglasses','strong glare','hands','obstructing','scarves','objects','Goal ACTIVE','independent final/app review','unpaired','manual','clearer'])
    write(final,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        publication_sha256=sha(ANALYSIS/'publication.json'),documents_checked=7,exact_historical_suffixes_checked=5,
        returned_file_bindings_checked=10,parent_hashes_checked=len(old_p['parent_required_sha256']),
        original_checkers_and_used_packet_preserved=True,pre_model_storage_failure_not_quality_failure=True,
        exact_archive_cleanup_independently_closed=True,retained_research_metadata_entries=cleanup['retained_research_metadata_entries'],
        retained_packet_hashes=cleanup['retained_packet_byte_hashes'],free_GiB_live=cleanup['free_GiB_live'],
        full_local_backup_retained_and_rehashed=True,prospective_importer_scope_and_auditor_bound=True,
        scientific_resource_limits_and_source_split_scope_preserved=True,new_manual_R1_measurement_unrun=True,
        five_manual_steps_two_uploads_three_single_source_explicit_downloads_checked=True,
        local_protected_bindings_checked=len(protected),local_neural_gradient_optimizer_or_displacement_calls=0,
        all_five_milestones_seven_families_and_independent_final_app_scope_preserved=True,
        no_new_model_qualification=True,goal_status='active',goal_complete=False))
    print(dict(complete=True,documents=7,protected_bindings=len(protected),parent_bindings=len(old_p['parent_required_sha256']),
        storage_closed=True,manual_R1_unrun=True,goal_status='active'),flush=True)


if __name__=='__main__':
    main()
