"""Prepare a fresh storage recovery only after independently verified cleanup."""
import ast
from datetime import datetime, timezone
from pathlib import Path
import shutil
import tarfile
from import_cctv_dgp_bank_group_gradients_v1_return_r1 import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write

NAME='cctv_dgp_bank_group_gradients_v1_r1_vm'
STEM='cctv-dgp-bank-group-gradients-v1-r1'


def main():
    packet=OUT/NAME;archive=OUT/(STEM+'-execution.tar.gz')
    receipt=OUT/'cctv_dgp_bank_group_gradients_v1_r1_preparation.json'
    assert not any(p.exists() for p in [packet,archive,receipt])
    failure=read(ANALYSIS/'failure_audit.json')
    cleanup_path=OUT/'cctv_dgp_bank_group_gradients_v1_storage_cleanup/independent_audit.json'
    cleanup=read(cleanup_path)
    assert failure['complete'] and failure['scope']=='pre-model storage preflight stop'
    assert failure['neural_gradient_optimizer_and_parameter_displacement_calls']==0
    assert cleanup['complete'] and cleanup['exact_deleted_archives']==1 and cleanup['free_GiB_live']>=4
    assert cleanup['checkpoints_research_caches_splits_logs_and_gate_failures_retained']
    old=read(RETURN/'protocol.json');assert sha(RETURN/'protocol.json')==PIN
    assert all(sha(RETURN/n)==h for n,h in read(RETURN/'export_manifest.json')['files'].items())
    assert all(sha(PARENT/n)==h for n,h in old['parent_required_sha256'].items())
    protected=read(ANALYSIS/'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    packet.mkdir();(packet/'scripts').mkdir()
    worker=RETURN/'scripts/cctv_dgp_bank_group_gradients_v1_vm.py'
    source=worker.read_text(encoding='utf-8').replace('group_gradients_v1','group_gradients_v1_r1').replace('group-gradients-v1','group-gradients-v1-r1')
    before="    assert shutil.disk_usage(root).free >= p['minimum_free_bytes'], 'Need4GiB free after installation'\n"
    assert source.count(before)==1
    after='''    available = shutil.disk_usage(root).free
    if available < p['minimum_free_bytes']:
        out = root/'outputs/group_gradients_v1_r1'
        out.mkdir(parents=True)
        write(out/'failure.json',dict(type='AssertionError',message='Need4GiB free after installation',
            stage='pre-model storage preflight',available_free_bytes=available,
            required_free_bytes=p['minimum_free_bytes'],optimizer_updates=0,
            neural_calls=0,gradient_queries=0,parameter_displacements=0,
            model_qualified=False,more_epochs_not_qualified=True))
        raise AssertionError('Need4GiB free after installation')
'''
    source=source.replace(before,after)
    needle="            groups=32,planned_gradient_queries=320,optimizer_updates=0,actual_neural_or_gradient_calls=0),flush=True)"
    assert source.count(needle)==1
    source=source.replace(needle,"            groups=32,planned_gradient_queries=320,optimizer_updates=0,actual_neural_or_gradient_calls=0,\n            available_free_bytes=shutil.disk_usage(root).free,minimum_free_bytes=p['minimum_free_bytes']),flush=True)")
    ast.parse(source,feature_version=(3,10))
    worker_name='scripts/cctv_dgp_bank_group_gradients_v1_r1_vm.py'
    (packet/worker_name).write_text(source,encoding='utf-8',newline='\n')
    for name in ['directions.npz','direction_review.json']:
        shutil.copyfile(RETURN/name,packet/name)
    shutil.copyfile(RETURN/'cctv_dgp_bank_group_gradients_v1_contract.py',packet/'cctv_dgp_bank_group_gradients_v1_r1_contract.py')
    shell=(RETURN/'scripts/run_groups.sh').read_text(encoding='utf-8').replace('group_gradients_v1_vm.py','group_gradients_v1_r1_vm.py')
    (packet/'scripts/run_groups.sh').write_text(shell,encoding='utf-8',newline='\n')
    p=dict(old)
    p.update(name=NAME,UTC=datetime.now(timezone.utc).isoformat(),
        recovery_from_zero_neural_storage_preflight=True,previous_protocol_sha256=PIN,
        previous_return_archive_sha256=read(ANALYSIS/'import.json')['archive_sha256'],
        previous_failure_audit_sha256=sha(ANALYSIS/'failure_audit.json'),
        verified_cleanup_sha256=sha(cleanup_path),
        cleanup_snapshot_free_GiB=cleanup['free_GiB_live'],snapshot_not_future_free_space_guarantee=True,
        retained_measurement_scope_and_all_scientific_resource_limits=True,
        startup_storage_bytes_printed_and_structured_storage_failure_written=True,
        fresh_directory_preserves_original_used_packet=True)
    p['assets_sha256']={f.relative_to(packet).as_posix():sha(f) for f in packet.rglob('*') if f.is_file()}
    assert len(p['assets_sha256'])==5
    write(packet/'protocol.json',p);pin=sha(packet/'protocol.json')
    with tarfile.open(archive,'x:gz',compresslevel=1) as tar:
        for file in sorted(f for f in packet.rglob('*') if f.is_file()):
            tar.add(file,arcname=NAME+'/'+file.relative_to(packet).as_posix(),recursive=False)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:
        stream.write(digest+'  '+archive.name+'\n')
    new_importer=ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_r1_return.py'
    assert not new_importer.exists()
    text=(ROOT/'scripts/import_cctv_dgp_bank_group_gradients_v1_return_r1.py').read_text(encoding='utf-8')
    text=text.replace('group_gradients_v1','group_gradients_v1_r1').replace('group-gradients-v1','group-gradients-v1-r1').replace(PIN,pin)
    # The recovery has a new preparation/transfer ledger, not the earlier used one.
    with new_importer.open('x',encoding='utf-8',newline='\n') as stream:stream.write(text)
    new_auditor=ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_r1_return.py'
    audit=(ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_return_r1.py').read_text(encoding='utf-8')
    audit=audit.replace('group_gradients_v1','group_gradients_v1_r1')
    audit=audit.replace('from import_cctv_dgp_bank_group_gradients_v1_r1_return_r1 import ',
                        'from import_cctv_dgp_bank_group_gradients_v1_r1_return import ')
    begin=audit.index("    if not (out/'run_results.json').exists():")
    end=audit.index("    run = read(out/'run_results.json')",begin)
    branch='''    if not (out/'run_results.json').exists():
        assert not manifest['complete_diagnostic']
        log=(RETURN/'review.log').read_text(encoding='utf-8')
        assert int((RETURN/'review_exit_code.txt').read_text().strip()) != 0
        if (out/'failure.json').exists():
            failure=read(out/'failure.json')
            assert manifest['failure_present'] and failure['optimizer_updates']==0
            if failure.get('stage')=='pre-model storage preflight':
                assert failure['available_free_bytes'] < failure['required_free_bytes']==p['minimum_free_bytes']==4294967296
                assert failure['neural_calls']==failure['gradient_queries']==failure['parameter_displacements']==0
            detail=dict(scope='structured retained diagnostic failure',failure=failure,failure_sha256=sha(out/'failure.json'))
        else:
            assert not out.exists() and not manifest['failure_present']
            detail=dict(scope='unstructured startup failure; no completed measurement',traceback=log,
                        no_neural_execution_claim_from_export_flag=True)
        write(ANALYSIS/'failure_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
            checker_sha256=sha(__file__),import_sha256=sha(ANALYSIS/'import.json'),**detail,
            review_log_sha256=sha(RETURN/'review.log'),independently_checked_all_return_file_and_parent_hashes=True,
            no_neural_reexecution_or_post_failure_RAM_state_hash_claimed=True,
            local_neural_or_optimizer_calls=0,no_training_qualification=True,model_qualified=False,goal_complete=False))
        print(dict(complete=True,retained_failure_audited=True,model_qualified=False))
        return
'''
    audit=audit[:begin]+branch+audit[end:]
    ast.parse(text);ast.parse(audit)
    with new_auditor.open('x',encoding='utf-8',newline='\n') as stream:stream.write(audit)
    checker=ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_r1_transfer.py'
    transfer=(ROOT/'scripts/audit_cctv_dgp_bank_group_gradients_v1_transfer.py').read_text(encoding='utf-8')
    transfer=transfer.replace('group_gradients_v1','group_gradients_v1_r1')
    needle="    assert sha(importer)==prep['importer_sha256']\n"
    assert transfer.count(needle)==1
    transfer=transfer.replace(needle,needle+"    imported=module(importer,'prospective_recovery_importer')\n    assert imported.NAME==prep['name'] and imported.STEM==prep['stem']\n    assert imported.RETURN==OUT/'cctv_dgp_bank_group_gradients_v1_r1_return'\n    assert imported.ANALYSIS==OUT/'cctv_dgp_bank_group_gradients_v1_r1_analysis'\n    previous=read(OUT/'cctv_dgp_bank_group_gradients_v1_return/protocol.json')\n    allowed={'name','UTC','assets_sha256','recovery_from_zero_neural_storage_preflight','previous_protocol_sha256','previous_return_archive_sha256','previous_failure_audit_sha256','verified_cleanup_sha256','cleanup_snapshot_free_GiB','snapshot_not_future_free_space_guarantee','retained_measurement_scope_and_all_scientific_resource_limits','startup_storage_bytes_printed_and_structured_storage_failure_written','fresh_directory_preserves_original_used_packet'}\n    assert {k:v for k,v in p.items() if k not in allowed}=={k:v for k,v in previous.items() if k not in allowed}\n    assert 'available_free_bytes=available' in source and source.index('available = shutil.disk_usage(root).free') < source.index('import torch')\n")
    ast.parse(transfer)
    with checker.open('x',encoding='utf-8',newline='\n') as stream:stream.write(transfer)
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(receipt,dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),preparer_sha256=sha(__file__),
        name=NAME,stem=STEM,protocol_sha256=pin,archive_sha256=digest,archive_bytes=archive.stat().st_size,
        packet_assets=5,parent_assets=5852,gradient_groups=32,planned_gradient_queries=320,
        previous_storage_failure_sha256=sha(ANALYSIS/'failure_audit.json'),verified_cleanup_sha256=sha(cleanup_path),
        importer_sha256=sha(new_importer),prospective_auditor_sha256=sha(new_auditor),transfer_checker_sha256=sha(checker),
        previous_immutable_return_and_packet_preserved=True,measurement_recipe_and_all_limits_unchanged=True,
        fresh_execution_root_and_structured_storage_log_added=True,actual_manual_VM_diagnostic_unrun=True,
        local_neural_gradient_optimizer_parameter_displacement_calls=0,model_qualified=False,goal_complete=False))
    print(dict(complete=True,name=NAME,protocol_sha256=pin,archive_sha256=digest,bytes=archive.stat().st_size),flush=True)


if __name__=='__main__':
    main()
