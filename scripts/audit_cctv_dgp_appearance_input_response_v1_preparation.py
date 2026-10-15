"""Independent package, dependency, guard and actual hash-only receipt audit."""
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

from import_cctv_dgp_appearance_input_response_v1_return import ROOT,OUT,NAME,STEM,PIN,SOURCE,BANK,read,sha,write,member_names


def main():
    packet=OUT/NAME;p=read(packet/'protocol.json');prepared=read(OUT/'cctv_dgp_appearance_input_response_v1_preparation.json')
    assert prepared['complete'] and PIN==sha(packet/'protocol.json')==prepared['protocol_sha256']
    assert prepared['packet_assets']==len(p['assets_sha256'])==16 and not (packet/'outputs').exists() and not (packet/'review.log').exists()
    assert all(sha(packet/n)==h for n,h in p['assets_sha256'].items())
    assert len(p['response_required_sha256'])==1246 and len(p['bank_required_sha256'])==5852
    assert 'export.log' not in p['response_required_sha256'] and (SOURCE/'export.log').exists()
    assert 'export_manifest.json' in p['response_required_sha256']
    assert p['minimum_free_bytes']==3*1024**3 and p['output_byte_cap']==512*1024**2 and p['disk_reserve_bytes']==1024**3
    assert p['maximum_gradient_queries']==200 and p['recognizer_forward_cap']==40
    assert p['restorer_forward_cap']==p['neural_parameter_displacements']==p['optimizer_updates']==p['accepted_updates']==0
    assert p['external_measurement_seconds']+p['external_export_seconds']+2*p['termination_grace_seconds']==750
    for n,row in p['component_origins'].items():assert row['source_sha256']==row['prepared_sha256']==sha(packet/n)
    archive=OUT/(STEM+'-execution.tar.gz');checksum=Path(str(archive)+'.sha256')
    assert archive.stat().st_size==prepared['archive_bytes'] and sha(archive)==prepared['archive_sha256']
    assert checksum.read_bytes()==(prepared['archive_sha256']+'  '+archive.name+'\n').encode('ascii')
    with tarfile.open(archive,'r:gz') as tar:
        names=member_names(tar.getmembers());expected={**p['assets_sha256'],'protocol.json':PIN}
        assert set(names)==set(expected)
        for n,m in names.items():
            assert m.size==(packet/n).stat().st_size
            with tar.extractfile(m) as f:assert hashlib.sha256(f.read()).hexdigest()==expected[n]
    for file in ['scripts/cctv_dgp_appearance_input_response_v1_vm.py','input_projection_contract.py','input_scopes.py','cctv_dgp_pilot.py','models/identity_loss.py']:
        ast.parse((packet/file).read_text(encoding='utf-8'),feature_version=(3,10))
    importer=ROOT/'scripts/import_cctv_dgp_appearance_input_response_v1_return.py'
    assert sha(importer)==prepared['prospective_importer_sha256'] and sha(ROOT/'scripts/audit_cctv_dgp_appearance_input_response_v1_return.py')==prepared['prospective_return_auditor_sha256']
    fixtures=read(OUT/'cctv_dgp_appearance_input_response_v1_batch_fixture.json')
    assert fixtures['complete'] and fixtures['prospective_batch_auditor_sha256']==prepared['prospective_return_auditor_sha256']
    assert fixtures['incorrect_projection_rejected'] and fixtures['changed_gradient_file_rejected'] and fixtures['fixture_restored_exactly']
    guard=subprocess.run([sys.executable,'-B',str(ROOT/'scripts/check_cctv_dgp_appearance_input_response_v1_guard.py')],cwd=ROOT,capture_output=True,text=True,timeout=20)
    assert guard.returncode==0,guard.stderr
    command=[sys.executable,'-B',str(packet/'scripts/cctv_dgp_appearance_input_response_v1_vm.py'),
        '--root',str(packet),'--response-root',str(SOURCE),'--bank-root',str(BANK),'--protocol-sha',PIN,'--verify-transfer']
    verified=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=180)
    assert verified.returncode==0,verified.stderr
    receipt=json.loads(verified.stdout.strip());assert receipt['complete'] and receipt['packet_assets']==16
    assert receipt['response_assets']==1246 and receipt['bank_assets']==5852 and receipt['cases']==100
    assert receipt['maximum_gradient_queries']==200 and receipt['maximum_recognizer_forwards']==40
    assert receipt['actual_neural_calls']==receipt['optimizer_updates']==receipt['neural_parameter_displacements']==0
    assert receipt['minimum_free_bytes']==3*1024**3
    shell=(packet/'scripts/run_input_response.sh').read_bytes();assert b'\r' not in shell
    for value in ['--kill-after=30s 480s','--kill-after=30s 210s','test ! -e review.log','Human manual tmux required']:
        assert value in shell.decode('ascii')
    protected=read(OUT/'cctv_dgp_conditioning_response_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(OUT/'cctv_dgp_appearance_input_response_v1_transfer_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),protocol_sha256=PIN,archive_sha256=prepared['archive_sha256'],archive_bytes=archive.stat().st_size,
        archive_bindings=len(names),response_file_bindings=1246,bank_file_bindings=5852,actual_hash_only_receipt=receipt,
        hash_only_receipt_measured_on_Windows_not_VM=True,real_updated_Windows_guard_rechecked=True,
        five_case_fixture_and_corruptions_pass=True,ASCII_LF_checksum_and_wrapper_checked=True,
        old_scientific_cases_sources_thresholds_parity_failures_and_protected_assets_preserved=True,
        local_neural_gradient_or_training_calls=0,manual_diagnostic_unrun=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,archive_bindings=len(names),response_bindings=1246,bank_bindings=5852,actual_hash_only_receipt_pass=True,manual_UNRUN=True))


if __name__=='__main__':main()
