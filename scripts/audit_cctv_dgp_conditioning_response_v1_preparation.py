"""Prospective packet, CLI, namespace, resource and return-checker verification."""
import ast
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile

from import_cctv_dgp_conditioning_response_v1_return import ROOT, OUT, NAME, STEM, PIN, PARENT, member_names, read, sha, write


def main():
    packet=OUT/NAME;prep=read(OUT/'cctv_dgp_conditioning_response_v1_preparation.json');p=read(packet/'protocol.json')
    assert prep['complete'] and prep['protocol_sha256']==PIN==sha(packet/'protocol.json') and len(PIN)==64
    bindings={**p['assets_sha256'],'protocol.json':PIN};assert len(bindings)==20
    assert all(sha(packet/n)==h for n,h in bindings.items())
    archive=OUT/(STEM+'-execution.tar.gz');checksum=Path(str(archive)+'.sha256')
    assert sha(archive)==prep['archive_sha256'] and archive.stat().st_size==prep['archive_bytes']
    raw=checksum.read_bytes();assert b'\r' not in raw and raw==f"{prep['archive_sha256']}  {archive.name}\n".encode('ascii')
    with tarfile.open(archive,'r:gz') as tar:
        names=member_names(tar.getmembers());assert set(names)==set(bindings)
        for n,m in names.items():
            with tar.extractfile(m) as stream:h=hashlib.file_digest(stream,'sha256').hexdigest()
            assert h==bindings[n] and m.size==(packet/n).stat().st_size
    worker=packet/'scripts/cctv_dgp_conditioning_response_v1_vm.py'
    assert sha(worker)==sha(ROOT/'scripts/cctv_dgp_conditioning_response_v1_vm.py')
    assert p['maximum_gradient_queries']==2*20*16==640
    assert p['model_forward_cap']==20+2*20+6*20+20==200
    assert p['recognizer_forward_cap']==40+2*20+6*40+40==360
    assert p['optimizer_updates']==p['accepted_updates']==0 and p['temporary_displacements']==6
    assert p['full_output_plan']==dict(snapshots=8,RAW_outputs=800,PNG_outputs=800,stage_metric_rows=1600,visual_sheets=20,visual_cells=1000)
    assert p['minimum_free_bytes']-2*(p['output_byte_cap']+64*1024**2)>=p['disk_reserve_bytes']
    assert p['worker_seconds']<p['external_diagnostic_seconds'] and p['export_seconds']<p['external_export_seconds']
    assert p['measurement_seconds']<p['worker_seconds'] and p['gradient_round_seconds']<=600
    assert not any(p[k] for k in ['new_direction_solve','optimizer_constructed','app_promotion','automatic_follow_on','failed_run_resume_allowed','model_qualified','goal_complete'])
    assert p['original_model_checkpoint_sha256']=='646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b'
    previous=OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_return'
    prior=read(previous/'protocol.json');old=read(PARENT/'protocol.json')
    for k in ['parent_required_sha256','case_ids','reference_blocks','raw_gradient_groups','conditioning_names','conditioning_shapes','initial_model_state','preserved_scientific_thresholds','all37_learning_names']:
        assert p[k]==prior[k],k
    assert p['preserved_scientific_thresholds']==old['scientific_thresholds']
    assert sha(packet/'saved_direction.npy')==sha(previous/'outputs/raw_conditioning_learning_v1_r1/round00/direction.npy')==p['saved_direction_sha256']
    assert p['base_signed_magnitude']==read(previous/'outputs/raw_conditioning_learning_v1_r1/r00_trial02/decision.json')['rate']
    for n,r in p['component_origins'].items():
        assert sha(previous/n)==r['source_sha256'] and sha(packet/n)==r['prepared_sha256']
        if n=='conditioning_model.py':
            original=(previous/n).read_text(encoding='utf-8')
            assert (packet/n).read_text(encoding='utf-8')==original.replace("NAME = 'cctv_dgp_raw_conditioning_learning_v1_r1_vm'",f"NAME = '{NAME}'",1)
        else:assert r['source_sha256']==r['prepared_sha256']
    for n,h in p['parent_required_sha256'].items():assert sha(PARENT/n)==h,n
    for script in [worker,ROOT/'scripts/import_cctv_dgp_conditioning_response_v1_return.py',ROOT/'scripts/audit_cctv_dgp_conditioning_response_v1_return.py']:
        source=script.read_text(encoding='utf-8');ast.parse(source,feature_version=(3,10));assert '__PROTOCOL_SHA256__' not in source
    tree=ast.parse(worker.read_text(encoding='utf-8'))
    assert not [n for n in ast.walk(tree) if isinstance(n,ast.Call) and ((isinstance(n.func,ast.Name) and n.func.id in ['solve_direction','Adam','AdamW','SGD','minimize','nnls']) or (isinstance(n.func,ast.Attribute) and n.func.attr=='step'))]
    assert '--diagnose' in (packet/'scripts/run_response.sh').read_text(encoding='utf-8')
    assert '--run' not in (packet/'scripts/run_response.sh').read_text(encoding='utf-8')
    r=subprocess.run([sys.executable,'-B',str(worker),'--root',str(packet),'--parent-root',str(PARENT),'--protocol-sha',PIN,'--verify-transfer'],cwd=ROOT,capture_output=True,text=True,timeout=120)
    assert r.returncode==0,(r.stdout,r.stderr)
    import json
    printed=json.loads(r.stdout.strip())
    assert printed['packet_assets']==19 and printed['parent_assets']==5852 and printed['cases']==100
    assert printed['maximum_gradient_queries']==640 and printed['temporary_displacements']==6 and printed['optimizer_updates']==printed['actual_neural_calls']==0
    assert printed['minimum_free_bytes']==p['minimum_free_bytes']
    protected=read(OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    checkers=['contract_green.json','guard_green.json','archive_members_green.json','snapshot_fixture.json']
    assert all(read(OUT/('cctv_dgp_conditioning_response_v1_'+n))['complete'] for n in checkers)
    write(OUT/'cctv_dgp_conditioning_response_v1_transfer_audit.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),auditor_sha256=sha(__file__),
        protocol_sha256=PIN,preparation_sha256=sha(OUT/'cctv_dgp_conditioning_response_v1_preparation.json'),archive_sha256=sha(archive),
        exact_archive_bindings=20,parent_bindings=5852,hash_only_worker_receipt=printed,source_and_component_origins_verified=True,
        unchanged_scientific_gates_and_failed_qualifications_retained=True,return_checker_and_boundary_fixtures_verified=True,
        no_local_neural_gradient_optimizer_or_model_parameter_displacement=True,manual_diagnostic_unrun=True,
        Bash_syntax_audit_pending=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,protocol_sha256=PIN,archive_bindings=20,parent_bindings=5852,worker_receipt=printed,manual_unrun=True))


if __name__=='__main__':main()
