"""Independent packet/source contracts and actual Windows execution rejection."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r1_vm'
OUT=ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r1_preparation'
PIN='f4a532587f4d722dae6348546ee37c6ed1f2801f9f37c9e7173ff412ba93ea09'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    start=time.monotonic();receipt=OUT/'independent_packet_audit.json'
    assert not receipt.exists()
    prep=read(OUT/'preparation.json');p=read(BUNDLE/'protocol.json')
    assert sha(BUNDLE/'protocol.json')==PIN==prep['protocol_sha256']
    archive=ROOT/'outputs/cctv-dgp-original-decoder-gradient-v1-r1-execution.tar.gz'
    assert archive.stat().st_size==prep['archive_bytes']==13223 and sha(archive)==prep['archive_sha256']
    assert Path(str(archive)+'.sha256').read_text(encoding='ascii').strip()==prep['archive_sha256']+'  '+archive.name
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name
    expected={'protocol.json',*p['assets_sha256']}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers()
        assert len(members)==4 and all(m.isfile() for m in members)
        assert {m.name for m in members}=={BUNDLE.name+'/'+name for name in expected}
        for member in members:
            name=member.name[len(BUNDLE.name)+1:]
            assert hashlib.sha256(tar.extractfile(member).read()).hexdigest()==sha(BUNDLE/name)
    review=read(ROOT/'outputs/cctv_dgp_original_decoder_review_v1/review.json')
    assert p['parameter_layout']==review['parameter_layout'] and p['original_DGP_state']==review['DGP_state_before_after']
    assert p['decoder_parameters']==609219 and p['decoder_parameter_tensors']==14 and p['cases']==50
    parent=ROOT/'outputs/cctv_dgp_feature_skips_vm_v27';base=read(parent/'protocol.json')
    assert sha(parent/'protocol.json')==p['closed_V27_protocol_sha256']
    assert p['retained_capacity_gates']==base['prospective_gates']
    assert all(c['role']=='train' for c in base['cases']) and len({c['source_person_or_reference'] for c in base['cases']})==10
    for name,digest in base['assets_sha256'].items():assert sha(parent/name)==digest,name
    for name,digest in p['closed_V27_evidence_sha256'].items():assert sha(ROOT/'outputs/cctv_dgp_feature_skips_v27_return'/name)==digest
    worker_path=BUNDLE/'scripts/cctv_dgp_original_decoder_gradient_v1_r1_vm.py'
    worker=worker_path.read_text(encoding='utf-8');tree=ast.parse(worker,feature_version=(3,10))
    restored=worker.replace('own-DGP-original-decoder-zero-update-gradient-proof-v1-r1','own-DGP-original-decoder-zero-update-gradient-proof-v1')
    restored=restored.replace('cctv_dgp_original_decoder_gradient_v1_r1_vm','cctv_dgp_original_decoder_gradient_v1_vm')
    restored=restored.replace('cctv-dgp-original-decoder-gradient-v1-r1-results','cctv-dgp-original-decoder-gradient-v1-results')
    restored=restored.replace('cctv_dgp_original_decoder_gradient_v1_r1_return','cctv_dgp_original_decoder_gradient_v1_return')
    restored=restored.replace('cctv-dgp-original-decoder-gradient-v1-r1-export','cctv-dgp-original-decoder-gradient-v1-export')
    corrected="z = torch.arange(-6,7,dtype=torch.float32)\n        kernel = torch.exp(-.5*(z/2).square()); kernel = kernel / kernel.sum()\n        fixed.kernel = kernel.cuda()"
    draft="z = torch.arange(-6,7,dtype=torch.float32,device='cuda')\n        fixed.kernel = torch.exp(-.5*(z/2).square()); fixed.kernel /= fixed.kernel.sum()"
    assert restored.count(corrected)==1
    restored=restored.replace(corrected,draft)
    old=(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_vm/scripts/cctv_dgp_original_decoder_gradient_v1_vm.py').read_text(encoding='utf-8')
    assert ast.dump(ast.parse(restored),include_attributes=False)==ast.dump(ast.parse(old),include_attributes=False)
    assert not any(isinstance(n,ast.Attribute) and n.attr in ['Adam','AdamW','SGD','backward','step'] for n in ast.walk(tree))
    assert sum(isinstance(n,ast.Call) and ast.unparse(n.func)=='torch.autograd.grad' for n in ast.walk(tree))==1
    assert 'range(0,50,5)' in worker and 'retain_graph=i<6,create_graph=False,allow_unused=False' in worker
    assert 'prediction = candidate(' in worker and 'prediction.requires_grad and not torch.is_inference(prediction)' in worker
    assert 'torch.equal(prediction.detach(),b[\'base\'])' in worker
    assert 'vm_scope(root, parent)  # Before importing any model or creating outputs.' in worker
    assert p['initial_preservation_terms_exact_zero']==['clear_baseline_anchor','pixel_regression','SSIM_regression','ArcFace_regression']
    assert p['optimizer_updates']==p['epochs']==0 and p['component_gradient_calls']==70 and p['batches']==10
    assert p['budgets']=={'worker_seconds':600,'external_seconds':660,'external_kill_grace_seconds':30,
        'export_seconds':90,'external_export_seconds':120,'external_export_kill_grace_seconds':10,
        'peak_vram_bytes':21474836480,'minimum_free_disk_bytes':2147483648,'maximum_export_uncompressed_bytes':900000000}
    assert (BUNDLE/'cctv_dgp_original_decoder_candidate_v1.py').read_bytes()==(ROOT/'scripts/cctv_dgp_original_decoder_candidate_v1.py').read_bytes()
    shell=(BUNDLE/'scripts/run_decoder_gradient.sh').read_bytes()
    historical=(ROOT/'outputs/cctv_dgp_v26_gradient_diagnostic_v1_vm/scripts/run_gradient.sh').read_bytes()
    restored_shell=shell.replace(b'cctv_dgp_original_decoder_gradient_v1_r1_vm.py',b'cctv_dgp_v26_gradient_diagnostic_v1_vm.py').replace(b'660s ',b'480s ').replace(b'120s ',b'60s ')
    assert restored_shell==historical
    before={path.relative_to(BUNDLE).as_posix():sha(path) for path in BUNDLE.rglob('*') if path.is_file()}
    assert set(before)==expected
    result=subprocess.run([sys.executable,'-B',str(worker_path),'--root',str(BUNDLE),'--protocol-sha',PIN,'--run'],
                          capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'Existing Linux VM only; no local gradients' in result.stderr
    assert {path.relative_to(BUNDLE).as_posix():sha(path) for path in BUNDLE.rglob('*') if path.is_file()}==before
    spec=importlib.util.spec_from_file_location('decoder_packet_verification',worker_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.verify(BUNDLE,PIN)==p
    try:module.verify(BUNDLE,'0'*64)
    except AssertionError:pass
    else:raise AssertionError('Changed protocol was accepted')
    audit_path=ROOT/'scripts/audit_cctv_dgp_original_decoder_gradient_v1_r1_return.py'
    ast.parse(audit_path.read_text(encoding='utf-8'),feature_version=(3,10))
    audit_tree=ast.parse(audit_path.read_text(encoding='utf-8'))
    assert not any(isinstance(n,ast.Attribute) and n.attr in ['grad','backward','AdamW','SGD','step'] for n in ast.walk(audit_tree))
    record={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':PIN,
        'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'packet_regular_files_verified':4,
        'original_assets_verified':len(base['assets_sha256']),'original_V27_failed_evidence_verified':4,
        'full_worker_AST_delta_and_historical_shell_behavior_verified':True,
        'original_filter_initializer_retained':True,'decoder_14_tensor_and_original_forward_contract_verified':True,
        'actual_Windows_execution_rejected_before_neural_work':True,'packet_unchanged_by_host_guard':True,
        'source_changed_protocol_rejection_verified':True,'prospective_return_audit_sha256':sha(audit_path),
        'local_neural_or_gradient_calls':0,'optimizer_updates':0,'VM_actions':False,
        'VM_gradient_proof_pending':True,'new_training_recipe_created':False,'app_promotion':False,'goal_complete':False,
        'seconds':time.monotonic()-start}
    with receipt.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
