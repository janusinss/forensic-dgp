"""Independent frozen packet/source contracts and actual pre-neural host rejection."""
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
BUNDLE=ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'
OUT=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_preparation'
PIN='27c140430673880f1aaab47a9df9af9b33758cc5d8adec53822cd9e05b13bbc8'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def main():
    start=time.monotonic();target=OUT/'independent_packet_audit_final.json';assert not target.exists()
    p=read(BUNDLE/'protocol.json');prep=read(OUT/'preparation.json')
    assert sha(BUNDLE/'protocol.json')==prep['protocol_sha256']==PIN
    archive=ROOT/'outputs/cctv-dgp-active-original-decoder-v28-execution.tar.gz'
    assert sha(archive)==prep['archive_sha256'] and archive.stat().st_size==prep['archive_bytes']
    assert Path(str(archive)+'.sha256').read_text(encoding='ascii').strip()==prep['archive_sha256']+'  '+archive.name
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name
    for name,digest in p['assets_sha256'].items():assert sha(BUNDLE/name)==digest,name
    expected={'protocol.json',*p['assets_sha256']}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==prep['packet_regular_files']==8
        assert all(m.isfile() and not m.issym() and not m.islnk() for m in members)
        assert {m.name for m in members}=={BUNDLE.name+'/'+name for name in expected}
        for m in members:
            name=m.name[len(BUNDLE.name)+1:]
            assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sha(BUNDLE/name)
    review=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_review/results.json')
    assert p['parameter_layout']==review['parameter_layout'] and p['decoder_parameters']==498627 and p['decoder_parameter_tensors']==12
    assert p['optimizer_updates']==800 and p['epochs']==80 and p['snapshot_updates']==p['snapshots']==[0,50,400,800]
    assert p['batches']==10 and p['batch_size']==5 and p['component_gradient_calls']==70
    parent=ROOT/'outputs/cctv_dgp_feature_skips_vm_v27';base=read(parent/'protocol.json')
    assert p['retained_capacity_gates']==base['prospective_gates'] and p['case_rows']==base['cases']
    assert all(c['role']=='train' for c in p['case_rows']) and len({c['source_person_or_reference'] for c in p['case_rows']})==10
    for name,digest in base['assets_sha256'].items():assert sha(parent/name)==digest,name
    for name,digest in p['closed_V27_evidence_sha256'].items():assert sha(ROOT/'outputs/cctv_dgp_feature_skips_v27_return'/name)==digest
    previous=ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return'
    for name,digest in p['closed_R2_evidence_sha256'].items():assert sha(previous/name)==digest,name
    assert p['closed_R2_protocol_sha256']=='81127e45a205c44ef685646a4f82911d02b2355dfe24c63772c23e62e66a41f2'
    assert read(previous/'outputs/failure.json')['component_gradient_calls']==70 and not (previous/'outputs/results.json').exists()
    assert (BUNDLE/'schedule.json').read_bytes()==(parent/'schedule.json').read_bytes()
    for name in ['cctv_dgp_original_decoder_candidate_v1.py','cctv_dgp_active_original_decoder_v28.py','cctv_dgp_active_decoder_v28_training.py']:
        assert (BUNDLE/name).read_bytes()==(ROOT/'scripts'/name).read_bytes()
        ast.parse((BUNDLE/name).read_text(encoding='utf-8'),feature_version=(3,10))
    app=function(ast.parse((ROOT/'dgp_face_workflow_v3.py').read_text(encoding='utf-8')),'canonical_tensor')
    shipped=function(ast.parse((BUNDLE/'cctv_dgp_app_input_v28.py').read_text(encoding='utf-8')),'canonical_tensor')
    assert ast.dump(app,include_attributes=False)==ast.dump(shipped,include_attributes=False)
    assert hashlib.sha256(ast.dump(app,include_attributes=False).encode()).hexdigest()==p['app_input_function_AST_sha256']
    worker_path=BUNDLE/'scripts/cctv_dgp_active_original_decoder_v28_vm.py';worker=worker_path.read_text(encoding='utf-8')
    tree=ast.parse(worker,feature_version=(3,10));run=function(tree,'run')
    oldtree=ast.parse((ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm/scripts/cctv_dgp_original_decoder_gradient_v1_r2_vm.py').read_text(encoding='utf-8'))
    # Fixed source-extraction/loss functions and kernel construction must retain
    # their original AST; initial proof is still seven terms in ten batches.
    assert ast.dump(function(tree,'original_functions'),include_attributes=False)==ast.dump(function(oldtree,'original_functions'),include_attributes=False)
    assert ast.unparse(run.body[0].value.func)=='vm_scope'
    calls=[ast.unparse(n.func) for n in ast.walk(run) if isinstance(n,ast.Call)]
    assert calls.count('torch.autograd.grad')==1 and not any(x.endswith(('backward','AdamW','step')) for x in calls)
    for literal in ["from cctv_dgp_app_input_v28 import canonical_tensor as as_tensor",
                    'from cctv_dgp_active_original_decoder_v28 import ActiveOriginalDecoderV28 as OriginalDecoderCandidate',
                    'retain_graph=i<6,create_graph=False,allow_unused=False',"torch.equal(prediction.detach(),b['base'])",
                    'all_selected12_improvement_gradients_nonzero',"'optimizer_constructed':False",'preflight_active = False']:
        assert literal in worker,literal
    assert worker.index("write(out/'gradient_preflight.json'")<worker.index('        train(root,parent,p,pin,')
    assert '609219' not in ast.unparse(run) and 'len(pieces) == 12' in worker
    helper=(BUNDLE/'cctv_dgp_active_decoder_v28_training.py').read_text(encoding='utf-8')
    training=ast.parse(helper,feature_version=(3,10))
    for literal in ["torch.optim.AdamW(parameters,lr=.00003,weight_decay=.01)",
                    'torch.nn.utils.clip_grad_norm_(parameters,1)',"assert len(selected)==12",'torch.equal(prediction,b[\'base\'])',
                    'canonical_tensor(a,\'cuda\')','snapshot_batch_size\':5',"if update in [50,400,800]",'assert gain>=.01',
                    'gain>=.1','photometric_fraction>.2','actual[metric]>base[metric]+1e-12','actual[metric]<base[metric]-1e-6',
                    "assert frozen_partition(candidate.net,names)==frozen_before",'recheck_parent()',
                    'time.monotonic()-fit_started<1500','assert len(schedule)==800',"progress['optimizer_updates']=update"]:
        assert literal in helper,literal
    assert p['optimizer']=={'type':'AdamW selected12 original decoder tensors','learning_rate':.00003,'weight_decay':.01,'gradient_clip_norm':1,'AMP':False,'EMA':False,
        'reason':'One fixed conservative fine-tuning rate,10x lower than the randomly initialized added-head rate; no sweep or adaptive threshold search. Existing original weights are preserved at initialization.'}
    assert p['budgets']=={'preflight_seconds':300,'fit_seconds':1500,'worker_seconds':1800,'external_seconds':2100,
        'external_kill_grace_seconds':30,'export_seconds':120,'external_export_seconds':150,'external_export_kill_grace_seconds':30,
        'peak_vram_bytes':20*1024**3,'minimum_free_disk_bytes':3*1024**3,'maximum_export_uncompressed_bytes':900_000_000,
        'timing_update':20,'timing_projection_safety_factor':1.25,'overhead_seconds':120}
    shell=(BUNDLE/'scripts/run_v28.sh').read_bytes()
    historical=(parent/'scripts/run_v27.sh').read_bytes()
    assert shell.replace(b'cctv_dgp_active_original_decoder_v28_vm.py',b'cctv_dgp_feature_skips_v27_vm.py')==historical
    before={path.relative_to(BUNDLE).as_posix():sha(path) for path in BUNDLE.rglob('*') if path.is_file()}
    assert set(before)==expected
    executed=subprocess.run([sys.executable,'-B',str(worker_path),'--root',str(BUNDLE),'--protocol-sha',PIN,'--run'],capture_output=True,text=True,timeout=30)
    assert executed.returncode!=0 and 'Existing Linux VM only; no local gradients' in executed.stderr
    assert {path.relative_to(BUNDLE).as_posix():sha(path) for path in BUNDLE.rglob('*') if path.is_file()}==before
    spec=importlib.util.spec_from_file_location('verified_v28_source',worker_path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.verify(BUNDLE,PIN)==p
    try:module.verify(BUNDLE,'0'*64)
    except AssertionError:pass
    else:raise AssertionError('Changed protocol accepted')
    auditor=ROOT/'scripts/audit_cctv_dgp_active_original_decoder_v28_return.py'
    audit_ast=ast.parse(auditor.read_text(encoding='utf-8'),feature_version=(3,10))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['grad','backward','AdamW','SGD','step'] for n in ast.walk(audit_ast))
    result={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':PIN,'archive_sha256':sha(archive),
            'archive_bytes':archive.stat().st_size,'packet_regular_files_verified':8,'original_assets_verified':246,
            'original_R2_all14_and_V27_failures_retained':True,'exact_app_encoder_AST_verified':True,
            'unchanged_finite_schedule_and_quality_gates_verified':True,'initial_app_reference_receipts':50,
            'only_active12_original_decoder_scope':True,'selected12_initial_gradient_proof_precedes_optimizer':True,
            'Windows_run_rejected_before_neural_work_and_outputs':True,'changed_protocol_rejected':True,
            'packet_unchanged_by_boundary_checks':True,'historical_full_shell_deadlines_retained':True,
            'prospective_return_audit_sha256':sha(auditor),'local_neural_or_gradient_calls':0,'VM_actions':False,
            'actual_training_started':False,'manual_VM_execution_required':True,'app_promotion':False,'goal_complete':False,
            'seconds':time.monotonic()-start}
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
