"""Independent transfer, finite-plan, regression and forward-only projection checks."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_mean_centered_decoder_vm_v29'
OLD=ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'
PREP=ROOT/'outputs/cctv_dgp_mean_centered_decoder_v29_preparation'
PIN='77565ba437959305f22cff4dd967fc6c3caadbf9dbd4ac91abf8a366577fa72f'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def function(tree,name):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def same_ast(a,b):return ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)


def main():
    start=time.monotonic();assert not (PREP/'independent_packet_audit.json').exists()
    prep=read(PREP/'preparation.json');p=read(BUNDLE/'protocol.json');old=read(OLD/'protocol.json')
    assert sha(BUNDLE/'protocol.json')==prep['protocol_sha256']==PIN
    archive=ROOT/'outputs/cctv-dgp-mean-centered-decoder-v29-execution.tar.gz'
    assert archive.stat().st_size==prep['archive_bytes']==30293 and sha(archive)==prep['archive_sha256']
    assert Path(str(archive)+'.sha256').read_text().strip()==sha(archive)+'  '+archive.name
    for name,digest in p['assets_sha256'].items():assert sha(BUNDLE/name)==digest,name
    for name,digest in p['local_basis_sha256'].items():assert sha(ROOT/name)==digest,name
    expected={'protocol.json',*p['assets_sha256']}
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();assert len(members)==7==prep['packet_regular_files']
        assert all(m.isfile() and not m.issym() and not m.islnk() for m in members)
        assert {m.name for m in members}=={BUNDLE.name+'/'+name for name in expected}
        for m in members:
            assert hashlib.sha256(tar.extractfile(m).read()).hexdigest()==sha(BUNDLE/m.name[len(BUNDLE.name)+1:])
    parent=ROOT/'outputs/cctv_dgp_feature_skips_vm_v27';base=read(parent/'protocol.json')
    for name,digest in base['assets_sha256'].items():assert sha(parent/name)==digest,name
    assert len(base['assets_sha256'])==246
    for folder,key in [(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return','closed_V28_evidence_sha256'),
                       (ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return','closed_diagnostic_evidence_sha256'),
                       (ROOT/'outputs/cctv_dgp_feature_skips_v27_return','closed_V27_evidence_sha256'),
                       (ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return','closed_R2_evidence_sha256')]:
        for name,digest in p[key].items():assert sha(folder/name)==digest,name
    for key in ['optimizer','optimizer_updates','epochs','parameter_layout','decoder_parameters','decoder_parameter_tensors',
                'terms','initial_preservation_terms_exact_zero','retained_capacity_gates','budgets','snapshot_updates','snapshots','case_rows']:
        assert p[key]==old[key],key
    assert p['case_rows']==base['cases'] and p['retained_capacity_gates']==base['prospective_gates']
    assert all(c['role']=='train' for c in p['case_rows']) and len({c['source_person_or_reference'] for c in p['case_rows']})==10
    assert (BUNDLE/'schedule.json').read_bytes()==(OLD/'schedule.json').read_bytes()
    assert (BUNDLE/'cctv_dgp_app_input_v28.py').read_bytes()==(OLD/'cctv_dgp_app_input_v28.py').read_bytes()
    assert (BUNDLE/'cctv_dgp_mean_centered_decoder_v29.py').read_bytes()==(ROOT/'scripts/cctv_dgp_mean_centered_decoder_v29.py').read_bytes()
    for path in BUNDLE.rglob('*.py'):ast.parse(path.read_text(encoding='utf-8'),feature_version=(3,10))
    worker=BUNDLE/'scripts/cctv_dgp_mean_centered_decoder_v29_vm.py';text=worker.read_text();tree=ast.parse(text)
    original_tree=ast.parse((OLD/'scripts/cctv_dgp_active_original_decoder_v28_vm.py').read_text())
    for name in ['parent_check','original_functions']:
        assert same_ast(function(tree,name),function(original_tree,name)),name
    run=function(tree,'run');assert ast.unparse(run.body[0].value.func)=='vm_scope'
    calls=[ast.unparse(n.func) for n in ast.walk(run) if isinstance(n,ast.Call)]
    assert calls.count('torch.autograd.grad')==1 and not any(x.endswith(('backward','AdamW','step')) for x in calls)
    assert text.index("write(out/'gradient_preflight.json'")<text.index('        train(root,parent,p,pin,')
    assert "candidate(b['x'],b['mask'],b['base'])" in text
    assert 'retain_graph=i<6,create_graph=False,allow_unused=False' in text and 'len(pieces) == 12' in text
    helper=(BUNDLE/'cctv_dgp_mean_centered_v29_training.py').read_text()
    old_helper=(OLD/'cctv_dgp_active_decoder_v28_training.py').read_text()
    assert helper.count("candidate(b['x'],b['mask'],b['base'])")==2
    assert same_ast(function(ast.parse(helper),'frozen_partition'),function(ast.parse(old_helper),'frozen_partition'))
    for literal in ['torch.optim.AdamW(parameters,lr=.00003,weight_decay=.01)',
                    'torch.nn.utils.clip_grad_norm_(parameters,1)','assert gain>=.01','gain>=.1',
                    'photometric_fraction>.2','actual[metric]>base[metric]+1e-12','actual[metric]<base[metric]-1e-6',
                    'time.monotonic()-fit_started<1500','assert len(schedule)==800','if update in [50,400,800]']:
        assert literal in helper,literal
    shell=(BUNDLE/'scripts/run_v29.sh').read_bytes()
    assert shell.replace(b'cctv_dgp_mean_centered_decoder_v29_vm.py',b'cctv_dgp_active_original_decoder_v28_vm.py')==(OLD/'scripts/run_v28.sh').read_bytes()
    assert b'2100s' in shell and b'150s' in shell and b'--kill-after=30s' in shell
    before={path.relative_to(BUNDLE).as_posix():sha(path) for path in BUNDLE.rglob('*') if path.is_file()}
    execution=subprocess.run([sys.executable,'-B',str(worker),'--root',str(BUNDLE),'--protocol-sha',PIN,'--run'],capture_output=True,text=True,timeout=30)
    assert execution.returncode!=0 and 'Existing Linux VM only; no local gradients' in execution.stderr
    assert not (BUNDLE/'outputs').exists() and before=={path.relative_to(BUNDLE).as_posix():sha(path) for path in BUNDLE.rglob('*') if path.is_file()}
    write(PREP/'actual_Windows_rejection.json',{'complete':True,'exit_code':execution.returncode,'stderr':execution.stderr,
          'outputs_created':False,'neural_calls':0,'gradient_calls':0,'packet_unchanged':True})
    spec=importlib.util.spec_from_file_location('verified_v29_worker',worker);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    assert module.verify(BUNDLE,PIN)==p
    try:module.verify(BUNDLE,'0'*64)
    except AssertionError:pass
    else:raise AssertionError('Changed protocol accepted')
    test=subprocess.run([sys.executable,'-B','-m','unittest','discover','-s','tests','-p','test_cctv_dgp_mean_centered_decoder_v29*.py'],cwd=ROOT,capture_output=True,text=True,timeout=60)
    assert test.returncode==0 and 'Ran 17 tests' in test.stderr
    write(PREP/'test_receipt.json',{'complete':True,'tests_passed':17,'stdout':test.stdout,'stderr':test.stderr,'local_gradient_calls':0,'optimizer_updates':0})
    sys.path.insert(0,str(OLD));sys.path.insert(0,str(BUNDLE))
    import torch
    from PIL import Image
    from cctv_dgp_mean_centered_decoder_v29 import center_observed_delta
    case_rows=[];largest=0.0
    with torch.no_grad():
        for case in p['case_rows']:
            cid=case['id'];v28=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_return/outputs'
            baseline=np.load(v28/'initial_baseline'/(cid+'.npy'),allow_pickle=False)
            final=np.load(v28/'update800'/(cid+'.npy'),allow_pickle=False)
            with Image.open(parent/case['input']) as im:camera=np.asarray(im.convert('RGB')).copy()
            with Image.open(parent/case['observed']) as im:support=np.asarray(im).copy()>0
            tensor=lambda a:torch.from_numpy(a.copy()).permute(2,0,1)[None]
            image=camera.astype(np.float32)/np.float32(255);mask=torch.from_numpy(support)[None,None]
            zero=center_observed_delta(tensor(baseline),tensor(baseline),tensor(image),mask)
            assert np.array_equal(zero[0].permute(1,2,0).numpy(),baseline)
            answer=center_observed_delta(tensor(final),tensor(baseline),tensor(image),mask)[0].permute(1,2,0).numpy()
            delta=final.astype(np.float64)-baseline.astype(np.float64)
            mean=delta[support].mean(0)
            reference=np.where(support[...,None],np.clip(final.astype(np.float64)-mean,0,1),image).astype(np.float32)
            error=float(np.abs(answer-reference).max());assert error<=2e-6
            assert np.array_equal(answer[~support],image[~support]) and np.isfinite(answer).all() and (answer>=0).all() and (answer<=1).all()
            largest=max(largest,error)
            case_rows.append({'id':cid,'initial_exact_parity':True,'maximum_float64_reference_error':error,
                              'postclip_RGB_delta_mean':(answer.astype(np.float64)-baseline.astype(np.float64))[support].mean(0).tolist(),
                              'transient_projected_array_sha256':hashlib.sha256(answer.tobytes()).hexdigest()})
    write(PREP/'all50_projection_arithmetic.json',{'complete':True,'rows':case_rows,'cases':50,'maximum_float64_reference_error':largest,
          'CNN_or_recognizer_calls':0,'local_gradient_calls':0,'not_a_new_quality_evaluation':True,'clipping_drift_is_recorded':True})
    a=read(ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_analysis/results.json')
    src=ROOT/'outputs/cctv_dgp_v28_preservation_diagnostic_v1_return/outputs'
    g=np.load(src/'gradient_components.npy',allow_pickle=False);delta=np.load(src/'parameter_displacement.npy',allow_pickle=False)
    assert np.array_equal(-(g@g[:7].sum(0)),np.asarray(a['directional_derivative_along_negative_original_objective_gradient']))
    assert np.array_equal(g@delta,np.asarray(a['component_dot_observed_original_to_final_displacement']))
    for name,digest in a['source_sha256'].items():assert sha(ROOT/name)==digest,name
    auditor=ROOT/'scripts/audit_cctv_dgp_mean_centered_decoder_v29_return.py'
    assert sha(auditor)==prep['prospective_auditor_sha256']
    audit_tree=ast.parse(auditor.read_text(),feature_version=(3,10))
    assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ['grad','backward','AdamW','SGD','step'] for n in ast.walk(audit_tree))
    assert same_ast(function(audit_tree,'capacity'),function(ast.parse((ROOT/'scripts/audit_cctv_dgp_active_original_decoder_v28_return.py').read_text()),'capacity'))
    runbook=ROOT/'CCTV_DGP_MEAN_CENTERED_DECODER_V29_VM.md';book=runbook.read_text()
    for number in range(1,6):assert '\n'+str(number)+'. ' in book
    assert book.count('gcloud compute scp')==5 and book.count('--project=forensic-dgp-thesis --zone=us-central1-a')==5
    assert book.count('janusdominic0@forensic-dgp-thesis:')==5 and 'tmux new-session -A -s dgp_v29' in book
    assert PIN in book and prep['archive_sha256'] in book and '30,293 bytes' in book
    for suffix in ['results.tar.gz','results.tar.gz.sha256','export.json']:
        assert 'cctv-dgp-mean-centered-decoder-v29-'+suffix+'" "."' in book
    receipt={'complete':True,'checker_sha256':sha(Path(__file__)),'protocol_sha256':PIN,'archive_sha256':sha(archive),'archive_bytes':30293,
        'packet_regular_files_verified':7,'original_assets_verified':246,'closed_V28_files_verified':11,'closed_diagnostic_files_verified':9,
        'loss_weights_optimizer_schedule_and_capacity_gates_unchanged':True,'selected12_initial_proof_precedes_optimizer':True,
        'same_input_baseline_without_target_profile_source_identity_forward_conditions':True,'zero_learned_projection_parameters':True,
        'exact_app_input_encoder_and_initial50_projection_parity':True,'all50_projection_reference_arithmetic_verified':True,
        'projection_reference_maximum_error':largest,'changed_protocol_rejected':True,'actual_Windows_rejection_before_neural_or_gradients':True,
        'saved_VM_gradient_analysis_arithmetic_verified':True,'return_and_projection_regressions_passed':17,
        'prospective_return_auditor_sha256':sha(auditor),'runbook_sha256':sha(runbook),'five_manual_steps_verified':True,
        'actual_VM_training_started':False,'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,
        'VM_actions':False,'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start}
    write(PREP/'independent_packet_audit.json',receipt);print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
