"""New finite manual-only pilot; never modify the failed R2/V27 protocols."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_active_original_decoder_vm_v28'
OUT=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_preparation'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def build_worker():
    original=(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm/scripts/cctv_dgp_original_decoder_gradient_v1_r2_vm.py').read_text(encoding='utf-8')
    header=original[:original.index('def run(')]
    header='"""Finite active-original-decoder L4 pilot. Historical all14 failure retained."""\n'+header[header.index('import argparse'):]
    header=header.replace('own-DGP-original-decoder-zero-update-gradient-proof-v1-r2','own-DGP-active-original-decoder-capacity-v28')
    header=header.replace("assert p['optimizer_updates'] == p['epochs'] == 0","assert p['optimizer_updates'] == 800 and p['epochs'] == 80")
    header=header.replace("p['decoder_parameters'] == 609219 and p['decoder_parameter_tensors'] == 14","p['decoder_parameters'] == 498627 and p['decoder_parameter_tensors'] == 12")
    header=header.replace("cctv_dgp_original_decoder_gradient_v1_r2_vm'","cctv_dgp_active_original_decoder_vm_v28'")
    header=header.replace('    return base\n',"""    previous = parent.parent / 'cctv_dgp_original_decoder_gradient_v1_r2_vm'
    for name,digest in p['closed_R2_evidence_sha256'].items():
        assert sha(previous/name)==digest, 'Retain original R2 failed evidence: '+name
    failure=read(previous/'outputs/failure.json')
    assert failure['component_gradient_calls']==70 and failure['optimizer_updates']==0
    assert 'All14 original decoder tensors' in failure['cause'] and not (previous/'outputs/results.json').exists()
    return base
""")
    prefix=original[original.index('def run('):original.index("        assert all(row['improvement_gradient_norm'] > 0 for row in partitions.values())")]
    prefix=prefix.replace('from cctv_dgp_original_decoder_candidate_v1 import OriginalDecoderCandidate','from cctv_dgp_active_original_decoder_v28 import ActiveOriginalDecoderV28 as OriginalDecoderCandidate')
    prefix=prefix.replace('from dgp_face_restoration import as_tensor','from cctv_dgp_app_input_v28 import canonical_tensor as as_tensor')
    prefix=prefix.replace('609219','498627').replace('len(parameters) == 14','len(parameters) == 12').replace('len(pieces) == 14','len(pieces) == 12').replace('all14','all12')
    prefix=prefix.replace('        def clock():','        preflight_active = True\n        def clock():')
    prefix=prefix.replace("assert time.monotonic() - start < p['budgets']['worker_seconds'], 'Decoder proof worker cap600 seconds'",
                          "assert time.monotonic() - start < (p['budgets']['preflight_seconds'] if preflight_active else p['budgets']['worker_seconds']), 'V28 preflight300s/worker1800s cap'")
    continuation="""        assert all(row['improvement_gradient_norm'] > 0 for row in partitions.values()), 'All12 selected original decoder tensors need finite nonzero improvement gradients before any optimizer'
        assert progress == {'reference_DGP_forwards':10,'candidate_DGP_forwards':10,'recognizer_forwards':20,
                            'component_gradient_calls':70,'optimizer_updates':0,'epochs':0}
        assert state_hash(original.net)==state_hash(candidate.net)==original_state and state_hash(identity)==identity_state
        assert all(value.grad is None for value in candidate.parameters())
        assert all(not value.requires_grad and value.grad is None for value in candidate.net.head4.parameters())
        parent_check(parent,p);clock()
        write(out/'gradient_preflight.json',{'complete':True,'protocol_sha256':pin,'seconds':time.monotonic()-start,
            'decoder_parameters':498627,'decoder_parameter_tensors':12,'parameter_layout':layout,
            'fresh_actual_app_input_initial_rows':initial,'raw_and_PNG_parity_exact_cases':50,
            'normalization':'Actual app NumPy float32 division before device transfer; same5-case batch reference/candidate',
            'all_selected12_improvement_gradients_nonzero':True,'initial_preservation_gradients_exact_zero_all_batches':True,
            'DGP_state_before_after':original_state,'recognizer_state_before_after':identity_state,**progress,
            'optimizer_constructed':False,'new_checkpoint_created':False,'historical_R2_all14_failure_retained':True,
            'peak_allocated_VRAM_bytes':torch.cuda.max_memory_allocated(),'native_or_reserved_used':False,
            'app_promotion':False,'goal_complete':False})
        preflight_active = False
        from cctv_dgp_active_decoder_v28_training import train
        train(root,parent,p,pin,original,candidate,identity,items,progress,start,clock,normalizers,ns,as_tensor,write,sha,
              lambda:parent_check(parent,p))
    except BaseException as exc:
        write(out/'failure.json',{'complete':False,'protocol_sha256':pin,'seconds':time.monotonic()-start,
            **progress,'cause':str(exc),'traceback':traceback.format_exc(),'resume_permitted':False,
            'optimizer_constructed':progress.get('optimizer_constructed',False),
            'new_checkpoint_created':progress.get('new_checkpoint_created',False),
            'app_promotion':False,'goal_complete':False})
        raise


"""
    tail=original[original.index('def export('):]
    tail=tail.replace('cctv-dgp-original-decoder-gradient-v1-r2-results','cctv-dgp-active-original-decoder-v28-results')
    tail=tail.replace('cctv-dgp-original-decoder-gradient-v1-r2-export','cctv-dgp-active-original-decoder-v28-export')
    tail=tail.replace('cctv_dgp_original_decoder_gradient_v1_r2_return/','cctv_dgp_active_original_decoder_v28_return/')
    tail=tail.replace("    receipt = {'complete':True", "    terminal=terminal_record(root)\n    receipt = {'complete':True")
    tail=tail.replace("'training_success_not_implied':True,'optimizer_updates':0,","'training_success_not_implied':True,'optimizer_updates':terminal.get('optimizer_updates',0),")
    tail=tail.replace("'cap_seconds':660,'kill_grace_seconds':30,'within_external_bound':elapsed<=690", "'cap_seconds':2100,'kill_grace_seconds':30,'within_external_bound':elapsed<=2130")
    tail=tail.replace("'optimizer_updates':0,'quality_acceptance_not_implied':True", "'optimizer_updates':terminal_record(root).get('optimizer_updates',0),'quality_acceptance_not_implied':True")
    tail=tail.replace("Decoder proof worker cap600 seconds", "V28 worker cap1800 seconds").replace('signal.alarm(600)','signal.alarm(1800)')
    terminal="""def terminal_record(root):
    paths=[root/'outputs'/name for name in ['results.json','failure.json']]
    existing=[path for path in paths if path.exists()]
    assert len(existing)<=1, 'Keep terminal result and failure separate'
    return read(existing[0]) if existing else {}


"""
    worker=header+prefix+continuation+terminal+tail
    ast.parse(worker,feature_version=(3,10))
    return worker


def main():
    start=time.monotonic();assert not BUNDLE.exists() and not OUT.exists()
    review=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_review/results.json')
    checked=read(ROOT/'outputs/cctv_dgp_active_original_decoder_v28_review/independent_readback.json')
    assert review['complete'] and checked['complete'] and review['initial_actual_app_reference_parity_cases']==50
    audit=read(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json')
    assert audit['complete'] and audit['VM_failure_retained'] and not audit['VM_proof_complete']
    BUNDLE.mkdir();(BUNDLE/'scripts').mkdir();OUT.mkdir()
    for name in ['cctv_dgp_original_decoder_candidate_v1.py','cctv_dgp_active_original_decoder_v28.py','cctv_dgp_active_decoder_v28_training.py']:
        (BUNDLE/name).write_bytes((ROOT/'scripts'/name).read_bytes())
    app=ast.parse((ROOT/'dgp_face_workflow_v3.py').read_text(encoding='utf-8'))
    fn=next(n for n in app.body if isinstance(n,ast.FunctionDef) and n.name=='canonical_tensor')
    (BUNDLE/'cctv_dgp_app_input_v28.py').write_text('"""Exact actual-app encoder AST; no device-side byte division."""\nimport numpy as np\nimport torch\n\n'+ast.unparse(fn)+'\n',encoding='utf-8',newline='\n')
    worker=build_worker();(BUNDLE/'scripts/cctv_dgp_active_original_decoder_v28_vm.py').write_text(worker,encoding='utf-8',newline='\n')
    historical_shell=(ROOT/'outputs/cctv_dgp_feature_skips_vm_v27/scripts/run_v27.sh').read_bytes()
    shell=historical_shell.replace(b'cctv_dgp_feature_skips_v27_vm.py',b'cctv_dgp_active_original_decoder_v28_vm.py')
    (BUNDLE/'scripts/run_v28.sh').write_bytes(shell)
    (BUNDLE/'schedule.json').write_bytes((ROOT/'outputs/cctv_dgp_feature_skips_vm_v27/schedule.json').read_bytes())
    old=read(ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm/protocol.json')
    p=copy.deepcopy(old)
    p.update({'format':'own-DGP-active-original-decoder-capacity-v28','date':'2026-10-06',
      'purpose':'One finite capacity pilot of the12 demonstrated active original decoder tensors; all14 R2 proof remains failed',
      'design':'Use the original DGPSynthesizer forward in a separate copy. Train head1-head3,smooth,smooth2,final only; freeze subnormal head4, complete encoder/FPN and all stored evaluation buffers. Actual app NumPy input encoding; fresh same5-case reference/candidate. No reinitialization, added head or derivative surrogate.',
      'difference_from_failed_recipes':'No added residual head is trained. The original decoder parameter path changes; the two demonstrated inactive head4 tensors remain frozen. Exact original output and preservation anchors survive initialization. R2 all14 is not accepted or changed.',
      'decoder_parameters':498627,'decoder_parameter_tensors':12,'parameter_layout':review['parameter_layout'],
      'optimizer_updates':800,'epochs':80,'snapshot_updates':[0,50,400,800],
      'initial_preservation_terms_exact_zero':old['initial_preservation_terms_exact_zero'],
      'initial_parity':'Every initial actual-app-input raw and PNG equals a fresh same5-case unchanged-DGP reference',
      'normalization':'Exact app canonical_tensor AST: NumPy float32 divide by float32 255 BEFORE transfer, every input/target/delivered image; frozen evaluation IN/BN. Actual CUDA selected12 initial gradient proof precedes any optimizer.',
      'app_input_function_AST_sha256':hashlib.sha256(ast.dump(fn,include_attributes=False).encode()).hexdigest(),
      'optimizer':{'type':'AdamW selected12 original decoder tensors','learning_rate':.00003,'weight_decay':.01,'gradient_clip_norm':1,'AMP':False,'EMA':False,
                   'reason':'One fixed conservative fine-tuning rate,10x lower than the randomly initialized added-head rate; no sweep or adaptive threshold search. Existing original weights are preserved at initialization.'},
      'original_checkpoint_sha256':'646fbb11674e8882ea41e251d6e6e7d468c963979b48c93132def8292834e24b',
      'budgets':{'preflight_seconds':300,'fit_seconds':1500,'worker_seconds':1800,'external_seconds':2100,
                 'external_kill_grace_seconds':30,'export_seconds':120,'external_export_seconds':150,
                 'external_export_kill_grace_seconds':30,'peak_vram_bytes':20*1024**3,
                 'minimum_free_disk_bytes':3*1024**3,'maximum_export_uncompressed_bytes':900_000_000,
                 'timing_update':20,'timing_projection_safety_factor':1.25,'overhead_seconds':120},
      'execution':'Human transfer/SSH/tmux on the existing L4/g2-standard-4 only; no assistant cloud action or automatic follow-on',
      'failure_policy':'Stop first failed source/state/gradient/time/structure/preservation requirement; retain every partial/checkpoint/log. No overwrite, resume, unchanged retry or threshold search.',
      'next':'Independently audit every returned source, matrix, checkpoint, raw/PNG and metric; review all50 faces together before any separately frozen broader native/app qualification.',
      'new_checkpoint_created':False,
      'new_checkpoint_policy':'Only separate VM candidate snapshots/stopped/final artifacts; original files never overwritten',
      'gradient_policy':'Initial actual-app-input selected12 improvement gradients finite, connected and nonzero; all four preservation values/gradients exact zero in every batch before an optimizer. Inactive head4 stays frozen; historical all14 R2 remains failed.',
      'snapshots':[0,50,400,800],
      'case_rows':read(ROOT/'outputs/cctv_dgp_feature_skips_vm_v27/protocol.json')['cases']})
    previous=ROOT/'outputs/cctv_dgp_original_decoder_gradient_v1_r2_return'
    p['closed_R2_protocol_sha256']=sha(previous/'protocol.json')
    for name in ['unissued_draft_protocol_sha256','unissued_draft_change','unissued_R1_protocol_sha256','unissued_R1_source_check_failure']:
        p.pop(name,None)
    p['closed_R2_evidence_sha256']={name:sha(previous/name) for name in ['protocol.json','outputs/failure.json','outputs/gradient_summary.json','trainer_exit_code.txt','supervisor_receipt.json']}
    p['assets_sha256']={path.relative_to(BUNDLE).as_posix():sha(path) for path in sorted(BUNDLE.rglob('*')) if path.is_file()}
    basis=['scripts/prepare_cctv_dgp_active_original_decoder_v28.py','dgp_face_workflow_v3.py',
           'outputs/cctv_dgp_active_original_decoder_v28_review/results.json',
           'outputs/cctv_dgp_active_original_decoder_v28_review/independent_readback.json',
           'outputs/cctv_dgp_original_decoder_head4_review_v1/results.json',
           'outputs/cctv_dgp_original_decoder_head4_review_v1/independent_readback.json',
           'outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json',
           'outputs/cctv_dgp_decoder_app_normalization_v1/results.json',
           'outputs/cctv_dgp_decoder_app_normalization_v1/independent_readback.json',
           'outputs/dgp_original_decoder_review_gradient_v1_r2_milestone/milestone.json']
    p['local_basis_sha256']={name:sha(ROOT/name) for name in basis}
    write(BUNDLE/'protocol.json',p);pin=sha(BUNDLE/'protocol.json')
    archive=ROOT/'outputs/cctv-dgp-active-original-decoder-v28-execution.tar.gz'
    assert not archive.exists()
    with tarfile.open(archive,'x:gz',compresslevel=6) as tar:
        for path in sorted(BUNDLE.rglob('*')):
            if path.is_file():tar.add(path,arcname=BUNDLE.name+'/'+path.relative_to(BUNDLE).as_posix(),recursive=False)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as f:f.write(sha(archive)+'  '+archive.name+'\n')
    record={'complete':True,'protocol_sha256':pin,'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
            'packet_regular_files':len(p['assets_sha256'])+1,'original_sources_data_weights_uploaded':False,
            'local_neural_or_gradient_calls':0,'actual_VM_training_started':False,'manual_VM_execution_required':True,
            'historical_R2_all14_failure_preserved':True,'new_finite_training_protocol_created':True,
            'app_promotion':False,'goal_complete':False,'seconds':time.monotonic()-start}
    write(OUT/'preparation.json',record);print(json.dumps(record,indent=2))


if __name__=='__main__':main()
