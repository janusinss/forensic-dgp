"""Independent original-copy scope/parity/history readback; no neural calls."""
import ast
import hashlib
import json
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/cctv_dgp_active_original_decoder_v28_review'


def read(p):return json.loads(p.read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    start=time.monotonic();target=OUT/'independent_readback.json';assert not target.exists()
    r=read(OUT/'results.json');plan=read(OUT/'plan.json')
    assert r['complete'] and plan['complete'] and plan['frozen_before_inference']
    assert r['plan_sha256']==sha(OUT/'plan.json') and r['source_bindings_sha256']==plan['source_bindings_sha256']
    for name,digest in r['source_bindings_sha256'].items():
        p=(ROOT/name).resolve();assert p.is_relative_to(ROOT) and sha(p)==digest,name
    assert len(r['source_bindings_sha256'])==254
    old=read(ROOT/'outputs/cctv_dgp_original_decoder_review_v1/review.json')
    layout=[];offset=0
    for row in old['parameter_layout']:
        if row['name'].startswith('head4.'):continue
        size=row['end']-row['start'];layout.append({**row,'start':offset,'end':offset+size});offset+=size
    assert layout==r['parameter_layout'] and offset==r['trainable_parameters']==498627
    assert r['trainable_tensors']==len(layout)==12 and r['frozen_head4_parameters']==110592
    source=ast.parse((ROOT/'scripts/cctv_dgp_active_original_decoder_v28.py').read_text(encoding='utf-8'),feature_version=(3,10))
    cls=next(n for n in source.body if isinstance(n,ast.ClassDef))
    assert [ast.unparse(n) for n in cls.bases]==['OriginalDecoderCandidate']
    assert [n.name for n in cls.body if isinstance(n,ast.FunctionDef)]==['__init__']
    calls=[ast.unparse(n.func) for n in ast.walk(source) if isinstance(n,ast.Call)]
    assert 'super().__init__' in calls and 'value.requires_grad_' in calls
    assert not any(x.endswith(('backward','grad','step','save','load_state_dict')) for x in calls)
    for node in ast.walk(source):
        if isinstance(node,ast.Call) and ast.unparse(node.func)=='value.requires_grad_':
            assert len(node.args)==1 and isinstance(node.args[0],ast.Constant) and node.args[0].value is False
    normalization=read(ROOT/'outputs/cctv_dgp_decoder_app_normalization_v1/results.json')
    assert [x['id'] for x in r['cases']]==[x['id'] for x in normalization['cases']]==plan['case_ids']
    assert len(r['cases'])==r['initial_actual_app_reference_parity_cases']==50
    for row,reference in zip(r['cases'],normalization['cases']):
        assert row['role']=='train' and row['parity_exact']
        for key in ['actual_app_input_raw_sha256','actual_app_input_PNG_sha256','source','profile']:
            assert row[key]==reference[key]
    assert r['DGP_state_before_after']==old['DGP_state_before_after']
    assert r['neural_forward_counts']=={'original':0,'candidate':50}
    assert r['all_encoder_head4_and_normalization_buffers_frozen'] and r['historical_all14_proof_remains_failed']
    assert r['new_CUDA_context_proof_required_before_optimizer'] and 0<r['seconds']<plan['cap_seconds']==300
    assert r['local_gradient_calls']==r['local_backward_calls']==r['local_optimizer_updates']==0
    for key in ['new_checkpoint_created','native_or_reserved_used','VM_actions','app_promotion','independent_final_review','goal_complete']:
        assert r[key] is False
    record={'complete':True,'checker_sha256':sha(Path(__file__)),'results_sha256':sha(OUT/'results.json'),
            'source_bindings_verified':254,'original_forward_inherited_without_override':True,
            'only_requires_grad_flags_for_head4_disabled':True,'selected_parameters':498627,'selected_tensors':12,
            'initial_actual_app_reference_receipts_verified':50,'historical_R2_all14_failure_preserved':True,
            'local_neural_or_gradient_calls':0,'VM_actions':False,'app_promotion':False,'goal_complete':False,
            'seconds':time.monotonic()-start}
    with target.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
