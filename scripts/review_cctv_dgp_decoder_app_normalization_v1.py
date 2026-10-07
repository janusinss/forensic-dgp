"""Finite forward-only comparison of the actual app and legacy byte encoders."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs/cctv_dgp_feature_skips_vm_v27'
OUT=ROOT/'outputs/cctv_dgp_decoder_app_normalization_v1'


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def function(path,name):
    tree=ast.parse(path.read_text(encoding='utf-8'))
    return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)


def main():
    started=time.monotonic();assert not OUT.exists(),'Preserve any previous normalization evidence'
    prior=read(ROOT/'outputs/cctv_dgp_original_decoder_review_v1/review.json')
    p=read(BUNDLE/'protocol.json')
    app_path=ROOT/'dgp_face_workflow_v3.py'
    app_function=function(app_path,'canonical_tensor')
    legacy_function=function(BUNDLE/'dgp_face_restoration.py','as_tensor')
    bindings={name:sha(ROOT/name) for name in ['dgp_face_workflow_v3.py','scripts/cctv_dgp_original_decoder_candidate_v1.py',
        'scripts/review_cctv_dgp_decoder_app_normalization_v1.py','outputs/cctv_dgp_original_decoder_review_v1/review.json',
        'outputs/dgp_original_decoder_review_gradient_v1_r2_milestone/milestone.json',
        'outputs/cctv_dgp_original_decoder_gradient_v1_r2_vm/protocol.json']}
    for name,digest in p['assets_sha256'].items():
        assert sha(BUNDLE/name)==digest,name
        bindings[(BUNDLE/name).relative_to(ROOT).as_posix()]=digest
    app_record=read(ROOT/'outputs/dgp_app_v3_integration_record.json')
    for name,digest in {**app_record['sources_sha256'],**app_record['evidence_sha256']}.items():assert sha(ROOT/name)==digest,name
    sys.path.insert(0,str(BUNDLE))
    import numpy as np
    import torch
    from PIL import Image
    from cctv_dgp_pilot import state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_original_decoder_candidate_v1 import OriginalDecoderCandidate
    torch.set_num_threads(4)
    namespace={'np':np,'torch':torch}
    exec(compile(ast.Module(body=[app_function,legacy_function],type_ignores=[]),'<verified-input-functions-only>','exec'),namespace)
    OUT.mkdir()
    write(OUT/'plan.json',{'complete':True,'frozen_before_new_inference':True,'date':'2026-10-06',
        'source_bindings_sha256':bindings,'case_ids':[c['id'] for c in p['cases']],
        'maximum_original_forwards':100,'maximum_candidate_forwards':50,'cap_seconds':300,
        'criteria':['Measure every byte0-255 encoder result on current CPU','Replay legacy receipt hashes',
                    'Exact original/candidate initial parity under the actual app input function for all50 TRAIN cases',
                    'Separate CPU encoding observations from unmeasured CUDA/app-flow or quality acceptance'],
        'training':False,'local_gradient_calls':0,'VM_actions':False,'native_or_reserved_used':False})
    counts={'original':0,'candidate':0}
    original,_=load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth',expected_sha256=p['assets_sha256']['weights/dgp_v2.pth'],device='cpu')
    before=state_hash(original.net)
    candidate=OriginalDecoderCandidate(original.net)
    original.net.register_forward_hook(lambda *_:counts.__setitem__('original',counts['original']+1))
    candidate.net.register_forward_hook(lambda *_:counts.__setitem__('candidate',counts['candidate']+1))
    saved={r['id']:r for r in prior['initial_cases']}
    rows=[]
    with torch.no_grad():
        probe=np.repeat(np.arange(256,dtype=np.uint8)[None,:,None],3,axis=2)
        app_bytes=namespace['canonical_tensor'](probe,'cpu')
        legacy_bytes=namespace['as_tensor'](probe,'cpu')
        mismatch=[int(i) for i in range(256) if not torch.equal(app_bytes[...,i],legacy_bytes[...,i])]
        maximum_byte_error=float((app_bytes-legacy_bytes).abs().max())
        for case in p['cases']:
            assert time.monotonic()-started<300,'App-normalization forward review cap300 seconds'
            assert case['role']=='train'
            with Image.open(BUNDLE/case['input']) as im:camera=np.asarray(im.convert('RGB')).copy()
            with Image.open(BUNDLE/case['observed']) as im:support=np.asarray(im).copy()>0
            mask=torch.from_numpy(support)[None,None]
            old_x=namespace['as_tensor'](camera,'cpu')
            new_x=namespace['canonical_tensor'](camera,'cpu')
            old=torch.where(mask,original(old_x),old_x)
            expected=torch.where(mask,original(new_x),new_x)
            current=candidate(new_x,mask)
            old_raw=old[0].permute(1,2,0).numpy().copy()
            new_raw=expected[0].permute(1,2,0).numpy().copy()
            assert hashlib.sha256(old_raw.tobytes()).hexdigest()==saved[case['id']]['initial_raw_rgb_sha256']
            assert torch.equal(current,expected),'Initial actual app-input candidate parity failed'
            old_png=np.where(support[...,None],np.floor(old_raw*np.float32(255)),camera).astype(np.uint8)
            new_png=np.where(support[...,None],np.floor(new_raw*np.float32(255)),camera).astype(np.uint8)
            candidate_png=np.where(support[...,None],np.floor(current[0].permute(1,2,0).numpy()*np.float32(255)),camera).astype(np.uint8)
            assert hashlib.sha256(old_png.tobytes()).hexdigest()==saved[case['id']]['initial_PNG_rgb_sha256']
            assert np.array_equal(candidate_png,new_png)
            rows.append({'id':case['id'],'source':case['source'],'profile':case['profile'],'role':'train',
                'input_values_differ':int(torch.count_nonzero(old_x!=new_x)),'maximum_input_difference':float((old_x-new_x).abs().max()),
                'maximum_raw_output_difference':float(np.abs(old_raw-new_raw).max()),
                'changed_delivered_RGB_components':int(np.count_nonzero(old_png!=new_png)),
                'changed_delivered_pixels':int(np.any(old_png!=new_png,axis=2).sum()),
                'maximum_delivered_byte_difference':int(np.abs(old_png.astype(np.int16)-new_png.astype(np.int16)).max()),
                'actual_app_input_raw_candidate_parity_exact':True,'actual_app_input_PNG_candidate_parity_exact':True,
                'legacy_raw_and_PNG_receipt_replayed':True,
                'actual_app_input_raw_sha256':hashlib.sha256(new_raw.tobytes()).hexdigest(),
                'actual_app_input_PNG_sha256':hashlib.sha256(new_png.tobytes()).hexdigest()})
    assert counts=={'original':100,'candidate':50}
    assert state_hash(original.net)==state_hash(candidate.net)==before
    assert all(v.grad is None for v in candidate.parameters()) and all(not v.requires_grad and v.grad is None for v in original.parameters())
    for name,digest in bindings.items():assert sha(ROOT/name)==digest,name
    record={'complete':True,'date':'2026-10-06','plan_sha256':sha(OUT/'plan.json'),'source_bindings_sha256':bindings,
        'app_input_function_AST_sha256':hashlib.sha256(ast.dump(app_function,include_attributes=False).encode()).hexdigest(),
        'legacy_input_function_AST_sha256':hashlib.sha256(ast.dump(legacy_function,include_attributes=False).encode()).hexdigest(),
        'byte_values_with_CPU_encoding_difference':mismatch,'maximum_CPU_byte_encoding_difference':maximum_byte_error,
        'cases':rows,'neural_forward_counts':counts,'DGP_state_before_after':before,
        'actual_app_input_exact_initial_candidate_parity_cases':50,'historical_legacy_receipts_replayed':50,
        'encoding_policy_clarification':'App: NumPy float32 division before device transfer. Released R2: torch division after transfer through legacy as_tensor. Its canonical wording does not name the actual app encoder.',
        'R2_packet_and_commands_changed':False,'R2_gradient_proof_still_valid_for_its_declared_source':'Own-DGP graph/initial same-context gradient diagnostic only; not actual app normalization or quality qualification',
        'future_training_requirement':'Use the actual app NumPy-float32-before-transfer encoder in a distinct finite training protocol; require initial parity in that context before optimization. Do not change the released R2 worker or waive old gates.',
        'CUDA_difference_measured':False,'canonical_full_app_flow_verified':False,
        'quality_gain_or_identity_claim':False,'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,
        'VM_actions':False,'native_or_reserved_used':False,'independent_final_review':False,'app_promotion':False,
        'goal_complete':False,'seconds':time.monotonic()-started}
    write(OUT/'results.json',record)
    print(json.dumps({k:record[k] for k in ['complete','byte_values_with_CPU_encoding_difference','maximum_CPU_byte_encoding_difference',
        'neural_forward_counts','actual_app_input_exact_initial_candidate_parity_cases','seconds']},indent=2))


if __name__=='__main__':main()
