"""Finite actual-app-input initial parity; no local derivatives or checkpoints."""
import ast
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
OUT = ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_review'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024**2), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    start = time.monotonic(); assert not OUT.exists()
    p = read(PARENT/'protocol.json')
    basis_names = ['scripts/cctv_dgp_active_original_decoder_v28.py',
                  'scripts/cctv_dgp_original_decoder_candidate_v1.py',
                  'scripts/review_cctv_dgp_active_original_decoder_v28.py', 'dgp_face_workflow_v3.py',
                  'outputs/cctv_dgp_original_decoder_head4_review_v1/results.json',
                  'outputs/cctv_dgp_original_decoder_head4_review_v1/independent_readback.json',
                  'outputs/cctv_dgp_decoder_app_normalization_v1/results.json',
                  'outputs/cctv_dgp_original_decoder_gradient_v1_r2_independent_audit.json']
    bindings = {name:sha(ROOT/name) for name in basis_names}
    for name,digest in p['assets_sha256'].items():
        assert sha(PARENT/name) == digest
        bindings[(PARENT/name).relative_to(ROOT).as_posix()] = digest
    basis = read(ROOT/'outputs/cctv_dgp_original_decoder_head4_review_v1/independent_readback.json')
    assert basis['complete'] and basis['failure_and_all14_gate_preserved']
    assert basis['initial_active_aggregate_tensors'] == 12
    OUT.mkdir()
    write(OUT/'plan.json', {'complete':True,'date':'2026-10-06','frozen_before_inference':True,
          'source_bindings_sha256':bindings,'case_ids':[c['id'] for c in p['cases']],
          'maximum_candidate_forwards':50,'cap_seconds':300,
          'criteria':'Only head4 requires_grad flags change in a separate original-decoder copy; exact actual-app initial reference parity every case. Historical all14 diagnostic remains failed.',
          'local_gradient_calls':0,'local_optimizer_updates':0,'native_or_reserved_used':False,'VM_actions':False})
    import numpy as np
    import torch
    from PIL import Image
    sys.path.insert(0,str(PARENT))
    from cctv_dgp_pilot import state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_active_original_decoder_v28 import ActiveOriginalDecoderV28
    torch.set_num_threads(4)
    original,_ = load_frozen_dgp_restorer(PARENT/'weights/dgp_v2.pth', expected_sha256=p['assets_sha256']['weights/dgp_v2.pth'],device='cpu')
    before = state_hash(original.net)
    candidate = ActiveOriginalDecoderV28(original.net)
    fn = next(n for n in ast.parse((ROOT/'dgp_face_workflow_v3.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='canonical_tensor')
    namespace = {'np':np,'torch':torch}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'<actual-app-encoder-only>','exec'),namespace)
    saved = {r['id']:r for r in read(ROOT/'outputs/cctv_dgp_decoder_app_normalization_v1/results.json')['cases']}
    selected = [];offset=0
    for name,value in candidate.net.named_parameters():
        if value.requires_grad:
            selected.append({'name':name,'shape':list(value.shape),'start':offset,'end':offset+value.numel()})
            offset += value.numel()
    assert offset==498627 and len(selected)==12
    counts={'original':0,'candidate':0}
    original.net.register_forward_hook(lambda *_:counts.__setitem__('original',counts['original']+1))
    candidate.net.register_forward_hook(lambda *_:counts.__setitem__('candidate',counts['candidate']+1))
    rows=[]
    with torch.no_grad():
        for case in p['cases']:
            assert case['role']=='train' and time.monotonic()-start<300
            with Image.open(PARENT/case['input']) as im:camera=np.asarray(im.convert('RGB')).copy()
            with Image.open(PARENT/case['observed']) as im:support=np.asarray(im).copy()>0
            raw=candidate(namespace['canonical_tensor'](camera,'cpu'),torch.from_numpy(support)[None,None])[0].permute(1,2,0).numpy().copy()
            delivered=np.where(support[...,None],np.floor(raw*np.float32(255)),camera).astype(np.uint8)
            raw_sha=hashlib.sha256(raw.tobytes()).hexdigest();png_sha=hashlib.sha256(delivered.tobytes()).hexdigest()
            assert raw_sha==saved[case['id']]['actual_app_input_raw_sha256']
            assert png_sha==saved[case['id']]['actual_app_input_PNG_sha256']
            rows.append({'id':case['id'],'source':case['source'],'profile':case['profile'],'role':'train',
                         'actual_app_input_raw_sha256':raw_sha,'actual_app_input_PNG_sha256':png_sha,'parity_exact':True})
    assert state_hash(original.net)==state_hash(candidate.net)==before
    assert counts=={'original':0,'candidate':50}
    assert all(v.grad is None for v in candidate.parameters())
    for name,digest in bindings.items(): assert sha(ROOT/name)==digest,name
    write(OUT/'results.json',{'complete':True,'date':'2026-10-06','plan_sha256':sha(OUT/'plan.json'),
          'source_bindings_sha256':bindings,'parameter_layout':selected,'trainable_parameters':498627,'trainable_tensors':12,
          'frozen_head4_parameters':110592,'DGP_state_before_after':before,'initial_actual_app_reference_parity_cases':50,
          'cases':rows,'neural_forward_counts':counts,'all_encoder_head4_and_normalization_buffers_frozen':True,
          'historical_all14_proof_remains_failed':True,'new_CUDA_context_proof_required_before_optimizer':True,
          'local_gradient_calls':0,'local_backward_calls':0,'local_optimizer_updates':0,'new_checkpoint_created':False,
          'native_or_reserved_used':False,'VM_actions':False,'app_promotion':False,'independent_final_review':False,
          'goal_complete':False,'seconds':time.monotonic()-start})
    print(json.dumps({'complete':True,'trainable_parameters':498627,'trainable_tensors':12,
          'exact_actual_app_initial_parity_cases':50,'seconds':time.monotonic()-start},indent=2))


if __name__=='__main__': main()
