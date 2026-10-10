"""Independent parameter-boundary and two-case replay of initialization only."""
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
OUT = ROOT/'outputs/cctv_dgp_head4_reactivation_parity_v1'
PHASE3 = ROOT/'checkpoints/dgp_zamboanga_final.pth'
sys.path.insert(0,str(BUNDLE))


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024**2),b''): h.update(block)
    return h.hexdigest()


def main():
    start = time.monotonic()
    plan,p,result = read(OUT/'plan.json'),read(BUNDLE/'protocol.json'),read(OUT/'results.json')
    assert not (OUT/'independent_audit.json').exists()
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name)==digest,name
    assert result['complete'] and result['plan_sha256']==sha(OUT/'plan.json')
    assert result['initialization_checkpoint_sha256']==sha(OUT/'initialization_only.pth')
    assert [r['id'] for r in result['rows']]==plan['case_ids']
    assert len(result['rows'])==100 and result['CNN_calls']==40
    assert result['states_before']==result['states_after']
    assert result['states_before']['original']==p['original_state']
    assert result['maximum_original_CPU_candidate_error']==0
    assert result['maximum_CPU_VM_error']==max(r['CPU_VM_max_abs'] for r in result['rows'])<=3e-6
    assert all(r['initial_output_equal_to_original'] for r in result['rows'])
    assert result['normal_scale_seed_response_cases']==sum(r['seed_live_positive_activations']>0 for r in result['rows'])
    assert result['local_learning_guards_refused']==['enable_gradients','grad_enabled_forward']
    for name in ['new_trained_checkpoint','native_or_final_used','app_adoption','model_qualification','goal_complete']:
        assert result[name] is False
    for name in ['local_gradient_queries','local_optimizer_updates']: assert result[name]==0
    import torch
    from cctv_dgp_pilot import state_hash,buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_head4_reactivation_v1 import ReactivatedDGP
    torch.set_num_threads(4)
    original,_ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    saved = torch.load(OUT/'initialization_only.pth',map_location='cpu',weights_only=True)
    assert saved['initialization_only'] and saved['optimizer'] is saved['scheduler'] is None
    assert saved['training_updates']==0 and saved['model_qualification'] is False
    assert saved['starting_checkpoint_sha256']==p['original_checkpoint_sha256']
    original_state = original.net.state_dict(); unchanged = 0
    changed = {'head4.block0.weight','head4.block1.weight','smooth.0.weight'}
    assert set(saved['net'])==set(original_state)
    for name,value in original_state.items():
        if name in changed: continue
        assert torch.equal(value,saved['net'][name]),name
        unchanged += 1
    assert torch.equal(saved['net']['smooth.0.weight'][:,64:],original_state['smooth.0.weight'][:,64:])
    assert torch.equal(saved['net']['head4.block0.weight'],original_state['head3.block0.weight'])
    assert torch.equal(saved['net']['head4.block1.weight'],original_state['head3.block1.weight'])
    assert torch.equal(saved['net']['smooth.0.weight'][:,:64],original_state['smooth.0.weight'][:,64:128])
    definitions = {
        'original_head4_0':original_state['head4.block0.weight'],
        'original_head4_1':original_state['head4.block1.weight'],
        'original_fusion4':original_state['smooth.0.weight'][:,:64],
        'anchor_head4_0':original_state['head3.block0.weight'],
        'anchor_head4_1':original_state['head3.block1.weight'],
        'anchor_fusion4':original_state['smooth.0.weight'][:,64:128]}
    assert set(saved['anchors'])==set(definitions)
    for name,value in definitions.items(): assert torch.equal(saved['anchors'][name],value),name
    # Explicit value comparison, rather than inference from equal printed norms.
    phase_sha = sha(PHASE3)
    assert phase_sha=='b6376f56c4161ef1f26a8f9efb0bb309b3014f4ecdeaca5399a47c1c4607586c'
    phase = torch.load(PHASE3,map_location='cpu',weights_only=True)
    phase_equal = {
        'head4.block0.weight':torch.equal(phase['head4.block0.weight'],original_state['head4.block0.weight']),
        'head4.block1.weight':torch.equal(phase['head4.block1.weight'],original_state['head4.block1.weight']),
        'fusion_head4_slice':torch.equal(phase['smooth.0.weight'][:,:64],original_state['smooth.0.weight'][:,:64])}
    candidate = ReactivatedDGP(original.net).eval().requires_grad_(False)
    candidate.net.load_state_dict(saved['net'],strict=True)
    with torch.no_grad():
        for name,value in saved['anchors'].items(): getattr(candidate,name).copy_(value)
    assert state_hash(candidate)==result['states_before']['candidate']
    assert buffer_hash(candidate)==result['states_before']['candidate_buffers']
    assert all(not m.training for m in candidate.modules())
    assert all(not v.requires_grad and v.grad is None for v in candidate.parameters())
    replay=[]
    cases = {c['id']:c for c in p['cases']}
    with torch.inference_mode():
        for cid in plan['independent_replay_case_ids']:
            with Image.open(BUNDLE/cases[cid]['input']) as image: array=np.asarray(image.convert('RGB')).copy()
            value=torch.from_numpy(array.astype(np.float32)/np.float32(255)).permute(2,0,1).unsqueeze(0)
            a,b=original.net(value),candidate(value)
            error=float((a-b).abs().max()); assert torch.equal(a,b)
            replay.append({'id':cid,'fresh_original_initialization_max_abs':error})
    assert state_hash(candidate)==result['states_before']['candidate']
    assert state_hash(original.net)==p['original_state'] and sha(PHASE3)==phase_sha
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name)==digest,name
    receipt = {'complete':True,'plan_sha256':sha(OUT/'plan.json'),'results_sha256':sha(OUT/'results.json'),
        'checker_sha256':sha(Path(__file__)),'source_bindings_verified':len(plan['source_bindings']),
        'unchanged_full_state_entries':unchanged,'changed_full_entries':sorted(changed),
        'unchanged_adjacent_fusion_channels':192,'six_fixed_anchors_exact':True,
        'phase3_dead_branch_value_equality':phase_equal,'Phase3_checkpoint_sha256':phase_sha,
        'owned_trainable_tensors_planned':160,'owned_trainable_elements_planned':2106627,
        'fresh_independent_replays':replay,'fresh_CNN_calls':4,'all100_independently_replayed':False,
        'worker100_parity_records_verified':True,'local_gradient_queries':0,'local_optimizer_updates':0,
        'gradient_trainability_not_yet_measured':True,'initialization_only':True,
        'model_qualification':False,'app_adoption':False,'goal_complete':False,'seconds':time.monotonic()-start}
    with (OUT/'independent_audit.json').open('x',encoding='utf-8',newline='\n') as stream:
        stream.write(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print({'complete':True,'unchanged_state_entries':unchanged,'fresh_replays':2,
        'Phase3_branch_equal':phase_equal,'local_updates':0,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__': main()
