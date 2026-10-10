"""Freeze initial repair parity and branch response; no local learning allowed."""
import argparse
from datetime import datetime,timezone
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
RETURN = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm_return'
OUT = ROOT/'outputs/cctv_dgp_head4_reactivation_parity_v1'
sys.path.insert(0,str(BUNDLE))
from cctv_dgp_finite_guard_v1_r1_contract import sha,read,write


def pixels(path,mode='RGB'):
    with Image.open(path) as image:
        assert image.size == (256,256)
        return np.asarray(image.convert(mode)).copy()


def prepare():
    assert not OUT.exists(); p = read(BUNDLE/'protocol.json')
    assert read(ROOT/'outputs/cctv_dgp_original_head4_trace_v1/results.json')['head4_all_zero_cases'] == 100
    assert read(ROOT/'outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json')['complete']
    bindings = {}
    def bind(path,expected=None):
        digest = sha(path)
        if expected is not None: assert digest == expected,path
        bindings[Path(path).relative_to(ROOT).as_posix()] = digest
    for name,digest in p['assets_sha256'].items(): bind(BUNDLE/name,digest)
    for c in p['cases']: bind(RETURN/'outputs/baseline'/(c['id']+'.npy'))
    for path in [Path(__file__),ROOT/'scripts/cctv_dgp_head4_reactivation_v1.py',
        ROOT/'scripts/audit_cctv_dgp_head4_reactivation_parity_v1.py',BUNDLE/'protocol.json',
        ROOT/'outputs/cctv_dgp_original_head4_trace_v1/results.json',
        ROOT/'outputs/cctv_dgp_original_head4_weight_audit_v1/statistics.json',
        ROOT/'outputs/cctv_dgp_full_training_coverage_v1/independent_audit.json']:
        bind(path)
    OUT.mkdir()
    write(OUT/'plan.json',{'complete':True,'prepared_before_new_forwards':True,'source_bindings':bindings,
        'case_ids':[c['id'] for c in p['cases']],'cases':100,'CNN_calls':40,'worker_seconds':240,
        'seed_source':'current retained head3 and its adjacent fusion slice, fixed before outputs',
        'original_CPU_to_initial_candidate_raw_max_abs':0.,'CPU_to_VM_raw_max_abs':3e-6,
        'independent_replay_case_ids':[p['cases'][0]['id'],p['cases'][51]['id']],
        'initializations_not_data_fitting':True,'no_random_or_external_pretrained_seed':True,
        'local_gradient_queries':0,'local_optimizer_updates':0,'native_or_final_used':False,
        'prepared_UTC':datetime.now(timezone.utc).isoformat()})
    print({'prepared':True,'cases':100,'CNN_calls':40,'seed':'current head3'},flush=True)


def run():
    start = time.monotonic(); plan,p = read(OUT/'plan.json'),read(BUNDLE/'protocol.json')
    assert not (OUT/'results.json').exists()
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    import torch
    from cctv_dgp_pilot import state_hash,buffer_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_head4_reactivation_v1 import ReactivatedDGP
    torch.set_num_threads(4)
    original,_ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    candidate = ReactivatedDGP(original.net); candidate.eval().requires_grad_(False)
    before = {'original':state_hash(original.net),'candidate':state_hash(candidate),
        'original_buffers':buffer_hash(original.net),'candidate_buffers':buffer_hash(candidate)}
    probes = {}
    def live_probe(module,inputs,output): probes['live_second_conv'] = output.detach().clone()
    hook = candidate.net.head4.block1.register_forward_hook(live_probe)
    rows,calls,worst = [],0,0.
    try:
        with torch.inference_mode():
            for begin in range(0,100,5):
                assert time.monotonic()-start < plan['worker_seconds']
                batch = p['cases'][begin:begin+5]; cameras = [pixels(BUNDLE/c['input']) for c in batch]
                tensor = torch.from_numpy(np.stack(cameras).astype(np.float32)/np.float32(255)).permute(0,3,1,2)
                base = original.net(tensor); fresh = candidate(tensor); calls += 2
                assert torch.equal(base,fresh), 'Initial repair changes original pixels; retain failure'
                array = fresh.permute(0,2,3,1).numpy().copy()
                for i,c in enumerate(batch):
                    observed = pixels(BUNDLE/c['observed'],'L') > 0
                    raw = array[i]; raw[~observed] = cameras[i][~observed].astype(np.float32)/np.float32(255)
                    old = np.load(RETURN/'outputs/baseline'/(c['id']+'.npy'),allow_pickle=False)
                    error = float(np.abs(raw-old).max()); worst = max(worst,error)
                    live = probes['live_second_conv'][i].numpy()
                    rows.append({'id':c['id'],'source':c['source'],'profile':c['profile'],
                        'initial_output_equal_to_original':True,'CPU_VM_max_abs':error,
                        'seed_live_positive_activations':int(np.count_nonzero(live>0)),
                        'seed_live_maximum':float(live.max())})
                if begin % 25 == 0: print({'initial_parity_cases':begin+5,'of':100,'CNN_calls':calls},flush=True)
    finally: hook.remove()
    after = {'original':state_hash(original.net),'candidate':state_hash(candidate),
        'original_buffers':buffer_hash(original.net),'candidate_buffers':buffer_hash(candidate)}
    assert after == before and calls == 40 and worst <= plan['CPU_to_VM_raw_max_abs']
    assert all(not v.requires_grad and v.grad is None for v in candidate.parameters())
    # These calls must fail before any autograd operation or model forward.
    refused = []
    for name,call in [('enable_gradients',lambda:candidate.enable_vm_gradients(ROOT)),
        ('grad_enabled_forward',lambda:candidate(torch.zeros(1,3,256,256)))]:
        try: call()
        except AssertionError: refused.append(name)
        else: raise AssertionError('Local learning guard did not refuse '+name)
    assert len(refused)==2 and after == {'original':state_hash(original.net),'candidate':state_hash(candidate),
        'original_buffers':buffer_hash(original.net),'candidate_buffers':buffer_hash(candidate)}
    snapshot = {'net':candidate.net.state_dict(),'anchors':{name:value for name,value in candidate.named_buffers() if not name.startswith('net.')},
        'starting_checkpoint_sha256':p['original_checkpoint_sha256'],'architecture':'anchored-head4-and-fusion-reactivation-v1',
        'initialization_only':True,'optimizer':None,'scheduler':None,'training_updates':0,'model_qualification':False}
    torch.save(snapshot,OUT/'initialization_only.pth')
    result = {'complete':True,'plan_sha256':sha(OUT/'plan.json'),'rows':rows,'states_before':before,'states_after':after,
        'initial_CPU_parity_cases':100,'maximum_original_CPU_candidate_error':0.,'maximum_CPU_VM_error':worst,
        'normal_scale_seed_response_cases':sum(r['seed_live_positive_activations']>0 for r in rows),
        'reachable_trainable_tensors_planned':160,'reachable_trainable_parameters_planned':2106627,
        'CNN_calls':calls,'local_learning_guards_refused':refused,'local_gradient_queries':0,'local_optimizer_updates':0,
        'new_trained_checkpoint':False,'initialization_checkpoint_sha256':sha(OUT/'initialization_only.pth'),
        'gradient_trainability_not_yet_measured':True,'native_or_final_used':False,'app_adoption':False,
        'model_qualification':False,'goal_complete':False,'seconds':time.monotonic()-start}
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    assert result['seconds'] < plan['worker_seconds']; write(OUT/'results.json',result)
    print({'complete':True,'exact_initial_parity':100,'live_seed_response_cases':result['normal_scale_seed_response_cases'],
        'model_training_updates':0,'seconds':time.monotonic()-start},flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    else:
        try: run()
        except Exception:
            import traceback
            if OUT.exists() and not (OUT/'failure.json').exists():
                write(OUT/'failure.json',{'complete':False,'traceback':traceback.format_exc(),
                    'local_gradient_queries':0,'local_optimizer_updates':0,'goal_complete':False})
            raise


if __name__ == '__main__': main()
