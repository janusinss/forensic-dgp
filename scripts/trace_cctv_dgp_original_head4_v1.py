"""Frozen current-model inference trace: parameter ownership and head activity only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm'
RETURN = ROOT/'outputs/cctv_dgp_finite_guard_v1_r1_vm_return'
OUT = ROOT/'outputs/cctv_dgp_original_head4_trace_v1'
sys.path.insert(0,str(BUNDLE))
from cctv_dgp_finite_guard_v1_r1_contract import read,write,sha


def pixels(path,mode='RGB'):
    with Image.open(path) as image:
        assert image.size == (256,256)
        return np.asarray(image.convert(mode)).copy()


def prepare():
    assert not OUT.exists(); p = read(BUNDLE/'protocol.json')
    assert sha(BUNDLE/'protocol.json') == 'efdd62759213136114a56f0aaa256278753cac8bac4bc6c6a52f1f8b7550598f'
    bindings = {}
    def bind(path,expected=None):
        digest = sha(path)
        if expected is not None: assert digest == expected,path
        bindings[Path(path).relative_to(ROOT).as_posix()] = digest
    for name,digest in p['assets_sha256'].items(): bind(BUNDLE/name,digest)
    for c in p['cases']: bind(RETURN/'outputs/baseline'/(c['id']+'.npy'))
    for path in [Path(__file__),BUNDLE/'protocol.json',ROOT/'outputs/cctv_dgp_post_finite_guard_parameter_inventory_v1/inventory.json']:
        bind(path)
    OUT.mkdir()
    write(OUT/'plan.json',{'complete':True,'prepared_before_new_forwards':True,
        'source_bindings':bindings,'case_ids':[c['id'] for c in p['cases']],
        'cases':100,'maximum_model_forward_calls':20,'batch_size':5,'wall_seconds':180,
        'raw_CPU_VM_replay_max_abs':3e-6,'parameter_or_buffer_updates':0,
        'gradient_queries':0,'optimizer_updates':0,'native_or_final_used':False,
        'claim_scope':'Reached module and activation statistics on100 exposed TRAIN cases; not a gradient test or global capacity proof',
        'prepared_UTC':datetime.now(timezone.utc).isoformat()})
    print({'prepared':True,'cases':100,'maximum_forwards':20,'neural_calls':0},flush=True)


def run():
    started = time.monotonic(); plan,p = read(OUT/'plan.json'),read(BUNDLE/'protocol.json')
    assert not (OUT/'results.json').exists()
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    import torch
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_pilot import state_hash,buffer_hash
    torch.set_num_threads(4)
    restorer,_ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth',
        expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    net = restorer.net; before = (state_hash(net),buffer_hash(net))
    assert before[0] == p['original_state']
    named = dict(net.named_parameters()); by_object = {id(value):name for name,value in named.items()}
    selected = {r['name'] for r in p['parameter_layout']}; reached,setups = set(),[]
    for module in net.modules():
        owned = [id(value) for value in module.parameters(recurse=False)]
        if owned:
            def touch(owner,inputs,outputs,owned=owned): reached.update(owned)
            setups.append(module.register_forward_hook(touch))
    activations = {}
    for number in range(1,5):
        def head_hook(module,inputs,output,number=number): activations[number] = output.detach().clone()
        setups.append(getattr(net,'head'+str(number)).register_forward_hook(head_hook))
    preactivations = {}
    for name in ['block0','block1']:
        def pre_hook(module,inputs,output,name=name): preactivations[name] = output.detach().clone()
        setups.append(getattr(net.head4,name).register_forward_hook(pre_hook))
    weights = {}
    for name in ['block0','block1']:
        value = getattr(net.head4,name).weight.detach().numpy()
        weights[name] = {'shape':list(value.shape),'nonzero':int(np.count_nonzero(value)),
            'positive':int(np.count_nonzero(value>0)),'negative':int(np.count_nonzero(value<0)),
            'minimum':float(value.min()),'maximum':float(value.max()),'L2':float(np.linalg.norm(value.astype(np.float64)))}
    records,calls,worst = [],0,0.
    try:
        with torch.inference_mode():
            for begin in range(0,100,5):
                assert time.monotonic()-started < plan['wall_seconds']
                batch = p['cases'][begin:begin+5]
                cameras = [pixels(BUNDLE/c['input']) for c in batch]
                value = torch.from_numpy(np.stack(cameras).astype(np.float32)/np.float32(255)).permute(0,3,1,2)
                activations.clear(); preactivations.clear()
                output = net(value).permute(0,2,3,1).numpy().copy(); calls += 1
                assert set(activations) == {1,2,3,4} and set(preactivations) == {'block0','block1'}
                for i,c in enumerate(batch):
                    observed = pixels(BUNDLE/c['observed'],'L') > 0
                    raw = output[i]; raw[~observed] = cameras[i][~observed].astype(np.float32)/np.float32(255)
                    cached = np.load(RETURN/'outputs/baseline'/(c['id']+'.npy'),allow_pickle=False)
                    error = float(np.abs(raw-cached).max()); worst = max(worst,error)
                    row = {'id':c['id'],'source':c['source'],'profile':c['profile'],'raw_CPU_VM_max_abs':error,
                        'heads':{},'head4_preactivations':{}}
                    for number,tensor in activations.items():
                        array = tensor[i].numpy()
                        row['heads'][str(number)] = {'shape':list(array.shape),'nonzero':int(np.count_nonzero(array)),
                            'zero_fraction':float(np.mean(array==0)),'minimum':float(array.min()),'maximum':float(array.max())}
                    for name,tensor in preactivations.items():
                        array = tensor[i].numpy()
                        row['head4_preactivations'][name] = {'shape':list(array.shape),'positive':int(np.count_nonzero(array>0)),
                            'nonzero':int(np.count_nonzero(array)),'minimum':float(array.min()),'maximum':float(array.max())}
                    records.append(row)
                if begin % 25 == 0: print({'traced_cases':begin+5,'of':100,'model_calls':calls},flush=True)
    finally:
        for hook in setups: hook.remove()
    assert calls == 20 and worst <= plan['raw_CPU_VM_replay_max_abs']
    assert before == (state_hash(net),buffer_hash(net))
    assert all(not value.requires_grad and value.grad is None for value in net.parameters())
    active = {by_object[v] for v in reached}; inactive = set(named)-active
    assert selected <= active
    result = {'complete':True,'plan_sha256':sha(OUT/'plan.json'),'rows':records,'weight_statistics':weights,
        'owned_tensors':len(named),'owned_parameters':sum(v.numel() for v in named.values()),
        'reached_tensors':len(active),'reached_parameters':sum(named[n].numel() for n in active),
        'reached_unselected_names':sorted(active-selected),'unreached_names':sorted(inactive),
        'head4_all_zero_cases':sum(r['heads']['4']['nonzero']==0 for r in records),
        'head4_block0_all_nonpositive_cases':sum(r['head4_preactivations']['block0']['positive']==0 for r in records),
        'head4_block1_all_nonpositive_cases':sum(r['head4_preactivations']['block1']['positive']==0 for r in records),
        'maximum_raw_replay_error':worst,'states_before':list(before),'states_after':list(before),
        'model_forward_calls':calls,'gradient_queries':0,'optimizer_updates':0,'parameter_or_buffer_updates':0,
        'neural_gradients_measured':False,'native_or_final_used':False,'new_checkpoint':False,
        'model_qualification':False,'goal_complete':False,'seconds':time.monotonic()-started}
    for name,digest in plan['source_bindings'].items(): assert sha(ROOT/name) == digest,name
    assert result['seconds'] < plan['wall_seconds']; write(OUT/'results.json',result)
    print({'complete':True,'head4_all_zero_cases':result['head4_all_zero_cases'],
        'reached_unselected':result['reached_unselected_names'],'maximum_raw_error':worst,'seconds':time.monotonic()-started},flush=True)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare',action='store_true'); args = parser.parse_args()
    if args.prepare: prepare()
    else:
        try: run()
        except Exception:
            import traceback
            if OUT.exists() and not (OUT/'failure.json').exists():
                write(OUT/'failure.json',{'complete':False,'traceback':traceback.format_exc(),
                    'gradient_queries':0,'optimizer_updates':0,'goal_complete':False})
            raise


if __name__ == '__main__': main()
