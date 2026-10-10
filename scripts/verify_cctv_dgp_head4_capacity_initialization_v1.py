"""CPU initialization/ownership proof only; no local learning."""
import json
from pathlib import Path
import sys
import time
import hashlib
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/cctv_dgp_head4_capacity_v1_parity'
BUNDLE=ROOT/'outputs/cctv_dgp_head4_reactivation_vm_v1'
sys.path.insert(0,str(BUNDLE))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    with Path(path).open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False)


def main():
    started=time.monotonic();plan=json.loads((OUT/'plan.json').read_text());p=json.loads((BUNDLE/'protocol.json').read_text())
    assert not (OUT/'results.json').exists()
    for n,d in plan['source_bindings'].items():assert sha(ROOT/n)==d,n
    write(OUT/'runner_binding.json',{'complete':True,'prepared_before_forwards':True,'runner_sha256':sha(Path(__file__)),
        'base_plan_sha256':sha(OUT/'plan.json'),'original_protocol_sha256':sha(BUNDLE/'protocol.json')})
    import torch
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_pilot import state_hash,buffer_hash
    from cctv_dgp_head4_capacity_model_v1 import Head4CapacityDGP
    torch.set_num_threads(4)
    original,_=load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth',expected_sha256=p['original_checkpoint_sha256'],device='cpu')
    candidate=Head4CapacityDGP(original.net).eval().requires_grad_(False)
    before={'original':state_hash(original.net),'candidate':state_hash(candidate),'buffers':buffer_hash(candidate)}
    old=original.net.state_dict();new=candidate.net.state_dict();changed={'head4.block0.weight','head4.block1.weight'}
    assert set(old)==set(new)
    assert all(torch.equal(v,new[n]) for n,v in old.items() if n not in changed)
    assert torch.equal(candidate.live_fusion4,old['smooth.0.weight'][:,64:128])
    assert torch.equal(new['head4.block0.weight'],old['head3.block0.weight']) and torch.equal(new['head4.block1.weight'],old['head3.block1.weight'])
    assert sum(v.numel() for v in candidate.learning_parameters())==147456 and len(candidate.learning_parameters())==3
    calls=0
    with torch.inference_mode():
        for begin in range(0,100,5):
            assert time.monotonic()-started<240
            values=[]
            for c in p['cases'][begin:begin+5]:
                with Image.open(BUNDLE/c['input']) as im:values.append(np.asarray(im.convert('RGB')).copy())
            x=torch.from_numpy(np.stack(values).astype(np.float32)/np.float32(255)).permute(0,3,1,2)
            a,b=original.net(x),candidate(x);calls+=2;assert torch.equal(a,b),'Initial output changed'
            if begin%25==0:print({'capacity_initializer_cases':begin+5,'of':100},flush=True)
    after={'original':state_hash(original.net),'candidate':state_hash(candidate),'buffers':buffer_hash(candidate)}
    assert after==before and calls==40 and all(v.grad is None and not v.requires_grad for v in candidate.parameters())
    refused=[]
    for name,fn in [('enable_learning',lambda:candidate.enable_vm_learning(ROOT)),('grad_enabled_forward',lambda:candidate(torch.zeros(1,3,256,256)))]:
        try:fn()
        except AssertionError:refused.append(name)
        else:raise AssertionError('Local learning was allowed')
    assert len(refused)==2
    for n,d in plan['source_bindings'].items():assert sha(ROOT/n)==d,n
    result={'complete':True,'plan_sha256':sha(OUT/'plan.json'),'runner_sha256':sha(Path(__file__)),
        'before':before,'after':after,'exact_initial_parity_cases':100,'maximum_initial_error':0.,'CNN_calls':40,
        'unchanged_full_net_state_entries':len(old)-2,'entire_original_fusion_tensor_unchanged':True,
        'three_independent_learning_pieces':True,'trainable_elements_planned':147456,'local_learning_guards_refused':refused,
        'local_gradient_queries':0,'optimizer_updates':0,'model_qualification':False,'goal_complete':False,'seconds':time.monotonic()-started}
    write(OUT/'results.json',result);print({'complete':True,'exact_parity_cases':100,'unchanged_original_fusion':True,'seconds':time.monotonic()-started},flush=True)


if __name__=='__main__':main()
