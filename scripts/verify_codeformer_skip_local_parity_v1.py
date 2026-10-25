"""Exactly two CPU forwards: local inference wrapper vs functional SFT adapter.

Requires the exported model and all closed scientific review pins. No training,
gradients, optimizer, VM, display enhancement or application promotion occurs.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/codeformer_skip_feature_adapter_return_review_v1/local_parity_v1'


def need(value,message):
    if not value:raise ValueError(message)


def pin(path):
    path=Path(path)
    with path.open('rb') as f:s=hashlib.file_digest(f,'sha256').hexdigest()
    return {'bytes':path.stat().st_size,'sha256':s}


def main():
    start=time.monotonic();need(not OUT.exists(),'Preserve previous/partial parity run')
    import research_skip_feature_restoration_v1 as app
    path,receipt=app.skip_file()  # Exact state/pixel/root-visual closure before Torch.
    import torch
    import numpy as np
    from PIL import Image
    from torch.nn import functional as F
    from third_party.codeformer.codeformer_arch import CodeFormer
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    production,provenance=app.engine.load('skip55',torch)
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)['model']
    reference=CodeFormer(dim_embd=512,codebook_size=1024,n_head=8,n_layers=9,connect_list=list(app.SITES))
    reference.load_state_dict({k.removeprefix('base.'):v for k,v in saved.items() if k.startswith('base.')},strict=True)
    reference.float().eval().requires_grad_(False)
    # Independent functional implementation: no production adapter class/helper.
    site_calls=[];handles=[]
    def hook(site):
        def transform(module,args):
            site_calls.append(site);root='adapters.'+site+'.'
            x=F.conv2d(args[0],saved[root+'down.weight'],saved[root+'down.bias'])
            x=F.silu(x,inplace=False)
            x=F.conv2d(x,saved[root+'depthwise.weight'],saved[root+'depthwise.bias'],padding=1,groups=32)
            x=F.silu(x,inplace=False)
            x=F.conv2d(x,saved[root+'up.weight'],saved[root+'up.bias'])
            return args[0]+x,args[1],args[2]
        return transform
    manifest=json.loads((ROOT/'outputs/cctv_codeformer_skip_feature_adapter_v1_vm/auxiliary_bundle.json').read_bytes())
    row=next(r for r in manifest['cases'] if r['id']=='tr_ffhq_03761__strong_fixed_blur3p2_lr64_jpeg50')
    input_path=ROOT/'outputs/cctv_codeformer_skip_feature_adapter_v1_vm'/row['input']['path']
    need(pin(input_path)=={k:row['input'][k] for k in ('bytes','sha256')},'Frozen input differs')
    with Image.open(input_path) as im:rgb=np.asarray(im).copy()
    x=torch.from_numpy(rgb.astype(np.float32)/np.float32(255)).permute(2,0,1)[None]
    sample=F.interpolate(x,(512,512),mode='bilinear',align_corners=False)*2-1
    before=app.base.state_sha(production.state_dict(),torch)
    OUT.mkdir()
    with (OUT/'started.json').open('x') as f:json.dump({'planned_CPU_forwards':2,'optimizer_updates':0,'source':pin(__file__)},f)
    calls=0
    try:
        for site in app.SITES:handles.append(reference.fuse_convs_dict[site].register_forward_pre_hook(hook(site)))
        calls+=1;expected=reference(sample,w=1.,detach_16=True,adain=True)[0]
    finally:
        for handle in handles:handle.remove()
    calls+=1;actual=production(sample,w=1.,detach_16=True,adain=True)[0]
    need(calls==2 and tuple(site_calls)==app.SITES and production.last_forward_sites==app.SITES,'Exact two-call/four-site parity required')
    need(torch.equal(actual,expected),'Functional reference and local candidate native512 differ')
    def view(value):
        raw=F.interpolate(((value+1)/2).clamp(0,1),(256,256),mode='bilinear',align_corners=False)[0].permute(1,2,0).contiguous().numpy().copy()
        return raw,(raw*255).astype(np.uint8)
    actual_raw,actual_png=view(actual);expected_raw,expected_png=view(expected)
    need(np.array_equal(actual_raw,expected_raw) and np.array_equal(actual_png,expected_png),'RAW/PNG views differ')
    need(app.base.state_sha(production.state_dict(),torch)==before and
         all(p.grad is None and not p.requires_grad for p in production.parameters()) and
         all(not m._forward_pre_hooks and not m._forward_hooks for m in production.base.modules()),'State/gradient/hook namespace changed')
    need(time.monotonic()-start<240,'Finite240s CPU parity cap')
    outputs={}
    for name,raw,png in (('production',actual_raw,actual_png),('functional_reference',expected_raw,expected_png)):
        with (OUT/(name+'.npy')).open('xb') as f:np.save(f,raw,allow_pickle=False)
        with (OUT/(name+'.png')).open('xb') as f:Image.fromarray(png).save(f,format='PNG')
        outputs[name]={'RAW':pin(OUT/(name+'.npy')),'PNG':pin(OUT/(name+'.png'))}
    result={'complete':True,'parity_pass':True,'exact_CPU_native512_RAW256_PNG256':True,'actual_neural_forwards':2,
        'actual_gradient_backward_optimizer_VM_calls':0,'input':row['input'],'export':pin(path),'source':pin(__file__),
        'model539_state_before_after':before,'all4_sites_executed':list(app.SITES),'outputs':outputs,
        'proof_limit':'Same local CPU functional graph/export parity, not CPU/CUDA bit parity, model usefulness or real CCTV identity evidence.',
        'model_qualified':False,'seconds':time.monotonic()-start}
    with (OUT/'parity_v1.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    print(json.dumps({'complete':True,'parity_pass':True,'actual_neural_forwards':2,'seconds':result['seconds']}))


if __name__=='__main__':main()
