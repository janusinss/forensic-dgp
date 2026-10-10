"""Pre-return fixed-filter arithmetic check on archived TRAIN floats; no models."""
import ast
from pathlib import Path
import time
from types import MethodType,SimpleNamespace
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from cctv_dgp_actual_step_review_v1_contract import NAME,read,write,sha
from audit_cctv_dgp_actual_step_review_v1_return import raw_terms
from cctv_dgp_spatial_fit_v40_contract import erode,feature_support

ROOT=Path(__file__).resolve().parents[1]
BUNDLE=ROOT/'outputs'/NAME
RETURN=ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return'
OUT=ROOT/'outputs/cctv_dgp_actual_step_review_v1_preparation'
ALLOWANCE=np.array([5e-5,5e-5,1e-5,1e-5,1e-5,5e-4,1e-5])


def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    p=read(BUNDLE/'protocol.draft.json');imported=read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_import.json')
    torch.set_num_threads(4);torch.set_grad_enabled(False)
    tree=ast.parse((BUNDLE/'frozen_definitions.py').read_text());names={'blur','high','mean','feature_errors','ssim'}
    selected=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    ns={'torch':torch,'F':F};exec(compile(ast.Module(body=selected,type_ignores=[]),'<fixed-filter-arithmetic-only>','exec'),ns)
    z=torch.arange(-6,7,dtype=torch.float32);kernel=torch.exp(-.5*(z/2).square());kernel/=kernel.sum()
    head=SimpleNamespace(kernel=kernel,reflect_indices=torch.cat((torch.arange(5,-1,-1),torch.arange(256),torch.arange(255,249,-1))))
    head.blur=MethodType(ns['blur'],head);head.high=MethodType(ns['high'],head);ns['head']=head
    errors=np.zeros(7);checked=0;bindings={};cases={c['id']:c for c in p['cases']}
    canonical=lambda a:torch.from_numpy(a).permute(2,0,1)[None]
    for cid in p['cohorts']['not_yet_optimized']:
        c=cases[cid];target8=np.asarray(Image.open(BUNDLE/c['target']).convert('RGB')).copy()
        mask=np.asarray(Image.open(BUNDLE/c['observed']).convert('L'))>0;support=feature_support(mask,c['landmarks5_canvas_xy'])
        target=target8.astype(np.float32)/np.float32(255)
        load=lambda folder,suffix:np.load(RETURN/f'outputs/{folder}/{cid}{suffix}.npy',allow_pickle=False)
        for folder in ['update0','update50']:
            for suffix in ['', '_embedding']:
                path=RETURN/f'outputs/{folder}/{cid}{suffix}.npy';name=path.relative_to(RETURN).as_posix()
                assert sha(path)==imported['files_sha256'][name];bindings[path.relative_to(ROOT).as_posix()]=sha(path)
        path=RETURN/f'outputs/update0/{cid}_target_embedding.npy';assert sha(path)==imported['files_sha256'][path.relative_to(RETURN).as_posix()]
        bindings[path.relative_to(ROOT).as_posix()]=sha(path)
        base=load('update0','');truth=load('update0','_target_embedding');reference=load('update0','_embedding')
        for folder in ['update0','update50']:
            assert time.monotonic()-start<180
            raw=load(folder,'');vector=load(folder,'_embedding')
            a={'rgb':raw,'truth':truth,'raw_vector':vector,'raw_reference_vector':reference}
            independent,anchor=raw_terms(a,base,target8,mask,support,p)
            actual_t=canonical(raw);target_t=canonical(target);base_t=canonical(base)
            mask_t=torch.from_numpy(mask.astype(np.float32))[None,None]
            feat_t=torch.from_numpy(support.astype(np.float32))[None,None]
            interior=torch.from_numpy(erode(mask,6).astype(np.float32))[None,None]
            valid=torch.from_numpy(erode(mask,3).astype(np.float32))[None,None]
            f,i=ns['feature_errors'](actual_t,target_t,feat_t,interior)
            pixel=ns['mean']((actual_t-target_t).square(),mask_t);bp=ns['mean']((base_t-target_t).square(),mask_t)
            bs=ns['ssim'](base_t,target_t,valid);score=ns['ssim'](actual_t,target_t,valid)
            bc=(torch.from_numpy(reference)*torch.from_numpy(truth)).sum();cosine=(torch.from_numpy(vector)*torch.from_numpy(truth)).sum()
            reward=0. if c['profile']=='clear' else 1.25
            values=[reward*f/np.float32(p['normalizers']['feature']),reward*.25*i/np.float32(p['normalizers']['interior']),
                reward*.05*pixel/bp.clamp_min(1e-5),.05*ns['mean']((actual_t-base_t).square(),mask_t)/bp.clamp_min(1e-5) if not reward else torch.zeros_like(pixel),
                2*F.relu((pixel-bp)/bp.clamp_min(1e-5)),5*F.relu(bs-score),5*F.relu(bc-cosine)]
            expected=np.array([v.item() for v in values],dtype=np.float64)
            if not reward:independent[:3]=0;independent[3]=anchor
            error=np.abs(independent-expected);errors=np.maximum(errors,error)
            assert np.all(error<=ALLOWANCE),(cid,folder,error.tolist())
            assert all(not v.requires_grad and v.grad_fn is None for v in values)
            checked+=1
    assert checked==100 and time.monotonic()-start<180
    write(OUT/'fixed_filter_arithmetic.json',{'complete':True,'archived_float_cases':100,'case_source':'unchanged preselected TRAIN cohort, V41 update0/update50',
          'maximum_component_errors':errors.tolist(),'prospective_component_allowances':ALLOWANCE.tolist(),
          'ArcFace_arithmetic_inputs':'Saved PNG vectors supplied to both formulas solely for reduction arithmetic; no raw embedding parity claim',
          'source_evidence_sha256':bindings,'checker_sha256':sha(Path(__file__)),'new_model_forwards':0,'model_parameters_loaded':0,
          'optimizer_updates':0,'gradient_queries':0,'VM_calls':0,'seconds':time.monotonic()-start,'cap_seconds':180})
    print({'complete':True,'cases':checked,'maximum_component_errors':errors.tolist(),'new_model_forwards':0,'seconds':time.monotonic()-start},flush=True)


if __name__=='__main__':main()
