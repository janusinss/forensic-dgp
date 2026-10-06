"""Independent frozen-feature receipts and CPU input-only re-extraction."""
import hashlib
import math
from import_cctv_dgp_spatial_features_v25 import read, require, sha

SHAPES = [(64,128,128),(128,64,64),(128,32,32),(128,16,16),(128,8,8)]


def bounded(value, cap, label):
    require(type(value) in (int,float) and math.isfinite(value) and 0 <= value <= cap, label)


def check_rows(rows, protocol):
    require(isinstance(rows,list) and len(rows)==50 and [r['id'] for r in rows]==[c['id'] for c in protocol['cases']], 'Frozen feature50-case order differs')
    for row in rows:
        require(set(row)=={'id','fresh_cached_raw_maximum','features'}, 'CUDA feature row schema differs')
        bounded(row['fresh_cached_raw_maximum'],2e-6,'Fresh all50 CUDA cache parity differs')
        require(isinstance(row['features'],list) and len(row['features'])==5,'Five feature scales required')
        for value,shape in zip(row['features'],SHAPES):
            require(set(value)=={'shape','dtype','sha256','requires_grad','inference_tensor'} and value['shape']==[1,*shape] and
                    value['dtype']=='float32' and value['requires_grad'] is False and value['inference_tensor'] is False and
                    isinstance(value['sha256'],str) and len(value['sha256'])==64 and all(c in '0123456789abcdef' for c in value['sha256']),
                    'Ordinary detached float32 feature receipt differs')


def check_CPU_rows(rows,p):
    require(isinstance(rows,list) and [r['id'] for r in rows]==p['fresh_DGP_parity_cases'], 'Four fixed CPU comparison cases differ')
    for row in rows:
        require(set(row)=={'id','CPU_DGP_raw_maximum','CPU_DGP_feature_maxima'} and len(row['CPU_DGP_feature_maxima'])==5,'CPU feature comparison schema differs')
        bounded(row['CPU_DGP_raw_maximum'],1e-5,'VM CPU raw comparison exceeded prospective bound')
        for value in row['CPU_DGP_feature_maxima']:bounded(value,5e-5,'VM CPU feature comparison exceeded prospective bound')


def load_arrays(returned,p,rows,receipt):
    import numpy as np
    import torch
    expected={c['id']+'_fpn'+str(i)+'.npy' for c in p['cases'] for i in range(5)}
    folder=returned/'outputs/frozen_DGP_features'
    require(receipt.get('complete') is True and receipt.get('cases')==50 and receipt.get('feature_arrays')==250 and
            receipt.get('optimizer_constructed') is False and set(receipt.get('files_sha256',{}))==expected,
            'Frozen feature-cache receipt/policy differs')
    require({f.name for f in folder.iterdir() if f.is_file()}==expected,'Frozen250 feature array file set differs')
    features={}
    for case,row in zip(p['cases'],rows):
        tensors=[]
        for index,(shape,reported) in enumerate(zip(SHAPES,row['features'])):
            name=case['id']+'_fpn'+str(index)+'.npy';path=folder/name
            require(sha(path)==receipt['files_sha256'][name],'Frozen feature array file hash differs')
            array=np.load(path,allow_pickle=False)
            require(array.shape==shape and array.dtype==np.float32 and np.isfinite(array).all(),'Frozen feature array schema differs')
            require(hashlib.sha256(array.tobytes()).hexdigest()==reported['sha256'],'Frozen feature tensor payload hash differs')
            value=torch.from_numpy(array.copy())[None]
            require(not value.requires_grad and not torch.is_inference(value),'Audit requires ordinary detached feature tensors')
            tensors.append(value)
        features[case['id']]=tuple(tensors)
    return features


def audit_features(bundle,returned,p,preflights,clock):
    from audit_cctv_dgp_degraded_detail_v24 import state_hash, rgb, raw_rgb
    cache_path=returned/'outputs/frozen_DGP_features.json'
    all_evidence=[]
    for path in sorted(returned.glob('feature_preflight_*.json')):
        evidence=read(path)
        require(evidence.get('DGP_state_before')==evidence.get('DGP_state_after') and evidence.get('optimizer_constructed') is False and evidence.get('backward_calls')==0,
                'Frozen DGP feature state or pre-gradient scope differs')
        require(evidence.get('CUDA_cache_raw_tolerance')==2e-6 and evidence.get('CPU_DGP_raw_tolerance')==1e-5 and evidence.get('CPU_DGP_feature_tolerance')==5e-5,
                'Prospective intermediate compatibility bounds changed')
        if evidence.get('complete') is True:
            require(evidence.get('cause') is None and evidence['counts']=={'detail_head':50,'DGP':50,'DGP_CPU':4,'fixed_recognizer':0},'Complete feature-preflight counts differ')
            check_rows(evidence['CUDA_feature_rows'],p);check_CPU_rows(evidence['CPU_comparison_rows'],p)
        else:
            require(evidence.get('complete') is False and isinstance(evidence.get('cause'),str),'Failed feature preflight must retain cause')
        all_evidence.append((path.name,evidence))
    for preflight in preflights:
        name=preflight.get('feature_preflight_file');matched=[x for n,x in all_evidence if n==name]
        require(len(matched)==1 and matched[0]['complete'] is True,'Successful preflight lacks original feature receipt')
        evidence=matched[0]
        require(evidence['CUDA_feature_rows']==preflight['frozen_feature_cache_rows'] and evidence['CPU_comparison_rows']==preflight['CPU_DGP_comparison_rows'] and
                evidence['head_state']==preflight['initial_head_state'] and evidence['DGP_state_after']==preflight['DGP_state_unchanged'] and
                preflight['all_feature_tensors_ordinary_detached'] is True,'Preflight feature/state linkage differs')
    if not cache_path.exists():
        require(not list((returned/'outputs').glob('update*')) and not (returned/'outputs/results.json').exists(),'Snapshots or result lack frozen features')
        return {}, {'cache_present':False,'feature_preflight_receipts':len(all_evidence),'DGP_CPU_forwards':0,'scope':'Preflight-only or interrupted extraction; no fitting acceptance'}
    cache=read(cache_path);check_rows(cache['CUDA_feature_rows'],p)
    require(preflights and all(r['frozen_feature_cache_rows']==cache['CUDA_feature_rows'] and r['DGP_state_unchanged']==cache['DGP_state_unchanged'] for r in preflights),
            'Frozen cache and successful preflights differ')
    features=load_arrays(returned,p,cache['CUDA_feature_rows'],cache)
    import sys
    import numpy as np
    import torch
    sys.path.insert(0,str(bundle))
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    dgp,provenance=load_frozen_dgp_restorer(bundle/'weights/dgp_v2.pth',expected_sha256=p['assets_sha256']['weights/dgp_v2.pth'],device='cpu')
    before=state_hash(dgp.net.state_dict());require(before==cache['DGP_state_unchanged'],'Local frozen DGP state differs from feature source')
    rows=[];maxraw=0.;maxfeatures=[0.]*5
    for case in p['cases']:
        clock();camera=rgb(bundle/case['input']);x=torch.from_numpy(camera.copy()).permute(2,0,1).float()[None]/255
        captured=[]
        def hook(_module,_input,output):
            require(not captured and isinstance(output,tuple) and len(output)==5,'CPU FPN extraction differs');captured.extend(output)
        handle=dgp.net.fpn.register_forward_hook(hook)
        try:
            with torch.no_grad():actual=dgp.net(x)
        finally:handle.remove()
        require(len(captured)==5 and not dgp.net.fpn._forward_hooks,'CPU feature capture hook left behind')
        raw_error=float(np.abs(actual[0].permute(1,2,0).numpy()-raw_rgb(bundle/case['raw_dgp'])).max())
        errors=[float((left-right).abs().max()) for left,right in zip(captured,features[case['id']])]
        bounded(raw_error,1e-5,'CPU original DGP raw compatibility failed: '+case['id'])
        for error in errors:bounded(error,5e-5,'CPU original DGP feature compatibility failed: '+case['id'])
        maxraw=max(maxraw,raw_error);maxfeatures=[max(a,b) for a,b in zip(maxfeatures,errors)]
        rows.append({'id':case['id'],'raw_maximum':raw_error,'feature_maxima':errors})
    require(state_hash(dgp.net.state_dict())==before and not any(v.grad is not None or v.requires_grad for v in dgp.net.parameters()),'CPU DGP state/gradients changed')
    del dgp
    return features, {'cache_present':True,'cases':50,'arrays':250,'feature_preflight_receipts':len(all_evidence),
        'CUDA_all50_raw_receipts_checked':True,'VM_four_CPU_receipts_checked':True,'all250_payload_and_file_hashes_verified':True,
        'CPU_original_DGP_state':before,'maximum_CPU_raw_error':maxraw,'maximum_CPU_feature_errors':maxfeatures,'CPU_rows':rows,
        'DGP_CPU_forwards':50,'DGP_provenance':provenance,'local_backward_calls':0,'local_optimizer_updates':0,
        'scope_limit':'Independent input-only CPU re-extraction and receipt compatibility, not original GPU re-execution, canonical app parity or useful restoration'}
