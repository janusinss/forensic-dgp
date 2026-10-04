"""Frozen V14 oracle-check diagnosis: six renders, no training/backward."""
import argparse
import json
from pathlib import Path
import sys
import time


def run(root, parent):
    sys.path.insert(0,str(parent));sys.path.insert(0,str(root))
    from cctv_dgp_direct_codes_v14 import require_vm, verify, require, read, write, sha, PARENT_RESULTS
    require_vm(root)
    pin='67b8df8e6cfc7857b65c4862988f34e740bbae26960b3b07af5d06852255ae4f'
    p=verify(root,parent,pin)
    import numpy as np
    import torch
    from dgp_direct_face_code_v14 import render_codes
    from third_party.codeformer.codeformer_arch import calc_mean_std
    from pretrained_face_restoration_portable_v12 import load_face_restorer
    from cctv_dgp_pilot import state_hash
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    started=time.monotonic()
    out=root/'outputs/cctv_dgp_stats_roundtrip_v14';require(not out.exists(),'Preserve diagnostic')
    out.mkdir()
    pp=read(parent/'face_code_fit_protocol_v12.json')
    source=parent/'outputs/cctv_dgp_face_code_fit_v12'
    require(sha(source/'results.json')==PARENT_RESULTS,'Parent result differs')
    artifacts=read(source/'results.json')['artifacts_sha256']
    prior,_=load_face_restorer(parent/pp['weights']['prior'],device='cuda')
    before=state_hash(prior.net);count=[0]
    def hook(*_):count[0]+=1
    handle=prior.net.generator.blocks[0].register_forward_hook(hook)
    selected=[next(r for r in p['references'] if r['source']==s)
              for s in ['dataset/asian_faces','dataset/thumbnails128x128']]
    rows=[];files={};used={}
    with torch.inference_mode():
        for ref in selected:
            rid=ref['id']
            def cached(kind):
                name='teacher/'+rid+'_'+kind+'.npy'
                require(sha(source/name)==artifacts[name],'Changed teacher cache');used[name]=artifacts[name]
                return torch.from_numpy(np.load(source/name,allow_pickle=False)).cuda()
            codes=cached('codes');old=cached('features')
            fresh=prior.net.quantize.get_codebook_feat(codes,[1,16,16,256])
            cm,cs=calc_mean_std(old);fm,fs=calc_mean_std(fresh)
            require(torch.equal(fresh,old),'Cached codebook values differ')
            raw={}
            raw['none']=render_codes(prior.net,codes)
            raw['cached_stats']=render_codes(prior.net,codes,mean=cm.flatten(1),std=cs.flatten(1))
            raw['fresh_stats']=render_codes(prior.net,codes,mean=fm.flatten(1),std=fs.flatten(1))
            row={'reference_id':rid,'source':ref['source'],'cached_stride':list(old.stride()),
                 'fresh_stride':list(fresh.stride()),'codebook_values_equal':True,
                 'mean_max_difference':float((cm-fm).abs().max()),'std_max_difference':float((cs-fs).abs().max()),
                 'latent_cached_roundtrip_max':float((((fresh-fm)/fs*cs+cm)-fresh).abs().max()),
                 'latent_fresh_roundtrip_max':float((((fresh-fm)/fs*fs+fm)-fresh).abs().max()),
                 'renders':{}}
            base=raw['none'].cpu().numpy()
            for mode,value in raw.items():
                array=value.cpu().numpy();name=rid+'_'+mode+'.npy'
                np.save(out/name,array,allow_pickle=False);files[name]=sha(out/name)
                difference=abs(array-base)
                px=np.floor(array*255).astype(np.uint8);bp=np.floor(base*255).astype(np.uint8)
                row['renders'][mode]={'raw':name,'max_float_difference':float(difference.max()),
                    'mean_float_difference':float(difference.mean()),'different_uint8_channels':int(np.count_nonzero(px!=bp)),
                    'max_uint8_difference':int(abs(px.astype(np.int16)-bp.astype(np.int16)).max())}
            rows.append(row)
            torch.cuda.synchronize();require(time.monotonic()-started<=120,'Diagnostic exceeded120 seconds')
    after=state_hash(prior.net);handle.remove()
    require(before==after and count[0]==6,'Frozen state/forward count differs')
    result={'complete':True,'diagnostic':'V14 oracle normalization numeric roundtrip; no fitting',
        'source_sha256':sha(Path(__file__)),'parent_protocol_sha256':pin,'parent_results_sha256':PARENT_RESULTS,
        'rows':rows,'artifacts_sha256':files,'used_parent_arrays':used,'prior_before':before,'prior_after':after,
        'generator_forwards':6,'encoder_forwards':0,'teacher_forwards':0,'backward_calls':0,'optimizer_updates':0,
        'native_used':False,'validation_used':False,'production_promoted':False,'seconds':time.monotonic()-started}
    write(out/'results.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['artifacts_sha256','used_parent_arrays']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--parent-bundle',type=Path,required=True)
    args=parser.parse_args();run(args.root.resolve(),args.parent_bundle.resolve())
