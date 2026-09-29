"""Check both matched placement arms without training or feature extraction."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from expanded_feature_data import ExpandedCoveringDataset
from completion_data import CompletionDataset


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parents[1]
    if Path.cwd().resolve()!=root:raise RuntimeError('Run from repository root')
    folder=root/'outputs/expanded_feature_data_v1'
    output=folder/'matched_check.json'
    if output.exists():raise RuntimeError('Existing matched check; preserve it')
    manifest=json.loads((folder/'manifest.json').read_text())
    a=ExpandedCoveringDataset(manifest,root,placement='anatomical')
    b=ExpandedCoveringDataset(manifest,root,placement='fixed')
    assert a.cases==b.cases and a.rejections==b.rejections
    generic=0;anatomy=0;shared_pixels=0
    legacy={i:CompletionDataset([r['path']],256,42,validation=True) for i,r in enumerate(a.sources)}
    for i in range(len(a)):
        x,y=a[i],b[i]
        assert torch.equal(x['target'],y['target']) and x['seed']==y['seed']
        for item in (x,y):
            assert torch.isfinite(item['input']).all() and 0<=item['input'].min()<=item['input'].max()<=1
            assert set(item['mask'].unique().tolist())<={0.,1.}
            assert bool(item['mask'].any())==(item['kind']!='none')
        if x['kind'] in ('eyes','lower'):
            anatomy+=1
            if not x['degraded']:
                shared=(x['geometry'][0]>0)&(y['geometry'][0]>0)
                shared_pixels+=int(shared.sum())
                assert torch.equal(x['input'][:,shared],y['input'][:,shared])
        else:
            old=legacy[x['source_index']][x['variant']]
            assert all(torch.equal(x[k],y[k]) and torch.equal(x[k],old[k]) for k in ('input','target','mask','geometry'))
            generic+=1
        if (i+1)%500==0:print('Matched data',i+1,len(a),flush=True)
    report={'complete':True,'cases_per_arm':len(a),'sources':len(a.sources),'identical_membership':True,
            'identical_rejections':len(a.rejections),'generic_cases_equal_to_legacy':generic,
            'anatomical_cases_per_arm':anatomy,'equal_shared_texture_pixels_clean':shared_pixels,
            'manifest_sha256':hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest(),
            'current_code_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['expanded_feature_data.py','anatomical_augmentation.py','completion.py','feature_disk_cache.py']},
            'original_manifest_code_hash_is_historical':True,'training_run':False,'vm_runner_ready':False,
            'scope':'Data and cache infrastructure verification only; no GPU forward or quality evidence'}
    output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
