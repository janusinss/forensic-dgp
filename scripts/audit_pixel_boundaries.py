"""Read-only error localization at a fixed four-pixel band, never a selection metric."""
import json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image
import torch
from feature_detector import FrozenFeatureHead
from detector_training import load_manifest,ReviewedMasks


def boundary_counts(prediction,target,radius=4):
    p=np.asarray(prediction,dtype=bool);t=np.asarray(target,dtype=bool)
    if p.shape!=t.shape or t.ndim!=2:raise ValueError('Matching 2D masks required')
    kernel=np.ones((2*radius+1,2*radius+1),np.uint8)
    expanded=cv2.dilate(t.astype('uint8'),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    interior=cv2.erode(t.astype('uint8'),kernel,borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    fp=p&~t;fn=~p&t
    return dict(tp=int((p&t).sum()),fp=int(fp.sum()),fn=int(fn.sum()),
                fp_edge=int((fp&expanded).sum()),fp_far=int((fp&~expanded).sum()),
                fn_edge=int((fn&~interior).sum()),fn_interior=int((fn&interior).sum()))


def main():
    torch.set_num_threads(4)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    checkpoint=Path('outputs/downloaded_feature_mixed_vm/outputs/feature_mixed_training/final_epoch_20.pth')
    state=torch.load(checkpoint,weights_only=True)
    assert sha(checkpoint)=='eaa16229f88d416ad09d813a6f08ca0ae72bba214cb8266648bed382603864a4'
    assert sha('feature_detector.py')==state['protocol']['head_code_sha256']
    model=FrozenFeatureHead().eval().requires_grad_(False);model.load_state_dict(state['pixel'])
    manifest=Path('dataset/detector_glare_review_v3/manifest.json')
    assert sha(manifest)==state['protocol']['v3_manifest_sha256']
    rows=load_manifest(manifest);train=[r for r in rows if r['split']=='train']
    cache=torch.load('outputs/frozen_feature_probe/train_features.pt',weights_only=True)
    assert cache['protocol']['train_images']==[r['image'] for r in train]
    oldpath=Path('dataset/detector_expanded_review_v2/manifest.json')
    assert sha(oldpath)==cache['protocol']['manifest_sha256']
    old=[r for r in load_manifest(oldpath) if r['split']=='train']
    assert [r['image_sha256'] for r in old]==[r['image_sha256'] for r in train]
    records=[]
    def add(name,domain,kind,degraded,p,t):
        counts=boundary_counts(p,t)
        assert counts['fp']==counts['fp_edge']+counts['fp_far'] and counts['fn']==counts['fn_edge']+counts['fn_interior']
        records.append(dict(name=name,domain=domain,kind=kind,degraded=degraded,**counts))
    data=ReviewedMasks(train,256)
    for i,r in enumerate(train):
        with torch.inference_mode():p=model(cache['features'][i:i+1],(256,256))[0,0].numpy()>=0
        add(r['image'],'real_train',r['kind'],None,p,data[i][1][0].numpy()>0)
    partial=Path('outputs/feature_mixed_training')
    for path in sorted((partial/'features').glob('*.pt')):
        cached=torch.load(path,weights_only=True);r=cached['record']
        assert r['partition']=='train' and r['encoder_sha256']==state['protocol']['encoder_sha256']
        mask=partial/'mask'/(path.stem+'.png');image=partial/'input'/(path.stem+'.png')
        assert sha(mask)==r['mask_sha256'] and sha(image)==r['input_sha256']
        with torch.inference_mode():p=model(cached['features'],(256,256))[0,0].numpy()>=0
        add(path.stem,'synthetic_train_partial',r['kind'],r['degraded'],p,np.array(Image.open(mask).convert('L'))>0)
    prior=json.loads(Path('outputs/feature_mixed_validation/results.json').read_text())
    assert prior['checkpoint_sha256']==sha(checkpoint)
    byname={r['image']:r for r in rows}
    bench=Path('outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm')
    for domain in ('real','synthetic'):
        for c in prior[domain+'_cases']:
            name=c['image']
            truth=Path(byname[name]['mask_path']) if domain=='real' else bench/'mask'/name
            p=np.array(Image.open(Path('outputs/feature_mixed_validation')/domain/'raw'/Path(name).name))>0
            t=np.array(Image.open(truth).convert('L'))>0
            counts=boundary_counts(p,t)
            assert all(counts[k]==c['raw'][k] for k in ('tp','fp','fn'))
            kind=byname[name]['kind'] if domain=='real' else c['stratum'].split('/')[0]
            degraded=None if domain=='real' else c['stratum'].split('/')[1]
            add(name,domain+'_validation',kind,degraded,p,t)
    groups={}
    for r in records:
        for key in (r['domain'],r['domain']+'/'+r['kind'],r['domain']+'/degraded='+str(r['degraded'])):
            g=groups.setdefault(key,dict(cases=0,**{k:0 for k in ('tp','fp','fn','fp_edge','fp_far','fn_edge','fn_interior')}))
            g['cases']+=1
            for k in ('tp','fp','fn','fp_edge','fp_far','fn_edge','fn_interior'):g[k]+=r[k]
    for g in groups.values():
        g['fp_far_fraction']=g['fp_far']/max(1,g['fp']);g['fn_interior_fraction']=g['fn_interior']/max(1,g['fn'])
    result={'checkpoint_sha256':sha(checkpoint),'radius':4,'scope':'diagnostic only; original gates unchanged',
            'limitation':'training synthetic subset is 77 cached examples from first eight Asian sources, not representative of full training',
            'groups':groups,'records':records}
    out=Path('outputs/pixel_boundary_audit');out.mkdir(exist_ok=False)
    (out/'results.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in groups.items() if '/' not in k},indent=2))


if __name__=='__main__':main()
