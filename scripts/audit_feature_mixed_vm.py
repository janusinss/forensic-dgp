"""Read-only training-feature audit and CPU/CUDA parity; zero optimizer updates."""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


def summarize(rows):
    return dict(cases=len(rows),covered=sum(r['area']>0 for r in rows),
                clear=sum(r['area']==0 for r in rows),
                missed_covered=sum(r['area']>0 and r['probability']<.5 for r in rows),
                false_positive_clear=sum(r['area']==0 and r['probability']>=.5 for r in rows))


def parity_metrics(a,b,pa,pb):
    delta=(a-b).abs()
    return dict(embedding_max_abs=float(delta.max()),embedding_mean_abs=float(delta.mean()),
                probability_abs_delta=abs(pa['probability']-pb['probability']),
                presence_decision_changed=(pa['probability']>=.5)!=(pb['probability']>=.5),
                mask_disagreement_fraction=float((pa['mask']!=pb['mask']).float().mean()))


def main(root):
    root=Path(root).resolve()
    if (root/'outputs/feature_mixed_audit').exists():
        raise RuntimeError('Audit output already exists; preserve and inspect it before rerunning')
    sys.path[:0]=[str(root),str(root/'outputs/vendor_sam2')]
    from feature_detector import FrozenFeatureHead
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    if not torch.cuda.is_available():raise RuntimeError('Run this diagnostic on the VM with CUDA')
    torch.set_num_threads(4)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    train=root/'outputs/feature_mixed_training'
    checkpoint=train/'final_epoch_20.pth'
    assert sha(checkpoint)=='eaa16229f88d416ad09d813a6f08ca0ae72bba214cb8266648bed382603864a4'
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert state['format']=='mixed-feature-diagnostic-v1'
    assert sha(root/'feature_detector.py')==state['protocol']['head_code_sha256']
    progress=json.loads((train/'cache_progress.json').read_text())
    assert progress['complete'] and len(progress['records'])==400
    pixel=FrozenFeatureHead().eval().requires_grad_(False)
    pixel.load_state_dict(state['pixel'])
    presence=torch.nn.Linear(256,1).eval().requires_grad_(False)
    presence.load_state_dict(state['presence'])
    def predict(f):
        with torch.inference_mode():
            return {'probability':float(presence(F.layer_norm(f.mean((2,3)),(256,))).sigmoid()[0,0]),
                    'mask':pixel(f,(256,256))[0,0]>=0}
    report={'checkpoint_sha256':sha(checkpoint),'scope':'training-source diagnostic only; no updates or threshold tuning',
            'runtime':{'torch':str(torch.__version__),'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name()},
            'cases':[],'parity':[]}
    # Fix selection before examining predictions: first clean/degraded case of each kind in each dataset.
    selected=[];seen=set()
    for i,r in enumerate(progress['records']):
        assert r['partition']=='train'
        key=(r['source']['source'],r['kind'],r['degraded'])
        if key not in seen:selected.append(i);seen.add(key)
    assert len(selected)==20
    report['parity_selection']=selected
    chosen_features={}
    for i,r in enumerate(progress['records']):
        stem=f'{i:04d}'
        input_path=train/'input'/(stem+'.png');mask_path=train/'mask'/(stem+'.png')
        assert sha(input_path)==r['input_sha256'] and sha(mask_path)==r['mask_sha256']
        cached=torch.load(train/'features'/(stem+'.pt'),map_location='cpu',weights_only=True)
        assert cached['record']==r and r['encoder_sha256']==state['protocol']['encoder_sha256']
        f=cached['features'];assert f.shape==(1,256,64,64) and torch.isfinite(f).all()
        p=predict(f);target=np.array(Image.open(mask_path))>0;area=float(target.mean())
        report['cases'].append({'index':i,'source':r['source']['source'],'kind':r['kind'],'degraded':r['degraded'],
                                'area':area,'area_bin':'empty' if area==0 else ('small_0_to_5pct' if area<=.05 else ('medium_5_to_20pct' if area<=.2 else 'large_over_20pct')),
                                'probability':p['probability'],'pixel_has_mask':bool(p['mask'].any())})
        if i in selected:chosen_features[i]=f
        if (i+1)%50==0:print('audited training cache',i+1,400,flush=True)
    report['summary']={'all':summarize(report['cases'])}
    for key in ('source','kind','degraded','area_bin'):
        report['summary'][key]={str(v):summarize([c for c in report['cases'] if c[key]==v]) for v in sorted({c[key] for c in report['cases']})}
    weights=root/'outputs/sam2.1_hiera_tiny.pt'
    assert sha(weights)==state['protocol']['encoder_sha256']
    features_by_device={}
    for device in ('cpu','cuda'):
        predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(weights),device=device,apply_postprocessing=False))
        predictor.model.eval().requires_grad_(False)
        features_by_device[device]={}
        for count,i in enumerate(selected):
            rgb=np.array(Image.open(train/'input'/f'{i:04d}.png').convert('RGB'))
            with torch.inference_mode():
                predictor.set_image(rgb)
                f=predictor.get_image_embedding().cpu().clone()
            assert torch.isfinite(f).all()
            features_by_device[device][i]=f
            print('parity encoder',device,count+1,20,flush=True)
        del predictor
        torch.cuda.empty_cache()
    for i in selected:
        cpu=features_by_device['cpu'][i];gpu=features_by_device['cuda'][i];old=chosen_features[i]
        report['parity'].append({'index':i,'cpu_vs_gpu':parity_metrics(cpu,gpu,predict(cpu),predict(gpu)),
                                'saved_gpu_vs_current_gpu':parity_metrics(old,gpu,predict(old),predict(gpu))})
    report['complete']=True
    out=root/'outputs/feature_mixed_audit'
    out.mkdir(exist_ok=False)
    (out/'results.json').write_text(json.dumps(report,indent=2))
    with tarfile.open(root/'feature-mixed-audit-results.tar.gz','w:gz') as tar:
        tar.add(out/'results.json',arcname='results.json')
    print(json.dumps(report['summary'],indent=2))
    print('DONE:',root/'feature-mixed-audit-results.tar.gz',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    main(parser.parse_args().root)
