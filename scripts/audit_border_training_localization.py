"""Inference-only audit of existing final heads on reviewed training examples."""
import sys,json,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch,numpy as np,cv2
from PIL import Image,ImageDraw
from detector_training import load_manifest,ReviewedMasks
from scripts.compare_pixel_heads_vm import PixelHead
from scripts.compare_presence_heads_vm import PresenceHead
from scripts.train_expanded_feature_vm import pixel_counts


def main():
    torch.set_num_threads(4);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    old_path=Path('dataset/detector_expanded_review_v2/manifest.json')
    path=Path('dataset/detector_glare_review_v3/manifest.json')
    old=[r for r in load_manifest(old_path) if r['split']=='train']
    rows=[r for r in load_manifest(path) if r['split']=='train'];data=ReviewedMasks(rows,256)
    cache_path=Path('outputs/frozen_feature_probe/train_features.pt')
    cache=torch.load(cache_path,weights_only=True,map_location='cpu')
    assert cache['protocol']['manifest_sha256']==sha(old_path)
    assert cache['protocol']['train_images']==[r['image'] for r in old]==[r['image'] for r in rows]
    assert [r['image_sha256'] for r in old]==[r['image_sha256'] for r in rows]
    assert cache['features'].shape==(68,256,64,64) and torch.isfinite(cache['features']).all()
    root=Path('outputs/downloaded_expanded_border')
    report=json.loads((root/'results.json').read_text());assert report['complete']
    assert sha(path)==json.loads(Path('outputs/expanded_feature_data_v1/manifest.json').read_text())['v3_manifest_sha256']
    out=Path('outputs/border_training_localization');out.mkdir(exist_ok=False)
    result={'scope':'Training fit only; local CPU inference, no optimizer updates','feature_cache_sha256':sha(cache_path),
            'feature_cache_note':'V2 image order/hashes verified against V3; targets freshly loaded from V3, not old cached targets',
            'cpu_gpu_caveat':'Features computed on CPU; aggregate agreement with VM is checked, not bitwise equality','arms':{}}
    for arm in ('control','border2'):
        cp=root/(arm+'_epoch_10.pth');assert sha(cp)==report['arms'][arm]['checkpoint_sha256']
        state=torch.load(cp,weights_only=True,map_location='cpu')
        assert state['protocol']['prior_protocol']['encoder_sha256']==cache['protocol']['checkpoint_sha256']
        pixel=PixelHead(3).eval().requires_grad_(False);pixel.load_state_dict(state['pixel'])
        gate=PresenceHead(4).eval().requires_grad_(False);gate.load_state_dict(state['presence'])
        for mode in ('raw','gated'):(out/arm/mode).mkdir(parents=True)
        cases=[]
        for i,r in enumerate(rows):
            f=cache['features'][i:i+1];target=data[i][1][None].bool()
            with torch.inference_mode():
                z=pixel(f);prob=float(gate(f).sigmoid()[0]);raw=z>=0
            masks={'raw':raw,'gated':raw&(prob>=.5)}
            c={'image':r['image'],'kind':r['kind'],'glare':r.get('glare_stratum')=='strong_lens_reflection','probability':prob,
               'target_pixels':int(target.sum()),'counts':{m:pixel_counts(p,target) for m,p in masks.items()}}
            truth=target[0,0].numpy()
            near=cv2.dilate(truth.astype('uint8'),np.ones((17,17),np.uint8)).astype(bool)&~truth
            c['location']={}
            for mode,pred in masks.items():
                p=pred[0,0].numpy();fp=p&~truth
                c['location'][mode]={'near_fp':int((fp&near).sum()),'far_fp':int((fp&~near).sum())}
            if c['glare']:
                previous=np.array(Image.open(old[i]['mask_path']).convert('L'))>0
                added=target[0,0].numpy()&~previous
                c['added_glare_pixels']=int(added.sum())
                c['added_glare_recovered']={m:int((p[0,0].numpy()&added).sum()) for m,p in masks.items()}
            for m,p in masks.items():Image.fromarray(p[0,0].numpy().astype('uint8')*255).save(out/arm/m/Path(r['image']).name)
            cases.append(c)
        groups={'all':cases,'glare':[c for c in cases if c['glare']],'clear':[c for c in cases if c['kind']=='uncovered']}
        metrics={}
        for group,cs in groups.items():
            metrics[group]={}
            for mode in ('raw','gated'):
                counts={k:sum(c['counts'][mode][k] for c in cs) for k in cases[0]['counts'][mode]}
                metrics[group][mode]={'cases':len(cs),'counts':counts,'iou':counts['tp']/max(1,counts['tp']+counts['fp']+counts['fn'])}
        vm=report['arms'][arm]['training_fit']['real/gated']['iou']
        result['arms'][arm]={'cases':cases,'metrics':metrics,'vm_real_gated_iou':vm,'cpu_vm_iou_delta':metrics['all']['gated']['iou']-vm}
        print(arm,json.dumps({'glare':metrics['glare'],'clear':metrics['clear'],'cpu_vm_delta':metrics['all']['gated']['iou']-vm}),flush=True)
    # Fixed category: six largest gated false-positive training cases in treatment.
    rank=sorted(result['arms']['border2']['cases'],key=lambda c:c['counts']['gated']['fp'],reverse=True)[:6]
    selected=[next(r for r in rows if r['image']==c['image']) for c in rank]
    canvas=Image.new('RGB',(1280,28+280*len(selected)),'white');d=ImageDraw.Draw(canvas)
    d.text((4,4),'Training input | V3 target | Control raw | Control gated | Border2 gated',fill='black')
    for i,r in enumerate(selected):
        y=28+i*280;name=Path(r['image']).name;d.text((4,y),name,fill='black')
        paths=[Path(r['image_path']),Path(r['mask_path']),out/'control/raw'/name,out/'control/gated'/name,out/'border2/gated'/name]
        for j,p in enumerate(paths):canvas.paste(Image.open(p).convert('RGB').resize((256,256)),(j*256,y+22))
    canvas.save(out/'preview.png');result['complete']=True
    (out/'results.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
