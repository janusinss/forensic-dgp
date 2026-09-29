"""GPU-only matched pointwise/context pixel-head experiment; no deployment."""
import argparse,hashlib,json,sys,tarfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class PixelHead(nn.Module):
    def __init__(self,kernel=1):
        super().__init__()
        if kernel not in (1,3):raise ValueError('Expected kernel 1 or 3')
        self.kernel=kernel
        self.head=nn.Sequential(nn.Conv2d(256,64,kernel,padding=kernel//2),nn.GELU(),nn.Conv2d(64,1,1))

    def initialize(self,state):
        with torch.no_grad():
            self.head[0].weight.zero_()
            k=self.kernel//2
            self.head[0].weight[:,:,k:k+1,k:k+1].copy_(state['head.0.weight'])
            self.head[0].bias.copy_(state['head.0.bias'])
            self.head[2].weight.copy_(state['head.2.weight'])
            self.head[2].bias.copy_(state['head.2.bias'])

    def forward(self,f,size=(256,256)):
        x=F.layer_norm(f.detach().float().permute(0,2,3,1),(256,)).permute(0,3,1,2)
        return F.interpolate(self.head(x),size=size,mode='bilinear',align_corners=False)


def main(root):
    if not torch.cuda.is_available():raise RuntimeError('Training requires the VM GPU; CPU execution disabled')
    root=Path(root).resolve();sys.path[:0]=[str(root),str(root/'outputs/vendor_sam2')]
    from feature_vm_runtime import require_cuda,balanced_batches
    from detector_training import load_manifest,ReviewedMasks
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device=require_cuda();torch.set_num_threads(4);torch.manual_seed(42)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    out=root/'outputs/pixel_architecture_comparison'
    if out.exists():raise RuntimeError('Output exists; preserve and inspect it before rerunning')
    train=root/'outputs/feature_mixed_training';checkpoint=train/'final_epoch_20.pth'
    assert sha(checkpoint)=='eaa16229f88d416ad09d813a6f08ca0ae72bba214cb8266648bed382603864a4'
    state=torch.load(checkpoint,weights_only=True,map_location='cpu');prior=state['protocol']
    presence_path=root/'outputs/presence_architecture_comparison/spatial_epoch_20.pth'
    assert sha(presence_path)=='b7af5b8568fc1fd1a6be289a6599794c6c2976e162a351be90a53344997b40c4'
    presence_state=torch.load(presence_path,weights_only=True,map_location='cpu')
    assert presence_state['grid']==4
    gate=nn.Linear(4096,1).to(device).eval().requires_grad_(False)
    gate.load_state_dict({k.removeprefix('linear.'):v for k,v in presence_state['model'].items()})
    source_path=train/'sources.json';assert sha(source_path)==prior['source_manifest_sha256']
    sources=json.loads(source_path.read_text());assert sources['partition']=='train'
    split_path=root/'outputs/downloaded_phase4/outputs/phase4_with_progress/split.json'
    assert sha(split_path)==sources['split_sha256']
    split=json.loads(split_path.read_text())
    train_names={p.replace('\\','/') for p in split['train']};val_names={p.replace('\\','/') for p in split['validation']}
    assert train_names.isdisjoint(val_names)
    for s in sources['sources']:
        assert s['path'] in train_names and s['path'] not in val_names and sha(root/s['path'])==s['sha256']
    manifest=root/'dataset/detector_glare_review_v3/manifest.json'
    assert sha(manifest)==prior['v3_manifest_sha256']
    rows=[r for r in load_manifest(manifest) if r['split']=='train'];assert len(rows)==68
    cache=json.loads((train/'cache_progress.json').read_text());assert cache['complete'] and len(cache['records'])==400
    weights=root/'outputs/sam2.1_hiera_tiny.pt';assert sha(weights)==prior['encoder_sha256']
    features=[];targets=[];metadata=[]
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(weights),device=str(device),apply_postprocessing=False))
    predictor.model.eval().requires_grad_(False)
    data=ReviewedMasks(rows,256)
    for i,r in enumerate(rows):
        x,t=data[i]
        with torch.inference_mode():
            predictor.set_image((x.permute(1,2,0).numpy()*255).round().astype('uint8'))
            f=predictor.get_image_embedding().cpu().clone()
        assert f.shape==(1,256,64,64) and torch.isfinite(f).all()
        features.append(f);targets.append(t[None]);metadata.append({'source':'real','image':r['image'],'kind':r['kind']})
        print('real features',i+1,68,flush=True)
    del predictor;torch.cuda.empty_cache()
    for i,r in enumerate(cache['records']):
        assert r['partition']=='train' and r['source']==sources['sources'][i//10]
        name=f'{i:04d}'
        assert sha(train/'input'/(name+'.png'))==r['input_sha256']
        mask=train/'mask'/(name+'.png');assert sha(mask)==r['mask_sha256']
        saved=torch.load(train/'features'/(name+'.pt'),weights_only=True,map_location='cpu')
        assert saved['record']==r and r['encoder_sha256']==prior['encoder_sha256']
        f=saved['features'];assert f.shape==(1,256,64,64) and torch.isfinite(f).all()
        t=torch.from_numpy(np.array(Image.open(mask).convert('L')).copy()).float()[None,None]/255
        assert bool(t.any())==(r['kind']!='none')
        features.append(f);targets.append(t);metadata.append({'source':r['source']['source'],'image':name,'kind':r['kind'],'degraded':r['degraded']})
    features=torch.cat(features);targets=torch.cat(targets);covered=targets.flatten(1).any(1)
    assert features.shape==(468,256,64,64) and targets.shape==(468,1,256,256)
    groups=[]
    for source in ('real','dataset/asian_faces','dataset/thumbnails128x128'):
        for positive in (True,False):groups.append([i for i,r in enumerate(metadata) if r['source']==source and bool(covered[i])==positive])
    assert [len(g) for g in groups]==[43,25,160,40,160,40]
    heads={a:PixelHead(k).to(device) for a,k in [('pointwise',1),('context',3)]}
    for h in heads.values():h.initialize(state['pixel'])
    with torch.inference_mode():
        x=features[:2].to(device)
        delta=float((heads['pointwise'](x)-heads['context'](x)).abs().max())
        torch.testing.assert_close(heads['pointwise'](x),heads['context'](x),atol=1e-4,rtol=1e-4)
    protocol={'epochs':20,'steps_per_epoch':80,'batch_size':12,'seed':42,'lr':.001,'weight_decay':.0001,
              'loss':'all-image BCE + nonempty per-image Dice / full batch size','thresholds':{'pixel':.5,'presence':.5},
              'clip_grad_norm':1.,'encoder_frozen':True,'presence_frozen':True,'selection':'final only',
              'parent_checkpoint_sha256':sha(checkpoint),'presence_sha256':sha(presence_path),
              'source_manifest_sha256':sha(source_path),'v3_manifest_sha256':sha(manifest),'encoder_sha256':sha(weights),
              'script_sha256':sha(Path(__file__)),'initial_max_logit_difference':delta,
              'parameter_counts':{a:sum(p.numel() for p in h.parameters()) for a,h in heads.items()},
              'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()}
    out.mkdir();(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    (out/'training_metadata.json').write_text(json.dumps(metadata,indent=2))
    def metrics(head,ids):
        totals={a:{k:0 for k in ('tp','fp','fn','visible','empty','negative_fp')} for a in ('raw','gated')}
        with torch.inference_mode():
            for start in range(0,len(ids),12):
                selected=ids[start:start+12];f=features[selected].to(device);t=targets[selected].to(device).bool()
                p=head(f)>=0
                pooled=F.adaptive_avg_pool2d(f,4).permute(0,2,3,1)
                keep=gate(F.layer_norm(pooled,(256,)).flatten(1))[:,0]>=0
                for arm,pred in [('raw',p),('gated',p&keep[:,None,None,None])]:
                    has=t.flatten(1).any(1);got=pred.flatten(1).any(1)
                    scores=dict(tp=(pred&t).sum(),fp=(pred&~t).sum(),fn=(~pred&t).sum(),visible=(~t).sum(),empty=(has&~got).sum(),negative_fp=(~has&got).sum())
                    for k,v in scores.items():totals[arm][k]+=int(v)
        return {a:dict(iou=s['tp']/max(1,s['tp']+s['fp']+s['fn']),missed_fraction=s['fn']/max(1,s['tp']+s['fn']),visible_false_positive=s['fp']/max(1,s['visible']),empty_mask_cases=s['empty'],negative_false_positive_cases=s['negative_fp']) for a,s in totals.items()}
    report={'protocol':protocol,'history':{},'complete':False}
    for arm,head in heads.items():
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001);history=[]
        for epoch in range(20):
            total=0.;seen=set()
            for ids in balanced_batches(groups,42+epoch):
                seen.update(ids);optimizer.zero_grad(set_to_none=True)
                z=head(features[ids].to(device));t=targets[ids].to(device);p=z.sigmoid()
                dice=(1-(2*(p*t).sum((1,2,3))+1)/(p.sum((1,2,3))+t.sum((1,2,3))+1))*covered[ids].to(device)
                loss=F.binary_cross_entropy_with_logits(z,t)+dice.mean();assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                total+=float(loss.detach())
            assert seen==set(range(468))
            entry={'epoch':epoch+1,'mean_loss':total/80,'real_train':metrics(head,list(range(68))),
                   'synthetic_train':metrics(head,list(range(68,468)))}
            history.append(entry);report['history'][arm]=history
            (out/'results.json').write_text(json.dumps(report,indent=2));print(arm,json.dumps(entry),flush=True)
        torch.save({'format':'pixel-context-diagnostic-v1','diagnostic_only':True,'kernel':head.kernel,
                    'model':{k:v.cpu() for k,v in head.state_dict().items()},'protocol':protocol},out/(arm+'_epoch_20.pth'))
    report['complete']=True;(out/'results.json').write_text(json.dumps(report,indent=2))
    with tarfile.open(root/'pixel-comparison-results.tar.gz','w:gz') as tar:
        for p in sorted(out.iterdir()):tar.add(p,arcname=p.name)
    print('DONE:',root/'pixel-comparison-results.tar.gz',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='.')
    main(parser.parse_args().root)
