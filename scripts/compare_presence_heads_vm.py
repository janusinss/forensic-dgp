"""VM-only fixed-budget global versus spatial presence comparison."""
import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F


class PresenceHead(nn.Module):
    def __init__(self,grid=1):
        super().__init__()
        if grid not in (1,4):raise ValueError('Fixed comparison supports grids 1 and 4')
        self.grid=grid
        self.linear=nn.Linear(256*grid*grid,1)

    def encode(self,x):
        x=F.adaptive_avg_pool2d(x.detach().float(),self.grid).permute(0,2,3,1)
        return F.layer_norm(x,(256,)).flatten(1)

    def forward(self,x):
        return self.linear(self.encode(x))[:,0]


def main(root):
    if not torch.cuda.is_available():raise RuntimeError('Training requires the VM GPU; CPU execution disabled')
    root=Path(root).resolve()
    sys.path[:0]=[str(root),str(root/'outputs/vendor_sam2')]
    from feature_vm_runtime import require_cuda,balanced_batches
    from detector_training import load_manifest,ReviewedMasks
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device=require_cuda()
    torch.set_num_threads(4)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    out=root/'outputs/presence_architecture_comparison'
    if out.exists():raise RuntimeError('Output exists; preserve it and inspect before rerunning')
    train=root/'outputs/feature_mixed_training'
    checkpoint=train/'final_epoch_20.pth'
    assert sha(checkpoint)=='eaa16229f88d416ad09d813a6f08ca0ae72bba214cb8266648bed382603864a4'
    prior=torch.load(checkpoint,map_location='cpu',weights_only=True)['protocol']
    source_path=train/'sources.json'
    assert sha(source_path)==prior['source_manifest_sha256']
    sources=json.loads(source_path.read_text())
    assert sources['partition']=='train'
    split_path=root/'outputs/downloaded_phase4/outputs/phase4_with_progress/split.json'
    assert sha(split_path)==sources['split_sha256']
    split=json.loads(split_path.read_text())
    train_names={p.replace('\\','/') for p in split['train']}
    val_names={p.replace('\\','/') for p in split['validation']}
    assert train_names.isdisjoint(val_names)
    for s in sources['sources']:
        assert s['path'] in train_names and s['path'] not in val_names
        assert sha(root/s['path'])==s['sha256']
    manifest=root/'dataset/detector_glare_review_v3/manifest.json'
    assert sha(manifest)==prior['v3_manifest_sha256']
    rows=[r for r in load_manifest(manifest) if r['split']=='train']
    assert len(rows)==68
    cache=json.loads((train/'cache_progress.json').read_text())
    assert cache['complete'] and len(cache['records'])==400
    weights=root/'outputs/sam2.1_hiera_tiny.pt'
    assert sha(weights)==prior['encoder_sha256']
    heads={}
    for arm,grid in [('global',1),('spatial',4)]:
        torch.manual_seed(42)
        heads[arm]=PresenceHead(grid).to(device)
    protocol={'seed':42,'epochs':20,'steps_per_epoch':80,'batch_size':12,'lr':.001,'weight_decay':.0001,
              'clip_grad_norm':1.,'threshold':.5,'encoder_frozen':True,'pixel_frozen':True,
              'selection':'final only; no validation fitting or threshold changes','training_cases':468,
              'initialization':'fresh seed 42 per arm; dimensions differ, so initial weights are not identical',
              'architecture':'global 1x1 pooled linear versus spatial 4x4 pooled linear; per-cell channel LayerNorm',
              'parameter_counts':{a:sum(p.numel() for p in h.parameters()) for a,h in heads.items()},
              'parent_checkpoint_sha256':sha(checkpoint),'source_manifest_sha256':sha(source_path),
              'v3_manifest_sha256':sha(manifest),'encoder_sha256':sha(weights),
              'script_sha256':sha(Path(__file__)),'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()}
    # Only pooled, detached features are retained; no encoder parameters enter either optimizer.
    encoded={a:[] for a in heads};metadata=[];targets=[]
    def add(f,record,target):
        assert f.shape==(1,256,64,64) and torch.isfinite(f).all()
        with torch.no_grad():
            for arm,h in heads.items():encoded[arm].append(h.encode(f).cpu())
        metadata.append(record);targets.append(float(target))
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(weights),device=str(device),apply_postprocessing=False))
    predictor.model.eval().requires_grad_(False)
    real=ReviewedMasks(rows,256)
    for i,r in enumerate(rows):
        rgb=(real[i][0].permute(1,2,0).numpy()*255).round().astype('uint8')
        with torch.inference_mode():
            predictor.set_image(rgb);f=predictor.get_image_embedding().cpu().clone()
        add(f,{'domain':'real','source':'real','image':r['image'],'kind':r['kind'],
               'glare':r['glare_stratum']=='strong_lens_reflection'},r['kind']=='covered')
        print('real feature',i+1,68,flush=True)
    del predictor
    torch.cuda.empty_cache()
    for i,r in enumerate(cache['records']):
        assert r['partition']=='train' and r['source']==sources['sources'][i//10]
        stem=f'{i:04d}'
        assert sha(train/'input'/(stem+'.png'))==r['input_sha256']
        mask=train/'mask'/(stem+'.png');assert sha(mask)==r['mask_sha256']
        target=bool(np.array(Image.open(mask)).any())
        assert target==(r['kind']!='none')
        saved=torch.load(train/'features'/(stem+'.pt'),map_location='cpu',weights_only=True)
        assert saved['record']==r and r['encoder_sha256']==prior['encoder_sha256']
        add(saved['features'],{'domain':'synthetic','source':r['source']['source'],'image':stem,
                              'kind':r['kind'],'degraded':r['degraded']},target)
    encoded={a:torch.cat(v).to(device) for a,v in encoded.items()}
    target=torch.tensor(targets,device=device)
    groups=[]
    for source in ('real','dataset/asian_faces','dataset/thumbnails128x128'):
        for covered in (True,False):groups.append([i for i,r in enumerate(metadata) if r['source']==source and bool(targets[i])==covered])
    assert [len(g) for g in groups]==[43,25,160,40,160,40]
    out.mkdir()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    report={'protocol':protocol,'history':{},'complete':False}
    for arm,head in heads.items():
        optimizer=torch.optim.AdamW(head.parameters(),lr=.001,weight_decay=.0001)
        history=[]
        for epoch in range(20):
            loss_sum=0.;seen=set()
            for ids in balanced_batches(groups,42+epoch):
                seen.update(ids);optimizer.zero_grad(set_to_none=True)
                logits=head.linear(encoded[arm][ids])[:,0]
                loss=F.binary_cross_entropy_with_logits(logits,target[ids])
                assert torch.isfinite(loss)
                loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                optimizer.step();loss_sum+=float(loss.detach())
            assert seen==set(range(468))
            with torch.inference_mode():probabilities=head.linear(encoded[arm])[:,0].sigmoid().cpu().tolist()
            metrics={}
            for source in ('real','dataset/asian_faces','dataset/thumbnails128x128'):
                ids=[i for i,r in enumerate(metadata) if r['source']==source]
                metrics[source]={'covered':sum(targets[i]>0 for i in ids),'clear':sum(targets[i]==0 for i in ids),
                    'missed_covered':sum(targets[i]>0 and probabilities[i]<.5 for i in ids),
                    'false_positive_clear':sum(targets[i]==0 and probabilities[i]>=.5 for i in ids)}
            entry={'epoch':epoch+1,'mean_loss':loss_sum/80,'metrics':metrics};history.append(entry)
            print(arm,json.dumps(entry),flush=True)
            report['history'][arm]=history
            (out/'results.json').write_text(json.dumps(report,indent=2))
        torch.save({'format':'presence-architecture-diagnostic-v1','diagnostic_only':True,'grid':head.grid,
                    'model':{k:v.cpu() for k,v in head.state_dict().items()},'protocol':protocol},out/(arm+'_epoch_20.pth'))
        (out/(arm+'_training_predictions.json')).write_text(json.dumps([dict(r,target=targets[i],probability=probabilities[i]) for i,r in enumerate(metadata)],indent=2))
    report['complete']=True
    (out/'results.json').write_text(json.dumps(report,indent=2))
    with tarfile.open(root/'presence-comparison-results.tar.gz','w:gz') as tar:
        for p in sorted(out.iterdir()):tar.add(p,arcname=p.name)
    print('DONE:',root/'presence-comparison-results.tar.gz',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    main(parser.parse_args().root)
