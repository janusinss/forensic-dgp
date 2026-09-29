"""Bounded mixed training of two heads on frozen features, never validation caches."""
import sys,json,hashlib,time
from pathlib import Path
sys.path[:0]=[str(Path.cwd()),str(Path('outputs/vendor_sam2').resolve())]
import torch,numpy as np
from PIL import Image
from torch.nn import functional as F
from completion_data import CompletionDataset
from detector_training import load_manifest,ReviewedMasks
from feature_detector import FrozenFeatureHead
import argparse, platform
from feature_vm_runtime import require_cuda, balanced_batches
def main(preflight=False):
    device=require_cuda()
    inventory=json.loads(Path('bundle_inventory.json').read_text())
    for name,digest in inventory.items():
        if hashlib.sha256(Path(name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'Bundle hash mismatch: {name}')
    if Path('outputs/feature_mixed_training/results.json').exists():
        raise RuntimeError('Existing run found; preserve it and use a fresh bundle directory')
    torch.set_num_threads(4);torch.manual_seed(42)
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    out=Path('outputs/feature_mixed_training');source=json.loads((out/'sources.json').read_text())
    assert source['partition']=='train' and len(source['sources'])==40
    split_path=Path('outputs/downloaded_phase4/outputs/phase4_with_progress/split.json')
    assert sha(split_path)==source['split_sha256']
    split=json.loads(split_path.read_text())
    normalize=lambda p:p.replace('\\','/')
    training={normalize(p) for p in split['train']}
    validation={normalize(p) for p in split['validation']}
    assert training.isdisjoint(validation)
    assert all(s['path'] in training and s['path'] not in validation for s in source['sources'])
    manifest=Path('dataset/detector_glare_review_v3/manifest.json');assert sha(manifest)==source['v3_manifest_sha256']
    for s in source['sources']:assert sha(s['path'])==s['sha256']
    pixel_path=Path('outputs/feature_empty_dice_comparison/nonempty_dice.pth');presence_path=Path('outputs/feature_presence_probe/presence.pth')
    weights=Path('outputs/sam2.1_hiera_tiny.pt');assert sha(weights)=='7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'
    protocol={'seed':42,'epochs':20,'steps_per_epoch':80,'batch_size':12,'lr_both_heads':.001,'weight_decay':.0001,
        'pixel_loss':'all-image BCE + nonempty Dice terms / batch size','presence_loss':'BCE; balanced by six-group sampler',
        'groups':['real_covered','real_uncovered','asian_covered','asian_uncovered','ffhq_covered','ffhq_uncovered'],
        'per_group_per_batch':2,'thresholds':{'pixel':.5,'presence':.5},'encoder_frozen':True,
        'source_manifest_sha256':sha(out/'sources.json'),'v3_manifest_sha256':sha(manifest),
        'pixel_initial_sha256':sha(pixel_path),'presence_initial_sha256':sha(presence_path),'encoder_sha256':sha(weights),
        'head_code_sha256':sha('feature_detector.py'),'selection':'final epoch only; unchanged external real/synthetic safeguards',
        'scope':'training-only run; no validation embeddings read, no deployment'}
    protocol['runtime']={'device':str(device),'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__),'cuda':torch.version.cuda,'python':platform.python_version(),'features':'all 468 recomputed on CUDA in float32; CPU/GPU numerics can differ'}
    rows=[r for r in load_manifest(manifest) if r['split']=='train'];assert len(rows)==68
    real_data=ReviewedMasks(rows,256)
    real_targets=torch.stack([real_data[i][1] for i in range(68)])
    data=CompletionDataset([s['path'] for s in source['sources']],256,42,validation=True)
    assert len(data)==400
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(weights),device=str(device),apply_postprocessing=False))
    predictor.model.eval().requires_grad_(False)
    # Preflight runs one frozen encoder/head forward, with no optimizer updates.
    real_features=[]
    with torch.inference_mode():
        rgb=(real_data[0][0].permute(1,2,0).numpy()*255).round().astype('uint8')
        predictor.set_image(rgb)
        probe=predictor.get_image_embedding().clone()
        assert probe.shape==(1,256,64,64) and torch.isfinite(probe).all()
        head=FrozenFeatureHead().to(device)
        head.load_state_dict(torch.load(pixel_path,weights_only=True,map_location='cpu')['model'])
        assert torch.isfinite(head(probe,(256,256))).all()
        del head,probe
    if preflight:
        print('PREFLIGHT PASS: GPU, hashes, reviewed split, 40 training sources, SAM/head forward; zero updates',flush=True)
        return
    for folder in ('features','input','mask'):(out/folder).mkdir(exist_ok=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    for i in range(68):
        rgb=(real_data[i][0].permute(1,2,0).numpy()*255).round().astype('uint8')
        with torch.inference_mode():
            predictor.set_image(rgb)
            f=predictor.get_image_embedding().cpu().clone()
        assert f.shape==(1,256,64,64) and torch.isfinite(f).all()
        real_features.append(f)
        print('real training feature',i+1,68,flush=True)
    features=[];targets=[];records=[];start=time.monotonic()
    for i,item in enumerate(data):
        rgb=(item['input'].permute(1,2,0).numpy()*255).round().astype('uint8')
        with torch.inference_mode():predictor.set_image(rgb);f=predictor.get_image_embedding().cpu().clone()
        assert f.shape==(1,256,64,64) and torch.isfinite(f).all()
        name=f'{i:04d}';Image.fromarray(rgb).save(out/'input'/(name+'.png'))
        Image.fromarray((item['mask'][0].numpy()*255).astype('uint8')).save(out/'mask'/(name+'.png'))
        record={'partition':'train','source':source['sources'][i//10],'kind':item['kind'],'degraded':item['degraded'],
                'input_sha256':sha(out/'input'/(name+'.png')),'mask_sha256':sha(out/'mask'/(name+'.png')),'encoder_sha256':protocol['encoder_sha256']}
        torch.save({'features':f,'record':record},out/'features'/(name+'.pt'))
        features.append(f);targets.append(item['mask'][None]);records.append(record)
        (out/'cache_progress.json').write_text(json.dumps({'expected':400,'complete':False,'records':records},indent=2));print('training feature',i+1,400,flush=True)
    del predictor
    features=torch.cat(real_features+features);targets=torch.cat([real_targets]+targets)
    del real_features
    torch.cuda.empty_cache()
    assert features.shape==(468,256,64,64) and targets.shape==(468,1,256,256)
    with torch.no_grad():global_features=F.layer_norm(features.mean((2,3)),(256,)).detach()
    covered=targets.flatten(1).any(1)
    groups=[[i for i,r in enumerate(rows) if r['kind']==k] for k in ('covered','uncovered')]
    for source_name in ('dataset/asian_faces','dataset/thumbnails128x128'):
        for positive in (True,False):groups.append([68+i for i,r in enumerate(records) if r['source']['source']==source_name and (r['kind']!='none')==positive])
    assert [len(g) for g in groups]==[43,25,160,40,160,40]
    (out/'cache_progress.json').write_text(json.dumps({'expected':400,'complete':True,'records':records},indent=2))
    pixel=FrozenFeatureHead().to(device);pixel.load_state_dict(torch.load(pixel_path,weights_only=True)['model'])
    presence=torch.nn.Linear(256,1).to(device);presence.load_state_dict(torch.load(presence_path,weights_only=True)['model'])
    optimizer=torch.optim.AdamW(list(pixel.parameters())+list(presence.parameters()),lr=.001,weight_decay=.0001)
    report={'protocol':protocol,'history':[]}
    def metrics(indices):
        totals={a:{k:0 for k in ('tp','fp','fn','visible','empty','negative_fp')} for a in ('raw','gated')}
        with torch.no_grad():
            for start in range(0,len(indices),12):
                ids=indices[start:start+12];raw=pixel(features[ids].to(device),(256,256))>=0;keep=presence(global_features[ids].to(device))[:,0]>=0;t=targets[ids].to(device).bool();has=covered[ids].to(device)
                for a,p in [('raw',raw),('gated',raw&keep[:,None,None,None])]:
                    got=p.flatten(1).any(1);s=totals[a]
                    for k,v in {'tp':(p&t).sum(),'fp':(p&~t).sum(),'fn':(~p&t).sum(),'visible':(~t).sum(),'empty':(has&~got).sum(),'negative_fp':(~has&got).sum()}.items():s[k]+=int(v)
        return {a:dict(iou=s['tp']/max(1,s['tp']+s['fp']+s['fn']),missed_fraction=s['fn']/max(1,s['tp']+s['fn']),visible_false_positive=s['fp']/max(1,s['visible']),empty_mask_cases=s['empty'],negative_false_positive_cases=s['negative_fp']) for a,s in totals.items()}
    for epoch in range(20):
        seen=set();total=0
        for ids in balanced_batches(groups,42+epoch):
            seen.update(ids)
            optimizer.zero_grad(set_to_none=True);z=pixel(features[ids].to(device),(256,256));t=targets[ids].to(device);p=z.sigmoid()
            dice=(1-(2*(p*t).sum((1,2,3))+1)/(p.sum((1,2,3))+t.sum((1,2,3))+1))*covered[ids].to(device)
            loss=F.binary_cross_entropy_with_logits(z,t)+dice.mean()+F.binary_cross_entropy_with_logits(presence(global_features[ids].to(device))[:,0],covered[ids].to(device).float())
            assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(list(pixel.parameters())+list(presence.parameters()),1.,error_if_nonfinite=True);optimizer.step();total+=float(loss.detach())
        assert seen==set(range(468))
        row={'epoch':epoch+1,'mean_loss':total/80,'real_train':metrics(list(range(68))),'synthetic_train':metrics(list(range(68,468)))}
        report['history'].append(row);(out/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
    torch.save({'format':'mixed-feature-diagnostic-v1','diagnostic_only':True,'pixel':{k:v.cpu() for k,v in pixel.state_dict().items()},'presence':{k:v.cpu() for k,v in presence.state_dict().items()},'protocol':protocol},out/'final_epoch_20.pth')
    report['complete']=True;report['seconds']=time.monotonic()-start
    (out/'results.json').write_text(json.dumps(report,indent=2));print('COMPLETE',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true')
    args=parser.parse_args()
    main(args.preflight)
