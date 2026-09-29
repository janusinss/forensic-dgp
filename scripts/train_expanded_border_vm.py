"""Bounded VM-only border-weight comparison on unchanged frozen training caches."""
import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F


def border_loss(logits, target, border, weight):
    if weight not in (1, 2) or border.shape != target.shape or torch.any(border & ~target.bool()):
        raise ValueError('Invalid border weight or target partition')
    weights = 1 + (weight-1)*border.float()
    bce = (F.binary_cross_entropy_with_logits(logits, target, reduction='none')*weights).flatten(1).sum(1)/weights.flatten(1).sum(1)
    p = logits.sigmoid()
    dice = (1-(2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1))*target.flatten(1).any(1)
    return bce.mean()+dice.mean()


def main(root, preflight=False):
    if not torch.cuda.is_available():
        raise RuntimeError('Training requires the VM GPU; CPU execution disabled')
    root = Path(root).resolve();sys.path.insert(0,str(root))
    from expanded_feature_data import ExpandedCoveringDataset, expanded_balanced_schedule, local_path, SOURCES
    from feature_disk_cache import FeatureDiskCache
    from scripts.compare_pixel_heads_vm import PixelHead
    from scripts.compare_presence_heads_vm import PresenceHead
    from scripts.train_expanded_feature_vm import head_digest, atomic_json, fit_metrics
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    train=root/'outputs/expanded_feature_training'
    parent=train/'fixed_epoch_20.pth'
    assert sha(parent)=='7846d84ac23849110604a6cead0678fa49f9a75b2cf6e95367e607631851eed4'
    state=torch.load(parent,map_location='cpu',weights_only=True)
    prior=json.loads((train/'protocol.json').read_text());assert state['protocol']==prior
    inventory_path=root/'expanded_inventory.json';assert sha(inventory_path)==prior['inventory_sha256']
    for name,digest in json.loads(inventory_path.read_text()).items():
        assert sha(local_path(root,name))==digest,name
    manifest_path=root/'outputs/expanded_feature_data_v1/manifest.json'
    assert sha(manifest_path)==prior['manifest_sha256']
    dataset=ExpandedCoveringDataset(json.loads(manifest_path.read_text()),root,placement='fixed')
    context={'arm':'fixed',**{k:prior[k] for k in ('manifest_sha256','inventory_sha256','encoder_sha256')}}
    cache=FeatureDiskCache.open(train/'fixed_cache',context)
    out=root/'outputs/expanded_border_training'
    archive=root/'expanded-border-results.tar.gz'
    if out.exists() or archive.exists():
        cache.close();raise RuntimeError('Output exists; preserve it, do not rerun')
    try:
        assert cache.state['count']==3540 and len(dataset)==3472
        records=[r['record'] for r in cache.state['rows']]
        groups=[[i for i,r in enumerate(records) if r['domain']=='real' and r['kind']==kind] for kind in ('covered','uncovered')]
        for source in SOURCES:
            for covered in (True,False):
                groups.append([i for i,r in enumerate(records) if r['domain']=='synthetic' and r['source']==source and (r['kind']!='none')==covered])
        schedule=list(expanded_balanced_schedule(groups,42,epochs=10,steps_per_epoch=80))
        assert {i for batches in schedule for batch in batches for i in batch}==set(range(3540))
        # Regeneration is label preparation, never encoder extraction or training.
        borders=np.zeros((3540,1,256,256),dtype=np.bool_)
        for index in range(len(dataset)):
            item=dataset[index];i=68+index;r=records[i]
            assert r['domain']=='synthetic' and r['path']==item['path'] and r['variant']==item['variant']
            rgb=(item['input'].permute(1,2,0).numpy()*255).round().astype('uint8')
            assert hashlib.sha256(rgb.tobytes()).hexdigest()==r['input_rgb_sha256']
            target=item['mask'].numpy().astype(bool);core=item['geometry'].numpy().astype(bool)
            assert np.array_equal(target,cache.targets[i].astype(bool)) and not np.any(core&~target)
            if item['degraded']:borders[i]=target&~core
            if index%500==0:print('Verified labels',index,'/3472',flush=True)
        def heads():
            torch.manual_seed(42)
            pixel=PixelHead(3).cuda();gate=PresenceHead(4).cuda().eval().requires_grad_(False)
            pixel.load_state_dict(state['pixel']);gate.load_state_dict(state['presence'])
            return pixel,gate
        pixel,gate=heads();initial=head_digest(pixel,gate)
        ids=schedule[0][0];f,t,_=cache.batch(ids)
        with torch.inference_mode():
            loss=border_loss(pixel(torch.from_numpy(f).cuda()),torch.from_numpy(t).cuda().float(),torch.from_numpy(borders[ids]).cuda(),2)
            assert torch.isfinite(loss)
        del pixel,gate
        if preflight:
            print('PREFLIGHT PASS: labels, cache batch, schedule and forward; zero updates',flush=True);return
        protocol={'format':'expanded-border-v1','parent_sha256':sha(parent),'prior_protocol':prior,
                  'script_sha256':sha(__file__),'arms':{'control':1,'border2':2},'epochs':10,
                  'steps_per_epoch':80,'batch_size':12,'seed':42,'lr':1e-4,'weight_decay':1e-4,
                  'gradient_clip':1.,'gate_frozen':True,'threshold':.5,'initial_heads_sha256':initial,
                  'schedule_sha256':hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
                  'border_sha256':hashlib.sha256(borders.tobytes()).hexdigest(),
                  'group_sizes':list(map(len,groups)),'selection':'final only; original external safeguards',
                  'torch':str(torch.__version__),'gpu':torch.cuda.get_device_name()}
        out.mkdir();atomic_json(out/'protocol.json',protocol)
        report={'complete':False,'protocol':protocol,'arms':{}};started=time.monotonic()
        for arm,weight in protocol['arms'].items():
            pixel,gate=heads();assert head_digest(pixel,gate)==initial
            optimizer=torch.optim.AdamW(pixel.parameters(),lr=1e-4,weight_decay=1e-4)
            history=[];seen=set()
            for epoch,batches in enumerate(schedule,1):
                pixel.train();total=0.
                for ids in batches:
                    f,t,_=cache.batch(ids)
                    optimizer.zero_grad(set_to_none=True)
                    loss=border_loss(pixel(torch.from_numpy(f).cuda()),torch.from_numpy(t).cuda().float(),torch.from_numpy(borders[ids]).cuda(),weight)
                    assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(pixel.parameters(),1.,error_if_nonfinite=True);optimizer.step()
                    total+=float(loss.detach());seen.update(ids)
                history.append({'epoch':epoch,'loss':total/80})
                report['arms'][arm]={'complete':False,'history':history};atomic_json(out/'results.json',report)
                print(arm,history[-1],flush=True)
            assert seen==set(range(3540))
            assert all(torch.equal(v.cpu(),state['presence'][k]) for k,v in gate.state_dict().items())
            pixel.eval();fit=fit_metrics(cache,pixel,gate,'cuda')
            # Report border/core fit across all domains, with actual synthetic border masks.
            region_totals={}
            with torch.inference_mode():
                for start in range(0,3540,12):
                    ids=list(range(start,min(start+12,3540)));f,t,rows=cache.batch(ids)
                    f=torch.from_numpy(f).cuda();target=torch.from_numpy(t).cuda().bool();border=torch.from_numpy(borders[ids]).cuda();core=target&~border
                    raw=pixel(f)>=0;gated=raw&(gate(f)>=0)[:,None,None,None]
                    for j,r in enumerate(rows):
                        group='real' if r['domain']=='real' else r['kind']+'/'+str(r['degraded'])
                        for mode,pred in (('raw',raw),('gated',gated)):
                            key=group+'/'+mode
                            counts={'core_pixels':int(core[j].sum()),'core_missed':int((core[j]&~pred[j]).sum()),'border_pixels':int(border[j].sum()),'border_missed':int((border[j]&~pred[j]).sum()),'outside_pixels':int((~target[j]).sum()),'outside_fp':int((pred[j]&~target[j]).sum())}
                            s=region_totals.setdefault(key,{k:0 for k in counts})
                            for k,v in counts.items():s[k]+=v
            path=out/(arm+'_epoch_10.pth')
            torch.save({'format':'expanded-border-v1','diagnostic_only':True,'arm':arm,'kernel':3,'grid':4,
                        'pixel':{k:v.detach().cpu() for k,v in pixel.state_dict().items()},
                        'presence':{k:v.detach().cpu() for k,v in gate.state_dict().items()},'protocol':protocol},path)
            report['arms'][arm].update(complete=True,updates=800,seen=len(seen),gate_unchanged=True,
                                       checkpoint_sha256=sha(path),training_fit=fit,regions=region_totals)
            atomic_json(out/'results.json',report)
            del pixel,gate,optimizer;torch.cuda.empty_cache()
        report.update(complete=True,seconds=time.monotonic()-started);atomic_json(out/'results.json',report)
        with tarfile.open(archive,'w:gz') as tar:
            for path in sorted(out.iterdir()):tar.add(path,arcname=path.name)
            tar.add(__file__,arcname='train_expanded_border_vm.py')
        print('DONE:',archive,flush=True)
    finally:
        cache.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='.')
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args()
    main(args.root,args.preflight)
