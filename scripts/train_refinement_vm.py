"""VM-only matched RGB ablation; frozen parent/gate, verified original inputs."""
import argparse
import hashlib
import json
import sys
import tarfile
import time
from pathlib import Path
import numpy as np
import torch


def verify_input(rgb, target, record, cached_target):
    if hashlib.sha256(rgb.tobytes()).hexdigest()!=record['input_rgb_sha256']:
        raise ValueError('RGB/cache mismatch')
    if not np.array_equal(target.astype(bool),cached_target.astype(bool)):
        raise ValueError('Target/cache mismatch')


def main(root,preflight=False):
    if not torch.cuda.is_available():raise RuntimeError('Training requires the VM GPU')
    root=Path(root).resolve();sys.path.insert(0,str(root));sys.path.insert(0,str(Path(__file__).resolve().parent))
    from refinement_head import RefinementHead
    import refinement_head
    from expanded_feature_data import ExpandedCoveringDataset,expanded_balanced_schedule,local_path,SOURCES
    from detector_training import load_manifest,ReviewedMasks
    from feature_disk_cache import FeatureDiskCache
    from scripts.compare_presence_heads_vm import PresenceHead
    from scripts.train_expanded_feature_vm import training_loss,pixel_counts,atomic_json,head_digest
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    parent=root/'outputs/expanded_border_training/control_epoch_10.pth'
    assert sha(parent)=='152735d14c6b1b84cc873401e99c4c8861a0fdefe9a7b83198a3737c5a44d2bc'
    state=torch.load(parent,weights_only=True,map_location='cpu');prior=state['protocol']['prior_protocol']
    inventory=root/'expanded_inventory.json';assert sha(inventory)==prior['inventory_sha256']
    for name,digest in json.loads(inventory.read_text()).items():assert sha(local_path(root,name))==digest,name
    manifest=root/'outputs/expanded_feature_data_v1/manifest.json';assert sha(manifest)==prior['manifest_sha256']
    manifest_data=json.loads(manifest.read_text());dataset=ExpandedCoveringDataset(manifest_data,root,placement='fixed')
    real_manifest=root/'dataset/detector_glare_review_v3/manifest.json';assert sha(real_manifest)==manifest_data['v3_manifest_sha256']
    rows=[r for r in load_manifest(real_manifest) if r['split']=='train'];real=ReviewedMasks(rows,256)
    assert len(rows)==68 and len(dataset)==3472
    out=root/'outputs/refinement_training';archive=root/'refinement-results.tar.gz'
    if out.exists() or archive.exists():raise RuntimeError('Output exists; preserve and inspect it')
    context={'arm':'fixed',**{k:prior[k] for k in ('manifest_sha256','inventory_sha256','encoder_sha256')}}
    cache=FeatureDiskCache.open(root/'outputs/expanded_feature_training/fixed_cache',context)
    try:
        assert cache.state['count']==3540
        records=[r['record'] for r in cache.state['rows']]
        # Store verified original RGB as uint8, about 664 MiB, not float features.
        images=np.empty((3540,3,256,256),dtype=np.uint8)
        for i,record in enumerate(records):
            if i<68:
                row=rows[i];assert record['domain']=='real' and record['image']==row['image']
                assert sha(row['image_path'])==record['image_sha256'] and sha(row['mask_path'])==record['mask_sha256']
                x,t=real[i]
            else:
                item=dataset[i-68];assert record['domain']=='synthetic' and record['path']==item['path'] and record['variant']==item['variant']
                x,t=item['input'],item['mask']
            rgb=(x.permute(1,2,0).numpy()*255).round().astype('uint8')
            verify_input(rgb,t.numpy(),record,cache.targets[i]);images[i]=rgb.transpose(2,0,1)
            if i%500==0:print('Verified RGB',i,'/3540',flush=True)
        groups=[[i for i,r in enumerate(records) if r['domain']=='real' and r['kind']==kind] for kind in ('covered','uncovered')]
        for source in SOURCES:
            for covered in (True,False):groups.append([i for i,r in enumerate(records) if r['domain']=='synthetic' and r['source']==source and (r['kind']!='none')==covered])
        schedule=list(expanded_balanced_schedule(groups,42,epochs=10,steps_per_epoch=80))
        assert {i for e in schedule for b in e for i in b}==set(range(3540))
        def heads(use_rgb):
            torch.manual_seed(42)
            model=RefinementHead(state['pixel'],use_rgb=use_rgb).cuda()
            gate=PresenceHead(4).cuda().eval().requires_grad_(False);gate.load_state_dict(state['presence'])
            return model,gate
        def batch(ids):
            f,t,_=cache.batch(ids)
            return torch.from_numpy(f).cuda(),torch.from_numpy(images[ids]).cuda().float()/255,torch.from_numpy(t).cuda().float()
        model,gate=heads(True);initial=head_digest(model,gate);f,x,t=batch(schedule[0][0])
        with torch.no_grad():assert torch.equal(model(f,x),model.parent(f))
        # Backward verifies GPU memory/gradient path without any optimizer update.
        loss=training_loss(model(f,x),gate(f),t);assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None for p in model.parent.parameters()) and all(p.grad is None for p in gate.parameters())
        peak=torch.cuda.max_memory_allocated();del model,gate,f,x,t,loss;torch.cuda.empty_cache()
        if preflight:print('PREFLIGHT PASS: 3540 input matches, full batch forward/backward, zero updates; peak bytes',peak,flush=True);return
        protocol={'format':'rgb-refinement-v1','parent_sha256':sha(parent),'prior_protocol':state['protocol'],
                  'script_sha256':sha(__file__),'module_sha256':sha(refinement_head.__file__),
                  'arms':{'semantic':False,'rgb':True},'seed':42,'epochs':10,'steps_per_epoch':80,'batch_size':12,
                  'lr':1e-3,'weight_decay':1e-4,'clip':1.,'parent_frozen':True,'gate_frozen':True,'threshold':.5,
                  'schedule_sha256':hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
                  'images_sha256':hashlib.sha256(images.tobytes()).hexdigest(),'initial_heads_sha256':initial,
                  'loss':'original BCE + covered Dice + frozen gate BCE constant','selection':'final only',
                  'gpu':torch.cuda.get_device_name(),'torch':str(torch.__version__)}
        out.mkdir();atomic_json(out/'protocol.json',protocol);report={'complete':False,'protocol':protocol,'arms':{}};start=time.monotonic()
        for arm,use_rgb in protocol['arms'].items():
            model,gate=heads(use_rgb);assert head_digest(model,gate)==initial
            params=[p for p in model.parameters() if p.requires_grad]
            optimizer=torch.optim.AdamW(params,lr=1e-3,weight_decay=1e-4);history=[];seen=set()
            for epoch,batches in enumerate(schedule,1):
                model.train();total=0.
                for ids in batches:
                    f,x,t=batch(ids);optimizer.zero_grad(set_to_none=True)
                    loss=training_loss(model(f,x),gate(f),t);assert torch.isfinite(loss)
                    loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                    seen.update(ids);total+=float(loss.detach())
                history.append({'epoch':epoch,'loss':total/80});report['arms'][arm]={'complete':False,'history':history}
                atomic_json(out/'results.json',report);print(arm,history[-1],flush=True)
            assert seen==set(range(3540))
            assert all(torch.equal(v.cpu(),state['pixel'][k]) for k,v in model.parent.state_dict().items())
            assert all(torch.equal(v.cpu(),state['presence'][k]) for k,v in gate.state_dict().items())
            model.eval();totals={}
            with torch.inference_mode():
                for offset in range(0,3540,12):
                    ids=list(range(offset,min(offset+12,3540)));f,x,t=batch(ids);target=t.bool();raw=model(f,x)>=0;gated=raw&(gate(f)>=0)[:,None,None,None]
                    for j,i in enumerate(ids):
                        r=records[i];group='real' if r['domain']=='real' else r['kind']+'/'+str(r['degraded'])
                        for mode,pred in (('raw',raw),('gated',gated)):
                            d=pixel_counts(pred[j:j+1],target[j:j+1]);s=totals.setdefault(group+'/'+mode,{k:0 for k in d})
                            for k,v in d.items():s[k]+=v
            path=out/(arm+'_epoch_10.pth')
            torch.save({'format':'rgb-refinement-v1','arm':arm,'use_rgb':use_rgb,'protocol':protocol,
                        'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                        'presence':{k:v.detach().cpu() for k,v in gate.state_dict().items()}},path)
            report['arms'][arm].update(complete=True,updates=800,seen=len(seen),parent_unchanged=True,gate_unchanged=True,checkpoint_sha256=sha(path),training_counts=totals)
            atomic_json(out/'results.json',report);del model,gate,optimizer,params;torch.cuda.empty_cache()
        report.update(complete=True,seconds=time.monotonic()-start);atomic_json(out/'results.json',report)
        with tarfile.open(archive,'w:gz') as tar:
            for p in sorted(out.iterdir()):tar.add(p,arcname=p.name)
            tar.add(__file__,arcname='train_refinement_vm.py');tar.add(refinement_head.__file__,arcname='refinement_head.py')
        print('DONE:',archive,flush=True)
    finally:cache.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='.')
    p.add_argument('--preflight',action='store_true');a=p.parse_args();main(a.root,a.preflight)
