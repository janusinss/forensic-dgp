"""GPU-only matched fixed/anatomical detector training on disk-backed features."""
import argparse,hashlib,json,os,platform,shutil,sys,tarfile,time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F

PARENT_PIXEL='eaa16229f88d416ad09d813a6f08ca0ae72bba214cb8266648bed382603864a4'
PARENT_PRESENCE='b7af5b8568fc1fd1a6be289a6599794c6c2976e162a351be90a53344997b40c4'
ENCODER='7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69'


def fixed_protocol():
    return {'format':'expanded-feature-placement-v1','arms':['fixed','anatomical'],'seed':42,
            'epochs':20,'steps_per_epoch':80,'batch_size':12,'lr':.001,'weight_decay':.0001,
            'gradient_clip':1.,'thresholds':{'pixel':.5,'presence':.5},'encoder_frozen':True,
            'pixel_kernel':3,'presence_grid':4,'both_heads_train':True,
            'loss':'all-image pixel BCE + nonempty Dice / whole batch + presence BCE',
            'selection':'final only; external unchanged safeguards',
            'parent_pixel_sha256':PARENT_PIXEL,'parent_presence_sha256':PARENT_PRESENCE,'encoder_sha256':ENCODER}


def make_heads(pixel_state,presence_state,device):
    from scripts.compare_pixel_heads_vm import PixelHead
    from scripts.compare_presence_heads_vm import PresenceHead
    pixel=PixelHead(3).to(device);pixel.initialize(pixel_state)
    gate=PresenceHead(4).to(device);gate.load_state_dict(presence_state)
    return pixel,gate


def head_digest(pixel,gate):
    digest=hashlib.sha256()
    for label,model in [('pixel',pixel),('presence',gate)]:
        for name,tensor in model.state_dict().items():
            digest.update((label+name+str(tuple(tensor.shape))).encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def training_loss(logits,presence,target):
    covered=target.flatten(1).any(1).float();p=logits.sigmoid()
    dice=(1-(2*(p*target).sum((1,2,3))+1)/(p.sum((1,2,3))+target.sum((1,2,3))+1))*covered
    return F.binary_cross_entropy_with_logits(logits,target)+dice.mean()+F.binary_cross_entropy_with_logits(presence,covered)


def pixel_counts(pred,target):
    has=target.flatten(1).any(1);got=pred.flatten(1).any(1)
    return {k:int(v) for k,v in dict(tp=(pred&target).sum(),fp=(pred&~target).sum(),fn=(~pred&target).sum(),
            visible=(~target).sum(),empty=(has&~got).sum(),negative_fp=(~has&got).sum(),positive=has.sum(),negative=(~has).sum()).items()}


def atomic_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)


def fit_metrics(cache,pixel,gate,device):
    totals={}
    with torch.inference_mode():
        for start in range(0,cache.state['count'],12):
            f,t,records=cache.batch(range(start,min(start+12,cache.state['count'])))
            f=torch.from_numpy(f).to(device);target=torch.from_numpy(t).to(device).bool()
            raw=pixel(f)>=0;keep=gate(f)>=0
            for j,r in enumerate(records):
                domain=r['domain']
                for mode,pred in [('raw',raw),('gated',raw&keep[:,None,None,None])]:
                    key=domain+'/'+mode;c=pixel_counts(pred[j:j+1],target[j:j+1])
                    if key not in totals:totals[key]={k:0 for k in c}
                    for k,v in c.items():totals[key][k]+=v
    return {k:dict(counts=s,iou=s['tp']/max(1,s['tp']+s['fp']+s['fn']),
                    visible_false_positive=s['fp']/max(1,s['visible']),missed_fraction=s['fn']/max(1,s['tp']+s['fn'])) for k,s in totals.items()}


def main(root,preflight=False):
    # Must precede workspace reads/imports, feature caching and optimizer creation.
    if not torch.cuda.is_available():raise RuntimeError('Training requires the VM GPU; CPU execution disabled')
    root=Path(root).resolve();os.chdir(root);sys.path[:0]=[str(root),str(root/'outputs/vendor_sam2')]
    from feature_vm_runtime import require_cuda
    from expanded_feature_data import ExpandedCoveringDataset,expanded_balanced_schedule,local_path,SOURCES
    from feature_disk_cache import FeatureDiskCache
    from detector_training import load_manifest,ReviewedMasks
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    device=require_cuda();torch.set_num_threads(4)
    torch.manual_seed(42);np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    inventory_path=root/'expanded_inventory.json';inventory=json.loads(inventory_path.read_text())
    for name,digest in inventory.items():
        if sha(local_path(root,name))!=digest:raise ValueError('Bundle hash mismatch: '+name)
    out=root/'outputs/expanded_feature_training'
    if out.exists():raise RuntimeError('Output exists; preserve it for inspection, do not rerun blindly')
    if shutil.disk_usage(root).free<35*1024**3:raise RuntimeError('At least 35 GiB free disk required for both caches')
    folder=root/'outputs/expanded_feature_data_v1';manifest_path=folder/'manifest.json'
    manifest=json.loads(manifest_path.read_text());check=json.loads((folder/'matched_check.json').read_text())
    if not check['complete'] or sha(manifest_path)!=check['manifest_sha256']:raise ValueError('Unverified dataset manifest')
    for name,digest in check['current_code_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Matched data code changed: '+name)
    data={arm:ExpandedCoveringDataset(manifest,root,placement=arm) for arm in ('fixed','anatomical')}
    if data['fixed'].cases!=data['anatomical'].cases or data['fixed'].rejections!=data['anatomical'].rejections:
        raise ValueError('Arms have different membership')
    if len(data['fixed'])!=3472:raise ValueError('Unexpected dataset size')
    real_path=root/'dataset/detector_glare_review_v3/manifest.json'
    if sha(real_path)!=manifest['v3_manifest_sha256']:raise ValueError('Reviewed manifest changed')
    rows=[r for r in load_manifest(real_path) if r['split']=='train']
    if len(rows)!=68:raise ValueError('Unexpected real training membership')
    real=ReviewedMasks(rows,256)
    pixel_path=root/'outputs/expanded_parents/mixed.pth';presence_path=root/'outputs/expanded_parents/spatial.pth'
    weights=root/'outputs/sam2.1_hiera_tiny.pt'
    for path,digest in [(pixel_path,PARENT_PIXEL),(presence_path,PARENT_PRESENCE),(weights,ENCODER)]:
        if sha(path)!=digest:raise ValueError('Parent/encoder hash mismatch')
    pixel_state=torch.load(pixel_path,weights_only=True,map_location='cpu')['pixel']
    presence_state=torch.load(presence_path,weights_only=True,map_location='cpu')['model']
    groups=[[i for i,r in enumerate(rows) if r['kind']==kind] for kind in ('covered','uncovered')]
    for source in SOURCES:
        for covered in (True,False):groups.append([68+i for i,c in enumerate(data['fixed'].cases) if c['source']==source and (c['kind']!='none')==covered])
    schedule=list(expanded_balanced_schedule(groups,42))
    predictor=SAM2ImagePredictor(build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(weights),device=str(device),apply_postprocessing=False))
    predictor.model.eval().requires_grad_(False)
    pixel,gate=make_heads(pixel_state,presence_state,device);initial=head_digest(pixel,gate)
    with torch.inference_mode():
        rgb=(real[0][0].permute(1,2,0).numpy()*255).round().astype('uint8')
        predictor.set_image(rgb);probe=predictor.get_image_embedding()
        if probe.shape!=(1,256,64,64) or not torch.isfinite(probe).all():raise ValueError('Invalid encoder probe')
        if not torch.isfinite(pixel(probe)).all() or not torch.isfinite(gate(probe)).all():raise ValueError('Invalid head forward')
    del pixel,gate,probe
    if preflight:
        print('PREFLIGHT PASS: CUDA, disk, hashes, splits, matched data and one forward; zero optimizer updates',flush=True)
        return
    out.mkdir();protocol=fixed_protocol()
    protocol.update(manifest_sha256=sha(manifest_path),inventory_sha256=sha(inventory_path),
                    initial_heads_sha256=initial,group_sizes=list(map(len,groups)),
                    schedule_sha256=hashlib.sha256(json.dumps(schedule).encode()).hexdigest(),
                    torch=str(torch.__version__),python=platform.python_version(),gpu=torch.cuda.get_device_name(),
                    tf32=False,cache_features='float32; per-row verified disk batches')
    atomic_json(out/'protocol.json',protocol)
    contexts={};start_time=time.monotonic()
    for arm,ds in data.items():
        context={'arm':arm,'manifest_sha256':protocol['manifest_sha256'],'inventory_sha256':protocol['inventory_sha256'],'encoder_sha256':ENCODER}
        contexts[arm]=context;cache=FeatureDiskCache.create(out/(arm+'_cache'),68+len(ds),(256,64,64),(1,256,256),context)
        try:
            for i in range(68+len(ds)):
                if i<68:
                    x,t=real[i];record={'domain':'real','image':rows[i]['image'],'kind':rows[i]['kind'],
                                        'image_sha256':rows[i]['image_sha256'],'mask_sha256':rows[i]['mask_sha256']}
                else:
                    item=ds[i-68];x,t=item['input'],item['mask']
                    record={k:item[k] for k in ('path','source','kind','degraded','variant','seed','augmentation')}
                    record['domain']='synthetic';record['source_sha256']=ds.sources[item['source_index']]['sha256']
                rgb=(x.permute(1,2,0).numpy()*255).round().astype('uint8')
                record['input_rgb_sha256']=hashlib.sha256(rgb.tobytes()).hexdigest()
                with torch.inference_mode():
                    predictor.set_image(rgb);feature=predictor.get_image_embedding()[0].cpu().numpy().copy()
                cache.append(feature,t.numpy(),record)
                if (i+1)%50==0:print(arm,'cache',i+1,68+len(ds),flush=True)
            cache.finish()
        finally:cache.close()
    del predictor;torch.cuda.empty_cache()
    report={'protocol':protocol,'complete':False,'arms':{}}
    for arm in protocol['arms']:
        torch.manual_seed(42);pixel,gate=make_heads(pixel_state,presence_state,device)
        if head_digest(pixel,gate)!=initial:raise ValueError('Initialization mismatch')
        cache=FeatureDiskCache.open(out/(arm+'_cache'),contexts[arm])
        params=list(pixel.parameters())+list(gate.parameters())
        optimizer=torch.optim.AdamW(params,lr=.001,weight_decay=.0001)
        history=[];seen=set()
        try:
            for epoch,batches in enumerate(schedule,1):
                total=0.;pixel.train();gate.train()
                for ids in batches:
                    f,t,_=cache.batch(ids);f=torch.from_numpy(f).to(device);t=torch.from_numpy(t).to(device).float()
                    optimizer.zero_grad(set_to_none=True);loss=training_loss(pixel(f),gate(f),t)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
                    loss.backward();torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);optimizer.step()
                    total+=float(loss.detach());seen.update(ids)
                history.append({'epoch':epoch,'mean_loss':total/80})
                report['arms'][arm]={'complete':False,'history':history}
                atomic_json(out/'results.json',report);print(arm,history[-1],flush=True)
            if seen!=set(range(cache.state['count'])):raise ValueError('Training schedule omitted examples')
            pixel.eval();gate.eval();metrics=fit_metrics(cache,pixel,gate,device)
            checkpoint={'format':'expanded-feature-placement-v1','diagnostic_only':True,'arm':arm,'kernel':3,'grid':4,
                        'pixel':{k:v.detach().cpu() for k,v in pixel.state_dict().items()},
                        'presence':{k:v.detach().cpu() for k,v in gate.state_dict().items()},'protocol':protocol}
            path=out/(arm+'_epoch_20.pth');tmp=path.with_suffix('.tmp');torch.save(checkpoint,tmp);tmp.replace(path)
            report['arms'][arm].update(complete=True,training_fit=metrics,checkpoint_sha256=sha(path),seen=len(seen))
            atomic_json(out/'results.json',report)
        finally:cache.close()
        del pixel,gate,optimizer,params;torch.cuda.empty_cache()
    report.update(complete=True,seconds=time.monotonic()-start_time);atomic_json(out/'results.json',report)
    archive=root/'expanded-feature-results.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in [out/'protocol.json',out/'results.json',out/'fixed_epoch_20.pth',out/'anatomical_epoch_20.pth',
                  out/'fixed_cache/state.json',out/'anatomical_cache/state.json',manifest_path,folder/'matched_check.json',inventory_path]:
            tar.add(p,arcname=p.relative_to(root).as_posix())
    print('DONE:',archive,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',default='.')
    parser.add_argument('--preflight',action='store_true');args=parser.parse_args();main(args.root,args.preflight)
