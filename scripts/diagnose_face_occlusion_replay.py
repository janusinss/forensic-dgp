"""Grouped replay fit and synthetic failures; CPU inference only, no optimizer."""
from collections import Counter
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from completion import KINDS
from scripts.compare_replay_fit import counts,aggregate
from scripts.package_face_occlusion_vm import sha
from scripts.audit_face_occlusion_results import require,read_json,PREFIX,REPLAY_SHA,PROTOCOL_SHA,PARENT_SHA
from scripts.evaluate_coverage_results import binary


def summarize_records(records,weights=None):
    ids=[r['case'] for r in records]
    require(records and len(ids)==len(set(ids)),'Require unique nonempty case records')
    weights={i:1 for i in ids} if weights is None else weights
    require(set(weights)==set(ids) and all(type(v) is int and v>0 for v in weights.values()),'Invalid case weights')
    def score(rows):
        result=aggregate([{k:v*weights[r['case']] for k,v in r['counts'].items()} for r in rows])
        result['cases']=sum(weights[r['case']] for r in rows)
        result['unique_cases']=len(rows)
        return result
    return {'all':score(records),
            'by_kind':{kind:score([r for r in records if r['kind']==kind]) for kind in KINDS if any(r['kind']==kind for r in records)},
            'by_kind_degradation':{kind+'/'+str(d):score([r for r in records if r['kind']==kind and r['degraded']==d])
                                   for kind in KINDS for d in (False,True) if any(r['kind']==kind and r['degraded']==d for r in records)},
            'by_source':{source:score([r for r in records if r['source']==source]) for source in sorted({r['source'] for r in records})}}


def verify_reference(records,cache,protocol):
    require({r['case'] for r in records}==set(cache) and len(records)==len(cache),'Prior replay membership differs')
    for r in records:
        i=r['case'];truth=cache[i][1].bool();c=r['counts'];fg=int(truth.sum());visible=truth.numel()-fg
        require(r['kind']==KINDS[i%5] and r['degraded']==(i%10>=5) and
                r['source']==protocol['sources'][i//10]['source'],'Prior replay metadata differs')
        require(set(c)=={'tp','fp','fn','visible','empty','positive','negative','false_positive'} and
                all(type(v) is int and v>=0 for v in c.values()),'Invalid prior confusion counts')
        require(c['tp']+c['fn']==fg and c['visible']==visible and c['fp']<=visible and
                c['positive']==int(fg>0) and c['negative']==int(fg==0) and
                c['empty']==int(fg>0 and c['tp']+c['fp']==0) and
                c['false_positive']==int(fg==0 and c['fp']>0),'Prior counts disagree with cached target')


def main():
    import torch
    import numpy as np
    from PIL import Image,ImageDraw
    torch.set_num_threads(4)
    out=ROOT/'outputs/face_occlusion_replay_fit'
    require(not out.exists(),'Preserve existing replay diagnostic')
    protocol_path=ROOT/'outputs/coverage_protocol_v1/protocol.json'
    cache_path=ROOT/'outputs/downloaded_coverage/outputs/coverage_training_vm/replay_pixels.pth'
    require(sha(protocol_path)==PROTOCOL_SHA and sha(cache_path)==REPLAY_SHA,'Protocol/cache changed')
    protocol=read_json(protocol_path);cache=torch.load(cache_path,weights_only=True,map_location='cpu')
    indices=sorted(cache)
    weights=Counter(i-73 for e in protocol['schedules']['extended']['batches'] for batch in e for i in batch if i>=73)
    require(set(weights)==set(cache) and len(cache)==638 and sum(weights.values())==840,'Replay exposure differs')
    for x,m in cache.values():
        require(x.dtype==m.dtype==torch.uint8 and x.shape==(3,256,256) and m.shape==(1,256,256) and
                ((m==0)|(m==1)).all(),'Invalid frozen replay tensors')
    prior_path=ROOT/'outputs/replay_fit_comparison/results.json';prior=read_json(prior_path)
    require(prior['optimizer_updates']==0 and prior['cache_sha256']==REPLAY_SHA and
            prior['checkpoints']['parent']==PARENT_SHA,'Prior parent replay provenance differs')
    parent_records=prior['arms']['parent']['records'];verify_reference(parent_records,cache,protocol)
    parent_unique=summarize_records(parent_records)
    require({k:v for k,v in parent_unique['all'].items() if k!='unique_cases'}==prior['arms']['parent']['groups']['all'],
            'Reused parent summary differs from its records')
    require(sha(ROOT/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth')==PARENT_SHA,'Original parent changed')
    audit_root=ROOT/'outputs/face_occlusion_validation'
    audit=read_json(audit_root/'results.json');reproduction=read_json(audit_root/'reproduction.json')
    require(sha(ROOT/'outputs/face-occlusion-results.tar.gz')==audit['archive_sha256']==reproduction['archive_sha256'],'Returned archive changed')
    extraction=ROOT/'outputs/downloaded_face_occlusion';returned=extraction/PREFIX;members=read_json(audit_root/'members.json')
    bench=ROOT/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm'
    benchmark_path=bench/'manifest.json'
    require(sha(benchmark_path)==protocol['input_hashes'][benchmark_path.relative_to(ROOT).as_posix()],'Benchmark changed')
    cases=read_json(benchmark_path)['cases'];require(len(cases)==400,'Unexpected validation membership')
    validation={}
    targets=[]
    original_inventory=read_json(ROOT/'outputs/coverage_protocol_v1/vm_inventory.json')
    for case in cases:
        path=bench/'mask'/case['file']
        require(sha(path)==original_inventory[path.relative_to(ROOT).as_posix()],
                'Benchmark target changed')
        targets.append(binary(path))
    choices={'parent':returned/'baseline_masks/synthetic'}
    choices.update({f'{arm}{epoch}':returned/arm/f'epoch_{epoch}_masks/synthetic' for arm in ('random','pretrained') for epoch in (1,5,10)})
    for key,folder in choices.items():
        records=[]
        for i,case in enumerate(cases):
            path=folder/f'{i:04}.png'
            require(sha(path)==members[path.relative_to(extraction).as_posix()],'Audited validation mask changed')
            records.append({'case':i,'kind':case['kind'],'degraded':case['degraded'],
                            'source':case['dataset_source'],'counts':counts(binary(path),targets[i])})
        validation[key]={'records':records,'groups':summarize_records(records)}
        logged=audit['baseline']['synthetic'] if key=='parent' else next(
            row['synthetic'] for row in audit['arms']['pretrained' if key.startswith('pretrained') else 'random']
            if row['epoch']==int(key.removeprefix('pretrained').removeprefix('random')))
        require({k:v for k,v in validation[key]['groups']['all'].items() if k not in ('cases','unique_cases')}==logged,
                'Grouped validation total differs from audit')
    out.mkdir()
    result={'script_sha256':sha(__file__),'optimizer_updates_locally':0,'evaluation_batch_size':8,
            'replay_sha256':REPLAY_SHA,'protocol_sha256':PROTOCOL_SHA,'archive_sha256':audit['archive_sha256'],
            'reused_parent_record_sha256':sha(prior_path),'parent_reinferred':False,
            'training':{'parent':{'records':parent_records,'unique':parent_unique,
                                  'scheduled_exposure':summarize_records(parent_records,weights)}},
            'validation':validation,'checkpoints':{'parent':PARENT_SHA},'promoted':False}
    sys.path.insert(0,str(ROOT/'outputs/face_extraction_dependencies'))
    from face_occlusion_adapter import load_adapter
    from scripts.train_face_occlusion_vm import dependency_versions
    result['dependencies']=dependency_versions(ROOT/'outputs/face_extraction_dependencies')
    for arm in ('random','pretrained'):
        key=arm+'10';path=returned/arm/'epoch_10.pth'
        require(sha(path)==reproduction['checkpoint_checks'][arm+'/10']['sha256'],'Audited detector changed')
        result['checkpoints'][key]=sha(path)
        model,_=load_adapter(path,'cpu');model.requires_grad_(False).eval()
        before={k:v.clone() for k,v in model.state_dict().items()};records=[]
        mask_dir=out/'train_masks'/key;mask_dir.mkdir(parents=True)
        with torch.inference_mode():
            for start in range(0,len(indices),8):
                batch=indices[start:start+8];x=torch.stack([cache[i][0].float()/255 for i in batch])
                predictions=(model.detect(x).sigmoid()[:,0]>=.5).numpy()
                for i,p in zip(batch,predictions):
                    t=cache[i][1][0].numpy().astype(bool)
                    records.append({'case':i,'kind':KINDS[i%5],'degraded':i%10>=5,
                                    'source':protocol['sources'][i//10]['source'],'counts':counts(p,t)})
                    Image.fromarray(p.astype('uint8')*255).save(mask_dir/f'{i:04}.png')
                if start%160==0:print(key,start,len(indices),flush=True)
        require(all(torch.equal(v,before[k]) for k,v in model.state_dict().items()),'Inference changed detector state')
        result['training'][key]={'records':records,'unique':summarize_records(records),
                                'scheduled_exposure':summarize_records(records,weights),'state_unchanged':True}
        del model,before
        (out/'partial.json').write_text(json.dumps(result,indent=2)+'\n')
    # Prediction-ranked training examples only; no examples are added to labels.
    preview_records=result['training']['pretrained10']['records'];selected=[]
    for kind in KINDS:
        for degraded in (False,True):
            subset=[r for r in preview_records if r['kind']==kind and r['degraded']==degraded]
            if kind=='none':row=min(subset,key=lambda r:(-r['counts']['fp'],r['case']))
            else:row=min(subset,key=lambda r:(r['counts']['tp']/max(1,r['counts']['tp']+r['counts']['fp']+r['counts']['fn']),r['case']))
            selected.append(row)
    sheet=Image.new('RGB',(640,1480),'white')
    for n,row in enumerate(selected):
        i=row['case'];image=cache[i][0].permute(1,2,0).numpy();t=cache[i][1][0].numpy().astype(bool)
        random=binary(out/'train_masks/random10'/f'{i:04}.png');pretrained=binary(out/'train_masks/pretrained10'/f'{i:04}.png')
        overlay=image.copy();overlay[t&~pretrained]=(220,55,65);overlay[~t&pretrained]=(55,135,220)
        tiles=[Image.fromarray(image),Image.fromarray(t.astype('uint8')*255).convert('RGB'),
               Image.fromarray(random.astype('uint8')*255).convert('RGB'),Image.fromarray(pretrained.astype('uint8')*255).convert('RGB'),Image.fromarray(overlay)]
        ImageDraw.Draw(sheet).text((2,n*148+2),f'{i} {row["kind"]}/{row["degraded"]}: input | target | random10 | pretrained10 | red=miss blue=extra',fill='black')
        for j,tile in enumerate(tiles):sheet.paste(tile.resize((128,128)),(j*128,n*148+20))
    sheet.save(out/'preview.png')
    result['preview_selection']={'training_only':True,'prediction_ranked':True,
                                  'rule':'worst IoU per covered stratum; most false pixels for clear; lowest case ID ties',
                                  'cases':[r['case'] for r in selected]}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value['unique']['all'] for key,value in result['training'].items()},indent=2),flush=True)


if __name__=='__main__':main()
