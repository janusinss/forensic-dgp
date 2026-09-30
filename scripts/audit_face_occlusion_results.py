"""Independent returned-mask, source and state audit; inference only, no fitting."""
import argparse
import hashlib
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.package_face_occlusion_vm import sha, sha_stream
from scripts.train_coverage_vm import real_gate, PROTOCOL_SHA, PARENT_SHA
from detector_replay import retention_passes, BenchmarkMasks
from detector_training import load_manifest, ReviewedMasks
from scripts.evaluate_coverage_results import binary, recount

PREFIX = 'outputs/face_occlusion_pilot_vm'
BUNDLE_SHA = '3af2f1b1fd72dd4506c4d1ca4634ab5d5721d1445c043987989522ef421ffea0'
INVENTORY_SHA = 'cb8485ef260bf83f8f4f3e6ce2f6fd45dc1a8726980130ddf4dfea956e83a4c1'
INITIAL_SHA = '7fb3fe1f5e4c2fc8a5266e87dfcd9e84d1bf07728145ebaebe5d487a52047650'
REPLAY_SHA = 'ad6fd64aed4f773c968cac7748be0fd99a7c8679d1019e32fc64a03b13662dc7'
OLD_INVENTORY_SHA = 'c831635933c9b615b4849ad23811a3b82ebde1883aa30a5c2249b8879fa95623'
EPOCHS = (1,5,10)


def require(condition,message):
    if not condition:raise ValueError(message)


def canonical_member(name):
    path = PurePosixPath(name)
    require(name and path.as_posix()==name and not path.is_absolute() and
            '..' not in path.parts and '\\' not in name and not PureWindowsPath(name).drive,
            'Archive path is not a canonical relative path')
    return name


def audit_history(steps,histories,schedule,baseline,complete):
    require(complete['complete'] is True and complete['parent_unchanged'] is True,'Missing completion/invariance marker')
    require(len(schedule)==10 and all(len(epoch)==21 for epoch in schedule),'Fixed schedule length differs')
    require(set(steps)==set(histories)==set(complete['arms'])=={'random','pretrained'},'Missing arm')
    result={'total_logged_updates':0,'arms':{}}
    flattened=[batch for epoch in schedule for batch in epoch]
    for arm,rows in steps.items():
        require(len(rows)==210,'Incomplete update log')
        for n,(row,batch) in enumerate(zip(rows,flattened),1):
            require((row['epoch'],row['step'],row['updates'])==((n-1)//21+1,(n-1)%21+1,n)
                    and row['indices']==batch,'Executed schedule differs')
            require(all(math.isfinite(row[key]) and row[key]>=0 for key in ('loss','pre_clip_norm')),
                    'Loss/gradient must be finite and nonnegative')
        history=histories[arm]
        require([row['epoch'] for row in history]==list(EPOCHS),'Candidate epochs differ')
        best=baseline['real']['iou'];selected=[]
        for row in history:
            epoch=row['epoch']
            require(row['updates']==epoch*21,'Candidate update count differs')
            mean=sum(r['loss'] for r in rows[(epoch-1)*21:epoch*21])/21
            require(math.isclose(mean,row['mean_epoch_loss'],abs_tol=1e-12,rel_tol=0),'Epoch loss differs from step log')
            real=real_gate(row['real'],baseline['real'],best)
            synthetic=retention_passes(row['synthetic'],baseline['synthetic'])
            require(all(type(row[key]) is bool for key in ('real_gate','synthetic_gate','selected')) and
                    (row['real_gate'],row['synthetic_gate'],row['selected'])==(real,synthetic,real and synthetic),
                    'Logged selection differs from unchanged gates')
            if real and synthetic:best=row['real']['iou'];selected.append(epoch)
        require(complete['arms'][arm]['updates']==210 and complete['arms'][arm]['selected_epochs']==selected,
                'Completion counters/selection differ')
        result['arms'][arm]={'logged_updates':210,'selected_epochs':selected}
        result['total_logged_updates']+=210
    return result


def read_json(path):return json.loads(Path(path).read_text())
def read_lines(path):return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify_bundle(bundle,inventory_path):
    require(sha(bundle)==BUNDLE_SHA and sha(inventory_path)==INVENTORY_SHA,'Sent bundle/inventory changed')
    inventory=read_json(inventory_path)
    expected={**inventory,inventory_path.relative_to(ROOT).as_posix():INVENTORY_SHA}
    seen=set()
    with tarfile.open(bundle,'r|gz') as tar:
        for member in tar:
            name=canonical_member(member.name)
            require(member.isfile() and name not in seen and name in expected,'Unexpected sent member')
            require(sha_stream(tar.extractfile(member))==expected[name],'Sent member hash differs')
            seen.add(name)
    require(seen==set(expected),'Missing sent file')
    for path,digest in inventory.items():require(sha(ROOT/path)==digest,f'Local sent reference changed: {path}')
    return inventory


def extract_return(archive,output,inventory):
    require(output.resolve().is_relative_to(ROOT) and not output.exists(),'Preserve existing extracted results')
    output.mkdir(parents=True)
    allowed_code={path:digest for path,digest in inventory.items() if not path.endswith('.pth')}
    allowed_code['outputs/face_occlusion_bundle_v1/inventory.json']=INVENTORY_SHA
    setup='outputs/face_occlusion_dependencies/setup.json'
    seen=set();hashes={}
    with tarfile.open(archive,'r|gz') as tar:
        for member in tar:
            name=canonical_member(member.name)
            require(name not in seen and (member.isfile() or member.isdir()),'Duplicate path or unsupported archive type')
            seen.add(name)
            if member.isdir():
                require(name==PREFIX or name.startswith(PREFIX+'/'),'Unexpected returned directory')
                continue
            require(name in allowed_code or name==setup or name.startswith(PREFIX+'/'),'Unexpected returned file')
            require(0<=member.size<=64*1024*1024,'Unexpected returned file size')
            destination=(output/name).resolve()
            require(destination.is_relative_to(output.resolve()),'Extraction escapes destination')
            destination.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256()
            with tar.extractfile(member) as stream,destination.open('xb') as saved:
                for chunk in iter(lambda:stream.read(1024*1024),b''):
                    digest.update(chunk);saved.write(chunk)
            hashes[name]=digest.hexdigest()
            if name in allowed_code:require(hashes[name]==allowed_code[name],f'Returned code differs: {name}')
    require(set(allowed_code).issubset(hashes),'Returned provenance files missing')
    return hashes


def verified_inputs():
    protocol_path=ROOT/'outputs/coverage_protocol_v1/protocol.json'
    old_inventory=ROOT/'outputs/coverage_protocol_v1/vm_inventory.json'
    require(sha(protocol_path)==PROTOCOL_SHA and sha(old_inventory)==OLD_INVENTORY_SHA,'Original protocol/inventory changed')
    for path,digest in read_json(old_inventory).items():require(sha(ROOT/path)==digest,f'Original reference changed: {path}')
    rows=load_manifest(ROOT/'dataset/detector_training_extension_v2/manifest.json')
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    require(len(training)==73 and len(validation)==25,'Real split changed')
    benchmark=BenchmarkMasks(ROOT/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm',256)
    require(len(benchmark)==400,'Synthetic benchmark changed')
    data={'real':ReviewedMasks(validation,256),'synthetic':benchmark,'training_real':ReviewedMasks(training,256)}
    return read_json(protocol_path),validation,data


def subsets(scores,predictions,truth,rows):
    for key,predicate in [('human_real',lambda r:Path(r['image']).name!='new_covered_40.png'),
                          ('mannequin',lambda r:Path(r['image']).name=='new_covered_40.png'),
                          ('glare',lambda r:r.get('glare_stratum')=='strong_lens_reflection')]:
        ids=[i for i,r in enumerate(rows) if predicate(r)]
        scores[key]=recount([predictions[i] for i in ids],[truth[i] for i in ids])
    return scores


def same_scores(actual,logged,label):
    for domain,metrics in actual.items():
        require(set(metrics)==set(logged[domain]),f'Metric fields differ: {label}/{domain}')
        for key,value in metrics.items():
            require(math.isclose(value,logged[domain][key],abs_tol=1e-12,rel_tol=0),f'Mask recount differs: {label}/{domain}/{key}')


def expected_files(inventory,complete,sizes):
    files={path for path in inventory if not path.endswith('.pth')}
    files.update(['outputs/face_occlusion_bundle_v1/inventory.json','outputs/face_occlusion_dependencies/setup.json'])
    files.update(f'{PREFIX}/{name}.json' for name in ('baseline','run','complete'))
    for domain in ('real','synthetic'):
        files.update(f'{PREFIX}/baseline_masks/{domain}/{i:04}.png' for i in range(sizes[domain]))
    for arm in ('random','pretrained'):
        files.update(f'{PREFIX}/{arm}/{name}.jsonl' for name in ('steps','metrics'))
        for epoch in EPOCHS:
            files.add(f'{PREFIX}/{arm}/epoch_{epoch}.pth')
            for domain,count in sizes.items():
                files.update(f'{PREFIX}/{arm}/epoch_{epoch}_masks/{domain}/{i:04}.png' for i in range(count))
        if complete['arms'][arm]['selected_epochs']:files.add(f'{PREFIX}/{arm}/best_detector.pth')
    return files


def audit():
    import numpy as np
    from PIL import Image,ImageDraw
    archive=ROOT/'outputs/face-occlusion-results.tar.gz'
    out=ROOT/'outputs/face_occlusion_validation'
    require(not out.exists(),'Preserve existing audit evidence')
    inventory=verify_bundle(ROOT/'outputs/face-occlusion-vm-code.tar.gz',ROOT/'outputs/face_occlusion_bundle_v1/inventory.json')
    protocol,rows,data=verified_inputs()
    extraction=ROOT/'outputs/downloaded_face_occlusion'
    hashes=extract_return(archive,extraction,inventory);returned=extraction/PREFIX
    run=read_json(returned/'run.json');complete=read_json(returned/'complete.json');baseline=read_json(returned/'baseline.json')
    fixed={'preflight':False,'protocol_sha256':PROTOCOL_SHA,'parent_sha256':PARENT_SHA,
           'old_inventory_sha256':OLD_INVENTORY_SHA,'bundle_inventory_sha256':INVENTORY_SHA,
           'initial_manifest_sha256':INITIAL_SHA,'replay_sha256':REPLAY_SHA,
           'encoder_lr':1e-5,'decoder_head_lr':1e-4,'weight_decay':1e-4,'clip':1.,
           'background_weight':.25,'consistency_weight':1.,'seed':42,'updates_per_arm':210,
           'check_epochs':list(EPOCHS),'validation_batch_size':1,
           'batchnorm_policy':'fixed_running_statistics_trainable_affine'}
    for key,value in fixed.items():require(run[key]==value,f'Fixed run field changed: {key}')
    for name,key in [('scripts/train_face_occlusion_vm.py','script_sha256'),
                     ('face_occlusion_adapter.py','adapter_sha256'),('FACE_OCCLUSION_PILOT.md','specification_sha256')]:
        require(hashes[name]==run[key],f'Logged source hash differs: {name}')
    setup_path='outputs/face_occlusion_dependencies/setup.json';setup=read_json(extraction/setup_path)
    require(hashes[setup_path]==run['setup_sha256'] and setup['packages']==run['dependencies'] and
            setup['torch']==run['torch'] and setup['optimizer_updates']==0,'Dependency setup provenance differs')
    registry=read_json(ROOT/'outputs/face_occlusion_initial_v1/manifest.json')
    require(run['source_sha256']==registry['source_sha256'],'External source initialization differs')
    steps={a:read_lines(returned/a/'steps.jsonl') for a in ('random','pretrained')}
    histories={a:read_lines(returned/a/'metrics.jsonl') for a in steps}
    log=audit_history(steps,histories,protocol['schedules']['extended']['batches'],baseline,complete)
    require(set(hashes)==expected_files(inventory,complete,{k:len(d) for k,d in data.items()}),'Returned file membership differs')
    for arm in steps:require(hashes[f'{PREFIX}/{arm}/epoch_10.pth']==complete['arms'][arm]['final_sha256'],'Final state hash differs')
    targets={domain:[dataset[i][1][0].numpy().astype(bool) for i in range(len(dataset))] for domain,dataset in data.items()}
    def saved_scores(folder,domains):
        predictions={domain:[binary(folder/domain/f'{i:04}.png') for i in range(len(targets[domain]))] for domain in domains}
        scores={domain:recount(predictions[domain],targets[domain]) for domain in domains}
        return subsets(scores,predictions['real'],targets['real'],rows),predictions
    base_scores,base_predictions=saved_scores(returned/'baseline_masks',('real','synthetic'))
    same_scores(base_scores,baseline,'baseline');total=425;arms={};previews={}
    for arm,history in histories.items():
        arms[arm]=[]
        for row in history:
            epoch=row['epoch'];scores,predictions=saved_scores(returned/arm/f'epoch_{epoch}_masks',data)
            same_scores(scores,row,f'{arm}/{epoch}');total+=sum(map(len,targets.values()))
            arms[arm].append({'epoch':epoch,'updates':row['updates'],'real_gate':row['real_gate'],
                              'synthetic_gate':row['synthetic_gate'],'selected':row['selected'],**scores})
            if epoch in (5,10):previews[(arm,epoch)]=predictions['real']
            print(f'Recounted {arm} epoch {epoch}: real IoU={scores["real"]["iou"]:.5f}, synthetic IoU={scores["synthetic"]["iou"]:.5f}',flush=True)
    out.mkdir()
    sheet=Image.new('RGB',(768,1480),'white')
    for i,row in enumerate(rows[:10]):
        masks=[targets['real'][i],base_predictions['real'][i],previews[('random',10)][i],
               previews[('pretrained',5)][i],previews[('pretrained',10)][i]]
        tiles=[Image.open(row['image_path']).convert('RGB')]+[Image.fromarray(m.astype('uint8')*255).convert('RGB') for m in masks]
        ImageDraw.Draw(sheet).text((2,i*148+2),f'{i}: input | target | parent | random10 | pretrained5 | pretrained10',fill='black')
        for j,tile in enumerate(tiles):sheet.paste(tile.resize((128,128)),(j*128,i*148+20))
    sheet.save(out/'preview.png')
    evidence={'archive_sha256':sha(archive),'bundle_sha256':BUNDLE_SHA,'log_audit':log,'saved_masks_recounted':total,
              'baseline':base_scores,'arms':arms,'runtime':complete,'optimizer_updates_locally':0,'promoted':False,
              'limitations':['Checkpoint-to-mask CPU reproduction pending','Remote parent invariance is an executed-code check/log claim; parent tensors were not returned',
                             'Optimizer intermediate states were not archived','External FFHQ pretraining overlap unresolved; development evidence only']}
    (out/'members.json').write_text(json.dumps(hashes,indent=2)+'\n')
    (out/'results.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps({'saved_masks_recounted':total,'total_logged_updates':log['total_logged_updates'],'promoted':False}),flush=True)


def reproduce():
    import numpy as np
    import torch
    from completion_inference import load_completion
    from face_occlusion_adapter import load_adapter
    torch.set_num_threads(4)
    out=ROOT/'outputs/face_occlusion_validation';audit_report=read_json(out/'results.json')
    destination=out/'reproduction.json';require(not destination.exists(),'Preserve completed reproduction')
    require(sha(ROOT/'outputs/face-occlusion-results.tar.gz')==audit_report['archive_sha256'],'Return archive changed')
    _,rows,data=verified_inputs();hashes=read_json(out/'members.json')
    extraction=ROOT/'outputs/downloaded_face_occlusion';returned=extraction/PREFIX
    for path,digest in hashes.items():require(sha(extraction/path)==digest,f'Extracted evidence changed: {path}')
    baseline=read_json(returned/'baseline.json');run=read_json(returned/'run.json')
    report={'optimizer_updates_locally':0,'checkpoint_checks':{},'reproductions':{},
            'parent_sha256':PARENT_SHA,'archive_sha256':audit_report['archive_sha256'],
            'torch':str(torch.__version__),'device':'cpu','promoted':False}
    sys.path.insert(0,str(ROOT/'outputs/face_extraction_dependencies'))
    from scripts.train_face_occlusion_vm import dependency_versions
    report['dependencies']=dependency_versions(ROOT/'outputs/face_extraction_dependencies')
    def infer(model,saved_folder,logged,label):
        model.requires_grad_(False).eval();scores={};differences={};real_predictions=real_truth=None
        with torch.inference_mode():
            for domain,dataset in data.items():
                predictions=[];truth=[];different=[]
                for i,(x,m) in enumerate(dataset):
                    p=(model.detect(x[None]).sigmoid()[0,0]>=.5).numpy();t=m[0].numpy().astype(bool)
                    predictions.append(p);truth.append(t)
                    saved=saved_folder/domain/f'{i:04}.png'
                    if saved.exists():different.append({'index':i,'different_pixels':int(np.count_nonzero(p!=binary(saved)))})
                    if (i+1)%100==0:print(label,domain,i+1,len(dataset),flush=True)
                scores[domain]=recount(predictions,truth)
                if different: differences[domain]={'cases':len(different),'all_exact':all(r['different_pixels']==0 for r in different),
                                                  'different_pixels':sum(r['different_pixels'] for r in different),'nonexact_cases':[r for r in different if r['different_pixels']]}
                if domain=='real':real_predictions,real_truth=predictions,truth
        subsets(scores,real_predictions,real_truth,rows)
        deltas={domain:{key:float(value-logged[domain][key]) for key,value in metrics.items()} for domain,metrics in scores.items()}
        return {'scores':scores,'mask_reproduction':differences,'metric_deltas_cpu_minus_logged':deltas}
    parent_path=ROOT/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    require(sha(parent_path)==PARENT_SHA,'Original parent changed')
    parent,_=load_completion(parent_path,'cpu')
    report['reproductions']['parent']=infer(parent,returned/'baseline_masks',baseline,'parent')
    del parent
    initial={}
    for arm in ('random','pretrained'):
        path=ROOT/f'outputs/face_occlusion_initial_v1/{arm}.pth'
        require(sha(path)==read_json(ROOT/'outputs/face_occlusion_initial_v1/manifest.json')['arms'][arm]['sha256'],'Initial checkpoint changed')
        initial[arm]=torch.load(path,map_location='cpu',weights_only=True)
        require(initial[arm]['optimizer_updates']==0 and initial[arm]['initialization']==arm,'Initial state metadata differs')
    phead={k:v for k,v in initial['pretrained']['model'].items() if k.startswith('network.segmentation_head.')}
    require(phead and all(torch.equal(v,initial['random']['model'][k]) for k,v in phead.items()),'Initial heads differ')
    report['initial_heads_equal']=True
    for arm in ('random','pretrained'):
        for logged in read_lines(returned/arm/'metrics.jsonl'):
            epoch=logged['epoch'];path=returned/arm/f'epoch_{epoch}.pth'
            model,payload=load_adapter(path,'cpu')
            require(payload['epoch']==epoch and payload['optimizer_updates']==21*epoch and payload['initialization']==arm and
                    payload['pilot_metadata']==run and payload['selection']==logged and payload['source_sha256']==initial[arm]['source_sha256'],
                    'Checkpoint metadata differs from audited run')
            before=initial[arm]['model'];state=payload['model']
            require(set(state)==set(before) and not any(k.startswith(('generator.','segmenter.')) for k in state),'Unexpected detector architecture')
            frozen=[k for k in state if k.startswith('reference_visible_head.') or k.endswith(('running_mean','running_var','num_batches_tracked'))]
            require(frozen and all(torch.equal(state[k],before[k]) for k in frozen),'Frozen state changed')
            changes={prefix:sum(not torch.equal(v,before[k]) for k,v in state.items() if k.startswith(prefix))
                     for prefix in ('network.encoder.','network.decoder.','network.segmentation_head.')}
            require(all(changes.values()),'No parameter changes in a declared trainable component')
            key=f'{arm}/{epoch}'
            report['checkpoint_checks'][key]={'sha256':sha(path),'frozen_tensors_verified':len(frozen),'changed_tensors':changes}
            report['reproductions'][key]=infer(model,returned/arm/f'epoch_{epoch}_masks',logged,key)
            del model,payload
            (out/'reproduction_partial.json').write_text(json.dumps(report,indent=2)+'\n')
    report['saved_masks_compared']=sum(d['cases'] for r in report['reproductions'].values() for d in r['mask_reproduction'].values())
    report['all_saved_masks_exact']=all(d['all_exact'] for r in report['reproductions'].values() for d in r['mask_reproduction'].values())
    require(report['saved_masks_compared']==3413,'Incomplete checkpoint-mask reproduction')
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'saved_masks_compared':report['saved_masks_compared'],'all_exact':report['all_saved_masks_exact'],'optimizer_updates_locally':0}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reproduce',action='store_true')
    args=parser.parse_args()
    reproduce() if args.reproduce else audit()
