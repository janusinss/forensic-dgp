"""VM-only matched reflection-coverage continuation; no automatic promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from face_occlusion_adapter import load_adapter,parameter_groups
from face_occlusion_focus import validate_source,validate_optimizer
from reflection_coverage import supported_segmentation_loss
from completion_inference import load_completion
from detector_training import load_manifest,ReviewedMasks,evaluate,segmentation_loss
from detector_replay import BenchmarkMasks,replay_consistency
from scripts.train_coverage_vm import require_vm_gpu,safe_path,mask_export,PARENT_SHA,PROTOCOL_SHA
from scripts.train_face_occlusion_vm import dependency_versions,selection,validate_schedule,REPLAY_SHA
from scripts.train_face_occlusion_focus_vm import MODEL_SHA,OPTIMIZER_SHA,OLD_INVENTORIES,restore_optimizer
from scripts.audit_reflection_coverage_data import audit_case

DATA_PATH='outputs/reflection_coverage_data_v1'
DATA_SHA='6696cee3a3acf48049f2f121a2a6328625c4351f558deabb5475da7fe948baf9'
PIXELS_SHA='ab47a88d79fe8d300ca92d572a2c1a05b92fa273b9577cb209df99e692009cc1'
AUDIT_SHA='21a77f47d2cdea74b4d04e0aef7433664422a5c8a923d4ac05e5f9468016fbb8'
INVENTORY_PATH='outputs/reflection_coverage_bundle_v1/inventory.json'
PRIOR_INVENTORIES={**OLD_INVENTORIES,
    'outputs/face_occlusion_focus_bundle_v1/inventory.json':'91b23aaa05c06c377a5c50c3c0bf53a5008e7ec7bd0d5ae657684d81b0b05471'}
CHECK_EPOCHS=(6,12)
SUPPLEMENTAL_WEIGHT=.25


def require(condition,message):
    if not condition:raise ValueError(message)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read_json(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def counters(epoch):
    require(type(epoch) is int and 1<=epoch<=12,'Epoch outside fixed12-epoch budget')
    return {'epoch':30+epoch,'optimizer_updates':630+21*epoch,'fresh_optimizer_updates':420+21*epoch,
            'additional_epoch':20+epoch,'experiment_epoch':epoch,'experiment_updates':21*epoch}


def supplemental_ids(arm,ids):
    require(arm in ('control','reflective') and len(ids)==2,'Unknown matched arm/supplemental pair')
    positive,clear=ids
    require(type(positive) is int and 0<=positive<280 and positive%10//2!=0
            and type(clear) is int and 0<=clear<280 and clear%10//2==0,'Wrong supplemental pair strata')
    return [positive if arm=='reflective' else positive//10*10+positive%2,clear]


@torch.inference_mode()
def supplemental_metrics(model,pixels,output=None):
    model.eval();tp=fp=fn=visible=0;covered=empty=clear=clear_errors=ignored=0;records=[]
    if output is not None:output.mkdir(parents=True)
    for i in sorted(pixels):
        case=pixels[i];x=case['input'].float().cuda()[None]/255
        prediction=(model.detect(x).sigmoid()[0,0]>=.5).cpu()
        target=case['mask'][0].bool();valid=case['valid'][0].bool()
        a=int((prediction&target&valid).sum());b=int((prediction&~target&valid).sum());c=int((~prediction&target&valid).sum())
        area=int((~target&valid).sum());has=bool(target.any());pred=bool((prediction&valid).any())
        tp+=a;fp+=b;fn+=c;visible+=area;covered+=has;empty+=has and not pred;clear+=not has;clear_errors+=not has and pred
        ignored+=int((prediction&~valid).sum())
        records.append({'case_id':i,'tp':a,'fp':b,'fn':c,'visible':area,'ignored_positive_pixels':int((prediction&~valid).sum())})
        if output is not None:Image.fromarray(prediction.numpy().astype('uint8')*255).save(output/f'{i:04}.png')
    return {'iou':tp/max(1,tp+fp+fn),'missed_fraction':fn/max(1,tp+fn),'visible_false_positive':fp/max(1,visible),
            'covered_cases':covered,'empty_mask_cases':empty,'negative_cases':clear,
            'negative_false_positive_cases':clear_errors,'ignored_positive_pixels':ignored,'records':records,
            'scope':'Training-only fixture diagnostics; excluded from selection'}


def main(args):
    require_vm_gpu() # Refuse local/CPU before paths, models, gradients or optimizers.
    root=Path.cwd().resolve();out=safe_path(root,args.output);archive=safe_path(root,args.archive)
    require(not out.exists() and (args.preflight or not archive.exists()),'Output/archive exists; inspect instead of restarting')
    require(torch.cuda.get_device_properties(0).total_memory>=4*1024**3,'At least4GiB total GPU memory required')
    inventory_path=root/INVENTORY_PATH;inventory=read_json(inventory_path)
    for name,digest in inventory.items():require(sha(safe_path(root,name))==digest,'New inventory input changed: '+name)
    for name,digest in PRIOR_INVENTORIES.items():
        require(sha(root/name)==digest,'Prior inventory changed')
        for member,expected in read_json(root/name).items():require(sha(safe_path(root,member))==expected,'Prior inventoried input changed: '+member)
    data_path=root/DATA_PATH
    require(sha(data_path/'manifest.json')==DATA_SHA and sha(data_path/'pixels.pth')==PIXELS_SHA
            and sha(data_path/'audit.json')==AUDIT_SHA,'Frozen supplemental evidence differs')
    data=read_json(data_path/'manifest.json');visual=read_json(data_path/'visual_review.json')
    require(visual['manifest_sha256']==DATA_SHA and visual['audit_sha256']==AUDIT_SHA
            and visual['accepted_for_bounded_pilot'] is True,'Native/augmentation visual review required')
    cache=torch.load(data_path/'pixels.pth',map_location='cpu',weights_only=True);pixels=cache['pixels']
    require(cache['format']=='dgp-reflection-coverage-pixels-v1' and set(pixels)==set(range(280)),'Supplemental cache membership differs')
    for row in data['cases']:
        i=row['case_id'];case=pixels[i];counted=audit_case(case,cache['references'][i//10],row['style'],row['degraded'])
        require(all(counted[k]==row[k] for k in ('hole_pixels','geometry_pixels','valid_pixels'))
                and all(hashlib.sha256(t.numpy().tobytes()).hexdigest()==row['pixel_sha256'][k] for k,t in case.items()),
                'Supplemental raw pixel/count registration differs')
    protocol_path=root/'outputs/coverage_protocol_v1/protocol.json'
    replay_path=root/'outputs/coverage_training_vm/replay_pixels.pth'
    require(sha(protocol_path)==PROTOCOL_SHA and sha(replay_path)==REPLAY_SHA,'Original protocol/cache changed')
    protocol=read_json(protocol_path);replay=torch.load(replay_path,map_location='cpu',weights_only=True)
    rows=load_manifest(root/'dataset/detector_training_extension_v2/manifest.json')
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    validate_schedule(protocol,training,replay);real=ReviewedMasks(training,256)
    schedule=data['core_schedule']['batches'];supplemental=data['supplemental_schedule']
    require(len(schedule)==12 and all(len(e)==21 and all(len(b)==8 for b in e) for e in schedule)
            and len(supplemental)==252,'Matched schedule/budget differs')
    for epoch in schedule:
        for batch in epoch:
            require(all(i<73 or i-73 in replay and (i-73)//10 not in data['core_schedule']['quarantined_sources'] for i in batch),
                    'Unsafe/unregistered core exposure')
    source_root=root/'outputs/face_occlusion_continuation_vm';source=source_root/'epoch_30.pth';opt_path=source_root/'final_optimizer.pth'
    require(sha(source)==MODEL_SHA and sha(opt_path)==OPTIMIZER_SHA,'Copied model/moments source differs')
    deps_path=root/'outputs/face_occlusion_dependencies';sys.path.insert(0,str(deps_path));deps=dependency_versions(deps_path)
    setup_path=deps_path/'setup.json';setup=read_json(setup_path)
    require(setup['packages']==deps and setup['torch']==str(torch.__version__),'Previous pinned runtime differs')
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model,payload=load_adapter(source,'cuda');validate_source(payload)
    initial={k:v.cpu().clone() for k,v in model.state_dict().items()}
    opt_payload=torch.load(opt_path,map_location='cpu',weights_only=True)
    optimizer_check=validate_optimizer(opt_payload,parameter_groups(model),MODEL_SHA)
    def core_tensors(batch):
        pairs=[real[i] if i<73 else (replay[i-73][0].float()/255,replay[i-73][1].float()) for i in batch]
        return (torch.stack([p[0] for p in pairs]).cuda(),torch.stack([p[1] for p in pairs]).cuda(),
                torch.tensor([i>=73 for i in batch],device='cuda'))
    def supplemental_tensors(ids):
        return tuple(torch.stack([pixels[i][key].float() for i in ids]).cuda()/(255 if key=='input' else 1)
                     for key in ('input','mask','valid'))
    out.mkdir(parents=True);started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    run={'preflight':args.preflight,'arms':['control','reflective'],'source_model_sha256':MODEL_SHA,
         'source_optimizer_sha256':OPTIMIZER_SHA,'source_model_updates':630,'source_optimizer_step':420,
         'optimizer_reset':False,'epochs_per_arm':12,'updates_per_arm':252,'check_epochs':list(CHECK_EPOCHS),
         'data_manifest_sha256':DATA_SHA,'data_audit_sha256':AUDIT_SHA,'supplemental_pixels_sha256':PIXELS_SHA,
         'inventory_sha256':sha(inventory_path),'script_sha256':sha(__file__),'module_sha256':sha(root/'reflection_coverage.py'),
         'encoder_lr':1e-5,'decoder_head_lr':1e-4,'weight_decay':1e-4,'clip':1.,'seed':42,
         'core_batch_size':8,'supplemental_batch_size':2,'background_weight':.25,'hard_fraction':.1,
         'core_consistency_weight':1.,'supplemental_weight':SUPPLEMENTAL_WEIGHT,'new_fixture_teacher_weight':0.,
         'original_parent_sha256':PARENT_SHA,'protocol_sha256':PROTOCOL_SHA,'replay_sha256':REPLAY_SHA,
         'quarantined_sources':data['core_schedule']['quarantined_sources'],'core_replacements':49,
         'supplemental_supervision':'Control two clear frames; reflective one reflection plus one matched clear frame. Valid-support reductions only.',
         'teacher_policy':'Original parent on core synthetic samples only; novel fixtures have direct synthetic truth.',
         'validation_policy':'Original real/synthetic gates at threshold0.5; all new fixture scores are training diagnostics.',
         'test_split':'No model score; prior development inspected test images. Identity/pretraining overlap unresolved.',
         'dependencies':deps,'setup_sha256':sha(setup_path),'torch':str(torch.__version__),
         'python':sys.version,'gpu':torch.cuda.get_device_name(),'opencv':str(__import__('cv2').__version__)}
    (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    if args.preflight:
        groups=parameter_groups(model);optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
        restored=restore_optimizer(optimizer,opt_payload,groups,MODEL_SHA)
        x,m,_=core_tensors(schedule[0][0]);terms={}
        with torch.inference_mode():
            terms['core']=float(segmentation_loss(model.detect(x),m,.25,.1))
            for arm in run['arms']:
                sx,sm,sv=supplemental_tensors(supplemental_ids(arm,supplemental[0]))
                terms[arm]=float(supported_segmentation_loss(model.detect(sx),sm,sv))
        require(all(np.isfinite(v) for v in terms.values()) and all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()),'Preflight loss/state failed')
        report={'losses':terms,'optimizer':optimizer_check,'restored_optimizer':restored,'optimizer_updates':0}
        (out/'preflight.json').write_text(json.dumps(report,indent=2)+'\n');print('CUDA forward/moments/support passed; zero updates',flush=True);return
    parent_path=root/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    require(sha(parent_path)==PARENT_SHA,'Original parent changed');parent,_=load_completion(parent_path,'cuda');parent.requires_grad_(False).eval()
    parent_state={k:v.cpu().clone() for k,v in parent.state_dict().items()}
    datasets={'real':ReviewedMasks(validation,256),'synthetic':BenchmarkMasks(root/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm',256),
              'training_real':real}
    for name,predicate in [('human_real',lambda r:Path(r['image']).name!='new_covered_40.png'),
                           ('mannequin',lambda r:Path(r['image']).name=='new_covered_40.png'),
                           ('glare',lambda r:r.get('glare_stratum')=='strong_lens_reflection')]:
        subset=[r for r in validation if predicate(r)];require(subset,'Missing reporting subset');datasets[name]=ReviewedMasks(subset,256)
    loaders={name:torch.utils.data.DataLoader(ds,batch_size=1) for name,ds in datasets.items()}
    baseline={name:evaluate(parent,loader,'cuda') for name,loader in loaders.items()}
    source_metrics={name:evaluate(model,loader,'cuda') for name,loader in loaders.items()}
    require(baseline==read_json(source_root/'baseline.json') and all(source_metrics[k]==payload['selection'][k] for k in source_metrics),'Baseline/source predictions differ from audited source30')
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n');(out/'source_metrics.json').write_text(json.dumps(source_metrics,indent=2)+'\n')
    shutil.copyfile(source,out/'initial.pth');shutil.copyfile(opt_path,out/'initial_optimizer.pth')
    for domain in ('real','synthetic'):mask_export(parent,datasets[domain],out/'baseline_masks'/domain)
    for domain in ('real','synthetic','training_real'):mask_export(model,datasets[domain],out/'initial_masks'/domain)
    initial_supp=supplemental_metrics(model,pixels,out/'initial_masks'/'training_reflection')
    (out/'source_reflection_metrics.json').write_text(json.dumps(initial_supp,indent=2)+'\n');completed={}
    for arm in run['arms']:
        if arm!='control':model,payload=load_adapter(source,'cuda')
        require(all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()),'Arm source weights differ')
        groups=parameter_groups(model);optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
        restored=restore_optimizer(optimizer,opt_payload,groups,MODEL_SHA)
        arm_out=out/arm;arm_out.mkdir();(arm_out/'restored_optimizer.json').write_text(json.dumps(restored,indent=2)+'\n')
        best=baseline['real']['iou'];selected=[];updates=0;torch.manual_seed(42)
        for epoch,batches in enumerate(schedule,1):
            model.train();sums={k:0. for k in ('loss','supervised','teacher','supplemental')}
            for step,batch in enumerate(batches,1):
                pair=supplemental_ids(arm,supplemental[updates]);x,m,tag=core_tensors(batch);sx,sm,sv=supplemental_tensors(pair)
                optimizer.zero_grad(set_to_none=True);logits=model.detect(x)
                with torch.no_grad():teacher_logits=parent.detect(x[tag])
                supervised=segmentation_loss(logits,m,.25,.1)
                teacher=replay_consistency(logits[tag],teacher_logits,torch.ones(int(tag.sum()),dtype=torch.bool,device='cuda'))
                extra=supported_segmentation_loss(model.detect(sx),sm,sv);loss=supervised+teacher+SUPPLEMENTAL_WEIGHT*extra
                require(torch.isfinite(loss),'Nonfinite loss');loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(model.network.parameters(),1.,error_if_nonfinite=True));optimizer.step();updates+=1
                values={'loss':float(loss.detach()),'supervised':float(supervised.detach()),'teacher':float(teacher.detach()),'supplemental':float(extra.detach())}
                for k,v in values.items():sums[k]+=v
                log={'experiment_epoch':epoch,'step':step,'experiment_updates':updates,'cumulative_model_updates':630+updates,
                     'optimizer_state_step':420+updates,'indices':batch,'supplemental_indices':pair,'supplemental_weight':SUPPLEMENTAL_WEIGHT,
                     'pre_clip_norm':norm,**values}
                with (arm_out/'steps.jsonl').open('a') as stream:stream.write(json.dumps(log)+'\n')
            if epoch in CHECK_EPOCHS:
                for k,v in model.state_dict().items():
                    if k.startswith('reference_visible_head.') or k.endswith(('running_mean','running_var','num_batches_tracked')):require(torch.equal(v.cpu(),initial[k]),'Frozen source state changed')
                require(all(s['step'].item()==420+updates for s in optimizer.state.values()),'Optimizer lifetime step differs')
                scores={name:evaluate(model,loader,'cuda') for name,loader in loaders.items()};real_ok,synth_ok,chosen=selection(scores,baseline,best)
                count=counters(epoch);require(count['experiment_updates']==updates,'Budget counter differs')
                fixture_scores=supplemental_metrics(model,pixels,arm_out/f'epoch_{30+epoch}_masks'/'training_reflection')
                (arm_out/f'epoch_{30+epoch}_reflection_metrics.json').write_text(json.dumps(fixture_scores,indent=2)+'\n')
                report={**count,'arm':arm,'mean_epoch_terms':{k:v/21 for k,v in sums.items()},'real_gate':real_ok,'synthetic_gate':synth_ok,'selected':chosen,**scores}
                with (arm_out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
                torch.save({**payload,'model':model.state_dict(),**count,'reflection_metadata':run,'reflection_arm':arm,'selection':report},arm_out/f'epoch_{30+epoch}.pth')
                if chosen:best=scores['real']['iou'];selected.append(30+epoch);shutil.copyfile(arm_out/f'epoch_{30+epoch}.pth',arm_out/'best_detector.pth')
                for domain in ('real','synthetic','training_real'):mask_export(model,datasets[domain],arm_out/f'epoch_{30+epoch}_masks'/domain)
                print(json.dumps(report),flush=True)
            else:print(f'{arm} epoch{epoch}/12; {updates}/252 updates',flush=True)
        require(updates==252,'Arm ended before bounded budget');final=arm_out/'epoch_42.pth';moment=arm_out/'final_optimizer.pth'
        torch.save({'state':optimizer.state_dict(),'model_sha256':sha(final),'experiment_updates':252,
                    'optimizer_state_step':672,'cumulative_model_updates':882},moment)
        completed[arm]={'experiment_updates':252,'cumulative_model_updates':882,'optimizer_state_step':672,'selected_epochs':selected,
                        'final_sha256':sha(final),'final_optimizer_sha256':sha(moment)}
        del optimizer,model
    require(all(torch.equal(v.cpu(),parent_state[k]) for k,v in parent.state_dict().items()),'Original parent/generator changed')
    result={'complete':True,'arms':completed,'total_experiment_updates':504,'parent_unchanged':True,
            'seconds':time.monotonic()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'promotion_requires_visual_review':True}
    (out/'complete.json').write_text(json.dumps(result,indent=2)+'\n')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(out,arcname=out.relative_to(root).as_posix())
        for name in inventory:tar.add(root/name,arcname=name)
        tar.add(inventory_path,arcname=INVENTORY_PATH)
    archive.with_name(archive.name+'.sha256').write_bytes((sha(archive)+'  '+archive.name+'\n').encode())
    print('Download',archive,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--output',default='outputs/reflection_coverage_vm')
    parser.add_argument('--archive',default='reflection-coverage-results.tar.gz')
    main(parser.parse_args())
