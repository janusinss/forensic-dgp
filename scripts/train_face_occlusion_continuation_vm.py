"""Fixed 420-update weight continuation with fresh AdamW; Linux CUDA VM only."""
import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from face_occlusion_adapter import load_adapter,parameter_groups
from completion_inference import load_completion
from detector_training import load_manifest,ReviewedMasks,evaluate,segmentation_loss
from detector_replay import BenchmarkMasks,replay_consistency
from scripts.train_coverage_vm import require_vm_gpu,safe_path,mask_export,PARENT_SHA,PROTOCOL_SHA
from scripts.train_face_occlusion_vm import dependency_versions,validate_schedule,selection,OLD_INVENTORY_SHA,REPLAY_SHA
from scripts.package_face_occlusion_vm import sha

START_SHA='cdd1752bce8e4a087ce2aac5af73ba81b6316ac9118e47f12acd7a24fcb62669'
FACE_INVENTORY_SHA='cb8485ef260bf83f8f4f3e6ce2f6fd45dc1a8726980130ddf4dfea956e83a4c1'
CHECK_ADDITIONAL_EPOCHS=(1,5,10,20)


def require(condition,message):
    if not condition:raise ValueError(message)


def continuation_schedule(protocol):
    base=protocol['schedules']['extended']['batches']
    require(len(base)==10 and all(len(epoch)==21 for epoch in base),'Original schedule differs')
    return copy.deepcopy(base)+copy.deepcopy(base)


def validate_start(payload):
    require(payload.get('initialization')=='pretrained' and payload.get('epoch')==10 and
            payload.get('optimizer_updates')==210,'Require the verified pretrained epoch10 source')
    require(not any(k in payload for k in ('optimizer','optimizer_state_dict','optimizer_state')),
            'Source optimizer state unexpectedly present; fixed pilot requires a fresh optimizer')


def checkpoint_counters(additional_epoch):
    require(type(additional_epoch) is int and 1<=additional_epoch<=20,'Additional epoch outside fixed budget')
    return {'epoch':10+additional_epoch,'optimizer_updates':210+additional_epoch*21,
            'additional_epoch':additional_epoch,'fresh_optimizer_updates':additional_epoch*21}


def main(args):
    require_vm_gpu()
    root=Path.cwd().resolve();out=safe_path(root,args.output)
    archive=root/'face-occlusion-continuation-results.tar.gz'
    require(not out.exists() and (args.preflight or not archive.exists()),'Preserve existing output/archive')
    for name,digest in [('outputs/coverage_protocol_v1/vm_inventory.json',OLD_INVENTORY_SHA),
                        ('outputs/face_occlusion_bundle_v1/inventory.json',FACE_INVENTORY_SHA)]:
        path=root/name;require(sha(path)==digest,'Existing inventory changed')
        for relative,expected in json.loads(path.read_text()).items():
            require(sha(safe_path(root,relative))==expected,f'Existing input changed: {relative}')
    inventory_path=root/'outputs/face_occlusion_continuation_bundle_v1/inventory.json'
    inventory=json.loads(inventory_path.read_text())
    required={'scripts/train_face_occlusion_continuation_vm.py','FACE_OCCLUSION_CONTINUATION.md','FACE_OCCLUSION_CONTINUATION_VM.md'}
    require(required.issubset(inventory),'Continuation package incomplete')
    for relative,expected in inventory.items():require(sha(safe_path(root,relative))==expected,f'Continuation input changed: {relative}')
    protocol_path=root/'outputs/coverage_protocol_v1/protocol.json'
    require(sha(protocol_path)==PROTOCOL_SHA,'Original protocol changed')
    protocol=json.loads(protocol_path.read_text())
    cache_path=root/'outputs/coverage_training_vm/replay_pixels.pth'
    require(sha(cache_path)==REPLAY_SHA,'Cached replay changed')
    cache=torch.load(cache_path,map_location='cpu',weights_only=True)
    rows=load_manifest(root/'dataset/detector_training_extension_v2/manifest.json')
    training=[r for r in rows if r['split']=='train'];validation=[r for r in rows if r['split']=='validation']
    validate_schedule(protocol,training,cache);schedule=continuation_schedule(protocol)
    source=root/'outputs/face_occlusion_pilot_vm/pretrained/epoch_10.pth'
    require(sha(source)==START_SHA,'Pretrained epoch10 source changed/missing')
    dependencies_path=root/'outputs/face_occlusion_dependencies'
    sys.path.insert(0,str(dependencies_path));dependencies=dependency_versions(dependencies_path)
    setup_path=dependencies_path/'setup.json';setup=json.loads(setup_path.read_text())
    require(setup['packages']==dependencies and setup['torch']==str(torch.__version__),'Active runtime differs from setup')
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model,payload=load_adapter(source,'cuda');validate_start(payload)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    real=ReviewedMasks(training,256)
    def tensors(batch):
        pairs=[real[i] if i<73 else (cache[i-73][0].float()/255,cache[i-73][1].float()) for i in batch]
        x,m=(torch.stack([p[j] for p in pairs]).cuda() for j in (0,1))
        return x,m,torch.tensor([i>=73 for i in batch],device='cuda')
    out.mkdir(parents=True);started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    run={'preflight':args.preflight,'source_checkpoint_sha256':START_SHA,'source_epoch':10,'source_optimizer_updates':210,
         'optimizer_reset':True,'optimizer_state_restored':False,'additional_epochs':20,'additional_updates':420,
         'check_additional_epochs':list(CHECK_ADDITIONAL_EPOCHS),'protocol_sha256':PROTOCOL_SHA,
         'replay_sha256':REPLAY_SHA,'parent_sha256':PARENT_SHA,'old_inventory_sha256':OLD_INVENTORY_SHA,
         'face_inventory_sha256':FACE_INVENTORY_SHA,'continuation_inventory_sha256':sha(inventory_path),
         'script_sha256':sha(__file__),'specification_sha256':sha(root/'FACE_OCCLUSION_CONTINUATION.md'),
         'adapter_sha256':sha(root/'face_occlusion_adapter.py'),'dependencies':dependencies,'setup_sha256':sha(setup_path),
         'torch':str(torch.__version__),'python':sys.version,'gpu':torch.cuda.get_device_name(),
         'encoder_lr':1e-5,'decoder_head_lr':1e-4,'weight_decay':1e-4,'clip':1.,'seed':42,
         'background_weight':.25,'consistency_weight':1.,'validation_batch_size':1,
         'batchnorm_policy':'fixed_running_statistics_trainable_affine',
         'test_split':'Not evaluated; previous development work inspected it',
         'overlap_limitation':'External FaceExtraction FFHQ pretraining overlap unresolved'}
    (out/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    if args.preflight:
        x,m,_=tensors(schedule[0][0])
        with torch.no_grad():loss=segmentation_loss(model.detect(x),m,.25,.1)
        require(torch.isfinite(loss),'Nonfinite CUDA preflight')
        require(all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()),'Forward changed source state')
        (out/'preflight.json').write_text(json.dumps({'finite_loss':float(loss),'additional_optimizer_updates':0,'source_verified':True},indent=2)+'\n')
        print('CUDA source forward passed; zero additional optimizer updates',flush=True)
        return
    parent_path=root/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    require(sha(parent_path)==PARENT_SHA,'Original common parent changed')
    parent,_=load_completion(parent_path,'cuda');parent.requires_grad_(False).eval()
    parent_state={k:v.detach().cpu().clone() for k,v in parent.state_dict().items()}
    datasets={'real':ReviewedMasks(validation,256),
              'synthetic':BenchmarkMasks(root/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm',256),
              'training_real':real}
    for name,predicate in [('human_real',lambda r:Path(r['image']).name!='new_covered_40.png'),
                           ('mannequin',lambda r:Path(r['image']).name=='new_covered_40.png'),
                           ('glare',lambda r:r.get('glare_stratum')=='strong_lens_reflection')]:
        subset=[r for r in validation if predicate(r)]
        if subset:datasets[name]=ReviewedMasks(subset,256)
    loaders={name:torch.utils.data.DataLoader(data,batch_size=1) for name,data in datasets.items()}
    baseline={name:evaluate(parent,loader,'cuda') for name,loader in loaders.items()}
    source_scores={name:evaluate(model,loader,'cuda') for name,loader in loaders.items()}
    require(all(scores==payload['selection'][name] for name,scores in source_scores.items()),'Source predictions do not reproduce original checkpoint metrics')
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    (out/'source_metrics.json').write_text(json.dumps(source_scores,indent=2)+'\n')
    shutil.copyfile(source,out/'initial.pth')
    require(sha(out/'initial.pth')==START_SHA,'Source copy differs')
    for domain in ('real','synthetic'):mask_export(parent,datasets[domain],out/'baseline_masks'/domain)
    for domain in ('real','synthetic','training_real'):mask_export(model,datasets[domain],out/'initial_masks'/domain)
    optimizer=torch.optim.AdamW(parameter_groups(model),weight_decay=1e-4)
    require(not optimizer.state,'Fresh optimizer unexpectedly contains prior moments')
    updates=0;best=baseline['real']['iou'];selected_epochs=[]
    for additional_epoch,batches in enumerate(schedule,1):
        model.train();loss_sum=0
        for step,batch in enumerate(batches,1):
            x,m,tag=tensors(batch);optimizer.zero_grad(set_to_none=True);logits=model.detect(x)
            with torch.no_grad():reference=parent.detect(x[tag])
            loss=segmentation_loss(logits,m,.25,.1)+replay_consistency(logits[tag],reference,torch.ones(int(tag.sum()),dtype=torch.bool,device='cuda'))
            require(torch.isfinite(loss),'Nonfinite continuation loss')
            loss.backward();norm=float(torch.nn.utils.clip_grad_norm_(model.network.parameters(),1.,error_if_nonfinite=True))
            optimizer.step();updates+=1;loss_sum+=float(loss.detach())
            row={'additional_epoch':additional_epoch,'global_epoch':10+additional_epoch,'step':step,
                 'fresh_optimizer_updates':updates,'cumulative_optimizer_updates':210+updates,
                 'indices':batch,'loss':float(loss.detach()),'pre_clip_norm':norm}
            with (out/'steps.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        if additional_epoch in CHECK_ADDITIONAL_EPOCHS:
            for key,value in model.state_dict().items():
                if key.startswith('reference_visible_head.') or key.endswith(('running_mean','running_var','num_batches_tracked')):
                    require(torch.equal(value.cpu(),initial[key]),'Frozen source head/statistics changed')
            metrics={name:evaluate(model,loader,'cuda') for name,loader in loaders.items()}
            real_ok,synthetic_ok,selected=selection(metrics,baseline,best)
            counters=checkpoint_counters(additional_epoch)
            require(counters['fresh_optimizer_updates']==updates,'Runtime update count differs')
            report={**counters,'mean_epoch_loss':loss_sum/21,'real_gate':real_ok,'synthetic_gate':synthetic_ok,'selected':selected,**metrics}
            with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
            exported={**payload,'model':model.state_dict(),**counters,'continuation_metadata':run,'selection':report}
            torch.save(exported,out/f'epoch_{10+additional_epoch}.pth')
            if selected:
                best=metrics['real']['iou'];selected_epochs.append(10+additional_epoch)
                torch.save(exported,out/'best_detector.pth')
            for domain in ('real','synthetic','training_real'):
                mask_export(model,datasets[domain],out/f'epoch_{10+additional_epoch}_masks'/domain)
            print(json.dumps(report),flush=True)
        else:print(f'Global epoch {10+additional_epoch}/30; fresh updates={updates}; loss={loss_sum/21:.6f}',flush=True)
    require(updates==420,'Continuation ended before fixed budget')
    require(all(torch.equal(v.cpu(),parent_state[k]) for k,v in parent.state_dict().items()),'Original parent/generator changed')
    optimizer_path=out/'final_optimizer.pth'
    torch.save({'state':optimizer.state_dict(),'model_sha256':sha(out/'epoch_30.pth'),
                'fresh_optimizer_updates':updates,'cumulative_model_updates':630},optimizer_path)
    complete={'complete':True,'fresh_optimizer_updates':updates,'cumulative_model_updates':630,
              'selected_epochs':selected_epochs,'source_sha256':START_SHA,'final_sha256':sha(out/'epoch_30.pth'),
              'final_optimizer_sha256':sha(optimizer_path),'parent_unchanged':True,'seconds':time.monotonic()-started,
              'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
              'peak_cuda_reserved_bytes':torch.cuda.max_memory_reserved(),'promotion_requires_visual_review':True}
    (out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(out,arcname=out.relative_to(root).as_posix())
        for path in inventory:tar.add(root/path,arcname=path)
        tar.add(inventory_path,arcname=inventory_path.relative_to(root).as_posix())
    archive.with_name(archive.name+'.sha256').write_bytes((sha(archive)+'  '+archive.name+'\n').encode())
    print('Download',archive,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/face_occlusion_continuation_vm')
    parser.add_argument('--preflight',action='store_true')
    main(parser.parse_args())
