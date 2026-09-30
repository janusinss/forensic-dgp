"""Matched ordinary/projected gradients on VM only; fixed existing data/cache."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tarfile
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from coverage_projection import project_real_against_replay, step_alignment
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks, evaluate, detector_optimizer, segmentation_loss
from detector_replay import replay_consistency, retention_passes, BenchmarkMasks
from scripts.train_coverage_vm import require_vm_gpu, sha, safe_path, real_gate, mask_export, PROTOCOL_SHA, PARENT_SHA


def flat_parameters(params):
    return torch.cat([p.detach().flatten() for p in params])


def assign_gradient(params, flat):
    if flat.numel()!=sum(p.numel() for p in params) or not torch.isfinite(flat).all():
        raise ValueError('Invalid flat gradient')
    offset=0
    for p in params:
        p.grad=flat[offset:offset+p.numel()].reshape_as(p).clone()
        offset+=p.numel()


def main(args):
    require_vm_gpu()
    root=Path.cwd().resolve();out=safe_path(root,args.output)
    if out.exists():raise ValueError('Preserve existing output; no automatic resume')
    train=root/'outputs/coverage_training_vm'
    previous=json.loads((train/'run.json').read_text())
    inventory=root/'outputs/coverage_protocol_v1/vm_inventory.json'
    assert sha(inventory)==previous['inventory_sha256']
    for path,digest in json.loads(inventory.read_text()).items():assert sha(safe_path(root,path))==digest
    protocol_path=root/'outputs/coverage_protocol_v1/protocol.json'
    assert sha(protocol_path)==PROTOCOL_SHA
    protocol=json.loads(protocol_path.read_text())
    schedule=protocol['schedules']['extended']['batches']
    assert len(schedule)==10 and all(len(epoch)==21 for epoch in schedule)
    assert sha(train/'replay_pixels.pth')==previous['replay_sha256']
    cache=torch.load(train/'replay_pixels.pth',map_location='cpu',weights_only=True)
    rows=load_manifest(root/'dataset/detector_training_extension_v2/manifest.json')
    real=ReviewedMasks([r for r in rows if r['split']=='train'],256)
    assert len(real)==73
    used={i-73 for epoch in schedule for batch in epoch for i in batch if i>=73}
    assert set(cache)==used
    parent=root/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    assert sha(parent)==PARENT_SHA
    torch.set_num_threads(4);torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    model,state=load_completion(parent,'cuda')
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    teacher=copy.deepcopy(model.segmenter).eval().requires_grad_(False)
    val_rows=[r for r in rows if r['split']=='validation']
    validations={'real':ReviewedMasks(val_rows,256),'synthetic':BenchmarkMasks(root/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm',256)}
    for name,predicate in [('human_real',lambda r:Path(r['image']).name!='new_covered_40.png'),
                           ('mannequin',lambda r:Path(r['image']).name=='new_covered_40.png'),
                           ('glare',lambda r:r.get('glare_stratum')=='strong_lens_reflection')]:
        subset=[r for r in val_rows if predicate(r)]
        if subset:validations[name]=ReviewedMasks(subset,256)
    loaders={k:torch.utils.data.DataLoader(v,batch_size=8) for k,v in validations.items()}
    def batch_tensors(batch):
        items=[real[i] if i<73 else (cache[i-73][0].float()/255,cache[i-73][1].float()) for i in batch]
        x,m=(torch.stack([item[j] for item in items]).cuda() for j in (0,1))
        tag=torch.tensor([i>=73 for i in batch],device='cuda')
        return x,m,tag
    out.mkdir(parents=True)
    torch.save(initial,out/'initial.pth')
    metadata={'protocol_sha256':PROTOCOL_SHA,'parent_sha256':PARENT_SHA,
        'inventory_sha256':sha(inventory),'replay_sha256':previous['replay_sha256'],
        'initial_sha256':sha(out/'initial.pth'),'script_sha256':sha(__file__),
        'helper_sha256':sha(root/'coverage_projection.py'),'torch':str(torch.__version__),
        'gpu':torch.cuda.get_device_name(),'lr':1e-5,'weight_decay':1e-4,'clip':1.,
        'background_weight':.25,'consistency_weight':1.,'preflight':args.preflight,
        'rule':'Project only conflicting real component against replay supervised plus teacher; then sum'}
    (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.preflight:
        x,m,tag=batch_tensors(schedule[0][0])
        with torch.no_grad():loss=segmentation_loss(model.detect(x),m,.25,.1)
        assert torch.isfinite(loss)
        (out/'preflight.json').write_text(json.dumps({'optimizer_updates':0,'finite_loss':float(loss)})+'\n')
        print('CUDA forward preflight passed; zero optimizer updates',flush=True);return
    baseline={k:evaluate(model,v,'cuda') for k,v in loaders.items()}
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    for arm in ('ordinary','projected'):
        model.load_state_dict(initial);torch.manual_seed(42)
        assert all(torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items())
        optimizer=detector_optimizer(model,1e-5);params=list(model.segmenter.parameters())
        arm_out=out/arm;arm_out.mkdir();best=baseline['real']['iou']
        for epoch,batches in enumerate(schedule,1):
            model.segmenter.train();loss_sum=0
            for step,batch in enumerate(batches):
                x,m,tag=batch_tensors(batch);logits=model.detect(x)
                real_loss=segmentation_loss(logits[~tag],m[~tag],.25,.1)*(~tag).sum()/len(tag)
                replay_loss=segmentation_loss(logits[tag],m[tag],.25,.1)*tag.sum()/len(tag)
                with torch.no_grad():reference=teacher(x[tag])
                replay_loss=replay_loss+replay_consistency(logits[tag],reference,torch.ones(int(tag.sum()),dtype=torch.bool,device='cuda'))
                full=segmentation_loss(logits,m,.25,.1)+replay_consistency(logits[tag],reference,torch.ones(int(tag.sum()),dtype=torch.bool,device='cuda'))
                def grad(loss):return torch.cat([v.detach().flatten() for v in torch.autograd.grad(loss,params,retain_graph=True)])
                real_grad=grad(real_loss);replay_grad=grad(replay_loss);ordinary=grad(full)
                error=float((real_grad.double()+replay_grad.double()-ordinary.double()).norm()/ordinary.double().norm().clamp_min(1e-12))
                if not torch.isfinite(full) or error>1e-4:raise ValueError('Loss/gradient decomposition failure')
                projected,projection=project_real_against_replay(real_grad,replay_grad)
                chosen=ordinary if arm=='ordinary' else projected
                optimizer.zero_grad(set_to_none=True);assign_gradient(params,chosen)
                pre_clip=float(torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True))
                before=flat_parameters(params).clone();optimizer.step()
                audit=step_alignment(before,flat_parameters(params),replay_grad)
                audit.update(epoch=epoch,step=step,indices=batch,projection=projection,
                    projection_applied=arm=='projected' and projection['applied'],gradient_relative_error=error,pre_clip_norm=pre_clip)
                with (arm_out/'steps.jsonl').open('a') as f:f.write(json.dumps(audit)+'\n')
                loss_sum+=float(full.detach())
            assert all(torch.equal(v.cpu(),initial['generator.'+k]) for k,v in model.generator.state_dict().items())
            metrics={k:evaluate(model,v,'cuda') for k,v in loaders.items()}
            r_ok=real_gate(metrics['real'],baseline['real'],best);s_ok=retention_passes(metrics['synthetic'],baseline['synthetic'])
            selected=r_ok and s_ok
            report={'epoch':epoch,'updates':epoch*21,'loss':loss_sum/21,'selected':selected,'real_gate':r_ok,'synthetic_gate':s_ok,**metrics}
            with (arm_out/'metrics.jsonl').open('a') as f:f.write(json.dumps(report)+'\n')
            export={**state,'model':model.state_dict(),'detector_only':True,'detector_finetune_epoch':epoch,'projection_arm':arm,'detector_training':metadata}
            torch.save(export,arm_out/f'epoch_{epoch}.pth')
            if selected:best=metrics['real']['iou'];torch.save(export,arm_out/'best_detector.pth')
            for domain in ('real','synthetic'):mask_export(model,validations[domain],arm_out/f'epoch_{epoch}_masks'/domain)
            print(arm,json.dumps(report),flush=True)
    (out/'complete.json').write_text(json.dumps({'complete':True,'updates_per_arm':210,'promotion_requires_visual_review':True})+'\n')
    archive=root/'projection-results.tar.gz'
    if archive.exists():raise ValueError('Result archive already exists; completed outputs preserved')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(out,arcname=out.relative_to(root).as_posix())
        tar.add(__file__,arcname='scripts/train_projection_vm.py')
        tar.add(root/'coverage_projection.py',arcname='coverage_projection.py')
    print('Download',archive,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='outputs/projection_training_vm');p.add_argument('--preflight',action='store_true')
    main(p.parse_args())
