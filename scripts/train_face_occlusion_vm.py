"""Matched direct-occlusion initialization pilot; optimization on Linux CUDA only."""
import argparse
from collections import Counter
import importlib
import importlib.metadata as metadata
import json
from pathlib import Path
import sys
import tarfile
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_occlusion_adapter import load_adapter, parameter_groups, SOURCE_SHA
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks, evaluate, segmentation_loss
from detector_replay import BenchmarkMasks, replay_consistency, retention_passes
from scripts.train_coverage_vm import require_vm_gpu, sha, safe_path, real_gate, mask_export, PROTOCOL_SHA, PARENT_SHA

OLD_INVENTORY_SHA = 'c831635933c9b615b4849ad23811a3b82ebde1883aa30a5c2249b8879fa95623'
REPLAY_SHA = 'ad6fd64aed4f773c968cac7748be0fd99a7c8679d1019e32fc64a03b13662dc7'
INITIAL_MANIFEST_SHA = '7fb3fe1f5e4c2fc8a5266e87dfcd9e84d1bf07728145ebaebe5d487a52047650'
CHECK_EPOCHS = (1, 5, 10)
DEPENDENCIES = {'segmentation-models-pytorch': ('segmentation_models_pytorch', '0.5.0'),
                'timm': ('timm', '1.0.15'), 'huggingface-hub': ('huggingface_hub', '0.29.3'),
                'safetensors': ('safetensors', '0.5.3'), 'PyYAML': ('yaml', '6.0.2')}


def dependency_versions(target):
    target = Path(target).resolve()
    report = {}
    for distribution, (name, expected) in DEPENDENCIES.items():
        actual = metadata.distribution(distribution).version
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        if actual != expected or not path.is_relative_to(target):
            raise ValueError(f'Pinned dependency differs: {distribution}')
        report[distribution] = {'version': actual, 'module_path': str(path)}
    return report


def validate_schedule(protocol, rows, cache):
    schedule = protocol['schedules']['extended']['batches']
    if len(rows) != 73 or len(schedule) != 10 or any(len(e) != 21 for e in schedule):
        raise ValueError('Unexpected fixed data/budget')
    used = {i-73 for e in schedule for b in e for i in b if i>=73}
    if set(cache) != used or len(used) != 638:
        raise ValueError('Replay membership differs')
    for i, (_,mask) in cache.items():
        if bool(mask.any()) != (i%5 != 0):
            raise ValueError('Replay type/target differs')
    for epoch in schedule:
        for batch in epoch:
            if len(batch) != 8 or any(type(i) is not int or i<0 for i in batch):
                raise ValueError('Invalid batch indices')
            classes = []
            for i in batch:
                if i<73:
                    classes.append('real/'+rows[i]['kind'])
                elif i-73 in cache:
                    classes.append('synthetic/'+('covered' if (i-73)%5 else 'uncovered'))
                else:
                    raise ValueError('Unknown replay index')
            if Counter(classes) != {'real/covered':2, 'real/uncovered':2,
                                    'synthetic/covered':2, 'synthetic/uncovered':2}:
                raise ValueError('Batch class balance differs')
    return schedule


def selection(scores, common_baseline, best_real):
    real = real_gate(scores['real'],common_baseline['real'],best_real)
    synthetic = retention_passes(scores['synthetic'],common_baseline['synthetic'])
    return real, synthetic, real and synthetic


def main(args):
    require_vm_gpu()
    root = Path.cwd().resolve()
    out = safe_path(root,args.output)
    archive = root/'face-occlusion-results.tar.gz'
    if out.exists() or (not args.preflight and archive.exists()):
        raise ValueError('Existing output/archive must be preserved; no automatic restart')
    old_inventory = root/'outputs/coverage_protocol_v1/vm_inventory.json'
    if sha(old_inventory) != OLD_INVENTORY_SHA:
        raise ValueError('Coverage inventory changed')
    for path,digest in json.loads(old_inventory.read_text()).items():
        if sha(safe_path(root,path)) != digest:
            raise ValueError(f'Coverage input differs: {path}')
    new_inventory = root/'outputs/face_occlusion_bundle_v1/inventory.json'
    package_inventory = json.loads(new_inventory.read_text())
    for path,digest in package_inventory.items():
        if sha(safe_path(root,path)) != digest:
            raise ValueError(f'Pilot code/initialization differs: {path}')
    required = {'face_occlusion_adapter.py', 'scripts/train_face_occlusion_vm.py',
                'FACE_OCCLUSION_PILOT.md', 'requirements_face_occlusion_vm.txt',
                'outputs/face_occlusion_initial_v1/manifest.json',
                'outputs/face_occlusion_initial_v1/pretrained.pth',
                'outputs/face_occlusion_initial_v1/random.pth'}
    if not required.issubset(package_inventory):
        raise ValueError('Bundle does not cover consumed files')
    initial_manifest = root/'outputs/face_occlusion_initial_v1/manifest.json'
    if sha(initial_manifest) != INITIAL_MANIFEST_SHA:
        raise ValueError('Initial registry differs')
    registry = json.loads(initial_manifest.read_text())
    if (registry['optimizer_updates'] != 0 or registry['source_sha256'] != SOURCE_SHA or
        not registry['heads_equal'] or registry['adapter_sha256'] != sha(root/'face_occlusion_adapter.py')):
        raise ValueError('Initial preparation provenance differs')
    protocol_path = root/'outputs/coverage_protocol_v1/protocol.json'
    if sha(protocol_path) != PROTOCOL_SHA:
        raise ValueError('Fixed protocol changed')
    protocol = json.loads(protocol_path.read_text())
    cache_path = root/'outputs/coverage_training_vm/replay_pixels.pth'
    if sha(cache_path) != REPLAY_SHA:
        raise ValueError('Cached replay changed')
    cache = torch.load(cache_path,map_location='cpu',weights_only=True)
    rows = load_manifest(root/'dataset/detector_training_extension_v2/manifest.json')
    train_rows = [r for r in rows if r['split']=='train']
    schedule = validate_schedule(protocol,train_rows,cache)
    real = ReviewedMasks(train_rows,256)
    dependency_target = root/'outputs/face_occlusion_dependencies'
    sys.path.insert(0,str(dependency_target))
    dependencies = dependency_versions(dependency_target)
    setup_path = dependency_target/'setup.json'
    setup = json.loads(setup_path.read_text())
    if setup['packages'] != dependencies or setup['torch'] != str(torch.__version__):
        raise ValueError('Setup environment differs from active runtime')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    def batch_tensors(batch):
        items = [real[i] if i<73 else (cache[i-73][0].float()/255,cache[i-73][1].float()) for i in batch]
        x,m = (torch.stack([item[j] for item in items]).cuda() for j in (0,1))
        return x,m,torch.tensor([i>=73 for i in batch],device='cuda')

    out.mkdir(parents=True)
    started=time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    free_vram,total_vram=torch.cuda.mem_get_info()
    metadata = dict(preflight=args.preflight, protocol_sha256=PROTOCOL_SHA,
                    replay_sha256=REPLAY_SHA, parent_sha256=PARENT_SHA,
                    old_inventory_sha256=OLD_INVENTORY_SHA, bundle_inventory_sha256=sha(new_inventory),
                    initial_manifest_sha256=INITIAL_MANIFEST_SHA, source_sha256=SOURCE_SHA,
                    script_sha256=sha(__file__), adapter_sha256=sha(root/'face_occlusion_adapter.py'),
                    specification_sha256=sha(root/'FACE_OCCLUSION_PILOT.md'),
                    torch=str(torch.__version__), gpu=torch.cuda.get_device_name(),
                    dependencies=dependencies, setup_sha256=sha(setup_path),
                    python=sys.version, python_executable=sys.executable,
                    free_vram_bytes=free_vram, total_vram_bytes=total_vram,
                    encoder_lr=1e-5, decoder_head_lr=1e-4, weight_decay=1e-4, clip=1.,
                    background_weight=.25, consistency_weight=1., seed=42,
                    updates_per_arm=210, check_epochs=list(CHECK_EPOCHS), validation_batch_size=1,
                    batchnorm_policy='fixed_running_statistics_trainable_affine',
                    overlap_limitation='External FaceExtraction FFHQ pretraining overlap unresolved',
                    test_split='Not evaluated; prior development work inspected it previously')
    (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.preflight:
        checks = {}
        for arm in ('random','pretrained'):
            entry = registry['arms'][arm];path=safe_path(root,entry['path'])
            if sha(path) != entry['sha256']:
                raise ValueError('Initial checkpoint differs')
            model,payload = load_adapter(path,'cuda',allow_initial=True)
            if payload['initialization'] != arm or payload['optimizer_updates'] != 0:
                raise ValueError('Unexpected initial checkpoint')
            x,m,_ = batch_tensors(schedule[0][0])
            with torch.no_grad():
                logits=model.detect(x);loss=segmentation_loss(logits,m,.25,.1)
            if not torch.isfinite(loss):raise ValueError('Nonfinite CUDA preflight')
            checks[arm]=dict(finite_loss=float(loss),shape=list(logits.shape),optimizer_updates=0)
            del model
        (out/'preflight.json').write_text(json.dumps(checks,indent=2)+'\n')
        print('Both CUDA forward preflights passed; zero optimizer updates',flush=True)
        return

    parent_path=root/'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    if sha(parent_path) != PARENT_SHA:raise ValueError('Common parent changed')
    parent,_=load_completion(parent_path,'cuda');parent.requires_grad_(False).eval()
    parent_state={k:v.detach().cpu().clone() for k,v in parent.state_dict().items()}
    val_rows=[r for r in rows if r['split']=='validation']
    datasets={'real':ReviewedMasks(val_rows,256),
              'synthetic':BenchmarkMasks(root/'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm',256),
              'training_real':real}
    for name,predicate in [('human_real',lambda r:Path(r['image']).name!='new_covered_40.png'),
                           ('mannequin',lambda r:Path(r['image']).name=='new_covered_40.png'),
                           ('glare',lambda r:r.get('glare_stratum')=='strong_lens_reflection')]:
        subset=[r for r in val_rows if predicate(r)]
        if subset:datasets[name]=ReviewedMasks(subset,256)
    loaders={k:torch.utils.data.DataLoader(data,batch_size=1) for k,data in datasets.items()}
    baseline={k:evaluate(parent,loader,'cuda') for k,loader in loaders.items()}
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    for domain in ('real','synthetic'):
        mask_export(parent,datasets[domain],out/'baseline_masks'/domain)
    arm_summaries={}
    for arm in ('random','pretrained'):
        entry=registry['arms'][arm];path=safe_path(root,entry['path'])
        if sha(path) != entry['sha256']:raise ValueError('Initial checkpoint differs')
        model,payload=load_adapter(path,'cuda',allow_initial=True)
        if payload['initialization'] != arm or payload['optimizer_updates'] != 0:
            raise ValueError('Unexpected initialization')
        initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.manual_seed(42)
        optimizer=torch.optim.AdamW(parameter_groups(model),weight_decay=1e-4)
        arm_out=out/arm;arm_out.mkdir();best=baseline['real']['iou'];updates=0;selected_epochs=[]
        for epoch,batches in enumerate(schedule,1):
            model.train();loss_sum=0
            for step,batch in enumerate(batches,1):
                x,m,tag=batch_tensors(batch)
                optimizer.zero_grad(set_to_none=True)
                logits=model.detect(x)
                with torch.no_grad():reference=parent.detect(x[tag])
                loss=segmentation_loss(logits,m,.25,.1)+replay_consistency(logits[tag],reference,torch.ones(int(tag.sum()),dtype=torch.bool,device='cuda'))
                if not torch.isfinite(loss):raise ValueError('Nonfinite pilot loss')
                loss.backward()
                norm=float(torch.nn.utils.clip_grad_norm_(model.network.parameters(),1.,error_if_nonfinite=True))
                optimizer.step();updates+=1;loss_sum+=float(loss.detach())
                row=dict(epoch=epoch,step=step,updates=updates,indices=batch,loss=float(loss.detach()),pre_clip_norm=norm)
                with (arm_out/'steps.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if epoch in CHECK_EPOCHS:
                for key,value in model.state_dict().items():
                    if key.startswith('reference_visible_head.') or key.endswith(('running_mean','running_var','num_batches_tracked')):
                        if not torch.equal(value.cpu(),initial[key]):raise ValueError('Frozen head/statistics changed')
                metrics={k:evaluate(model,loader,'cuda') for k,loader in loaders.items()}
                r_ok,s_ok,selected=selection(metrics,baseline,best)
                report=dict(epoch=epoch,updates=updates,mean_epoch_loss=loss_sum/21,
                            real_gate=r_ok,synthetic_gate=s_ok,selected=selected,**metrics)
                with (arm_out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(report)+'\n')
                export={**payload,'model':model.state_dict(),'optimizer_updates':updates,
                        'epoch':epoch,'pilot_metadata':metadata,'selection':report}
                torch.save(export,arm_out/f'epoch_{epoch}.pth')
                if selected:
                    best=metrics['real']['iou'];selected_epochs.append(epoch)
                    torch.save(export,arm_out/'best_detector.pth')
                for domain in ('real','synthetic','training_real'):
                    mask_export(model,datasets[domain],arm_out/f'epoch_{epoch}_masks'/domain)
                print(arm,json.dumps(report),flush=True)
            else:print(f'{arm} epoch{epoch}/10 updates={updates} loss={loss_sum/21:.6f}',flush=True)
        arm_summaries[arm]=dict(updates=updates,selected_epochs=selected_epochs,final_sha256=sha(arm_out/'epoch_10.pth'))
        del optimizer,model
    if not all(torch.equal(v.cpu(),parent_state[k]) for k,v in parent.state_dict().items()):
        raise ValueError('Original parent/generator changed')
    complete=dict(complete=True,arms=arm_summaries,parent_unchanged=True,
                  seconds=time.monotonic()-started,
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),
                  peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),
                  promotion_requires_visual_review=True)
    (out/'complete.json').write_text(json.dumps(complete,indent=2)+'\n')
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(out,arcname=out.relative_to(root).as_posix())
        for path in package_inventory:
            if not path.endswith('.pth'):
                tar.add(root/path,arcname=path)
        tar.add(new_inventory,arcname=new_inventory.relative_to(root).as_posix())
        tar.add(setup_path,arcname=setup_path.relative_to(root).as_posix())
    archive.with_name(archive.name+'.sha256').write_bytes((sha(archive)+'  '+archive.name+'\n').encode())
    print('Download',archive,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='outputs/face_occlusion_pilot_vm')
    parser.add_argument('--preflight',action='store_true')
    main(parser.parse_args())
