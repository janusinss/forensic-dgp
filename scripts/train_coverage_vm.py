"""VM-only, fixed-budget real-data coverage comparison. No local training."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from completion_inference import load_completion
from detector_replay import SyntheticMasks, BenchmarkMasks, retention_passes, replay_consistency
from detector_training import load_manifest, ReviewedMasks, evaluate, detector_optimizer, segmentation_loss

PROTOCOL_SHA = '46c438f6e72e9bd4709102dd8147dc926fff84de0e7e8a89f3bee1c22a9590fd'
PARENT_SHA = 'c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93'
MANIFESTS = {'control': 'dataset/detector_glare_review_v3/manifest.json',
             'extended': 'dataset/detector_training_extension_v2/manifest.json'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_vm_gpu():
    if sys.platform != 'linux' or not torch.cuda.is_available():
        raise RuntimeError('This runner requires the Linux VM CUDA device; no local training')


def safe_path(root, relative):
    path = (root / relative.replace('\\', '/')).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Inventory path escapes workspace')
    return path


def freeze_pixels(x, m):
    q = (x * 255).round().to(torch.uint8)
    if not torch.equal(x, q.float() / 255) or not torch.isfinite(m).all() or not ((m == 0) | (m == 1)).all():
        raise ValueError('Replay tensors are not losslessly representable as RGB bytes/binary masks')
    return q, m.to(torch.uint8)


def real_gate(metrics, baseline, best):
    return metrics['iou'] > best and all(metrics[k] <= baseline[k] for k in
        ('visible_false_positive', 'empty_mask_cases', 'negative_false_positive_cases'))


@torch.inference_mode()
def mask_export(model, dataset, out):
    out.mkdir(parents=True)
    for index in range(len(dataset)):
        x, _ = dataset[index]
        mask = (model.detect(x[None].cuda()).sigmoid()[0,0] >= .5).cpu().numpy()
        Image.fromarray(mask.astype('uint8') * 255).save(out / f'{index:04}.png')


def main(args):
    require_vm_gpu()  # Before creating files, models, gradients or optimizers.
    root = Path.cwd().resolve()
    out = safe_path(root, args.output)
    if out.exists():
        raise ValueError('Output exists; inspect it rather than overwrite/restart')
    protocol_path = root / 'outputs/coverage_protocol_v1/protocol.json'
    if sha(protocol_path) != PROTOCOL_SHA or sha(args.checkpoint) != PARENT_SHA:
        raise ValueError('Protocol or initial checkpoint mismatch')
    protocol = json.loads(protocol_path.read_text())
    inventory = json.loads(Path(args.inventory).read_text())
    for relative, digest in inventory.items():
        if sha(safe_path(root, relative)) != digest:
            raise ValueError(f'Inventory mismatch: {relative}')
    required = set(protocol['input_hashes']) | set(MANIFESTS.values())
    required.update(r['path'] for r in protocol['sources'])
    required.update(['scripts/train_coverage_vm.py','completion.py','completion_data.py',
                     'completion_inference.py','detector_training.py','detector_replay.py'])
    all_rows = {arm: load_manifest(root / path) for arm, path in MANIFESTS.items()}
    for arm, rows in all_rows.items():
        for r in rows:
            for k in ('image_path','mask_path'):
                required.add(Path(r[k]).relative_to(root).as_posix())
    benchmark_root = root / 'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm'
    bench_manifest = json.loads((benchmark_root / 'manifest.json').read_text())
    for case in bench_manifest['cases']:
        for folder in ('input','mask'):
            required.add((benchmark_root / folder / case['file']).relative_to(root).as_posix())
    normalized_inventory = {p.replace('\\','/') for p in inventory}
    if not {p.replace('\\','/') for p in required}.issubset(normalized_inventory):
        raise ValueError('Inventory does not cover all consumed code/data')
    for path, digest in protocol['input_hashes'].items():
        if sha(safe_path(root,path)) != digest:
            raise ValueError('Protocol input changed')
    for r in protocol['sources']:
        if sha(safe_path(root,r['path'])) != r['sha256']:
            raise ValueError('Replay source changed')
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, state = load_completion(args.checkpoint, 'cuda')
    initial = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    teacher = copy.deepcopy(model.segmenter).eval().requires_grad_(False)
    real = {arm:ReviewedMasks([r for r in rows if r['split']=='train'],state['size']) for arm,rows in all_rows.items()}
    val_rows = [r for r in all_rows['control'] if r['split']=='validation']
    # Compare immutable metadata, not resolved paths in the copied directories.
    expected_val = [(r['image_sha256'],r['mask_sha256'],r['group']) for r in val_rows]
    if expected_val != [(r['image_sha256'],r['mask_sha256'],r['group']) for r in all_rows['extended'] if r['split']=='validation']:
        raise ValueError('Validation differs across arms')
    validations = {'real':ReviewedMasks(val_rows,state['size']), 'synthetic':BenchmarkMasks(benchmark_root,state['size'])}
    for name, predicate in (
        ('human_real',lambda r:Path(r['image']).name!='new_covered_40.png'),
        ('mannequin',lambda r:Path(r['image']).name=='new_covered_40.png'),
        ('glare',lambda r:r.get('glare_stratum') not in (None,'no_added_glare'))):
        subset = [r for r in val_rows if predicate(r)]
        if subset:
            validations[name] = ReviewedMasks(subset,state['size'])
    loaders = {k:torch.utils.data.DataLoader(v,batch_size=8) for k,v in validations.items()}
    # Only cases in the frozen schedules need materialization; store exact bytes.
    used = sorted({i-len(real['control']) for epoch in protocol['schedules']['control']['batches'] for batch in epoch for i in batch if i>=len(real['control'])})
    other = sorted({i-len(real['extended']) for epoch in protocol['schedules']['extended']['batches'] for batch in epoch for i in batch if i>=len(real['extended'])})
    if used != other:
        raise ValueError('Replay case membership differs')
    synthetic = SyntheticMasks([r['path'] for r in protocol['sources']],state['size'],42)
    cache = {index:freeze_pixels(*synthetic[index]) for index in used}
    out.mkdir(parents=True)
    torch.save({'state_dict':initial,'parent_sha256':PARENT_SHA},out/'initial.pth')
    torch.save(cache,out/'replay_pixels.pth')
    metadata = {'protocol_sha256':PROTOCOL_SHA,'parent_sha256':PARENT_SHA,
        'initial_sha256':sha(out/'initial.pth'),'replay_sha256':sha(out/'replay_pixels.pth'),
        'inventory_sha256':sha(args.inventory),'torch':str(torch.__version__),
        'gpu':torch.cuda.get_device_name(),'lr':1e-5,'background_weight':.25,
        'consistency_weight':1.,'clip_norm':1.,'weight_decay':1e-4,
        'preflight':args.preflight,'synthetic_cases_cached':len(cache),
        'test_split':'Not evaluated here; previously inspected, not untouched test'}
    (out/'run.json').write_text(json.dumps(metadata,indent=2)+'\n')
    if args.preflight:
        batch = protocol['schedules']['control']['batches'][0][0]
        items = [real['control'][i] if i<len(real['control']) else
                 (cache[i-len(real['control'])][0].float()/255,cache[i-len(real['control'])][1].float()) for i in batch]
        x,m = (torch.stack([item[j] for item in items]).cuda() for j in (0,1))
        with torch.no_grad():
            loss = segmentation_loss(model.detect(x),m,.25,.1)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite preflight')
        (out/'preflight.json').write_text(json.dumps({'finite_loss':float(loss),'optimizer_updates':0})+'\n')
        print('Preflight complete: CUDA forward only, zero optimizer updates',flush=True)
        return
    baseline = {k:evaluate(model,v,'cuda') for k,v in loaders.items()}
    (out/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    started = time.monotonic()
    for arm in ('control','extended'):
        model.load_state_dict(initial,strict=True)
        if any(not torch.equal(v.cpu(),initial[k]) for k,v in model.state_dict().items()):
            raise ValueError('Initial state mismatch')
        torch.manual_seed(42)
        optimizer = detector_optimizer(model,1e-5)
        arm_out = out/arm
        arm_out.mkdir()
        best = baseline['real']['iou']
        for epoch,batches in enumerate(protocol['schedules'][arm]['batches'],1):
            model.segmenter.train()
            losses = []
            for batch in batches:
                items = [real[arm][i] if i<len(real[arm]) else
                         (cache[i-len(real[arm])][0].float()/255,cache[i-len(real[arm])][1].float()) for i in batch]
                x,m = (torch.stack([item[j] for item in items]).cuda() for j in (0,1))
                tagged = torch.tensor([i>=len(real[arm]) for i in batch],device='cuda')
                logits = model.detect(x)
                with torch.no_grad():
                    reference = teacher(x[tagged])
                loss = segmentation_loss(logits,m,.25,.1) + replay_consistency(
                    logits[tagged],reference,torch.ones(int(tagged.sum()),dtype=torch.bool,device='cuda'))
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.segmenter.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                losses.append(float(loss.detach()))
            if any(not torch.equal(v.cpu(),initial['generator.'+k]) for k,v in model.generator.state_dict().items()):
                raise ValueError('Frozen generator changed')
            metrics = {k:evaluate(model,v,'cuda') for k,v in loaders.items()}
            real_ok = real_gate(metrics['real'],baseline['real'],best)
            synthetic_ok = retention_passes(metrics['synthetic'],baseline['synthetic'])
            selected = real_ok and synthetic_ok
            report = {'epoch':epoch,'updates':epoch*21,'loss':sum(losses)/len(losses),
                      'real_gate':real_ok,'synthetic_gate':synthetic_ok,'selected':selected,**metrics}
            with (arm_out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(report)+'\n')
            export = {**state,'model':model.state_dict(),'detector_only':True,
                      'detector_finetune_epoch':epoch,'detector_training':metadata,'coverage_arm':arm}
            torch.save(export,arm_out/f'epoch_{epoch}.pth')
            if selected:
                best = metrics['real']['iou']
                torch.save(export,arm_out/'best_detector.pth')
            for name in ('real','synthetic'):
                mask_export(model,validations[name],arm_out/f'epoch_{epoch}_masks'/name)
            print(arm,json.dumps(report),flush=True)
    (out/'complete.json').write_text(json.dumps({'complete':True,'seconds':time.monotonic()-started,
        'updates_per_arm':210,'promotion_requires_visual_review':True})+'\n')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--inventory',required=True)
    parser.add_argument('--output',default='outputs/coverage_training_vm')
    parser.add_argument('--preflight',action='store_true')
    main(parser.parse_args())
