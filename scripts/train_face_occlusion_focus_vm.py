"""Matched auxiliary-loss pilot from epoch30 and saved AdamW; Linux CUDA only."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tarfile
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from face_occlusion_adapter import load_adapter, parameter_groups
from face_occlusion_focus import component_weights, focus_loss, validate_source, validate_optimizer, WEIGHT
from completion_inference import load_completion
from detector_training import load_manifest, ReviewedMasks, evaluate, segmentation_loss
from detector_replay import BenchmarkMasks, replay_consistency
from scripts.train_coverage_vm import require_vm_gpu, safe_path, mask_export, PARENT_SHA, PROTOCOL_SHA
from scripts.train_face_occlusion_vm import dependency_versions, validate_schedule, selection, OLD_INVENTORY_SHA, REPLAY_SHA
from scripts.package_face_occlusion_vm import sha

MODEL_SHA = '9ba74a30719f01bfafd7ee9c060dc090c507c26eeeecb117b4cb342a8bf4df42'
OPTIMIZER_SHA = '258a01cc4698668333e191d6d01bd2866bd1e847b1f9df84e1a33c8ec8204b00'
CHECK_EPOCHS = (5, 10)
OLD_INVENTORIES = {
    'outputs/coverage_protocol_v1/vm_inventory.json': OLD_INVENTORY_SHA,
    'outputs/face_occlusion_bundle_v1/inventory.json': 'cb8485ef260bf83f8f4f3e6ce2f6fd45dc1a8726980130ddf4dfea956e83a4c1',
    'outputs/face_occlusion_continuation_bundle_v1/inventory.json': '5d155a4373f8a251ecd9d0bb6caa91cf7f0b549d96bdea0b13f7760fbb2c1888',
}
INVENTORY_PATH = 'outputs/face_occlusion_focus_bundle_v1/inventory.json'
MAP_PATH = 'outputs/face_occlusion_focus_bundle_v1/target_maps_v2.json'
PRIORITY_PATH = 'outputs/face_occlusion_focus_bundle_v1/priority_regions.json'


def require(condition, message):
    if not condition: raise ValueError(message)


def fixed_schedule(protocol):
    batches = protocol['schedules']['extended']['batches']
    require(len(batches) == 10 and all(len(e) == 21 for e in batches), 'Fixed ten-epoch schedule differs')
    return copy.deepcopy(batches)


def counters(epoch):
    require(type(epoch) is int and 1 <= epoch <= 10, 'Epoch outside fixed budget')
    return {'epoch': 30 + epoch, 'optimizer_updates': 630 + epoch * 21,
            'additional_epoch': 20 + epoch, 'fresh_optimizer_updates': 420 + epoch * 21,
            'experiment_epoch': epoch, 'experiment_updates': epoch * 21}


def restore_optimizer(optimizer, payload, groups, model_sha):
    summary = validate_optimizer(payload, groups, model_sha)
    optimizer.load_state_dict(copy.deepcopy(payload['state']))
    restored = optimizer.state_dict()
    # PyTorch moves moments to the current parameter devices. Their values must
    # remain exact; this check performs no backward pass or optimizer update.
    for i, moment in restored['state'].items():
        for name, tensor in moment.items():
            require(torch.equal(tensor.detach().cpu(), payload['state']['state'][i][name]), 'Restored moment differs')
    for actual, expected in zip(restored['param_groups'], payload['state']['param_groups']):
        require(all(actual.get(k) == v for k, v in expected.items()), 'Restored optimizer group differs')
    return summary


def priority_masks(registry, training):
    require(len(training) == 73 and isinstance(registry.get('records'), list), 'Training reflection registry differs')
    records = registry['records']
    require(len(records) == 2 and {r.get('batch_index') for r in records} == {15, 46},
            'Only the two original training reflection cases are allowed')
    maps = {}
    for record in records:
        index = record['batch_index']; require(type(index) is int, 'Reflection index must be an integer')
        row = training[index]
        require(row['split'] == 'train' and
                row.get('glare_stratum') == 'strong_lens_reflection' and
                record['image_sha256'] == row['image_sha256'] and
                record['mask_sha256'] == row['mask_sha256'], 'Reflection source/mask/split differs')
        pixels = record.get('flat_indices')
        require(record.get('shape') == [256, 256] and isinstance(pixels, list) and pixels and
                all(type(i) is int and 0 <= i < 256 * 256 for i in pixels) and
                pixels == sorted(set(pixels)), 'Reflection pixel registration invalid')
        mask = np.zeros((256, 256), dtype=bool); mask.reshape(-1)[pixels] = True
        maps[index] = mask
    return maps


def target_maps(real, cache, priorities=None):
    require(len(real) == 73 and len(cache) == 638, 'Fixed training target counts differ')
    priorities = priorities or {}
    require(set(priorities).issubset({15, 46}), 'Reflection maps must be training-only')
    maps = {}; records = []
    targets = [(i, real[i][1][0].numpy().astype(bool)) for i in range(73)]
    targets.extend((73 + i, cache[i][1][0].numpy().astype(bool)) for i in sorted(cache))
    for index, target in targets:
        weights, info = component_weights(target, priorities.get(index))
        maps[index] = torch.from_numpy(weights[None])
        records.append({'batch_index': index, 'domain': 'real_train' if index < 73 else 'synthetic_train',
                        'target_sha256': hashlib.sha256(target.astype('uint8').tobytes()).hexdigest(),
                        'weight_sha256': hashlib.sha256(weights.astype('<f4', copy=False).tobytes()).hexdigest(), **info})
    return maps, records


def main(args):
    require_vm_gpu()  # Refuse CPU/Windows before reading files or creating models.
    root = Path.cwd().resolve(); out = safe_path(root, args.output)
    archive = root / 'face-occlusion-focus-results.tar.gz'
    require(not out.exists() and (args.preflight or not archive.exists()), 'Preserve existing output/archive')
    for name, digest in OLD_INVENTORIES.items():
        path = root / name; require(sha(path) == digest, 'Prior immutable inventory changed')
        for relative, expected in json.loads(path.read_text()).items():
            require(sha(safe_path(root, relative)) == expected, f'Prior input changed: {relative}')
    inventory_path = root / INVENTORY_PATH; inventory = json.loads(inventory_path.read_text())
    required = {'face_occlusion_focus.py', 'scripts/train_face_occlusion_focus_vm.py',
                'FACE_OCCLUSION_FOCUS.md', 'FACE_OCCLUSION_FOCUS_VM.md', MAP_PATH, PRIORITY_PATH}
    require(required.issubset(inventory), 'Focus package incomplete')
    for relative, expected in inventory.items():
        require(sha(safe_path(root, relative)) == expected, f'New input changed: {relative}')
    protocol_path = root / 'outputs/coverage_protocol_v1/protocol.json'
    cache_path = root / 'outputs/coverage_training_vm/replay_pixels.pth'
    require(sha(protocol_path) == PROTOCOL_SHA and sha(cache_path) == REPLAY_SHA, 'Protocol/cache changed')
    protocol = json.loads(protocol_path.read_text()); cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    rows = load_manifest(root / 'dataset/detector_training_extension_v2/manifest.json')
    training = [r for r in rows if r['split'] == 'train']; validation = [r for r in rows if r['split'] == 'validation']
    validate_schedule(protocol, training, cache); schedule = fixed_schedule(protocol); real = ReviewedMasks(training, 256)
    priority_registry = json.loads((root / PRIORITY_PATH).read_text())
    require(priority_registry['format'] == 'dgp-reviewed-training-reflections-v1' and
            priority_registry['training_manifest_sha256'] == sha(root / 'dataset/detector_training_extension_v2/manifest.json'),
            'Reflection registration provenance differs')
    priorities = priority_masks(priority_registry, training)
    maps, map_records = target_maps(real, cache, priorities)
    registered = json.loads((root / MAP_PATH).read_text())
    require(registered['format'] == 'dgp-component-focus-targets-v2' and
            registered['model_sha256'] == MODEL_SHA and registered['optimizer_sha256'] == OPTIMIZER_SHA and
            registered['optimizer_updates_locally'] == 0 and
            registered['priority_manifest_sha256'] == sha(root / PRIORITY_PATH) and
            registered['records'] == map_records and registered['protocol_sha256'] == PROTOCOL_SHA and
            registered['replay_sha256'] == REPLAY_SHA and registered['training_manifest_sha256'] ==
            sha(root / 'dataset/detector_training_extension_v2/manifest.json'), 'Training-only focus map registration differs')
    source_root = root / 'outputs/face_occlusion_continuation_vm'
    source = source_root / 'epoch_30.pth'; optimizer_path = source_root / 'final_optimizer.pth'
    require(sha(source) == MODEL_SHA and sha(optimizer_path) == OPTIMIZER_SHA, 'Verified model/optimizer source changed')
    dependency_path = root / 'outputs/face_occlusion_dependencies'
    sys.path.insert(0, str(dependency_path)); dependencies = dependency_versions(dependency_path)
    setup_path = dependency_path / 'setup.json'; setup = json.loads(setup_path.read_text())
    require(setup['packages'] == dependencies and setup['torch'] == str(torch.__version__), 'Runtime differs from prior setup')
    torch.set_num_threads(4); torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model, payload = load_adapter(source, 'cuda'); validate_source(payload)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    optimizer_payload = torch.load(optimizer_path, map_location='cpu', weights_only=True)
    optimizer_check = validate_optimizer(optimizer_payload, parameter_groups(model), MODEL_SHA)
    def tensors(batch):
        pairs = [real[i] if i < 73 else (cache[i - 73][0].float() / 255, cache[i - 73][1].float()) for i in batch]
        x, m = (torch.stack([p[j] for p in pairs]).cuda() for j in (0, 1))
        return x, m, torch.stack([maps[i] for i in batch]).cuda(), torch.tensor([i >= 73 for i in batch], device='cuda')
    out.mkdir(parents=True); started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    run = {'preflight': args.preflight, 'source_model_sha256': MODEL_SHA, 'source_optimizer_sha256': OPTIMIZER_SHA,
           'source_epoch': 30, 'source_model_updates': 630, 'source_optimizer_step': 420,
           'optimizer_reset': False, 'optimizer_state_restored': True, 'epochs_per_arm': 10, 'updates_per_arm': 210,
           'arms': {'control': 0., 'component_focus': WEIGHT}, 'check_epochs': list(CHECK_EPOCHS),
           'protocol_sha256': PROTOCOL_SHA, 'replay_sha256': REPLAY_SHA, 'parent_sha256': PARENT_SHA,
           'prior_inventories': OLD_INVENTORIES, 'inventory_sha256': sha(inventory_path), 'map_manifest_sha256': sha(root / MAP_PATH),
           'priority_manifest_sha256': sha(root / PRIORITY_PATH),
           'script_sha256': sha(__file__), 'specification_sha256': sha(root / 'FACE_OCCLUSION_FOCUS.md'),
           'focus_module_sha256': sha(root / 'face_occlusion_focus.py'), 'adapter_sha256': sha(root / 'face_occlusion_adapter.py'),
           'dependencies': dependencies, 'setup_sha256': sha(setup_path), 'opencv': str(__import__('cv2').__version__),
           'torch': str(torch.__version__), 'python': sys.version, 'gpu': torch.cuda.get_device_name(),
           'encoder_lr': 1e-5, 'decoder_head_lr': 1e-4, 'weight_decay': 1e-4, 'clip': 1., 'seed': 42,
           'background_weight': .25, 'consistency_weight': 1., 'component_ring_radius': 3, 'clear_hard_fraction': .1,
           'validation_batch_size': 1, 'batchnorm_policy': 'fixed_running_statistics_trainable_affine',
           'test_split': 'Not scored; previous development work inspected it',
           'overlap_limitation': 'External FaceExtraction FFHQ pretraining overlap unresolved'}
    (out / 'run.json').write_text(json.dumps(run, indent=2) + '\n')
    if args.preflight:
        groups = parameter_groups(model); optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        restored = restore_optimizer(optimizer, optimizer_payload, groups, MODEL_SHA)
        x, m, w, _ = tensors(schedule[0][0])
        with torch.no_grad():
            logits = model.detect(x); supervised = segmentation_loss(logits, m, .25, .1); auxiliary = focus_loss(logits, m, w)
        require(torch.isfinite(supervised + WEIGHT * auxiliary), 'Nonfinite preflight loss')
        require(all(torch.equal(v.cpu(), initial[k]) for k, v in model.state_dict().items()), 'Preflight changed model')
        report = {'supervised_loss': float(supervised), 'focus_loss': float(auxiliary), 'experiment_updates': 0,
                  'registered_training_targets': len(map_records), 'optimizer_snapshot': optimizer_check,
                  'restored_optimizer': restored}
        (out / 'preflight.json').write_text(json.dumps(report, indent=2) + '\n')
        print('CUDA forward and source/moments/maps passed; zero optimizer updates', flush=True)
        return
    parent_path = root / 'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'
    require(sha(parent_path) == PARENT_SHA, 'Original parent changed')
    parent, _ = load_completion(parent_path, 'cuda'); parent.requires_grad_(False).eval()
    parent_state = {k: v.detach().cpu().clone() for k, v in parent.state_dict().items()}
    datasets = {'real': ReviewedMasks(validation, 256),
                'synthetic': BenchmarkMasks(root / 'outputs/downloaded_completion_benchmark/outputs/completion_pretrained_vm', 256),
                'training_real': real}
    for name, predicate in [('human_real', lambda r: Path(r['image']).name != 'new_covered_40.png'),
                            ('mannequin', lambda r: Path(r['image']).name == 'new_covered_40.png'),
                            ('glare', lambda r: r.get('glare_stratum') == 'strong_lens_reflection')]:
        subset = [r for r in validation if predicate(r)]
        require(subset, f'Missing required {name} evaluation subset'); datasets[name] = ReviewedMasks(subset, 256)
    loaders = {name: torch.utils.data.DataLoader(dataset, batch_size=1) for name, dataset in datasets.items()}
    baseline = {name: evaluate(parent, loader, 'cuda') for name, loader in loaders.items()}
    source_scores = {name: evaluate(model, loader, 'cuda') for name, loader in loaders.items()}
    require(all(v == payload['selection'][name] for name, v in source_scores.items()), 'Source predictions differ from audited epoch30')
    require(baseline == json.loads((source_root / 'baseline.json').read_text()), 'Original common baseline differs')
    (out / 'baseline.json').write_text(json.dumps(baseline, indent=2) + '\n')
    (out / 'source_metrics.json').write_text(json.dumps(source_scores, indent=2) + '\n')
    shutil.copyfile(source, out / 'initial.pth'); shutil.copyfile(optimizer_path, out / 'initial_optimizer.pth')
    for domain in ('real', 'synthetic'): mask_export(parent, datasets[domain], out / 'baseline_masks' / domain)
    for domain in ('real', 'synthetic', 'training_real'): mask_export(model, datasets[domain], out / 'initial_masks' / domain)
    completed = {}
    for arm, weight in run['arms'].items():
        if arm != 'control': model, payload = load_adapter(source, 'cuda')
        require(all(torch.equal(v.cpu(), initial[k]) for k, v in model.state_dict().items()), 'Arm did not start from exact source')
        groups = parameter_groups(model); optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        restored = restore_optimizer(optimizer, optimizer_payload, groups, MODEL_SHA)
        torch.manual_seed(42); arm_out = out / arm; arm_out.mkdir(); best = baseline['real']['iou']; selected_epochs = []
        (arm_out / 'restored_optimizer.json').write_text(json.dumps(restored, indent=2) + '\n')
        updates = 0
        for epoch, batches in enumerate(schedule, 1):
            model.train(); sums = {'loss': 0., 'supervised': 0., 'teacher': 0., 'focus': 0.}
            for step, batch in enumerate(batches, 1):
                x, m, w, tag = tensors(batch); optimizer.zero_grad(set_to_none=True); logits = model.detect(x)
                with torch.no_grad(): reference = parent.detect(x[tag])
                supervised = segmentation_loss(logits, m, .25, .1)
                teacher = replay_consistency(logits[tag], reference, torch.ones(int(tag.sum()), dtype=torch.bool, device='cuda'))
                focused = focus_loss(logits, m, w)
                loss = supervised + teacher + weight * focused
                require(torch.isfinite(loss), 'Nonfinite pilot loss')
                loss.backward(); norm = float(torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1., error_if_nonfinite=True))
                optimizer.step(); updates += 1
                values = {'loss': float(loss.detach()), 'supervised': float(supervised.detach()),
                          'teacher': float(teacher.detach()), 'focus': float(focused.detach())}
                for key, value in values.items(): sums[key] += value
                row = {'experiment_epoch': epoch, 'global_epoch': 30 + epoch, 'step': step,
                       'experiment_updates': updates, 'cumulative_model_updates': 630 + updates,
                       'optimizer_state_step': 420 + updates, 'indices': batch,
                       'focus_weight': weight, 'pre_clip_norm': norm, **values}
                with (arm_out / 'steps.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
            if epoch in CHECK_EPOCHS:
                for key, value in model.state_dict().items():
                    if key.startswith('reference_visible_head.') or key.endswith(('running_mean', 'running_var', 'num_batches_tracked')):
                        require(torch.equal(value.cpu(), initial[key]), 'Frozen source state changed')
                require(all(state['step'].item() == 420 + updates for state in optimizer.state.values()), 'Optimizer lifetime step differs')
                scores = {name: evaluate(model, loader, 'cuda') for name, loader in loaders.items()}
                real_ok, synthetic_ok, chosen = selection(scores, baseline, best)
                count = counters(epoch); require(count['experiment_updates'] == updates, 'Experiment counter differs')
                report = {**count, 'arm': arm, 'mean_epoch_terms': {k: v / 21 for k, v in sums.items()},
                          'real_gate': real_ok, 'synthetic_gate': synthetic_ok, 'selected': chosen, **scores}
                with (arm_out / 'metrics.jsonl').open('a') as stream: stream.write(json.dumps(report) + '\n')
                exported = {**payload, 'model': model.state_dict(), **count,
                            'focus_metadata': run, 'focus_arm': arm, 'selection': report}
                torch.save(exported, arm_out / f'epoch_{30 + epoch}.pth')
                if chosen:
                    best = scores['real']['iou']; selected_epochs.append(30 + epoch)
                    torch.save(exported, arm_out / 'best_detector.pth')
                for domain in ('real', 'synthetic', 'training_real'):
                    mask_export(model, datasets[domain], arm_out / f'epoch_{30 + epoch}_masks' / domain)
                print(json.dumps(report), flush=True)
            else: print(f'{arm} global epoch {30 + epoch}/40; {updates}/210 experiment updates', flush=True)
        require(updates == 210, 'Arm ended before fixed budget')
        final = arm_out / 'epoch_40.pth'; final_optimizer = arm_out / 'final_optimizer.pth'
        torch.save({'state': optimizer.state_dict(), 'model_sha256': sha(final), 'experiment_updates': 210,
                    'optimizer_state_step': 630, 'cumulative_model_updates': 840}, final_optimizer)
        completed[arm] = {'experiment_updates': updates, 'cumulative_model_updates': 840, 'optimizer_state_step': 630,
                          'selected_epochs': selected_epochs, 'final_sha256': sha(final),
                          'final_optimizer_sha256': sha(final_optimizer)}
        del optimizer, model
    require(all(torch.equal(v.cpu(), parent_state[k]) for k, v in parent.state_dict().items()), 'Original parent/generator changed')
    complete = {'complete': True, 'arms': completed, 'total_experiment_updates': 420,
                'source_model_sha256': MODEL_SHA, 'source_optimizer_sha256': OPTIMIZER_SHA, 'parent_unchanged': True,
                'seconds': time.monotonic() - started, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
                'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(), 'promotion_requires_visual_review': True}
    (out / 'complete.json').write_text(json.dumps(complete, indent=2) + '\n')
    with tarfile.open(archive, 'w:gz') as tar:
        tar.add(out, arcname=out.relative_to(root).as_posix())
        for name in inventory: tar.add(root / name, arcname=name)
        tar.add(inventory_path, arcname=INVENTORY_PATH)
    archive.with_name(archive.name + '.sha256').write_bytes((sha(archive) + '  ' + archive.name + '\n').encode())
    print('Download', archive, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output', default='outputs/face_occlusion_focus_vm')
    main(parser.parse_args())
