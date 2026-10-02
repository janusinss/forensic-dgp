"""VM-only bounded native expert fit; no held-out scoring or promotion."""
import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from native_expert import (MODEL_SHA, OPTIMIZER_SHA, PARENT_SHA, NATIVE_COVERING,
                           EPOCHS, STEPS, require, require_vm_gpu, fixed_schedule,
                           counters, validate_source, validate_source_optimizer, expert_loss, fit_decision)
from supported_real_data import SupportedMasks

SPEC = 'inputs/native_expert_protocol.json'
PINNED = {'segmentation-models-pytorch': ('segmentation_models_pytorch', '0.5.0'),
          'timm': ('timm', '1.0.15'), 'huggingface-hub': ('huggingface_hub', '0.29.3'),
          'safetensors': ('safetensors', '0.5.3'), 'PyYAML': ('yaml', '6.0.2')}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
    return digest.hexdigest()


def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2) + '\n')


def safe_path(root, name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name
            and not name.startswith('/') and PurePosixPath(name).as_posix() == name
            and '..' not in PurePosixPath(name).parts, 'Canonical in-root POSIX member required')
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), 'Member escapes root')
    return path


def verify_fixture(case, row):
    require(set(case) == {'input', 'mask', 'geometry', 'valid'}, 'Fixture tensor fields differ')
    for name, value in case.items():
        require(isinstance(value, torch.Tensor) and value.device.type == 'cpu'
                and value.dtype == torch.uint8 and value.shape == ((3 if name == 'input' else 1), 256, 256),
                'Fixture dtype/shape differs')
        require(hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest() == row['pixel_sha256'][name],
                'Fixture raw bytes changed')
        if name != 'input': require(((value == 0) | (value == 1)).all(), 'Fixture maps must be binary')
    require(case['valid'].any() and not (case['mask'].bool() & ~case['valid'].bool()).any(),
            'Fixture target outside supervised support')
    for key, tensor in (('hole_pixels', 'mask'), ('geometry_pixels', 'geometry'), ('valid_pixels', 'valid')):
        require(int(case[tensor].sum()) == row[key], 'Fixture registered denominator differs')


def dependency_versions(target):
    versions = {}
    for distribution, (name, expected) in PINNED.items():
        actual = importlib.metadata.distribution(distribution).version
        module = importlib.import_module(name)
        require(module.__file__ is not None, 'Pinned module has no concrete file: ' + name)
        path = Path(module.__file__).resolve()
        require(actual == expected and path.is_relative_to(target), 'Pinned dependency differs: ' + distribution)
        versions[distribution] = {'version': actual, 'module_path': str(path)}
    return versions


def counts(prediction, target, valid):
    require(prediction.shape == target.shape == valid.shape == (256, 256)
            and prediction.dtype == target.dtype == valid.dtype == torch.bool
            and valid.any() and not (target & ~valid).any(), 'Invalid scored mask/support')
    covered, present = bool(target.any()), bool(prediction[valid].any())
    return {'tp': int((prediction & target & valid).sum()), 'fp': int((prediction & ~target & valid).sum()),
            'fn': int((~prediction & target & valid).sum()), 'visible': int((~target & valid).sum()),
            'covered_cases': int(covered), 'empty_mask_cases': int(covered and not present),
            'negative_cases': int(not covered), 'negative_false_positive_cases': int(not covered and present),
            'ignored_positive_pixels': int((prediction & ~valid).sum())}


def aggregate(rows):
    require(rows, 'Empty scored group')
    total = {key: sum(r[key] for r in rows) for key in rows[0]}
    return {**total, 'cases': len(rows), 'iou': total['tp'] / max(1, total['tp'] + total['fp'] + total['fn']),
            'missed_fraction': total['fn'] / max(1, total['tp'] + total['fn']),
            'visible_false_positive': total['fp'] / max(1, total['visible'])}


@torch.inference_mode()
def measure(model, real, fixture_rows, pixels, lenses, output):
    model.eval(); groups = {k: [] for k in ('original_real', 'native_real', 'all_real', 'reflection')}
    cases, native, lens, hashes = [], {}, {'old': 0, 'new': 0}, {}
    output.mkdir(parents=True)
    for pool, size in (('real', len(real)), ('reflection', len(fixture_rows))):
        folder = output / pool; folder.mkdir()
        for i in range(size):
            if pool == 'real':
                x, target, valid = real[i]; source = real.rows[i].get('source_index')
            else:
                item = pixels[i]; x = item['input'].float() / 255
                target, valid = item['mask'].float(), item['valid'].float(); source = None
            logits = model.detect(x.cuda()[None])
            require(logits.shape == (1, 1, 256, 256) and torch.isfinite(logits).all(), 'Invalid measured logits')
            prediction = (logits.sigmoid()[0, 0] >= .5).cpu()
            counted = counts(prediction, target[0].bool(), valid[0].bool())
            name = f'{pool}/{i:04}'; path = folder / f'{i:04}.png'
            Image.fromarray(prediction.numpy().astype('uint8') * 255).save(path)
            hashes[name + '.png'] = sha(path); cases.append({'id': name, 'counts': counted})
            if pool == 'real':
                groups['all_real'].append(counted)
                groups['original_real' if i < 73 else 'native_real'].append(counted)
                if i >= 73 and source in NATIVE_COVERING: native[str(source)] = counted
                if i in lenses:
                    lens['old' if i < 73 else 'new'] += int((prediction & lenses[i]).sum())
            else: groups['reflection'].append(counted)
    return {**{name: aggregate(rows) for name, rows in groups.items()},
            'native_cases': native, 'lens': lens, 'case_counts': cases, 'mask_sha256': hashes,
            'image_forward_count': len(cases), 'held_out_forward_images': 0}


def preview(real, pixels, ids, before, after, destination):
    width, row_height = 192, 218
    canvas = Image.new('RGB', (4 * width, 42 + row_height * len(ids)), 'white'); draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(('input', 'target', 'source42', 'native expert48')):
        draw.text((column * width + 3, 3), label, fill='black')
    draw.text((3, 19), 'Green=TP red=FP yellow=FN purple=ignored. Training masks only.', fill='black')
    for row, name in enumerate(ids):
        pool, index = name.split('/'); index = int(index)
        if pool == 'real': x, target, valid = real[index]
        else:
            item = pixels[index]; x = item['input'].float() / 255
            target, valid = item['mask'].float(), item['valid'].float()
        rgb = np.rint(x.permute(1, 2, 0).numpy() * 255).astype('uint8')
        truth, support = target[0].numpy().astype(bool), valid[0].numpy().astype(bool)
        def overlay(prediction):
            colors = rgb.copy(); marked = ~support | truth | prediction
            colors[~support] = (130, 90, 165)
            colors[truth & prediction & support] = (30, 220, 70)
            colors[~truth & prediction & support] = (240, 50, 50)
            colors[truth & ~prediction & support] = (255, 210, 30)
            result = rgb.copy(); result[marked] = (.5 * rgb[marked] + .5 * colors[marked]).astype('uint8')
            return result
        predictions = []
        for folder in (before, after):
            with Image.open(folder / (name + '.png')) as mask: predictions.append(np.array(mask).astype(bool))
        for column, panel in enumerate((rgb, overlay(truth), *(overlay(p) for p in predictions))):
            y = 42 + row * row_height
            canvas.paste(Image.fromarray(panel).resize((width, width)), (column * width, y))
            draw.text((column * width + 3, y + width + 3), name, fill='black')
    canvas.save(destination)


def main(args):
    require_vm_gpu()  # Must precede paths, output creation, models and optimizers.
    root = ROOT.resolve(); existing = Path(args.existing).expanduser().resolve() if args.existing else root.parent / 'coverage_vm_bundle'
    require(existing != root and existing.is_dir(), 'Require the separate completed coverage VM workspace')
    require(args.batch_size == (1 if args.dry_run else 8), 'Dry run requires batch1; full fixed pilot requires batch8')
    out = safe_path(root, args.output or ('outputs/native_expert_preflight' if args.dry_run else 'outputs/native_expert_vm'))
    archive = root / 'native-expert-results.tar.gz'
    require(not out.exists() and (args.dry_run or not archive.exists()), 'Preserve existing output/archive; inspect before any restart')
    inventory = read_json(root / 'inventory.json')
    require(inventory and all(sha(safe_path(root, name)) == digest for name, digest in inventory.items()), 'Sent code/data inventory changed')
    protocol = read_json(root / SPEC)
    require(protocol['format'] == 'dgp-native-expert-pilot-v1' and protocol['epochs'] == EPOCHS
            and protocol['steps_per_epoch'] == STEPS and protocol['held_out_forward_images'] == 0
            and protocol['promoted'] is False, 'Predeclared experiment scope differs')
    assets = {}
    for role, record in protocol['existing_assets'].items():
        path = safe_path(existing, record['vm_path']); require(sha(path) == record['sha256'], 'Existing asset changed: ' + role)
        assets[role] = path
    require(sha(assets['model']) == MODEL_SHA and sha(assets['optimizer']) == OPTIMIZER_SHA
            and sha(assets['parent']) == PARENT_SHA, 'Fixed source binding differs')
    require(all((existing.parent / 'dataset' / name).is_dir() for name in ('asian_faces', 'thumbnails128x128')),
            'Require original Asian/FFHQ dataset directories in the VM repository')
    split_path = root / 'inputs/phase4_split.json'; require(sha(split_path) == protocol['phase4_split_sha256'], 'Original split snapshot differs')
    split = read_json(split_path); training_paths = {p.replace('\\', '/') for p in split['train']}
    require(not (training_paths & {p.replace('\\', '/') for p in split['validation']}), 'Original split membership overlaps')
    repository_split = existing.parent / 'outputs/phase4_with_progress/split.json'
    if repository_split.exists(): require(sha(repository_split) == protocol['phase4_split_sha256'], 'VM repository split differs from audited snapshot')
    real = SupportedMasks(root / 'dataset/detector_supported_review_v1/manifest.json', split='train')
    require(len(real) == 83 and [r.get('source_index') for r in real.rows[73:]] == protocol['native_source_order'], 'Supported training order differs')
    fixture_manifest = read_json(assets['fixture_manifest'])
    fixture_rows = fixture_manifest['cases']
    require(all(r['source'].replace('\\', '/') in training_paths for r in real.rows[73:] + fixture_rows), 'Native/fixture source outside original Phase4 training')
    require(fixed_schedule(real.rows, fixture_rows) == protocol['schedule'], 'Predeclared training schedule differs')
    torch.set_num_threads(4); cache = torch.load(assets['fixture_pixels'], map_location='cpu', weights_only=True)
    require(cache['format'] == 'dgp-reflection-coverage-pixels-v1' and set(cache['pixels']) == set(range(280)), 'Fixture cache membership differs')
    pixels = cache['pixels']
    for i, record in enumerate(fixture_rows): verify_fixture(pixels[i], record)
    lenses = {}
    for row in protocol['lens_maps']:
        path = safe_path(root, row['path']); require(sha(path) == row['sha256'], 'Lens map changed')
        with Image.open(path) as image: array = np.array(image)
        require(array.shape == (256, 256) and array.dtype == np.uint8 and np.isin(array, [0, 255]).all(), 'Lens map format differs')
        mask = torch.from_numpy(array.astype(bool)); index = row['real_index']; _, target, valid = real[index]
        require(mask.any() and not (mask & ~target[0].bool()).any() and not (mask & ~valid[0].bool()).any()
                and int(mask.sum()) == row['pixels'], 'Lens target/support denominator differs')
        lenses[index] = mask
    require(len(lenses) == 6 and sum(int(m.sum()) for m in lenses.values()) == 21853, 'Fixed six-lens cohort differs')
    deps_path = existing / 'outputs/face_occlusion_dependencies'; sys.path.insert(0, str(deps_path))
    deps = dependency_versions(deps_path)
    setup_path = deps_path / 'setup.json'; setup = read_json(setup_path)
    require(sha(setup_path) == protocol['setup_sha256'] and setup['packages'] == deps
            and setup['torch'] == str(torch.__version__) == '2.9.1+cu129', 'Existing pinned CUDA runtime differs')
    require(torch.cuda.get_device_properties(0).total_memory >= 4 * 1024**3, 'At least4GiB total GPU memory required')
    free_vram, total_vram = torch.cuda.mem_get_info()
    require(free_vram >= 2 * 1024**3, 'At least2GiB free VRAM required before constructing the model')
    from face_occlusion_adapter import load_adapter, parameter_groups
    torch.manual_seed(42); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    model, payload = load_adapter(assets['model'], 'cuda'); validate_source(payload)
    initial = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    frozen = {key: value for key, value in initial.items() if key.startswith('reference_visible_head.')
              or key.endswith(('running_mean', 'running_var', 'num_batches_tracked'))}
    groups = parameter_groups(model); moments = torch.load(assets['optimizer'], map_location='cpu', weights_only=True)
    moment_check = validate_source_optimizer(moments, groups, MODEL_SHA)
    out.mkdir(parents=True); started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    write_json(out / 'run.json', {'dry_run': args.dry_run, 'protocol_sha256': sha(root / SPEC),
               'inventory_sha256': sha(root / 'inventory.json'), 'torch': str(torch.__version__), 'python': sys.version,
               'gpu': torch.cuda.get_device_name(), 'dependencies': deps, 'existing_root': str(existing),
               'free_vram_before_model_bytes': free_vram, 'total_vram_bytes': total_vram,
               'source_model_sha256': MODEL_SHA, 'source_optimizer_sha256': OPTIMIZER_SHA,
               'parent_sha256': PARENT_SHA, 'moment_check': moment_check, 'held_out_forward_images': 0,
               'updates_budget': 0 if args.dry_run else 336, 'training_fit_only': True, 'promoted': False})
    def check_frozen():
        require(all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in frozen.items()), 'Frozen reference head/BN state changed')
        require(all(sha(path) == protocol['existing_assets'][role]['sha256'] for role, path in assets.items()), 'Read-only existing assets changed')
    if args.dry_run:
        from reflection_coverage import supported_segmentation_loss
        index = next(i for i, row in enumerate(real.rows) if row.get('source_index') == 171)
        x, target, valid = real[index]
        with torch.inference_mode(): loss = supported_segmentation_loss(model.detect(x.cuda()[None]), target.cuda()[None], valid.cuda()[None])
        require(torch.isfinite(loss) and all(torch.equal(v.detach().cpu(), initial[k]) for k, v in model.state_dict().items()), 'Dry-run finite loss/state failed')
        check_frozen()
        write_json(out / 'preflight.json', {'complete': True, 'batch_size': 1, 'loss': float(loss), 'source171_valid_pixels': int(valid.sum()),
                   'optimizer_constructed': False, 'optimizer_updates': 0, 'held_out_forward_images': 0,
                   'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'training_output_quality_verified': False})
        print('Native supported CUDA dry run passed; zero updates. This is not trained quality evidence.', flush=True)
        return
    before = measure(model, real, fixture_rows, pixels, lenses, out / 'initial_masks')
    write_json(out / 'baseline.json', before)
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    optimizer.load_state_dict(copy.deepcopy(moments['state']))
    restored = optimizer.state_dict()
    require(restored['param_groups'] == moments['state']['param_groups']
            and all(torch.equal(t.detach().cpu(), moments['state']['state'][i][key])
                    for i, record in restored['state'].items() for key, t in record.items()), 'Restored moment values/groups differ')
    write_json(out / 'restored_optimizer.json', {**moment_check, 'values_exact': True, 'optimizer_reset': False})
    def tensors(batch):
        values = [real[i] for i in batch['real']]
        real_tensors = tuple(torch.stack([r[j] for r in values]).cuda() for j in range(3))
        fixture_tensors = tuple(torch.stack([pixels[i][key].float() for i in batch['fixture']]).cuda() / (255 if key == 'input' else 1)
                               for key in ('input', 'mask', 'valid'))
        return real_tensors, fixture_tensors
    real_batch, fixture_batch = tensors(protocol['schedule'][0][0])
    model.train()
    with torch.inference_mode():
        logits = model.detect(torch.cat((real_batch[0], fixture_batch[0])))
        probe, _, _ = expert_loss(logits[:3], *real_batch[1:], logits[3:], *fixture_batch[1:])
    require(torch.isfinite(probe) and all(torch.equal(v.detach().cpu(), initial[k]) for k, v in model.state_dict().items()), 'Full batch allocation/loss changed source state')
    updates = 0
    for epoch, batches in enumerate(protocol['schedule'], 1):
        model.train(); loss_sum = 0.
        for step, batch in enumerate(batches, 1):
            r, f = tensors(batch); optimizer.zero_grad(set_to_none=True)
            logits = model.detect(torch.cat((r[0], f[0])))
            loss, real_loss, fixture_loss = expert_loss(logits[:3], *r[1:], logits[3:], *f[1:])
            require(torch.isfinite(loss), 'Nonfinite actual training loss'); loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.network.parameters(), 1., error_if_nonfinite=True))
            optimizer.step(); updates += 1; loss_sum += float(loss.detach())
            record = {'experiment_epoch': epoch, 'step': step, 'experiment_updates': updates,
                      'optimizer_state_step': 672 + updates, 'cumulative_model_updates': 882 + updates,
                      'real_indices': batch['real'], 'fixture_indices': batch['fixture'],
                      'loss': float(loss.detach()), 'real_loss': float(real_loss.detach()),
                      'fixture_loss': float(fixture_loss.detach()), 'pre_clip_norm': norm}
            with (out / 'steps.jsonl').open('a', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(record) + '\n')
        require(updates == counters(epoch)['experiment_updates'] and len(optimizer.state) == 92
                and all(s['step'].item() == 672 + updates for s in optimizer.state.values()), 'Update/moment counter differs')
        check_frozen(); print(f'Native expert epoch{epoch}/6: {updates}/336 updates; mean loss={loss_sum/STEPS:.6f}', flush=True)
    after = measure(model, real, fixture_rows, pixels, lenses, out / 'final_masks')
    write_json(out / 'final_metrics.json', after); decision = fit_decision(before, after)
    write_json(out / 'fit_decision.json', decision); count = counters(6)
    final = out / 'native_expert_epoch_48.pth'
    torch.save({**payload, 'model': model.state_dict(), **count, 'detector_only': True,
                'native_expert_protocol_sha256': sha(root / SPEC), 'native_expert_source_selection': payload['selection'],
                'selection': {'selected': False, 'development_evaluated': False, 'training_fit': decision},
                'native_expert_metadata': {'optimizer_reset': False, 'source_model_sha256': MODEL_SHA,
                    'source_optimizer_sha256': OPTIMIZER_SHA, 'held_out_forward_images': 0, 'promoted': False}}, final)
    moment_path = out / 'final_optimizer.pth'
    torch.save({'state': optimizer.state_dict(), 'model_sha256': sha(final), 'experiment_updates': 336,
                'optimizer_state_step': 1008, 'cumulative_model_updates': 1218}, moment_path)
    preview(real, pixels, protocol['preview_ids'], out / 'initial_masks', out / 'final_masks', out / 'preview.png')
    check_frozen()
    write_json(out / 'complete.json', {'complete': True, **count, 'final_sha256': sha(final),
               'final_optimizer_sha256': sha(moment_path), 'preview_sha256': sha(out / 'preview.png'),
               'baseline_sha256': sha(out / 'baseline.json'), 'final_metrics_sha256': sha(out / 'final_metrics.json'),
               'parent_unchanged': True, 'fit_decision': decision, 'held_out_forward_images': 0,
               'training_fit_only': True, 'promoted': False, 'seconds': time.monotonic() - started,
               'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved()})
    with tarfile.open(archive, 'x:gz') as tar:
        tar.add(out, arcname=out.relative_to(root).as_posix())
        for name in sorted(inventory): tar.add(safe_path(root, name), arcname=name, recursive=False)
        tar.add(root / 'inventory.json', arcname='inventory.json', recursive=False)
    with archive.with_name(archive.name + '.sha256').open('x', encoding='ascii', newline='\n') as stream:
        stream.write(sha(archive) + '  ' + archive.name + '\n')
    print(json.dumps(decision), flush=True); print('Download ' + str(archive), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry_run', action='store_true'); parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--existing'); parser.add_argument('--output')
    main(parser.parse_args())
