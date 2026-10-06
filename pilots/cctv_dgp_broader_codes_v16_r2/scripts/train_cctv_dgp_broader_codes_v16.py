"""Streamed, finite reset-code pilot; every graph/backward requires the existing L4."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

CONTEXT = {'out': None, 'model': None, 'updates': 0, 'backwards': 0}


def train(root, parent, mixed, baseline, pin):
    sys.path.insert(0, str(parent)); sys.path.insert(0, str(root))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)  # Before output creation, model construction, optimizer or graph.
    import cctv_dgp_broader_codes_v16 as v
    p = v.verify(root, parent, mixed, baseline, pin); d = p['design']
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from dgp_broader_code_conditioner_v16 import DGPBroaderCodePrior
    from dgp_direct_face_code_v14 import render_codes
    from dgp_face_code_conditioner_v11 import load_teacher, teacher_codes
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from pretrained_face_restoration_portable_v12 import load_face_restorer
    from cctv_dgp_pilot import FixedObservedIdentity, state_hash, grid112
    from face_prior_grid_v10 import render_grid

    torch.set_num_threads(4); torch.manual_seed(d['seed'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic(); cache_deadline = start + d['cache_cap_seconds']
    out = root / 'outputs/broader_codes_v16_r2'
    v.require(not out.exists(), 'Preserve prior/partial V16; no resume')
    v.require(shutil.disk_usage(root).free >= d['free_disk_required_bytes'], 'Need12GiB free disk')
    out.mkdir(parents=True); CONTEXT['out'] = out
    pp = v.read(parent / 'face_code_fit_protocol_v12.json')
    dgp, dp = load_frozen_dgp_restorer(parent / pp['weights']['dgp'],
        expected_sha256=pp['assets_sha256'][pp['weights']['dgp']], device='cuda')
    prior, cp = load_face_restorer(parent / pp['weights']['prior'], device='cuda')
    teacher = load_teacher(parent / pp['weights']['teacher'], pp['assets_sha256'][pp['weights']['teacher']], 'cuda')
    model = DGPBroaderCodePrior(dgp.net, prior.net).cuda().eval(); CONTEXT['model'] = model
    identity = FixedObservedIdentity(parent / pp['weights']['arcface'], 'cuda')
    frozen = {'dgp': model.core.dgp, 'prior': model.core.prior, 'teacher': teacher,
              'unused_v11': model.core.conditioner, 'recognizer': identity}
    before = {k: state_hash(net) for k, net in frozen.items()}
    initial = state_hash(model.conditioner)
    v.require(sum(x.numel() for x in model.conditioner.parameters()) == d['trainable_parameters'], 'Head size differs')
    v.require(all(not m.training and not any(x.requires_grad for x in m.parameters())
                  for m in frozen.values()), 'Frozen component enabled')
    counts = {}; handles = []
    for name, mod in [('dgp', model.core.dgp), ('prior_encoder', model.core.prior.encoder.blocks[0]),
            ('prior_classifier', model.core.prior.idx_pred_layer), ('head', model.conditioner),
            ('prior_generator', model.core.prior.generator.blocks[0]), ('teacher_encoder', teacher.encoder),
            ('teacher_quantizer', teacher.quantize), ('unused_v11', model.core.conditioner),
            ('recognizer', identity.encoder)]:
        counts[name] = 0
        def count(_a, _b, _c, name=name): counts[name] += 1
        handles.append(mod.register_forward_hook(count))
    artifacts = {}; caches = {}; targets = {}; parity = []
    refs = {r['id']: r for r in p['references']}
    cases = {c['id']: c for c in p['training_cases'] + p['validation_cases']}
    br = v.read(baseline / 'results.json'); base_rows = {r['id']: r for r in br['rows']}
    steps = v.read(root / 'schedule_v16.json')['steps']; fit_deadline = None

    def clock():
        torch.cuda.synchronize()
        deadline = cache_deadline if fit_deadline is None else fit_deadline
        v.require(time.monotonic() <= deadline, 'Finite cache/fit cap exceeded')
        v.require(torch.cuda.max_memory_allocated() <= d['peak_vram_cap_bytes'], 'VRAM exceeds20GiB')

    def tensor(image):
        return torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()

    def support(rid):
        with Image.open(mixed / refs[rid]['observed']) as image:
            value = np.asarray(image).copy() > 0
        v.require(value.shape == (256, 256) and value.any(), 'Invalid observation support')
        return value

    def save(name, value):
        path = v.safe(out, name); path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.npy'):
            with path.open('xb') as f: np.save(f, value, allow_pickle=False)
        else:
            v.require(not path.exists(), 'No artifact overwrite'); Image.fromarray(value).save(path)
        artifacts[name] = v.sha(path); return name

    def bind(name): artifacts[name] = v.sha(out / name)

    def embedding(image, rid):
        return identity.embedding(tensor(image), torch.from_numpy(support(rid).astype(np.float32))[None, None].cuda(),
            torch.from_numpy(grid112(refs[rid]['matrix112']))[None].cuda())[0].cpu().numpy().copy()

    def batch(ids):
        # At most ten cases live on the CPU/GPU, independent of the cohort size.
        values = [v.load_cache(out, cid, caches) for cid in ids]
        cameras = [v.rgb(mixed / cases[cid]['input']) for cid in ids]
        images = torch.cat([tensor(image) for image in cameras])
        arrays = [torch.from_numpy(np.stack([x[i] for x in values])).cuda() for i in range(3)]
        return [images] + arrays

    def objective(ids, capture=False):
        logits = model.conditioner(*batch(ids))
        rids = [cases[cid]['reference_id'] for cid in ids]
        v.require(all(refs[rid]['role'] == 'train' for rid in rids), 'Teacher/CE cannot consume validation')
        labels = torch.from_numpy(np.stack([targets[rid] for rid in rids])).cuda()
        mask = torch.from_numpy(np.stack([support(rid)[::16, ::16].reshape(256) for rid in rids])).cuda()
        tokens = F.cross_entropy(logits.transpose(1, 2), labels, reduction='none')
        loss = tokens[mask].mean(); accuracy = (logits.argmax(2) == labels)[mask].float().mean()
        v.require(torch.isfinite(loss).item() and torch.isfinite(logits).all().item(), 'Nonfinite code objective')
        return loss, {'code_ce': loss.item(), 'code_accuracy': accuracy.item()}, logits.detach() if capture else None

    v.write(out / 'execution.json', {'protocol_sha256': pin, 'design': d, 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0), 'frozen_before': before, 'initial_state': initial,
        'provenance': {'dgp': dp, 'prior': cp}, 'teacher_roles': ['train'],
        'native_used': False, 'native_reserved_used': False, 'production_promoted': False})
    from cctv_dgp_cache_timing_v16_r2 import cache_order, projection, METHOD
    clock(); initialization_seconds = time.monotonic() - start
    v.write(out / 'cache_initialization.json', {'seconds': initialization_seconds, 'initial_state': initial, 'method': METHOD})
    reference_timings = []; warmed = set()
    with torch.no_grad():
        for index, ref in enumerate(cache_order(p), 1):
            reference_start = time.monotonic()
            clock(); rid = ref['id']
            if ref['role'] == 'train':
                clean = tensor(v.rgb(mixed / ref['target']))
                codes, _ = teacher_codes(teacher, clean)
                targets[rid] = codes[0].cpu().numpy().copy().astype(np.int64)
                save('teacher/' + rid + '_codes.npy', targets[rid])
                own_cases = [c for c in p['training_cases'] if c['reference_id'] == rid]
            else:
                own_cases = [c for c in p['validation_cases'] if c['reference_id'] == rid]
            chunk = d['cache_training_batch_size'] if ref['role'] == 'train' else d['cache_validation_batch_size']
            for offset in range(0, len(own_cases), chunk):
                clock(); chosen = own_cases[offset:offset + chunk]
                x = torch.cat([tensor(v.rgb(mixed / c['input'])) for c in chosen])
                dgprgb, features, logits = model.frozen_inputs(x)
                zero = model.conditioner(x, dgprgb, features, logits)
                v.require(torch.equal(zero, logits), 'Reset head must exactly preserve starting logits')
                arrays = [y.cpu().numpy() for y in [dgprgb, features, logits]]
                for j, c in enumerate(chosen):
                    values = [y[j].copy().astype(np.float32) for y in arrays]; v.cache_arrays(*values)
                    path = v.cache_path(out, c['id']); path.parent.mkdir(exist_ok=True)
                    with path.open('xb') as stream:
                        np.savez(stream, dgp=values[0], features=values[1], logits=values[2])
                    caches[c['id']] = {'sha256': v.sha(path), 'bytes': path.stat().st_size,
                        'reference_id': rid, 'role': ref['role'], 'source': ref['source'], 'profile': c['profile'],
                        'input_sha256': p['data_assets_sha256'][c['input']], 'zero_logits_equal': True}
            clock()
            if index <= d['cache_timing_at_reference']:
                key = (ref['source'], ref['role'])
                reference_timings.append({'id': rid, 'source': ref['source'], 'role': ref['role'],
                    'cases': len(own_cases), 'seconds': time.monotonic() - reference_start, 'warmup': key not in warmed})
                warmed.add(key)
            if index == d['cache_timing_at_reference']:
                timing = projection(p, reference_timings, time.monotonic() - start, initialization_seconds)
                v.write(out / 'cache_timing.json', timing)
                print({'cache_projection_seconds': timing['projected_seconds'], 'initialization_seconds': initialization_seconds,
                    'cap_seconds': timing['cap_seconds']}, flush=True)
                v.require(timing['passed'], 'Cache projection exceeds900s')
            if index == 1 or index % 100 == 0:
                print(f'V16 r2 cache references {index}/885 cases {len(caches)}/4425', flush=True)
    clock()
    v.require(len(caches) == 4425 and set(targets) == {r['id'] for r in refs.values() if r['role'] == 'train'},
              'Cache/teacher cohort differs')
    v.write(out / 'cache_manifest.json', {'complete': True, 'protocol_sha256': pin, 'cases': caches,
        'teacher_ids': sorted(targets), 'float_dtype': 'float32', 'cache_bytes': sum(x['bytes'] for x in caches.values()),
        'seconds': time.monotonic() - start, 'zero_logits_parity_cases': 4425})
    bind('cache_manifest.json'); bind('cache_timing.json'); bind('cache_initialization.json'); bind('execution.json')
    cache_seconds = time.monotonic() - start; fit_start = time.monotonic()
    fit_deadline = fit_start + d['fit_cap_seconds']
    train_preview = [c for c in p['training_cases'] if c['reference_id'] in p['train_preview_reference_ids']]
    for rid in p['train_preview_reference_ids']:
        with torch.no_grad(): vec = embedding(v.rgb(mixed / refs[rid]['target']), rid)
        save('target_embeddings/' + rid + '.npy', vec)

    def snapshot(epoch):
        clock(); prefix = 'epoch' + str(epoch); rows = []
        checkpoint = prefix + '/conditioner.pth'; path = out / checkpoint; path.parent.mkdir(parents=True)
        torch.save({k: x.detach().cpu().clone() for k, x in model.conditioner.state_dict().items()}, path); bind(checkpoint)
        with torch.no_grad():
            for c in train_preview + p['validation_cases']:
                clock(); cid = c['id']; rid = c['reference_id']; ref = refs[rid]; mask = support(rid)
                camera = v.rgb(mixed / c['input']); target = v.rgb(mixed / ref['target'])
                is_train = ref['role'] == 'train'
                if is_train:
                    _, code_stats, logits = objective([cid], capture=True)
                else:
                    logits = model.conditioner(*batch([cid])); code_stats = {}
                v.require(torch.isfinite(logits).all().item(), 'Nonfinite snapshot logits')
                raw = render_codes(model.core.prior, logits.argmax(2))[0].permute(1, 2, 0).cpu().numpy().copy()
                image = v.png(raw, camera, mask); vec = embedding(image, rid)
                preview = rid in p[('train' if is_train else 'validation') + '_preview_reference_ids']
                if is_train:
                    truth = np.load(out / ('target_embeddings/' + rid + '.npy'), allow_pickle=False)
                else:
                    truth = np.load(baseline / ('target_embeddings/' + rid + '.npy'), allow_pickle=False)
                item = {**{k: c[k] for k in ['id', 'reference_id', 'source', 'profile']}, 'role': ref['role'],
                    **v.metrics(image, target, mask), **code_stats,
                    'ArcFace_observed_fixed': float(np.clip(vec @ truth, -1, 1)),
                    'prediction': save(prefix + '/images/' + cid + '.png', image),
                    'embedding': save(prefix + '/embeddings/' + cid + '.npy', vec)}
                if preview:
                    item['raw'] = save(prefix + '/raw/' + cid + '.npy', raw)
                    item['logits'] = save(prefix + '/logits/' + cid + '.npy', logits[0].cpu().numpy().copy())
                if is_train and epoch == 0:
                    value = v.load_cache(out, cid, caches)[0].transpose(1, 2, 0).copy()
                    item['dgp_preview'] = save('train_dgp/' + cid + '.png', v.png(value, camera, mask))
                    item['dgp_raw'] = save('train_dgp_raw/' + cid + '.npy', value)
                if epoch == 0 and not is_train:
                    old = base_rows[cid]['arms']['starting_prior_none']
                    v.require(np.array_equal(image, v.rgb(baseline / old['prediction'])), 'Starting validation PNG differs from V15')
                    if 'raw' in old:
                        delta = float(np.max(np.abs(raw - np.load(baseline / old['raw'], allow_pickle=False))))
                        v.require(delta <= d['float_parity_tolerance'], 'Starting validation raw differs from V15')
                    item['baseline_png_equal'] = True
                if epoch in d['fresh_parity_epochs'] and not is_train and preview:
                    fresh_inputs = model.frozen_inputs(tensor(camera))
                    fresh_logits = model.conditioner(tensor(camera), *fresh_inputs)
                    fresh_raw = render_codes(model.core.prior, fresh_logits.argmax(2))[0].permute(1, 2, 0).cpu().numpy().copy()
                    delta = float(np.max(np.abs(fresh_raw - raw)))
                    v.require(delta <= d['float_parity_tolerance'] and
                              np.array_equal(v.png(fresh_raw, camera, mask), image), 'Fresh/cache inference parity failed')
                    parity.append({'epoch': epoch, 'id': cid, 'maximum_float_difference': delta, 'png_equal': True,
                        'raw': save(prefix + '/fresh_raw/' + cid + '.npy', fresh_raw)})
                rows.append(item)
        validations = [r for r in rows if r['role'] == 'validation']
        summaries = {a: br['summaries'][a] for a in ['retained_dgp_v2', 'starting_prior_none', 'input']}
        summaries['trained_conditioner_none'] = v.aggregate(validations)
        train_rows = [r for r in rows if r['role'] == 'train']
        data = {'epoch': epoch, 'update': epoch * 391, 'rows': rows, 'checkpoint': checkpoint,
                'state_hash': state_hash(model.conditioner), 'summaries': summaries, 'guard_report': v.guard_report(summaries),
                'training_preview': {k: float(np.mean([r[k] for r in train_rows])) for k in ['code_ce', 'code_accuracy']}}
        name = prefix + '/metrics.json'; v.write(out / name, data); bind(name)
        print(f'V16 snapshot epoch{epoch}:50 training previews +520 validation outputs', flush=True)
        return {'epoch': epoch, 'metrics': name, 'checkpoint': checkpoint, 'state_hash': data['state_hash']}

    snap_start = time.monotonic(); snapshots = [snapshot(0)]; snapshot_seconds = time.monotonic() - snap_start
    model.conditioner.enable_vm_training(root)
    loss, pre_metrics, pred = objective(steps[0]['case_ids'], capture=True)
    save('preflight_logits.npy', pred.cpu().numpy().copy())
    loss.backward(); CONTEXT['backwards'] += 1
    norms = {k: None if x.grad is None else x.grad.norm().item() for k, x in model.conditioner.named_parameters()}
    v.require(all(x is not None and np.isfinite(x) for x in norms.values()) and
              norms['code_projection.weight'] > 0 and norms['code_projection.bias'] > 0, 'Invalid head gradient')
    v.require(all(x.grad is None for m in frozen.values() for x in m.parameters()), 'Frozen gradient detected')
    v.require(state_hash(model.conditioner) == initial and {k: state_hash(m) for k, m in frozen.items()} == before,
              'Zero-update backward preflight changed weights')
    for x in model.conditioner.parameters(): x.grad = None
    del loss, pred
    v.write(out / 'cuda_preflight.json', {'complete': True, 'optimizer_updates': 0, 'backward_calls': 1,
        'metrics': pre_metrics, 'gradient_norms': norms, 'state_hash': initial, 'case_ids': steps[0]['case_ids']})
    bind('cuda_preflight.json'); clock()
    optimizer = torch.optim.AdamW(model.conditioner.parameters(), lr=d['learning_rate'],
        weight_decay=d['weight_decay'], betas=tuple(d['betas']))
    update_start = time.monotonic()
    with (out / 'update_trace.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for step in steps:
            clock(); optimizer.zero_grad(set_to_none=True)
            loss, stats, _ = objective(step['case_ids']); loss.backward(); CONTEXT['backwards'] += 1
            norm = torch.nn.utils.clip_grad_norm_(model.conditioner.parameters(), d['gradient_clip_norm']).item()
            v.require(np.isfinite(norm) and all(x.grad is None for m in frozen.values() for x in m.parameters()),
                      'Nonfinite/frozen gradient')
            optimizer.step(); CONTEXT['updates'] = step['update']
            row = {**step, **stats, 'loss': loss.item(), 'gradient_norm_before_clip': norm, 'seconds': time.monotonic() - fit_start}
            stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush(); del loss
            if step['update'] == d['timing_update']:
                elapsed = time.monotonic() - update_start
                projected = time.monotonic() - fit_start + elapsed / step['update'] * (d['updates'] - step['update']) * d['timing_safety_factor'] + 2 * snapshot_seconds + 45
                v.write(out / 'fit_timing.json', {'update': step['update'], 'seconds': elapsed,
                    'projected_seconds': projected, 'cap_seconds': d['fit_cap_seconds']})
                v.require(projected <= d['fit_cap_seconds'], 'Fit projection exceeds1200s')
            if step['update'] == 1 or step['update'] % 100 == 0:
                print(f'V16 r2 update {step["update"]}/3128 CE={stats["code_ce"]:.4f} accuracy={stats["code_accuracy"]:.3f}', flush=True)
            if step['update'] in [1564, 3128]:
                snapshots.append(snapshot(step['epoch']))
                if step['epoch'] == 4:
                    old = v.read(out / snapshots[0]['metrics'])['training_preview']['code_ce']
                    new = v.read(out / snapshots[-1]['metrics'])['training_preview']['code_ce']
                    improvement = (old - new) / old
                    v.write(out / 'epoch4_fit_stop.json', {'relative_code_ce_improvement': improvement,
                        'required': d['minimum_fit_ce_improvement_epoch4'], 'passed': improvement >= d['minimum_fit_ce_improvement_epoch4']})
                    v.require(improvement >= d['minimum_fit_ce_improvement_epoch4'], 'Training-preview fitting fails epoch4 stop rule')
    bind('update_trace.jsonl'); bind('fit_timing.json'); bind('epoch4_fit_stop.json')
    grids = []; byepoch = {s['epoch']: v.read(out / s['metrics']) for s in snapshots}
    rowmaps = {e: {r['id']: r for r in m['rows']} for e, m in byepoch.items()}
    for role in ['train', 'validation']:
        selected = train_preview if role == 'train' else p['validation_cases']
        for profile in v.PROFILES:
            cells = []
            for rid in p[role + '_preview_reference_ids']:
                c = next(c for c in selected if c['reference_id'] == rid and c['profile'] == profile); cid = c['id']
                retained = out / rowmaps[0][cid]['dgp_preview'] if role == 'train' else baseline / base_rows[cid]['arms']['retained_dgp_v2']['prediction']
                images = [v.rgb(mixed / c['input']), v.rgb(retained)] + [v.rgb(out / rowmaps[e][cid]['prediction']) for e in [0, 4, 8]] + [v.rgb(mixed / refs[rid]['target'])]
                cells.append({'id': cid, 'images': images})
            name = f'grids/{role}_{profile}_10_rows.png'; (out / name).parent.mkdir(exist_ok=True)
            render_grid(['camera', 'retained DGP', 'starting prior', 'our epoch4', 'our epoch8', 'clean reference'], cells).save(out / name)
            bind(name); grids.append(name)
    clock(); after = {k: state_hash(m) for k, m in frozen.items()}
    expected = {'dgp': 1401, 'prior_encoder': 1401, 'prior_classifier': 1401, 'head': 6240,
                'prior_generator': 1810, 'teacher_encoder': 781, 'teacher_quantizer': 781, 'unused_v11': 0, 'recognizer': 1720}
    v.require(counts == expected and before == after and state_hash(model.conditioner) != initial,
              'Frozen state/count/change proof differs')
    v.require(CONTEXT['updates'] == 3128 and CONTEXT['backwards'] == 3129 and len(parity) == 100, 'Execution count differs')
    v.write(out / 'fresh_image_parity.json', {'complete': True, 'cases': parity}); bind('fresh_image_parity.json')
    v.write(out / 'neural_execution_receipt.json', {'complete': True, 'protocol_sha256': pin,
        'frozen_before': before, 'frozen_after': after, 'initial_state': initial,
        'final_state': state_hash(model.conditioner), 'counts': counts, 'expected_counts': expected,
        'backward_calls': CONTEXT['backwards'], 'optimizer_updates': CONTEXT['updates'],
        'peak_vram_bytes': torch.cuda.max_memory_allocated()}); bind('neural_execution_receipt.json')
    v.verify(root, parent, mixed, baseline, pin); clock()
    v.write(out / 'results.json', {'complete': True, 'protocol_sha256': pin, 'snapshots': snapshots,
        'artifacts_sha256': artifacts, 'grids': grids, 'cache_seconds': cache_seconds,
        'fit_seconds': time.monotonic() - fit_start, 'seconds': time.monotonic() - start,
        'optimizer_updates': CONTEXT['updates'], 'backward_calls': CONTEXT['backwards'], 'training_exposures': 31280,
        'native_used': False, 'native_reserved_used': False, 'production_promoted': False,
        'checkpoint_selected': False, 'best_checkpoint_created': False,
        'limitation': 'Engineering component diagnostic on development photographic proxies, not native CCTV truth, blind final review or Zamboanga evidence. Pretrained prior declared; no output path adoption.'})
    for handle in handles: handle.remove()
    print({'complete': True, 'updates': 3128, 'seconds': time.monotonic() - start}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'parent', 'mixed', 'baseline']: parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True); args = parser.parse_args()
    try:
        train(args.root.resolve(), args.parent.resolve(), args.mixed.resolve(), args.baseline.resolve(), args.expected_sha)
    except BaseException as error:
        if CONTEXT['out'] is not None:
            import cctv_dgp_broader_codes_v16 as v
            if CONTEXT['model'] is not None:
                import torch
                torch.save({k: x.detach().cpu().clone() for k, x in CONTEXT['model'].conditioner.state_dict().items()},
                           CONTEXT['out'] / 'partial_conditioner.pth')
            v.write(CONTEXT['out'] / 'failure.json', {'complete': False, 'error_type': type(error).__name__,
                'error': str(error), 'optimizer_updates': CONTEXT['updates'], 'backward_calls': CONTEXT['backwards'], 'resume_permitted': False})
        raise
