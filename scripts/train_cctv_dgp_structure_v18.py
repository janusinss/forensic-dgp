"""Finite existing-L4-only RGB/structure capacity pilot; preserve every failure."""
import argparse
from pathlib import Path
import shutil
import sys
import time

CONTEXT = {'out': None, 'head': None, 'updates': 0, 'backwards': 0, 'autograd_grad_calls': 0}


def train(root, parent, r2, mixed, baseline, pin):
    sys.path.insert(0, str(parent))
    from cctv_dgp_targets_v6 import require_vm
    require_vm(root)  # Before model/output/optimizer/graph construction.
    sys.path.insert(0, str(root))
    import cctv_dgp_structure_v18 as q
    p, v = q.verify(root, parent, r2, mixed, baseline, pin)
    sys.path.insert(0, str(root))
    import numpy as np
    from PIL import Image
    import torch
    from dgp_broader_code_conditioner_v16 import DGPBroaderCodePrior
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from pretrained_face_restoration_portable_v12 import load_face_restorer
    from dgp_structure_conditioner_v18 import DGPStructureResidualHead, prior_features64
    from dgp_structure_objective_v18 import reconstruction
    from cctv_dgp_pilot import FixedObservedIdentity, state_hash, grid112
    from face_prior_grid_v10 import render_grid
    d = p['design']
    torch.set_num_threads(4)
    torch.manual_seed(d['seed'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.monotonic()
    out = root / 'outputs/structure_v18'
    v.require(not out.exists(), 'Preserve prior/partial V18; no resume')
    v.require(shutil.disk_usage(root).free >= d['free_disk_required_bytes'], 'Need4GiB free disk')
    out.mkdir(parents=True)
    CONTEXT['out'] = out
    pp = v.read(parent / 'face_code_fit_protocol_v12.json')
    dgp, dp = load_frozen_dgp_restorer(parent / pp['weights']['dgp'],
        expected_sha256=pp['assets_sha256'][pp['weights']['dgp']], device='cuda')
    prior, cp = load_face_restorer(parent / pp['weights']['prior'], 'cuda')
    core = DGPBroaderCodePrior(dgp.net, prior.net).eval().cuda()
    core.conditioner.load_state_dict(torch.load(root / 'weights/r2_conditioner_epoch8.pth',
                                              map_location='cuda', weights_only=True), strict=True)
    head = DGPStructureResidualHead().cuda().eval()
    CONTEXT['head'] = head
    identity = FixedObservedIdentity(parent / pp['weights']['arcface'], 'cuda')
    frozen = {'dgp': core.core.dgp, 'prior': core.core.prior, 'r2_classifier': core.conditioner,
              'unused_v11': core.core.conditioner, 'recognizer': identity}
    before = {name: state_hash(net) for name, net in frozen.items()}
    initial = state_hash(head)
    v.require(sum(x.numel() for x in head.parameters()) == d['trainable_parameters'], 'New decoder size differs')
    v.require(all(not net.training and not any(x.requires_grad for x in net.parameters())
                  for net in frozen.values()), 'Frozen module enabled')
    counts, handles = {}, []
    for name, net in [('dgp', core.core.dgp), ('prior_encoder', core.core.prior.encoder.blocks[0]),
        ('prior_classifier', core.core.prior.idx_pred_layer), ('r2_head', core.conditioner),
        ('prior_generator', core.core.prior.generator.blocks[0]),
        ('prior_RGB_tail', core.core.prior.generator.blocks[-1]), ('residual_head', head),
        ('recognizer', identity.encoder), ('unused_v11', core.core.conditioner)]:
        counts[name] = 0
        def count(_a, _b, _c, key=name): counts[key] += 1
        handles.append(net.register_forward_hook(count))
    refs = {ref['id']: ref for ref in p['references']}
    cases = {case['id']: case for case in p['training_cases']}
    batches = v.read(root / 'schedule_v18.json')['batches']
    artifacts, data, targets, truth = {}, {}, {}, {}
    deadline = start + d['cache_cap_seconds']

    def clock():
        torch.cuda.synchronize()
        v.require(time.monotonic() <= deadline, 'Finite V18 cache/fit limit exceeded')
        v.require(torch.cuda.max_memory_allocated() <= d['peak_vram_cap_bytes'], 'VRAM exceeds20GiB')

    def tensor(array):
        return torch.from_numpy(array.copy()).cuda()

    def image(array):
        return tensor(array).permute(2, 0, 1).float()[None] / 255

    def save(name, value):
        path = v.safe(out, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.npy'):
            with path.open('xb') as stream: np.save(stream, value, allow_pickle=False)
        else:
            v.require(not path.exists(), 'No output overwrite')
            Image.fromarray(value).save(path)
        artifacts[name] = v.sha(path)
        return name

    def bind(name):
        artifacts[name] = v.sha(out / name)

    def embedding(rgb_image, rid):
        target = targets[rid]
        return identity.embedding(image(rgb_image), tensor(target['mask'])[None, None],
            tensor(target['grid'])[None])[0].cpu().numpy().copy()

    v.write(out / 'execution.json', {'complete': True, 'protocol_sha256': pin, 'design': d,
        'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(0), 'initial_state': initial,
        'frozen_before': before, 'provenance': {'dgp': dp, 'prior': cp},
        'teacher_used': False, 'validation_used': False, 'native_used': False,
        'native_reserved_used': False, 'production_promoted': False})
    bind('execution.json')
    with torch.no_grad():
        for rid, ref in refs.items():
            clock()
            target = v.rgb(mixed / ref['target'])
            with Image.open(mixed / ref['observed']) as mask:
                observed = (np.asarray(mask).copy() > 0).astype(np.float32)
            targets[rid] = {'rgb': target, 'mask': observed, 'grid': grid112(ref['matrix112'])}
            truth[rid] = embedding(target, rid)
            save('target_embeddings/' + rid + '.npy', truth[rid])
    cache_rows, durations, means = {}, [], {source + '/' + profile: [] for source in v.SOURCES for profile in v.PROFILES}
    with torch.no_grad():
        for index, case in enumerate(p['training_cases']):
            clock()
            step = time.monotonic()
            cid, rid = case['id'], case['reference_id']
            camera = v.rgb(mixed / case['input'])
            x = image(camera)
            base, features, logits = core.frozen_inputs(x)
            own_logits = core.conditioner(x, base, features, logits)
            feature64 = prior_features64(core.core.prior, own_logits)
            arrays = {'dgp_base': base[0].cpu().numpy().copy(), 'prior64': feature64[0].cpu().numpy().copy()}
            path = out / ('cache/' + cid + '.npz')
            path.parent.mkdir(exist_ok=True)
            with path.open('xb') as stream: np.savez(stream, **arrays)
            name = path.relative_to(out).as_posix()
            bind(name)
            cache_rows[cid] = {'file': name, 'sha256': artifacts[name], 'bytes': path.stat().st_size,
                'reference_id': rid, 'role': 'train', 'source': case['source'], 'profile': case['profile'],
                'input_sha256': p['data_assets_sha256'][case['input']]}
            data[cid] = {**arrays, 'camera': camera, 'target': targets[rid]}
            mask = targets[rid]['mask'] > 0
            error = arrays['dgp_base'].transpose(1, 2, 0) - targets[rid]['rgb'].astype(np.float32) / 255
            means[case['source'] + '/' + case['profile']].append(float(np.square(error[mask]).mean()))
            clock()
            durations.append(time.monotonic() - step)
            if index + 1 == d['cache_timing_case']:
                elapsed = time.monotonic() - start
                projection = elapsed + (50 - 10) * float(np.mean(durations[1:])) * 1.25 + 30
                v.write(out / 'cache_timing.json', {'cases': 10, 'seconds': elapsed,
                    'steady_sample_seconds': durations[1:], 'projected_seconds': projection,
                    'cap_seconds': d['cache_cap_seconds']})
                bind('cache_timing.json')
                v.require(projection <= d['cache_cap_seconds'], 'V18 cache projection exceeds300s')
            if index == 0 or (index + 1) % 10 == 0:
                print({'cache_cases': index + 1, 'total': 50, 'seconds': time.monotonic() - start}, flush=True)
    cache_seconds = time.monotonic() - start
    v.write(out / 'cache_manifest.json', {'complete': True, 'cases': cache_rows, 'seconds': cache_seconds})
    bind('cache_manifest.json')
    weights = q.normalized_group_weights({key: float(np.mean(values)) for key, values in means.items()})
    v.write(out / 'training_group_weights.json', {'raw_MSE_means': {key: float(np.mean(values)) for key, values in means.items()},
                                               'weights': weights})
    bind('training_group_weights.json')
    fit_start = time.monotonic()
    deadline = fit_start + d['fit_cap_seconds']

    def batch(ids):
        entries = [data[cid] for cid in ids]
        x = torch.cat([image(entry['camera']) for entry in entries])
        base = tensor(np.stack([entry['dgp_base'] for entry in entries]))
        prior64 = tensor(np.stack([entry['prior64'] for entry in entries]))
        target = torch.cat([image(entry['target']['rgb']) for entry in entries])
        mask = tensor(np.stack([entry['target']['mask'] for entry in entries]))[:, None]
        grid = tensor(np.stack([entry['target']['grid'] for entry in entries]))
        vec = tensor(np.stack([truth[cases[cid]['reference_id']] for cid in ids]))
        weight = torch.tensor([weights[cases[cid]['source'] + '/' + cases[cid]['profile']] for cid in ids], device='cuda')
        return x, base, prior64, target, mask, grid, vec, weight

    def components(ids):
        require_vm(root)
        x, base, features, target, mask, grid, vec, weight = batch(ids)
        raw = head(x, base, features)
        recon, pixel, coarse = reconstruction(raw, target, mask, root=root)
        actual = identity.embedding(raw * mask + x * (1 - mask), mask, grid)
        ident = (1 - (actual * vec).sum(1)).clamp_min(0)
        return (recon * weight).mean(), (ident * weight).mean(), raw, {
            'pixel': pixel.mean().item(), 'coarse': coarse.mean().item()}

    head.enable_vm_training(root)
    reconstruction_loss, identity_loss, _, _ = components(batches[0])
    parameters = tuple(head.parameters())
    require_vm(root)
    CONTEXT['autograd_grad_calls'] += 1
    gr = torch.autograd.grad(reconstruction_loss, parameters, retain_graph=True, allow_unused=False)
    require_vm(root)
    CONTEXT['autograd_grad_calls'] += 1
    gi = torch.autograd.grad(identity_loss, parameters, allow_unused=False)
    norm = lambda gradients: float(torch.stack([value.double().square().sum() for value in gradients]).sum().sqrt().item())
    nr, ni = norm(gr), norm(gi)
    weight_identity = q.calibrated_identity_weight(nr, ni)
    v.write(out / 'gradient_calibration.json', {'complete': True, 'case_ids': batches[0],
        'reconstruction_gradient_norm': nr, 'identity_gradient_norm': ni,
        'lambda_identity': weight_identity, 'autograd_grad_calls': 2,
        'method': d['identity_calibration'], 'reconstruction': reconstruction_loss.item(),
        'identity': identity_loss.item(), 'optimizer_updates': 0})
    bind('gradient_calibration.json')
    del gr, gi, reconstruction_loss, identity_loss
    reconstruction_loss, identity_loss, _, _ = components(batches[0])
    require_vm(root)
    CONTEXT['backwards'] += 1
    (reconstruction_loss + weight_identity * identity_loss).backward()
    gradients = [value.grad for value in parameters]
    v.require(all(value is not None and torch.isfinite(value).all() for value in gradients)
              and norm(gradients) > 0, 'New RGB decoder gradient path missing/nonfinite')
    v.require(all(value.grad is None for net in frozen.values() for value in net.parameters()), 'Frozen prior/DGP graph leaked')
    v.write(out / 'gradient_preflight.json', {'complete': True, 'gradient_norm': norm(gradients),
        'parameter_tensors': len(gradients), 'nonzero_gradient_tensors': sum(bool(x.abs().sum()) for x in gradients),
        'optimizer_updates': 0, 'frozen_gradients_absent': True})
    bind('gradient_preflight.json')
    head.zero_grad(set_to_none=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=d['learning_rate'], weight_decay=d['weight_decay'], betas=tuple(d['betas']))
    snapshots, parity, baseline_rows, base_summary = [], [], {}, None

    def fresh(cid, update, expected):
        case = cases[cid]
        x = image(data[cid]['camera'])
        with torch.no_grad():
            base, features, logits = core.frozen_inputs(x)
            own = core.conditioner(x, base, features, logits)
            features64 = prior_features64(core.core.prior, own)
            raw = head(x, base, features64)[0].permute(1, 2, 0).cpu().numpy().copy()
        delta = float(np.max(np.abs(raw - expected)))
        v.require(delta <= d['VM_float_parity_tolerance'], 'Fresh/cache VM pixel parity failed')
        name = save('fresh/update' + str(update) + '/' + cid + '.npy', raw)
        parity.append({'id': cid, 'update': update, 'raw': name, 'maximum_float_difference': delta})

    def snapshot(update):
        nonlocal base_summary
        clock()
        head.eval()
        prefix = 'update' + str(update)
        checkpoint = prefix + '/decoder.pth'
        (out / prefix).mkdir()
        torch.save({key: value.detach().cpu().clone() for key, value in head.state_dict().items()}, out / checkpoint)
        bind(checkpoint)
        rows, ablations, objective_rows = [], [], []
        with torch.no_grad():
            for case in p['training_cases']:
                clock()
                cid, rid = case['id'], case['reference_id']
                x, base, features, target, mask, grid, truth_vec, weight = batch([cid])
                raw_tensor = head(x, base, features)
                if update == 0:
                    v.require(torch.equal(raw_tensor, base), 'Initial DGP raw parity failed')
                raw = raw_tensor[0].permute(1, 2, 0).cpu().numpy().copy()
                camera = data[cid]['camera']
                support = targets[rid]['mask'] > 0
                delivered = v.png(raw, camera, support)
                vec = embedding(delivered, rid)
                recon, _, _ = reconstruction(raw_tensor, target, mask, root=root)
                ident_raw = max(0., 1 - float(vec @ truth[rid]))
                # Fit stopping uses delivered-PNG identity consistently; raw RGB
                # reconstruction is retained separately from exported metrics.
                objective_rows.append(float(weight.item() * (recon.item() + weight_identity * ident_raw)))
                row = {**{key: case[key] for key in ['id', 'reference_id', 'source', 'profile']}, 'role': 'train',
                    'raw': save(prefix + '/' + cid + '.npy', raw),
                    'prediction': save(prefix + '/' + cid + '.png', delivered),
                    'embedding': save(prefix + '/' + cid + '_embedding.npy', vec),
                    **v.metrics(delivered, targets[rid]['rgb'], support),
                    'ArcFace_observed_fixed': float(vec @ truth[rid]),
                    'raw_reconstruction': recon.item(), 'fit_stop_objective': objective_rows[-1]}
                rows.append(row)
                if update == 0:
                    baseline_rows[cid] = row.copy()
                    camera_vec = embedding(camera, rid)
                    save('input_embeddings/' + cid + '.npy', camera_vec)
                if update in [0, 600] and rid in [p['reference_ids'][0], p['reference_ids'][5]] and case['profile'] in ['clear', 'blur_lr24']:
                    fresh(cid, update, raw)
                if update == 600:
                    raw_ablation = head(x, base, features, prior_ablation=True)[0].permute(1, 2, 0).cpu().numpy().copy()
                    png_ablation = v.png(raw_ablation, camera, support)
                    vec_ablation = embedding(png_ablation, rid)
                    ablations.append({**{key: case[key] for key in ['id', 'reference_id', 'source', 'profile']}, 'role': 'train',
                        'raw': save('zero_prior/' + cid + '.npy', raw_ablation),
                        'prediction': save('zero_prior/' + cid + '.png', png_ablation),
                        'embedding': save('zero_prior/' + cid + '_embedding.npy', vec_ablation),
                        **v.metrics(png_ablation, targets[rid]['rgb'], support),
                        'ArcFace_observed_fixed': float(vec_ablation @ truth[rid])})
        summary = v.aggregate(rows)
        if update == 0: base_summary = summary
        report = q.strict_preservation(summary, base_summary)
        name = prefix + '/metrics.json'
        v.write(out / name, {'rows': rows, 'summary': summary, 'preservation': report,
                            'mean_fit_stop_objective': float(np.mean(objective_rows)), 'zero_prior_rows': ablations,
                            'zero_prior_summary': v.aggregate(ablations) if ablations else None})
        bind(name)
        snapshots.append({'update': update, 'metrics': name, 'checkpoint': checkpoint,
                          'state_hash': state_hash(head), 'mean_fit_stop_objective': float(np.mean(objective_rows))})
        print({'snapshot': update, 'training_cases': 50, 'preservation': report}, flush=True)
        clock()

    clock()
    snapshot_start = time.monotonic()
    snapshot(0)
    initial_snapshot_seconds = time.monotonic() - snapshot_start
    head.enable_vm_training(root)
    step_seconds = []
    with (out / 'trace.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        import json
        for update, ids in enumerate(batches, 1):
            clock()
            step_start = time.monotonic()
            require_vm(root)
            optimizer.zero_grad(set_to_none=True)
            recon, ident, _, details = components(ids)
            loss = recon + weight_identity * ident
            v.require(torch.isfinite(loss).item(), 'Nonfinite spatial objective')
            require_vm(root)
            CONTEXT['backwards'] += 1
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), d['gradient_clip_norm'])
            v.require(torch.isfinite(gradient_norm).item() and gradient_norm.item() > 0, 'Gradient vanished/nonfinite')
            require_vm(root)
            optimizer.step()
            CONTEXT['updates'] = update
            clock()
            seconds = time.monotonic() - step_start
            step_seconds.append(seconds)
            record = {'update': update, 'case_ids': ids, 'loss': loss.item(),
                'reconstruction': recon.item(), 'identity': ident.item(), **details,
                'gradient_norm': gradient_norm.item(), 'seconds': seconds}
            stream.write(json.dumps(record, allow_nan=False) + '\n')
            stream.flush()
            if update == d['fit_timing_update']:
                elapsed = time.monotonic() - fit_start
                projection = elapsed + (600 - 20) * float(np.mean(step_seconds[1:])) * 1.25 + 4 * initial_snapshot_seconds * 1.25 + 60
                v.write(out / 'fit_timing.json', {'update': 20, 'seconds': elapsed,
                    'steady_sample_seconds': step_seconds[1:], 'initial_snapshot_seconds': initial_snapshot_seconds,
                    'projected_seconds': projection, 'cap_seconds': d['fit_cap_seconds']})
                bind('fit_timing.json')
                v.require(projection <= d['fit_cap_seconds'], 'Fit projection exceeds900s')
            if update in d['snapshot_updates']:
                snapshot(update)
                if update == d['fit_stop_update']:
                    ratio = snapshots[-1]['mean_fit_stop_objective'] / snapshots[0]['mean_fit_stop_objective']
                    stop = {'update': update, 'full_cohort_loss_ratio': ratio,
                        'minimum_improvement': .01, 'passed': ratio <= .99}
                    v.write(out / 'fitting_stop.json', stop)
                    bind('fitting_stop.json')
                    v.require(stop['passed'], '50-update full-cohort fitting stop failed')
                if update < d['updates']: head.enable_vm_training(root)
            if update == 1 or update % 50 == 0:
                print({'update': update, 'total': 600, 'loss': loss.item(), 'seconds': time.monotonic() - fit_start}, flush=True)
    bind('trace.jsonl')
    head.eval()
    after = {name: state_hash(net) for name, net in frozen.items()}
    v.require(before == after and state_hash(head) != initial, 'Frozen state changed or new decoder did not learn')
    v.require(counts == q.expected_counts(), 'Neural counts differ')
    v.require(CONTEXT['updates'] == 600 and CONTEXT['backwards'] == 601
              and CONTEXT['autograd_grad_calls'] == 2, 'Finite optimizer/graph count differs')
    for handle in handles: handle.remove()
    grids = []
    views = {s['update']: {row['id']: row for row in v.read(out / s['metrics'])['rows']} for s in snapshots}
    ablations = {row['id']: row for row in v.read(out / snapshots[-1]['metrics'])['zero_prior_rows']}
    for profile in v.PROFILES:
        normal_rows, ablation_rows = [], []
        for rid in p['reference_ids']:
            case = next(c for c in p['training_cases'] if c['reference_id'] == rid and c['profile'] == profile)
            cid = case['id']
            base_png = v.rgb(out / views[0][cid]['prediction'])
            normal_rows.append({'id': cid, 'images': [data[cid]['camera'], base_png] +
                [v.rgb(out / views[u][cid]['prediction']) for u in [50, 200, 600]] + [targets[rid]['rgb']]})
            ablation_rows.append({'id': cid, 'images': [data[cid]['camera'], base_png,
                v.rgb(out / views[600][cid]['prediction']), v.rgb(out / ablations[cid]['prediction']), targets[rid]['rgb']]})
        for suffix, labels, rows in [('stages', ['camera', 'retained DGP', 'own update50', 'own update200', 'own update600', 'clean proxy'], normal_rows),
            ('zero_prior', ['camera', 'retained DGP', 'own update600', 'zero-prior sensitivity', 'clean proxy'], ablation_rows)]:
            name = 'grids/' + profile + '_' + suffix + '.png'
            (out / 'grids').mkdir(exist_ok=True)
            render_grid(labels, rows).save(out / name)
            bind(name)
            grids.append(name)
    clock()
    v.write(out / 'neural_execution_receipt.json', {'complete': True, 'counts': counts,
        'frozen_before': before, 'frozen_after': after, 'initial_state': initial,
        'final_state': state_hash(head), 'optimizer_updates': 600, 'backward_calls': 601,
        'autograd_grad_calls': 2, 'total_autograd_traversals': 603,
        'peak_allocated_vram_bytes': torch.cuda.max_memory_allocated()})
    bind('neural_execution_receipt.json')
    v.write(out / 'results.json', {'complete': True, 'protocol_sha256': pin, 'snapshots': snapshots,
        'fresh_parity': parity, 'grids': grids, 'artifacts_sha256': artifacts,
        'optimizer_updates': 600, 'exposures': 6000, 'backward_calls': 601, 'autograd_grad_calls': 2,
        'cache_seconds': cache_seconds, 'fit_seconds': time.monotonic() - fit_start,
        'seconds': time.monotonic() - start, 'teacher_used': False, 'validation_used': False,
        'native_used': False, 'native_reserved_used': False, 'checkpoint_selected': False,
        'production_promoted': False, 'zero_prior_scope': 'Input sensitivity, not a separately trained no-prior control.',
        'limitation': 'Ten photographic training references only. No generalization/native/Zamboanga/identity-recovery or app-readiness claim.'})
    print({'complete': True, 'updates': 600, 'counts': counts, 'seconds': time.monotonic() - start}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'parent', 'r2', 'mixed', 'baseline']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    args = parser.parse_args()
    try:
        train(args.root.resolve(), args.parent.resolve(), args.r2.resolve(), args.mixed.resolve(), args.baseline.resolve(), args.expected_sha)
    except BaseException as error:
        if CONTEXT['out'] is not None and not (CONTEXT['out'] / 'trainer_failure.json').exists():
            import json
            with (CONTEXT['out'] / 'trainer_failure.json').open('x', encoding='utf-8') as stream:
                json.dump({'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                    'optimizer_updates': CONTEXT['updates'], 'backward_calls': CONTEXT['backwards'],
                    'autograd_grad_calls': CONTEXT['autograd_grad_calls'], 'resume_permitted': False}, stream, indent=2)
        raise
