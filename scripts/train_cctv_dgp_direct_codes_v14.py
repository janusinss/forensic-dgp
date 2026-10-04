"""VM-only direct code/statistics capacity fit; fixed ten training references."""
import argparse
from pathlib import Path
import sys
import time

CONTEXT = {'out': None, 'model': None, 'updates': 0, 'backwards': 0}


def train(root, parent, expected_sha):
    sys.path.insert(0, str(parent)); sys.path.insert(0, str(root))
    from cctv_dgp_direct_codes_v14 import verify, require_vm, require, read, write, sha, PARENT_RESULTS
    require_vm(root)  # Before output/model construction or any training graph.
    p = verify(root, parent, expected_sha); design = p['design']
    import json
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    from dgp_direct_face_code_v14 import DGPDirectFaceCodePrior, render_codes
    from third_party.codeformer.codeformer_arch import calc_mean_std
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from pretrained_face_restoration_portable_v12 import load_face_restorer
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash, exported_pixel_metrics
    from face_prior_grid_v10 import render_grid

    torch.set_num_threads(4); torch.manual_seed(design['seed'])
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    started = time.monotonic(); deadline = started + design['trainer_cap_seconds']
    out = root / 'outputs/cctv_dgp_direct_codes_v14'
    require(not out.exists(), 'Preserve previous/partial run; no resume')
    out.mkdir(parents=True); CONTEXT['out'] = out
    pp = read(parent / 'face_code_fit_protocol_v12.json')
    source = parent / 'outputs/cctv_dgp_face_code_fit_v12'
    require(sha(source / 'results.json') == PARENT_RESULTS, 'Parent results differ')
    pr = read(source / 'results.json')
    dgp, _ = load_frozen_dgp_restorer(parent / pp['weights']['dgp'],
        expected_sha256=pp['assets_sha256'][pp['weights']['dgp']], device='cuda')
    prior, provenance = load_face_restorer(parent / pp['weights']['prior'], device='cuda')
    model = DGPDirectFaceCodePrior(dgp.net, prior.net).cuda(); CONTEXT['model'] = model
    identity = FixedObservedIdentity(parent / pp['weights']['arcface'], 'cuda')
    frozen = {'dgp': model.core.dgp, 'prior': model.core.prior,
              'unused_v11_conditioner': model.core.conditioner, 'recognizer': identity}
    before = {k: state_hash(v) for k, v in frozen.items()}
    initial = state_hash(model.conditioner)
    require(sum(v.numel() for v in model.conditioner.parameters()) == design['trainable_parameters'], 'Head size differs')
    counts = {}; handles = []; used_parent = {}; artifacts = {}
    for key, module in [('dgp', model.core.dgp), ('direct_conditioner', model.conditioner),
                        ('prior_encoder', model.core.prior.encoder.blocks[0]),
                        ('prior_transformer_head', model.core.prior.idx_pred_layer),
                        ('prior_generator', model.core.prior.generator.blocks[0]),
                        ('unused_v11_conditioner', model.core.conditioner), ('recognizer', identity.encoder)]:
        counts[key] = 0
        def hook(_, __, ___, key=key): counts[key] += 1
        handles.append(module.register_forward_hook(hook))

    def clock():
        torch.cuda.synchronize()
        require(time.monotonic() <= deadline, 'V14 trainer exceeded600 seconds')
        require(torch.cuda.max_memory_allocated() <= design['peak_vram_cap_bytes'], 'V14 VRAM exceeds20 GiB')

    def parent_array(name):
        pin = sha(source / name)
        require(pin == pr['artifacts_sha256'][name], 'Parent cached array changed: ' + name)
        used_parent[name] = pin
        return np.load(source / name, allow_pickle=False)

    def save(name, array):
        (out / name).parent.mkdir(parents=True, exist_ok=True)
        np.save(out / name, array, allow_pickle=False); artifacts[name] = sha(out / name)

    def bind(name): artifacts[name] = sha(out / name)

    def rgb(path):
        with Image.open(path) as im:
            require(im.mode == 'RGB' and im.size == (256, 256), 'Require exact RGB256')
            return np.asarray(im).copy()

    def tensor(image): return torch.from_numpy(image.astype(np.float32) / 255).permute(2, 0, 1)[None].cuda()

    refs = {r['id']: r for r in p['references']}; cases = {c['id']: c for c in p['cases']}
    target = {}; support = {}; token_support = {}; codes = {}; stats = {}; grids = {}; embeddings = {}; cache = {}
    write(out / 'execution.json', {'protocol_sha256': expected_sha, 'torch': torch.__version__,
        'gpu': torch.cuda.get_device_name(0), 'frozen_before': before, 'initial_state': initial,
        'design': design, 'prior_provenance': provenance, 'native_used': False, 'validation_used': False,
        'production_promoted': False})
    with torch.no_grad():
        for ref in p['references']:
            rid = ref['id']; target[rid] = rgb(parent / ref['target'])
            support[rid] = np.asarray(Image.open(parent / ref['observed'])) > 0
            token_support[rid] = torch.from_numpy(support[rid][::16, ::16].reshape(1, 256)).cuda()
            codes[rid] = torch.from_numpy(parent_array('teacher/' + rid + '_codes.npy')).cuda()
            features = torch.from_numpy(parent_array('teacher/' + rid + '_features.npy')).cuda()
            mean, std = calc_mean_std(features); stats[rid] = (mean.flatten(1), std.flatten(1))
            embeddings[rid] = parent_array('teacher/' + rid + '_embedding.npy')
            grids[rid] = torch.from_numpy(grid112(ref['matrix112']))[None].cuda()
        for c in p['cases']:
            clock(); cid = c['id']; image = tensor(rgb(parent / c['input']))
            dgprgb = model.core.dgp(image).clamp(0, 1)
            original = torch.from_numpy(parent_array('update0/code_probes/' + cid + '_features.npy')).cuda()
            logits = torch.from_numpy(parent_array('update0/code_probes/' + cid + '_logits.npy')).cuda()
            cache[cid] = (image, dgprgb, original, logits)
            save('dgp_cache/' + cid + '.npy', dgprgb.cpu().numpy())

    def objective(ids, capture=False):
        batch = [torch.cat([cache[cid][i] for cid in ids], 0) for i in range(4)]
        prediction = model.conditioner(*batch)
        require(all(torch.isfinite(prediction[k]).all() for k in ['logits','mean','logstd','std']), 'Nonfinite prediction')
        require(prediction['logstd'].abs().max() <= 8 and prediction['std'].min() > 0, 'Unsafe predicted statistics range')
        rids = [cases[cid]['reference_id'] for cid in ids]
        labels = torch.cat([codes[r] for r in rids], 0)
        mask = torch.cat([token_support[r] for r in rids], 0)
        target_mean = torch.cat([stats[r][0] for r in rids], 0)
        target_logstd = torch.cat([stats[r][1].log() for r in rids], 0)
        ce_tokens = F.cross_entropy(prediction['logits'].transpose(1, 2), labels, reduction='none')
        ce = ce_tokens[mask].mean()
        mm = (prediction['mean'] - target_mean).square().mean()
        sm = (prediction['logstd'] - target_logstd).square().mean()
        loss = design['lambda_code_ce'] * ce + design['lambda_mean_mse'] * mm + design['lambda_logstd_mse'] * sm
        accuracy = (prediction['logits'].argmax(2) == labels)[mask].float().mean()
        metrics = {'code_ce': ce.item(), 'mean_mse': mm.item(), 'logstd_mse': sm.item(), 'code_accuracy': accuracy.item()}
        return loss, metrics, prediction if capture else None

    steps = read(root / 'schedule_v14.json')['steps']
    model.conditioner.enable_vm_training(root)
    require(all(not v.requires_grad for net in frozen.values() for v in net.parameters()), 'Frozen parameter enabled')
    loss, metrics, _ = objective(steps[0]['case_ids']); loss.backward(); CONTEXT['backwards'] += 1
    norms = {name: None if param.grad is None else param.grad.norm().item()
             for name, param in model.conditioner.named_parameters()}
    require(all(v is not None and np.isfinite(v) for v in norms.values()), 'Missing/nonfinite head gradient')
    require(all(norms[k] > 0 for k in ['code_projection.weight','code_projection.bias',
                                     'stats_projection.weight','stats_projection.bias']), 'Gradient does not reach both heads')
    require(all(v.grad is None for net in frozen.values() for v in net.parameters()), 'Frozen gradient detected')
    require(state_hash(model.conditioner) == initial and {k: state_hash(v) for k, v in frozen.items()} == before,
            'Zero-update preflight changed state')
    for param in model.conditioner.parameters(): param.grad = None
    del loss
    clock(); write(out / 'cuda_preflight.json', {'complete': True, 'metrics': metrics,
        'gradient_norms': norms, 'head_state_unchanged': True, 'frozen_states_unchanged': True,
        'backward_calls': 1, 'optimizer_updates': 0, 'peak_vram_bytes': torch.cuda.max_memory_allocated(),
        'seconds': time.monotonic() - started})
    print('V14 CUDA preflight passed: both heads reached, one backward, zero updates', flush=True)

    require(sha(source/'update0/metrics.json') == pr['artifacts_sha256']['update0/metrics.json'], 'Parent initial metrics differ')
    used_parent['update0/metrics.json'] = pr['artifacts_sha256']['update0/metrics.json']
    parent_initial = {(r['id'], r['fidelity']): r for r in read(source / 'update0/metrics.json')['rows']}
    parity = []; oracle_parity = []
    with torch.inference_mode():
        selected = [next(r for r in p['references'] if r['source'] == s)
                    for s in ['dataset/asian_faces', 'dataset/thumbnails128x128']]
        for ref in selected:
            rid = ref['id']; mean, std = stats[rid]
            plain = render_codes(model.core.prior, codes[rid])
            standardized = render_codes(model.core.prior, codes[rid], mean=mean, std=std)
            delta = (plain - standardized).abs().max().item()
            require(delta <= 2e-6, 'Clean codebook-statistics oracle differs from no-stat oracle')
            for mode, value in [('none', plain), ('target_stats', standardized)]:
                save('oracle_parity/' + rid + '_' + mode + '.npy', value.cpu().numpy())
            oracle_parity.append({'reference_id': rid, 'maximum_float_difference': delta})

    def snapshot(update):
        clock(); prefix = 'update' + str(update); rows = []
        checkpoint = prefix + '/direct_conditioner.pth'
        (out / checkpoint).parent.mkdir(parents=True)
        torch.save({k: v.detach().cpu().clone() for k, v in model.conditioner.state_dict().items()}, out / checkpoint)
        bind(checkpoint)
        with torch.no_grad():
            for c in p['cases']:
                clock(); cid = c['id']; rid = c['reference_id']
                _, metrics, pred = objective([cid], capture=True)
                probe = {}
                for key in ['logits', 'mean', 'std', 'logstd']:
                    name = prefix + '/code_probes/' + cid + '_' + key + '.npy'
                    save(name, pred[key].cpu().numpy()); probe[key] = name
                predicted_codes = pred['logits'].argmax(2)
                for mode in design['statistics_modes']:
                    if mode == 'none':
                        render = render_codes(model.core.prior, predicted_codes)
                    else:
                        m = pred['mean'] if mode == 'predicted' else pred['observed_mean']
                        s = pred['std'] if mode == 'predicted' else pred['observed_std']
                        render = render_codes(model.core.prior, predicted_codes, mean=m, std=s)
                    raw = render[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
                    require(np.isfinite(raw).all() and 0 <= raw.min() <= raw.max() <= 1, 'Invalid raw renderer output')
                    rn = prefix + '/' + mode + '/raw/' + cid + '.npy'; save(rn, raw)
                    camera = rgb(parent / c['input']); png = np.floor(raw * 255).astype(np.uint8)
                    png[~support[rid]] = camera[~support[rid]]
                    name = prefix + '/' + mode + '/images/' + cid + '.png'
                    (out / name).parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(png).save(out / name); bind(name)
                    if update == 0 and mode != 'none':
                        previous = parent_initial[(cid, 0.0)]
                        old_raw = parent_array(previous['raw'])
                        require(sha(source/previous['prediction']) == pr['artifacts_sha256'][previous['prediction']], 'Parent initial PNG differs')
                        used_parent[previous['prediction']] = pr['artifacts_sha256'][previous['prediction']]
                        maximum = float(np.max(np.abs(raw - old_raw)))
                        require(maximum <= 2e-6 and np.array_equal(png, rgb(source / previous['prediction'])),
                                'Starting observed/predicted output differs from cached baseline')
                        parity.append({'case_id': cid, 'mode': mode, 'maximum_float_difference': maximum,
                                       'png_equal': True})
                    mask = torch.from_numpy(support[rid].astype(np.float32))[None, None].cuda()
                    vec = identity.embedding(tensor(png), mask, grids[rid])[0].cpu().numpy()
                    vn = prefix + '/' + mode + '/embeddings/' + cid + '.npy'; save(vn, vec)
                    rows.append({**{k: c[k] for k in ['id','reference_id','source','profile']},
                        'statistics': mode, **metrics, **exported_pixel_metrics(png, target[rid], support[rid]),
                        'ArcFace_observed_fixed': float(np.clip(vec @ embeddings[rid], -1, 1)),
                        'probes': probe, 'prediction': name, 'raw': rn, 'embedding': vn})
        metrics_name = prefix + '/metrics.json'
        write(out / metrics_name, {'update': update, 'rows': rows,
              'checkpoint': checkpoint, 'state_hash': state_hash(model.conditioner)})
        bind(metrics_name)
        print(f'V14 snapshot{update}:150 training-only renders elapsed={time.monotonic()-started:.1f}s', flush=True)
        return {'update': update, 'checkpoint': checkpoint, 'metrics': metrics_name}

    baseline_start = time.monotonic(); snapshots = [snapshot(0)]
    baseline_seconds = time.monotonic() - baseline_start
    optimizer = torch.optim.Adam(model.conditioner.parameters(), lr=design['learning_rate'],
                                 betas=tuple(design['betas']))
    traces = []; training_start = time.monotonic()
    for i, step in enumerate(steps, 1):
        clock(); optimizer.zero_grad(set_to_none=True)
        loss, metrics, _ = objective(step['case_ids']); loss.backward(); CONTEXT['backwards'] += 1
        norm = torch.nn.utils.clip_grad_norm_(model.conditioner.parameters(), design['gradient_clip_norm']).item()
        require(np.isfinite(norm), 'Nonfinite direct-conditioner gradient')
        require(all(v.grad is None for net in frozen.values() for v in net.parameters()), 'Frozen gradient detected')
        optimizer.step(); CONTEXT['updates'] = i
        traces.append({'update': i, 'epoch': step['epoch'], 'case_ids': step['case_ids'],
                       **metrics, 'loss': loss.item(), 'gradient_norm_before_clip': norm,
                       'seconds': time.monotonic() - started})
        with (out / 'update_trace.jsonl').open('a' if i > 1 else 'x', encoding='utf-8', newline='\n') as stream:
            stream.write(json.dumps(traces[-1], allow_nan=False) + '\n')
        del loss
        if i == design['timing_update']:
            elapsed = time.monotonic() - training_start
            projected = time.monotonic() - started + elapsed / i * (design['updates'] - i) + 2 * baseline_seconds + 20
            write(out / 'timing.json', {'update': i, 'training_seconds': elapsed,
                'projected_total_seconds': projected, 'cap_seconds': design['trainer_cap_seconds']})
            require(projected <= design['trainer_cap_seconds'], 'Projected V14 fit exceeds600 seconds')
        if i == 1 or i % 100 == 0:
            print(f'V14 update{i}/1000 CE={metrics["code_ce"]:.4f} accuracy={metrics["code_accuracy"]:.3f} elapsed={time.monotonic()-started:.1f}s', flush=True)
        if i in design['snapshots'][1:]: snapshots.append(snapshot(i))
    clock(); after = {k: state_hash(v) for k, v in frozen.items()}
    require(before == after, 'Frozen state changed')
    require(CONTEXT['updates'] == 1000 and CONTEXT['backwards'] == 1001, 'Actual fitting count differs')
    expected = {'dgp': 50, 'direct_conditioner': 1151, 'prior_encoder': 0,
                'prior_transformer_head': 0, 'prior_generator': 454,
                'unused_v11_conditioner': 0, 'recognizer': 450}
    require(counts == expected, 'Actual forward count differs')
    require(len(parity) == 100 and len(oracle_parity) == 2, 'Parity checks missing')
    require(state_hash(model.conditioner) != initial, 'Direct head did not change')
    write(out / 'used_parent_artifacts.json', used_parent); bind('used_parent_artifacts.json')
    write(out / 'update_trace.json', {'steps': traces}); bind('update_trace.json'); bind('update_trace.jsonl')
    write(out / 'neural_execution_receipt.json', {'complete': True, 'protocol_sha256': expected_sha,
        'frozen_before': before, 'frozen_after': after, 'initial_state': initial,
        'final_state': state_hash(model.conditioner), 'counts': counts, 'expected_counts': expected,
        'backward_calls': 1001, 'optimizer_updates': 1000, 'peak_vram_bytes': torch.cuda.max_memory_allocated(),
        'seconds_before_grids': time.monotonic() - started, 'baseline_parity': parity,
        'oracle_codebook_statistics_parity': oracle_parity})
    bind('neural_execution_receipt.json'); bind('execution.json'); bind('cuda_preflight.json'); bind('timing.json')
    grid_names = []
    for profile in p['profiles']:
        rows = []
        for ref in p['references']:
            c = next(c for c in p['cases'] if c['reference_id'] == ref['id'] and c['profile'] == profile)
            cid = c['id']
            images = [rgb(parent / c['input']), rgb(out / ('update0/observed/images/' + cid + '.png')),
                      rgb(out / ('update300/predicted/images/' + cid + '.png'))]
            images.extend(rgb(out / ('update1000/' + mode + '/images/' + cid + '.png'))
                          for mode in design['statistics_modes'])
            images.append(target[ref['id']]); rows.append({'id': cid, 'images': images})
        name = 'grids/' + profile + '_10_rows.png'; (out / name).parent.mkdir(exist_ok=True)
        render_grid(['camera','initial_observed','direct300_predstats','direct1000_observed',
                     'direct1000_no_stats','direct1000_predstats','clean_training_target'], rows).save(out / name)
        bind(name); grid_names.append(name)
    clock(); verify(root, parent, expected_sha)
    result = {'complete': True, 'protocol_sha256': expected_sha, 'snapshots': snapshots,
        'grids': grid_names, 'artifacts_sha256': artifacts, 'counts': counts,
        'optimizer_updates': 1000, 'backward_calls': 1001, 'training_exposures': 2000,
        'seconds': time.monotonic() - started, 'peak_vram_bytes': torch.cuda.max_memory_allocated(),
        'native_used': False, 'validation_used': False, 'native_reserved_used': False,
        'production_promoted': False, 'checkpoint_selected': False, 'best_checkpoint_created': False,
        'limitation': 'Training-cohort capacity on ten references only; not generalization or established useful CCTV restoration.'}
    write(out / 'results.json', result)
    for handle in handles: handle.remove()
    print({'complete': True, 'updates': 1000, 'seconds': result['seconds'], 'production_promoted': False}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-bundle', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha', required=True)
    args = parser.parse_args()
    try:
        train(args.root.resolve(), args.parent_bundle.resolve(), args.expected_protocol_sha)
    except BaseException as error:
        if CONTEXT['out'] is not None:
            import json
            data = {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                    'optimizer_updates': CONTEXT['updates'], 'backward_calls': CONTEXT['backwards'],
                    'resume_permitted': False}
            if CONTEXT['model'] is not None:
                import torch
                path = CONTEXT['out'] / 'partial_direct_conditioner.pth'
                torch.save({k: v.detach().cpu().clone() for k, v in
                            CONTEXT['model'].conditioner.state_dict().items()}, path)
            (CONTEXT['out'] / 'failure.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
        raise
