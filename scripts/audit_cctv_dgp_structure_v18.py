"""Saved-output audit and optional frozen CPU decoder replay; no training."""
import argparse
from pathlib import Path
import sys
import time


def raw_reconstruction(raw, target, support):
    """Independent NumPy reference for masked RGB/coarse arithmetic."""
    import numpy as np
    mask = support.astype(np.float32)
    target = target.astype(np.float32) / 255
    mse = float(np.square(raw - target)[support].mean())
    coarse = []
    for kernel in [8, 4, 2]:
        size = 256 // kernel
        coverage = mask.reshape(size, kernel, size, kernel).mean((1, 3))
        a = (raw * mask[..., None]).reshape(size, kernel, size, kernel, 3).mean((1, 3)) / coverage[..., None].clip(1e-6)
        b = (target * mask[..., None]).reshape(size, kernel, size, kernel, 3).mean((1, 3)) / coverage[..., None].clip(1e-6)
        coarse.append(float((np.square(a - b) * coverage[..., None]).sum() / max(1., 3 * coverage.sum())))
    return mse + .5 * float(np.mean(coarse))


def audit(root, parent, r2, mixed, baseline, pin, out, receipt, replay_head=False):
    sys.path.insert(0, str(root))
    import cctv_dgp_structure_v18 as q
    start = time.monotonic()
    p, v = q.verify(root, parent, r2, mixed, baseline, pin)
    sys.path.insert(0, str(root))
    import numpy as np
    from PIL import Image
    import torch
    from dgp_structure_conditioner_v18 import DGPStructureResidualHead
    from cctv_dgp_pilot import state_hash
    torch.set_num_threads(4)
    r = v.read(out / 'results.json')
    v.require(r['complete'] and r['protocol_sha256'] == pin, 'Completed V18 results required')
    for key in ['teacher_used', 'validation_used', 'native_used', 'native_reserved_used',
                'checkpoint_selected', 'production_promoted']:
        v.require(r[key] is False, 'V18 scope differs:' + key)
    v.require(r['optimizer_updates'] == 600 and r['exposures'] == 6000
              and r['backward_calls'] == 601 and r['autograd_grad_calls'] == 2, 'Update/graph scope differs')
    v.require(0 < r['cache_seconds'] <= 300 and 0 < r['fit_seconds'] <= 900
              and 0 < r['seconds'] <= 1200, 'Cache/fit timing differs')
    for name, digest in r['artifacts_sha256'].items():
        v.require(v.sha(v.safe(out, name)) == digest, 'Returned artifact differs:' + name)
    def clock():
        v.require(time.monotonic() - start <= 300, 'V18 independent audit exceeds300s')
    cache = v.read(out / 'cache_manifest.json')
    cases = {c['id']: c for c in p['training_cases']}
    refs = {ref['id']: ref for ref in p['references']}
    v.require(cache['complete'] and set(cache['cases']) == set(cases) and cache['seconds'] == r['cache_seconds'],
              'Full training-only cache differs')
    cached, masks, targets, truth, cameras = {}, {}, {}, {}, {}
    for rid, ref in refs.items():
        with Image.open(mixed / ref['observed']) as image: masks[rid] = np.asarray(image).copy() > 0
        targets[rid] = v.rgb(mixed / ref['target'])
        vec = np.load(out / ('target_embeddings/' + rid + '.npy'), allow_pickle=False)
        v.require(vec.dtype == np.float32 and vec.shape == (512,) and np.isfinite(vec).all()
                  and abs(float(np.linalg.norm(vec)) - 1) <= 2e-6, 'Target embedding invalid')
        truth[rid] = vec
    for cid, item in cache['cases'].items():
        clock()
        case = cases[cid]
        v.require(item['role'] == 'train' and all(item[key] == case[key] for key in
                  ['reference_id', 'source', 'profile']) and item['input_sha256'] == p['data_assets_sha256'][case['input']],
                  'Cache role/input binding differs')
        path = v.safe(out, item['file'])
        v.require(path.stat().st_size == item['bytes'] and v.sha(path) == item['sha256'], 'Cache byte binding differs')
        with np.load(path, allow_pickle=False) as arrays:
            v.require(set(arrays.files) == {'dgp_base', 'prior64'}, 'Cache schema differs')
            value = {key: arrays[key].copy() for key in arrays.files}
        v.require(value['dgp_base'].shape == (3, 256, 256) and value['prior64'].shape == (256, 64, 64)
                  and all(x.dtype == np.float32 and np.isfinite(x).all() for x in value.values())
                  and 0 <= value['dgp_base'].min() <= value['dgp_base'].max() <= 1, 'Cache tensors invalid')
        cached[cid] = value
        cameras[cid] = v.rgb(mixed / case['input'])
    means = {source + '/' + profile: [] for source in v.SOURCES for profile in v.PROFILES}
    for cid, case in cases.items():
        error = cached[cid]['dgp_base'].transpose(1, 2, 0) - targets[case['reference_id']].astype(np.float32) / 255
        means[case['source'] + '/' + case['profile']].append(float(np.square(error[masks[case['reference_id']]]).mean()))
    means = {key: float(np.mean(values)) for key, values in means.items()}
    weighting = v.read(out / 'training_group_weights.json')
    expected_weights = q.normalized_group_weights(means)
    v.require(weighting['raw_MSE_means'] == means and weighting['weights'] == expected_weights, 'Training-only weighting differs')
    calibration = v.read(out / 'gradient_calibration.json')
    lam = q.calibrated_identity_weight(calibration['reconstruction_gradient_norm'], calibration['identity_gradient_norm'])
    v.require(calibration['complete'] and calibration['lambda_identity'] == lam
              and calibration['autograd_grad_calls'] == 2 and calibration['optimizer_updates'] == 0
              and calibration['case_ids'] == v.read(root / 'schedule_v18.json')['batches'][0], 'Gradient calibration receipt differs')
    grad = v.read(out / 'gradient_preflight.json')
    v.require(grad['complete'] and grad['frozen_gradients_absent'] and grad['optimizer_updates'] == 0
              and np.isfinite(grad['gradient_norm']) and grad['gradient_norm'] > 0
              and grad['nonzero_gradient_tensors'] > 0, 'New RGB gradient preflight receipt differs')
    ct, ft = v.read(out / 'cache_timing.json'), v.read(out / 'fit_timing.json')
    v.require(ct['cases'] == 10 and ct['cap_seconds'] == 300 and len(ct['steady_sample_seconds']) == 9
              and ft['update'] == 20 and ft['cap_seconds'] == 900 and len(ft['steady_sample_seconds']) == 19,
              'Finite timing sample counts differ')
    for item, expected in [(ct, ct['seconds'] + 40 * float(np.mean(ct['steady_sample_seconds'])) * 1.25 + 30),
        (ft, ft['seconds'] + 580 * float(np.mean(ft['steady_sample_seconds'])) * 1.25 + 4 * ft['initial_snapshot_seconds'] * 1.25 + 60)]:
        v.require(all(np.isfinite(x) and x > 0 for x in item['steady_sample_seconds'])
                  and abs(item['projected_seconds'] - expected) <= 1e-6
                  and 0 < item['seconds'] <= item['projected_seconds'] <= item['cap_seconds'], 'Timing arithmetic/cap differs')
    import json
    trace = [json.loads(line) for line in (out / 'trace.jsonl').read_text().splitlines()]
    scope = q.verify_trace(p, trace, complete=True)
    neural = v.read(out / 'neural_execution_receipt.json')
    execution = v.read(out / 'execution.json')
    v.require(neural['complete'] and neural['counts'] == q.expected_counts()
              and neural['frozen_before'] == neural['frozen_after'] == execution['frozen_before']
              and neural['initial_state'] == execution['initial_state']
              and neural['optimizer_updates'] == 600 and neural['backward_calls'] == 601
              and neural['autograd_grad_calls'] == 2 and neural['total_autograd_traversals'] == 603
              and 0 < neural['peak_allocated_vram_bytes'] <= 20 * 1024**3, 'Neural/state receipt differs')
    v.require([s['update'] for s in r['snapshots']] == [0, 50, 200, 600], 'Every fixed snapshot required')
    head = DGPStructureResidualHead().eval()
    head_count = [0]
    if replay_head:
        def count(_a, _b, _c): head_count[0] += 1
        head.register_forward_hook(count)
    all_rows, max_replay, changed = {}, 0., {}

    def verify_row(row, prefix, is_ablation=False):
        nonlocal max_replay
        cid, rid = row['id'], row['reference_id']
        case = cases[cid]
        v.require(row['role'] == 'train' and all(row[key] == case[key] for key in
                  ['reference_id', 'source', 'profile']), 'Output cohort/role differs')
        raw = np.load(v.safe(out, row['raw']), allow_pickle=False)
        v.require(row['raw'] == prefix + '/' + cid + '.npy' and row['prediction'] == prefix + '/' + cid + '.png'
                  and raw.shape == (256, 256, 3) and raw.dtype == np.float32 and np.isfinite(raw).all()
                  and 0 <= raw.min() <= raw.max() <= 1, 'Raw/schema/path differs')
        delivered = v.rgb(v.safe(out, row['prediction']))
        v.require(np.array_equal(delivered, v.png(raw, cameras[cid], masks[rid])), 'Raw/PNG supported composition differs')
        for key, value in v.metrics(delivered, targets[rid], masks[rid]).items():
            v.require(row[key] == value if value is None or isinstance(value, bool)
                      else abs(row[key] - value) <= 1e-9, 'PNG metric differs:' + key)
        vec = np.load(out / row['embedding'], allow_pickle=False)
        v.require(vec.shape == (512,) and vec.dtype == np.float32 and np.isfinite(vec).all()
                  and abs(float(np.linalg.norm(vec)) - 1) <= 2e-6
                  and abs(float(vec @ truth[rid]) - row['ArcFace_observed_fixed']) <= 1e-7, 'Embedding arithmetic differs')
        if prefix == 'update0':
            v.require(np.array_equal(raw, cached[cid]['dgp_base'].transpose(1, 2, 0)), 'All50 initial DGP pixels must match')
        if not is_ablation:
            recon = raw_reconstruction(raw, targets[rid], masks[rid])
            ident = max(0., 1 - float(vec @ truth[rid]))
            expected = expected_weights[case['source'] + '/' + case['profile']] * (recon + lam * ident)
            v.require(abs(row['raw_reconstruction'] - recon) <= 1e-6
                      and abs(row['fit_stop_objective'] - expected) <= 2e-6, 'Full-cohort fitting objective differs')
        if replay_head:
            with torch.inference_mode():
                camera = torch.from_numpy(cameras[cid].copy()).permute(2, 0, 1).float()[None] / 255
                prediction = head(camera, torch.from_numpy(cached[cid]['dgp_base'])[None],
                    torch.from_numpy(cached[cid]['prior64'])[None], prior_ablation=is_ablation)
                replay = prediction[0].permute(1, 2, 0).numpy().copy()
            delta = float(np.max(np.abs(replay - raw)))
            max_replay = max(max_replay, delta)
            v.require(delta <= 5e-5, 'CPU cached-head replay exceeds declared5e-5; preserve audit failure')

    initial_state = None
    objectives = {}
    for snapshot in r['snapshots']:
        clock()
        update = snapshot['update']
        state = torch.load(out / snapshot['checkpoint'], map_location='cpu', weights_only=True)
        v.require(all(isinstance(value, torch.Tensor) and torch.isfinite(value).all() for value in state.values()), 'Decoder state invalid')
        head.load_state_dict(state, strict=True)
        v.require(state_hash(head) == snapshot['state_hash'], 'Checkpoint state hash differs')
        if update == 0:
            initial_state = {key: value.clone() for key, value in state.items()}
            v.require(snapshot['state_hash'] == neural['initial_state'], 'Initial zero state differs')
        else:
            changed[update] = sum(not torch.equal(value, initial_state[key]) for key, value in state.items())
            v.require(changed[update] > 0, 'New decoder checkpoint unchanged')
        metrics = v.read(out / snapshot['metrics'])
        rows = metrics['rows']
        v.require([row['id'] for row in rows] == [c['id'] for c in p['training_cases']], 'Snapshot case order differs')
        for row in rows:
            clock()
            verify_row(row, 'update' + str(update))
        summary = v.aggregate(rows)
        v.require(summary == metrics['summary'], 'Snapshot aggregation differs')
        if update == 0: baseline_summary = summary
        v.require(q.strict_preservation(summary, baseline_summary) == metrics['preservation'], 'Unchanged preservation report differs')
        objective = float(np.mean([row['fit_stop_objective'] for row in rows]))
        v.require(objective == metrics['mean_fit_stop_objective'] == snapshot['mean_fit_stop_objective'], 'Fit-stop aggregation differs')
        objectives[update] = objective
        all_rows[update] = {row['id']: row for row in rows}
        ablations = metrics['zero_prior_rows']
        v.require((len(ablations) == 50) if update == 600 else not ablations, 'Zero-prior control scope differs')
        for row in ablations:
            clock()
            verify_row(row, 'zero_prior', True)
        if update == 600:
            v.require(v.aggregate(ablations) == metrics['zero_prior_summary']
                      and state_hash(head) == neural['final_state'], 'Final/zero-prior summary differs')
            final_ablations = {row['id']: row for row in ablations}
    stop = v.read(out / 'fitting_stop.json')
    v.require(stop == {'update': 50, 'full_cohort_loss_ratio': objectives[50] / objectives[0],
                       'minimum_improvement': .01, 'passed': True}
              and stop['full_cohort_loss_ratio'] <= .99, '50-update stop gate differs')
    chosen = {ref['id'] for ref in p['references'] if ref['id'] in [p['reference_ids'][0], p['reference_ids'][5]]}
    expected_parity = {(update, c['id']) for update in [0, 600] for c in p['training_cases']
                       if c['reference_id'] in chosen and c['profile'] in ['clear', 'blur_lr24']}
    v.require({(item['update'], item['id']) for item in r['fresh_parity']} == expected_parity
              and len(r['fresh_parity']) == 8, 'Fresh-image parity scope differs')
    for item in r['fresh_parity']:
        raw = np.load(out / item['raw'], allow_pickle=False)
        old = np.load(out / all_rows[item['update']][item['id']]['raw'], allow_pickle=False)
        delta = float(np.max(np.abs(raw - old)))
        v.require(delta == item['maximum_float_difference'] and delta <= 2e-6, 'Actual fresh/cache pixel parity differs')
    expected_grids = []
    for profile in v.PROFILES:
        for suffix, width in [('stages', 1560), ('zero_prior', 1300)]:
            name = 'grids/' + profile + '_' + suffix + '.png'
            expected_grids.append(name)
            with Image.open(out / name) as grid:
                v.require(grid.mode == 'RGB' and grid.size == (width, 2904), 'Original256-cell grid size differs')
                for i, rid in enumerate(p['reference_ids']):
                    case = next(c for c in p['training_cases'] if c['reference_id'] == rid and c['profile'] == profile)
                    cid = case['id']
                    names = [all_rows[0][cid]['prediction']] + ([all_rows[u][cid]['prediction'] for u in [50, 200, 600]]
                        if suffix == 'stages' else [all_rows[600][cid]['prediction'], final_ablations[cid]['prediction']])
                    cells = [cameras[cid]] + [v.rgb(out / filename) for filename in names] + [targets[rid]]
                    for j, expected in enumerate(cells):
                        x, y = j * 260 + 2, 24 + i * 288 + 28
                        v.require(np.array_equal(np.asarray(grid.crop((x, y, x + 256, y + 256))), expected), 'Grid cell changed/resized')
    v.require(r['grids'] == expected_grids, 'Every predeclared grid required')
    v.require(head_count[0] == (250 if replay_head else 0), 'Local replay neural scope differs')
    clock()
    v.write(receipt, {'complete': True, 'protocol_sha256': pin, 'results_sha256': v.sha(out / 'results.json'),
        'scope': scope, 'raw_PNG_metrics': 250, 'cosines': 250, 'initial_exact_DGP_cases': 50,
        'fresh_VM_parity_cases': 8, 'original_grid_cells': 550, 'checkpoint_changed_tensors': changed,
        'head_forwards': head_count[0], 'maximum_head_replay_difference': max_replay,
        'DGP_prior_recognizer_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'scope_limit': 'Saved arithmetic/provenance and optional cached CPU decoder replay; no CUDA optimizer/gradient or recognizer replay. Capacity uses photographic training data only.',
        'seconds': time.monotonic() - start})
    print({'complete': True, 'head_forwards': head_count[0], 'raw_PNG_metrics': 250, 'seconds': time.monotonic() - start}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'parent', 'r2', 'mixed', 'baseline', 'results', 'receipt']:
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--replay-head', action='store_true')
    args = parser.parse_args()
    audit(args.root.resolve(), args.parent.resolve(), args.r2.resolve(), args.mixed.resolve(), args.baseline.resolve(),
          args.expected_sha, args.results.resolve(), args.receipt.resolve(), args.replay_head)
