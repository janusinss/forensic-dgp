"""Independent returned file/array/metric/gate/full-state and CPU spot audit.

Never imports the VM trainer. Saved TRAIN capacity, native visual usefulness,
final generalization, completion and app qualification remain separate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def image(path, mode='RGB'):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        return np.asarray(im.convert(mode)).copy()


def equal_metrics(a, b):
    assert set(a) == set(b)
    for key in a:
        if key == 'cases' or a[key] is None or isinstance(a[key], bool):
            assert a[key] == b[key]
        else:
            # Recorded float32 dot products can differ by an ULP across CPUs.
            tolerance = 2e-7 if key == 'ArcFace_observed_fixed' else 1e-13
            assert np.isclose(a[key], b[key], rtol=0, atol=tolerance), (key, a[key], b[key])


def grouped(rows, stage):
    result = {}
    for row in rows:
        source, profile = row['source'], row['profile']
        keys = {'all', 'clear' if profile == 'clear' else 'degraded', source+'/all',
                source+('/clear' if profile == 'clear' else '/degraded'), source+'/'+profile}
        for key in keys:
            result.setdefault(key, []).append(row[stage])
    return {key: {'cases': len(items), **{m: float(np.mean([v[m] for v in items])) for m in
        ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE', 'constant_mean_shift_only_MSE']}}
        for key, items in sorted(result.items())}


def gate(before, after):
    # Threshold decisions are recomputed independently and stay at original values.
    failures = []
    for key, old in before.items():
        new = after[key]; assert old['cases'] == new['cases']
        for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
            bad = new[metric] > old[metric]+1e-12 if metric == 'MSE' else new[metric] < old[metric]-1e-6
            if bad:
                failures.append((key, metric))
    gain = 1-after['degraded']['landmark_high_frequency_MSE']/before['degraded']['landmark_high_frequency_MSE']
    source_gains = {key: 1-after[key]['landmark_high_frequency_MSE']/old['landmark_high_frequency_MSE']
                    for key, old in before.items() if key.endswith('/degraded')}
    old, new = before['degraded'], after['degraded']
    brightness = max(0, old['MSE']-new['constant_mean_shift_only_MSE'])/max(old['MSE']-new['MSE'], 1e-12)
    if brightness > .2:
        failures.append(('degraded', 'brightness_gain_fraction'))
    return {'pass': not failures and gain >= .01 and all(v >= 0 for v in source_gains.values()),
            'failures': sorted(failures), 'gain': gain, 'source_gains': source_gains, 'brightness': brightness}


def audit(root, pin, receipt, neural_replay=True):
    stamp = time.monotonic(); root = root.resolve()
    assert not receipt.exists()
    sys.path.insert(0, str(root))
    from cctv_dgp_bank_comparison_v1_contract import verify, read, BUDGETS
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support, exported_pixel_metrics, detail_metric
    from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only
    p = verify(root, pin)
    assert digest(root/'scripts/audit_cctv_dgp_bank_comparison_return_v1.py') == '427d32f9c6a8a7837026c5c9daa93e7c9ed2ece8f74418f63f08c9abcb6cf784', 'Retain the original packet checker'
    manifest = read(root/'export_manifest.json')
    assert manifest['protocol_sha256'] == pin
    assert {f.relative_to(root).as_posix() for f in root.rglob('*') if f.is_file()} == set(manifest['files']) | {'export_manifest.json'}
    for name, expected in manifest['files'].items():
        assert digest(root/name) == expected, name
    assert manifest['uncompressed_bytes'] == sum((root/n).stat().st_size for n in manifest['files'])
    out = root/'outputs/bank_comparison_v1'; cases = p['cases']; refs = {r['id']: r for r in p['references']}
    for r in p['references']:
        a = image(root/r['target'])
        assert hashlib.sha256(a.tobytes()).hexdigest() == p['canonical_target_RGB_sha256'][r['id']]
    saved = {}; replay_cache = {}; array_count = 0; native_count = 0
    labels = [name for name in ['baseline', 'A50', 'B50', 'B50_bank_disabled'] if (out/name/'metrics.json').exists()]
    for label in labels:
        metrics = read(out/label/'metrics.json'); assert metrics['complete'] and metrics['label'] == label
        expected_ids = [c['id'] for c in cases] if label != 'B50_bank_disabled' else [c['id'] for c in cases if c['id'] in p['preview_case_ids']]
        assert [r['id'] for r in metrics['rows']] == expected_ids
        row_map = {r['id']: r for r in metrics['rows']}; rows = []
        for begin in range(0, 3905, 5):
            cs = cases[begin:begin+5]; ids = [c['id'] for c in cs]
            if ids[0] not in row_map:
                continue
            baseline, _ = unpack(out/'baseline/packs'/f'b{begin//5:03d}.npz', ids)
            raw, emb = unpack(out/label/'packs'/f'b{begin//5:03d}.npz', ids, baseline=None if label == 'baseline' else baseline)
            assert np.allclose(np.linalg.norm(emb, axis=2), 1, rtol=0, atol=1e-5)
            for i, c in enumerate(cs):
                source_ref = refs[c['source_person_or_reference']]
                camera, target = image(root/c['input']), image(root/source_ref['target'])
                mask = image(root/source_ref['observed'], 'L') > 0
                support = feature_support(mask, c['landmarks5_canvas_xy'])
                a = raw[i]; assert 0 <= a.min() <= a.max() <= 1
                assert np.array_equal(a[~mask], (camera.astype(np.float32)/np.float32(255))[~mask])
                png = deliver(a, camera, mask)
                mr, mp, shift = mean_only(a, baseline[i], camera, mask)
                rm, pm = pixel_metrics(a, target, mask), exported_pixel_metrics(png, target, mask)
                rm.update({'ArcFace_observed_fixed': float(emb[i, 0] @ emb[i, 2]),
                    'landmark_high_frequency_MSE': detail_float(a, target, support),
                    'constant_mean_shift_only_MSE': pixel_metrics(mr, target, mask)['MSE']})
                pm.update({'ArcFace_observed_fixed': float(emb[i, 1] @ emb[i, 2]),
                    'landmark_high_frequency_MSE': detail_metric(png, target, support),
                    'constant_mean_shift_only_MSE': exported_pixel_metrics(mp, target, mask)['MSE']})
                row = row_map[c['id']]
                assert row['source'] == c['source'] and row['profile'] == c['profile']
                equal_metrics(rm, row['raw']); equal_metrics(pm, row['png'])
                assert np.allclose(shift, row['postclip_mean_RGB_shift'], rtol=0, atol=1e-14)
                assert row['raw_RGB_float32_sha256'] == hashlib.sha256(a.tobytes()).hexdigest()
                assert row['PNG_RGB_sha256'] == hashlib.sha256(png.tobytes()).hexdigest()
                if c['id'] in p['preview_case_ids']:
                    assert np.array_equal(png, image(out/label/'previews'/(c['id']+'.png')))
                rows.append({**row, 'raw': rm, 'png': pm}); array_count += 1
            if begin in [p['schedule'][0][0], p['schedule'][0][1]]:
                replay_cache[(label, begin)] = (raw.copy(), emb.copy())
        for stage in ['raw', 'png']:
            groups = grouped(rows, stage)
            assert set(groups) == set(metrics['groups'][stage])
            for key in groups:
                equal_metrics(groups[key], metrics['groups'][stage][key])
        assert [r['id'] for r in metrics['native_unpaired_rows']] == [c['id'] for c in p['native_development']]
        for c, row in zip(p['native_development'], metrics['native_unpaired_rows']):
            camera = image(root/c['input']); mask = image(root/c['observed'], 'L') > 0
            raw = np.load(out/label/'native'/(c['id']+'.npy'), allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
            assert np.array_equal(raw[~mask], (camera.astype(np.float32)/np.float32(255))[~mask])
            png = deliver(raw, camera, mask)
            assert np.array_equal(png, image(out/label/'native'/(c['id']+'.png')))
            assert row['raw_RGB_float32_sha256'] == hashlib.sha256(raw.tobytes()).hexdigest()
            assert row['PNG_RGB_sha256'] == hashlib.sha256(png.tobytes()).hexdigest()
            native_count += 1
        saved[label] = metrics
    decisions = {}
    for route in ['A', 'B']:
        if (out/('outcome_'+route+'.json')).exists():
            outcome = read(out/('outcome_'+route+'.json'))
            for stage in ['raw', 'png']:
                g = gate(saved['baseline']['groups'][stage], saved[route+'50']['groups'][stage])
                old = outcome['gates'][stage]
                assert g['pass'] == old['pass']
                assert g['failures'] == sorted((f['group'], f['metric']) for f in old['preservation_failures'])
                assert np.isclose(g['gain'], old['relative_feature_gain'], rtol=0, atol=1e-13)
                assert np.isclose(g['brightness'], old['brightness_gain_fraction'], rtol=0, atol=1e-13)
                assert set(g['source_gains']) == set(old['source_feature_gains'])
                assert all(np.isclose(value, old['source_feature_gains'][key], rtol=0, atol=1e-13)
                           for key, value in g['source_gains'].items())
                decisions[route+'_'+stage] = g
            assert outcome['capacity_pass'] == all(decisions[route+'_'+s]['pass'] for s in ['raw', 'png'])
            assert (out/('quality_stop_'+route+'.json')).exists() == (not outcome['capacity_pass'])
    if (out/'results.json').exists():
        result = read(out/'results.json'); assert result['protocol_sha256'] == pin
        updates = sum(r['updates'] for r in result['routes'])
        assert updates == result['counts']['optimizer_updates'] <= 100
        for route_row in result['routes']:
            label = route_row['route']; h = read(out/('history_'+label+'.json'))
            assert [row['update'] for row in h['rows']] == list(range(1, route_row['updates']+1))
            assert [row['reference_starts'] for row in h['rows']] == p['schedule'][:route_row['updates']]
        for key in ['optimizer_updates', 'gradient_queries', 'backwards']:
            assert result['counts'][key] <= BUDGETS['maximum_'+key]
    else:
        assert (out/'failure.json').exists(); result = read(out/'failure.json')
        updates = result['progress']['counts']['optimizer_updates']
    fixture_count = 0
    if (out/'prior_CPU_CUDA_fixture.json').exists():
        q = read(out/'prior_CPU_CUDA_fixture.json')
        assert q['atol'] == q['rtol'] == 1e-4
        for row in q['rows']:
            name = row['name']; filename = 'call0_raw512.npy' if name == 'raw512' else 'seed0_'+name+'.npy'
            a = np.load(out/'prior_cuda_fixture'/filename, allow_pickle=False)
            b = np.load(root/'prior_fixtures'/filename, allow_pickle=False)
            assert np.allclose(a, b, rtol=1e-4, atol=1e-4)
            assert float(np.abs(a-b).max()) == row['maximum_absolute_error']; fixture_count += 1
    neural = []; state_checks = []; replayed = set()
    state_files = sorted(out.glob('state_*/full_state.pth'))
    if state_files:
        import torch
        from cctv_dgp_bank_comparison_v1_model import CorrectedCurrentDGP, ConditionedBankDGP
        from cctv_dgp_generative_bank_v1 import load_prior
        from cctv_dgp_pilot import state_hash, grid112, FixedObservedIdentity
        from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
        torch.set_num_threads(4); torch.set_num_interop_threads(1)
        original, _ = load_frozen_dgp_restorer(root/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'])
        identity = FixedObservedIdentity(root/'weights/w600k_r50.onnx', 'cpu') if neural_replay else None
        for f in state_files:
            state = torch.load(f, map_location='cpu', weights_only=True)
            label = state['route']; assert label in ['A', 'B']
            initial = torch.load(root/('initializers/initial_'+label+'.pth'), map_location='cpu', weights_only=True)
            if label == 'A':
                model = CorrectedCurrentDGP(original.net)
            else:
                prior = load_prior(root/'prior_source', root/'bank_implementation', p['standalone_prior_sha256'])
                model = ConditionedBankDGP(original.net, prior,
                    torch.from_numpy(np.load(root/'initializers/mean_style.npy', allow_pickle=False)))
            model.load_state_dict(state['model'], strict=True); model.eval().requires_grad_(False)
            names = set(model.learning_names())
            assert state_hash({n: v for n, v in state['model'].items() if n not in names}) == state_hash({n: v for n, v in initial.items() if n not in names})
            assert 0 <= state['updates'] <= 50 and all(torch.isfinite(v).all() for v in state['model'].values())
            r = read(f.parent/'receipt.json')
            assert r['model_state'] == state_hash(model) and r['full_state_sha256'] == digest(f)
            if state['optimizer'] is not None:
                assert state['updates'] == state['scheduler']['last_epoch']
                if state['updates']:
                    assert len(state['optimizer']['state']) == len(names)
                    assert all(float(s['step']) == state['updates'] for s in state['optimizer']['state'].values())
            state_checks.append({'path': f.relative_to(root).as_posix(), 'route': label, 'updates': state['updates'],
                                 'frozen_tensors_unchanged': True, 'strict_model_state': True})
            if not neural_replay or state['updates'] != 50 or label+'50' not in saved or label in replayed:
                del model
                continue
            assert state_hash(model) == saved[label+'50']['model_state']
            replayed.add(label)
            with torch.inference_mode():
                for begin in [p['schedule'][0][0], p['schedule'][0][1]]:
                    cs = cases[begin:begin+5]
                    cameras = [image(root/c['input']) for c in cs]
                    masks = [image(root/refs[c['source_person_or_reference']]['observed'], 'L') > 0 for c in cs]
                    x = torch.from_numpy(np.stack(cameras).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2)
                    mm = torch.from_numpy(np.stack(masks))[:, None]
                    pred = torch.where(mm, model(x), x)
                    expected, embeddings = replay_cache[(label+'50', begin)]
                    actual = pred.permute(0, 2, 3, 1).numpy()
                    assert np.allclose(actual, expected, rtol=1e-4, atol=1e-4), 'Limited CPU/CUDA model spot replay differs'
                    targets = [image(root/refs[c['source_person_or_reference']]['target']) for c in cs]
                    grids = torch.from_numpy(np.stack([grid112(refs[c['source_person_or_reference']]['matrix112']) for c in cs]))
                    # Replay the *saved* arrays, not CPU model output, for recognizer parity.
                    raw_tensor = torch.from_numpy(expected).permute(0, 3, 1, 2)
                    pngs = [deliver(a, camera, mask) for a, camera, mask in zip(expected, cameras, masks)]
                    png_tensor = torch.from_numpy(np.stack(pngs).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2)
                    target_tensor = torch.from_numpy(np.stack(targets).astype(np.float32)/np.float32(255)).permute(0, 3, 1, 2)
                    vec = identity.embedding(torch.cat([raw_tensor, png_tensor, target_tensor]), torch.cat([mm.float()]*3), torch.cat([grids]*3)).numpy()
                    rebuilt = np.stack([vec[:5], vec[5:10], vec[10:15]], 1)
                    assert np.allclose(rebuilt, embeddings, rtol=2e-4, atol=2e-4)
                    neural.append({'route': label, 'reference': cs[0]['reference_id'],
                        'model_maximum_absolute_error': float(np.abs(actual-expected).max()),
                        'recognizer_maximum_absolute_error': float(np.abs(rebuilt-embeddings).max())})
            del model
    for name, expected in manifest['files'].items():
        assert digest(root/name) == expected, 'Evidence changed during audit'
    record = {'complete': True, 'protocol_sha256': pin, 'manifest_sha256': digest(root/'export_manifest.json'),
        'checker_sha256': digest(__file__), 'exported_file_bindings': len(manifest['files']),
        'portable_metadata_tolerances': {'float32_embedding_dot': 2e-7, 'float64_RGB_shift': 1e-14},
        'original_thresholds_and_gate_decisions_unchanged_required': True,
        'saved_paired_raw_PNG_cases_checked': array_count, 'native_compositions_checked': native_count,
        'metric_and_gate_decisions': decisions, 'CPU_CUDA_prior_fixtures_rechecked': fixture_count,
        'limited_CPU_replays': neural, 'full_3905_neural_replay_claimed': False,
        'strict_full_state_checks': state_checks,
        'saved_capacity_checks_not_native_or_final_qualification': True, 'optimizer_updates_in_VM': updates,
        'local_optimizer_updates': 0, 'local_gradient_queries': 0, 'local_backwards': 0,
        'native_visual_review_pending': True, 'model_qualified': False, 'app_promotion': False,
        'all_seven_completion_families_pending': True, 'goal_complete': False, 'seconds': time.monotonic()-stamp}
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: record[k] for k in ['complete', 'saved_paired_raw_PNG_cases_checked', 'optimizer_updates_in_VM', 'model_qualified', 'seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True); parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--saved-artifacts-only', action='store_true')
    a = parser.parse_args(); audit(a.root, a.protocol_sha, a.receipt, not a.saved_artifacts_only)
