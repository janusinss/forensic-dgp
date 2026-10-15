"""Independent saved-array return checks. Never builds a neural candidate."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time

from cctv_dgp_spatial_readout_v1_contract import read, write, sha, verify_transfer, STEM, PARENT_NAME


def safe_member(member):
    name = member.name
    assert member.isreg() and name and ':' not in name and '\\' not in name
    assert not PurePosixPath(name).is_absolute() and '..' not in PurePosixPath(name).parts
    assert str(PurePosixPath(name)) == name and member.size >= 0


def independent_gate(before, after):
    assert len(before) == len(after) == 17 and before.keys() == after.keys()
    failures = []
    for key in sorted(before):
        old, new = before[key], after[key]
        assert old['cases'] == new['cases']
        for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
            a, b = new[metric], old[metric]
            assert math.isfinite(a) and math.isfinite(b)
            if (metric == 'MSE' and a > b+1e-12) or (metric != 'MSE' and a < b-1e-6):
                failures.append(dict(group=key, metric=metric, baseline=b, candidate=a))
    old, new = before['degraded'], after['degraded']
    gain = 1-new['landmark_high_frequency_MSE']/old['landmark_high_frequency_MSE']
    source_gains = {key: 1-after[key]['landmark_high_frequency_MSE']/before[key]['landmark_high_frequency_MSE']
                    for key in before if key.endswith('/degraded')}
    assert len(source_gains) == 2
    brightness = max(0., old['MSE']-new['constant_mean_shift_only_MSE'])/max(old['MSE']-new['MSE'], 1e-12)
    if brightness > .2:
        failures.append(dict(group='degraded', metric='brightness_gain_fraction', candidate=brightness, maximum=.2))
    return dict(**{'pass': not failures and gain >= .01 and all(v >= .01 for v in source_gains.values())},
        global_structure_gain=gain, source_structure_gains=source_gains,
        brightness_gain_fraction=brightness, preservation_failures=failures)


def grouped(rows, stage):
    import numpy as np
    values = {}
    for row in rows:
        source, profile = row['source'], row['profile']
        for key in {'all', 'clear' if profile == 'clear' else 'degraded', source+'/all',
                    source+('/clear' if profile == 'clear' else '/degraded'), source+'/'+profile}:
            values.setdefault(key, []).append(row[stage])
    metrics = ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE', 'constant_mean_shift_only_MSE']
    assert len(values) == 17
    return {key: dict(cases=len(items), **{m: float(np.mean([v[m] for v in items])) for m in metrics})
            for key, items in sorted(values.items())}


def state_hash(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous()
        result.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0')
        result.update(str(tuple(a.shape)).encode()+b'\0'+a.numpy().tobytes())
    return result.hexdigest()


def import_archive(archive, sidecar, exported, target, expected_pin):
    assert not target.exists(), 'Retain existing return; do not overwrite it'
    digest = sha(archive)
    words = sidecar.read_text(encoding='utf-8-sig').strip().split()
    record = read(exported)
    assert words == [digest, archive.name] and record['archive_sha256'] == digest
    assert record['complete'] is True and record['bytes'] == archive.stat().st_size
    assert record['protocol_sha256'] == expected_pin and record['training_success_not_implied'] is True
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        for member in members:
            safe_member(member)
        names = [m.name for m in members]
        assert len(names) == len(set(names)) and sum(m.size for m in members) <= 3*1024**3+16*1024**2
        manifest = json.load(stream.extractfile('result_manifest.json'))
        assert set(names) == set(manifest['files']) | {'result_manifest.json'}
        assert manifest['protocol_sha256'] == expected_pin
        target.mkdir()
        for member in members:
            path = target/member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            assert path.resolve().is_relative_to(target.resolve())
            with path.open('xb') as destination:
                shutil.copyfileobj(stream.extractfile(member), destination)
            if member.name != 'result_manifest.json':
                bound = manifest['files'][member.name]
                assert path.stat().st_size == bound['bytes'] and sha(path) == bound['sha256']
    return dict(complete=True, archive_sha256=digest, files=len(members),
                protocol_sha256=expected_pin, no_model_or_training_calls=True)


def audit_saved(root, parent, p, old, receipt):
    start = time.monotonic()
    import numpy as np
    import torch  # read-only tensor deserialization, no model constructors
    from PIL import Image
    sys.path.insert(1, str(parent))
    from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only
    from frozen_capacity_contract import feature_support, exported_pixel_metrics, detail_metric
    from cctv_dgp_head4_lossless_v1 import unpack
    out = root/'outputs/spatial_readout_v1'
    refs = {r['id']: r for r in old['references']}
    cases = {c['id']: c for c in old['cases']}
    if not (out/'run_results.json').exists():
        assert (out/'failure.json').exists(), 'Missing completed result and failure evidence'
        record = read(out/'failure.json')
        write(receipt, dict(complete=True, partial_failure_evidence_only=True, failure=record,
            returned_files={path.relative_to(root).as_posix(): sha(path) for path in out.rglob('*') if path.is_file()},
            neural_calls=0, optimizer_updates=0, model_qualified=False, visual_review_pending=True))
        return
    result = read(out/'run_results.json')
    assert result['complete'] and result['full_corpus_epochs'] == 0 and not result['model_qualified']
    assert result['counts']['optimizer_updates'] == 100 and result['counts']['backwards'] == 100
    assert result['counts']['gradient_queries'] == 4
    for key, value in result['counts'].items():
        cap = p['budgets'].get('maximum_'+key, p['budgets'].get(key))
        assert value <= cap
    environment = read(out/'environment.json')
    assert environment['GPU'] == 'NVIDIA L4' and environment['packages']['torch'] == '2.9.1+cu129'
    assert environment['original_state'] == p['original_state'] and environment['recognizer_state'] == p['recognizer_state']
    provider = torch.load(out/'frozen_feature_provider.pth', map_location='cpu', weights_only=True)
    original_provider = torch.load(parent/'initializers/initial_B.pth', map_location='cpu', weights_only=True)
    assert state_hash(provider) == state_hash(original_provider) == p['feature_provider_initial_state']
    bounds = read(out/'distance_bound.json')
    common = .001*min(bounds['norms_A_B_historical_conditioning'])
    assert abs(common-bounds['common_radius']) < 1e-12
    reductions, cells, checked = {}, {}, 0

    def pixels(path, mode='RGB'):
        with Image.open(path) as image:
            assert image.size == (256, 256)
            return np.asarray(image.convert(mode)).copy()
    for label in ['baseline', 'A0', 'B0', 'A50', 'B50']:
        rows = []
        saved_metrics = read(out/label/'metrics.json')
        saved_rows = {r['case']: r for r in saved_metrics['rows']}
        assert set(saved_rows) == set(p['case_ids'])
        for index, block in enumerate(p['reference_blocks']):
            ref = refs[block['reference']]
            retained, retained_embeddings = unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz", block['case_ids'])
            with np.load(out/label/'packs'/f'r{index:02d}.npz', allow_pickle=False) as archive:
                assert set(archive.files) == {'ids', 'raw', 'embeddings_raw', 'embeddings_png', 'target_embeddings'}
                assert archive['ids'].tolist() == block['case_ids']
                raw, er, ep, target_vec = [archive[k].copy() for k in ['raw', 'embeddings_raw', 'embeddings_png', 'target_embeddings']]
            assert raw.shape == (5, 256, 256, 3) and raw.dtype == np.float32 and np.isfinite(raw).all()
            assert raw.min() >= 0 and raw.max() <= 1
            assert all(v.dtype == np.float32 and v.shape == (15, 512) and np.isfinite(v).all() for v in [er, ep])
            assert np.array_equal(target_vec, retained_embeddings[:, 2])
            for vectors, slot in [(er, 0), (ep, 1)]:
                np.testing.assert_allclose(vectors[5:10], retained_embeddings[:, slot], rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(vectors[10:15], retained_embeddings[:, 2], rtol=1e-6, atol=1e-6)
            if label in ['baseline', 'A0', 'B0']:
                assert np.array_equal(raw, retained)
            for i, case_id in enumerate(block['case_ids']):
                c = cases[case_id]
                camera, target = pixels(parent/c['input']), pixels(parent/ref['target'])
                mask = pixels(parent/ref['observed'], 'L') > 0
                support = feature_support(mask, c['landmarks5_canvas_xy'])
                png = pixels(out/label/'png'/(case_id+'.png'))
                assert np.array_equal(png, deliver(raw[i], camera, mask))
                assert np.array_equal(raw[i][~mask], camera[~mask].astype(np.float32)/np.float32(255))
                assert np.array_equal(png[~mask], camera[~mask])
                mean, mean_png, _ = mean_only(raw[i], retained[i], camera, mask)
                rm = pixel_metrics(raw[i], target, mask)
                pm = exported_pixel_metrics(png, target, mask)
                for stage, values, metric in [('raw', er, rm), ('png', ep, pm)]:
                    slot = 0 if stage == 'raw' else 1
                    v = retained_embeddings[i, slot] if label in ['baseline', 'A0', 'B0'] else values[i]
                    tv = retained_embeddings[i, 2] if label in ['baseline', 'A0', 'B0'] else values[10+i]
                    metric['ArcFace_observed_fixed'] = float(v@tv)
                rm.update(landmark_high_frequency_MSE=detail_float(raw[i], target, support),
                          constant_mean_shift_only_MSE=pixel_metrics(mean, target, mask)['MSE'])
                pm.update(landmark_high_frequency_MSE=detail_metric(png, target, support),
                          constant_mean_shift_only_MSE=exported_pixel_metrics(mean_png, target, mask)['MSE'])
                row = dict(case=case_id, source=c['source'], profile=c['profile'], raw=rm, png=pm)
                assert all(row[k] == saved_rows[case_id][k] for k in ['case', 'source', 'profile'])
                for stage in ['raw', 'png']:
                    for key, value in row[stage].items():
                        other = saved_rows[case_id][stage][key]
                        if value is None or isinstance(value, bool):
                            assert value == other
                        else:
                            assert abs(value-other) <= 1e-13, (label, case_id, stage, key)
                rows.append(row)
                cells[(label, case_id)] = png
                checked += 1
        reductions[label] = {stage: grouped(rows, stage) for stage in ['raw', 'png']}
        for stage in ['raw', 'png']:
            for key, values in reductions[label][stage].items():
                for metric, value in values.items():
                    assert abs(value-saved_metrics['groups'][stage][key][metric]) <= 1e-13

    trajectories = []
    outcome_checks = []
    with np.load(root/'readouts.npz', allow_pickle=False) as readouts:
        readout_arrays = {key: readouts[key].copy() for key in readouts.files}
    for route in ['A', 'B']:
        initial = np.load(out/(route+'_initial_vector.npy'), allow_pickle=False)
        layout = read(out/(route+'_parameter_layout.json'))
        assert [r['name'] for r in layout] == p['trainable_parameter_names'][route]
        assert len(layout) == p['routes'][route]['tensors'] and len(initial) == p['routes'][route]['elements']
        state0 = torch.load(out/('state_'+route+'0/full_state.pth'), map_location='cpu', weights_only=True)
        state50 = torch.load(out/('state_'+route+'50/full_state.pth'), map_location='cpu', weights_only=True)
        names = set(p['trainable_parameter_names'][route])
        for name, value in state0['model'].items():
            if name.startswith('current.'):
                assert torch.equal(value, original_provider[name])
            elif name.startswith(('projectors.', 'anchor_projectors.')):
                _, size, term = name.split('.')
                expected = readout_arrays[f'f{size}_{term}']
                if term == 'weight':
                    padded = np.zeros(expected.shape[:2]+(3, 3), np.float32)
                    padded[:, :, 1:2, 1:2] = expected
                    expected = padded
                assert np.array_equal(value.numpy(), expected)
        assert set(state0['model']) == set(state50['model'])
        for name in state0['model']:
            if name not in names:
                assert torch.equal(state0['model'][name], state50['model'][name]), name
        actual0 = np.concatenate([state0['model'][name].numpy().reshape(-1) for name in p['trainable_parameter_names'][route]])
        assert np.array_equal(initial, actual0)
        assert abs(np.linalg.norm(initial.astype(np.float64))-bounds['norms_A_B_historical_conditioning'][0 if route == 'A' else 1]) < 1e-10
        fit = read(out/('fit_'+route+'.json'))
        trace = [json.loads(line) for line in (out/('trajectory_'+route)/'trace.jsonl').read_text(encoding='utf-8').splitlines()]
        assert trace == fit['rows'] and fit['updates'] == 50 and len(trace) == 50
        current = initial.copy()
        for index, row in enumerate(trace, 1):
            assert row['update'] == index and row['cache_index'] == p['schedule'][index-1]
            assert row['reference'] == p['reference_blocks'][row['cache_index']]['reference']
            with np.load(out/('trajectory_'+route)/f'update_{index:03d}.npz', allow_pickle=False) as saved:
                assert set(saved.files) == {'clipped_gradient', 'unconstrained', 'actual'}
                grad, proposal, actual = [saved[k].copy() for k in ['clipped_gradient', 'unconstrained', 'actual']]
            assert all(v.shape == initial.shape and v.dtype == np.float32 and np.isfinite(v).all() for v in [grad, proposal, actual])
            assert np.linalg.norm(grad.astype(np.float64)) <= 1.00001
            distance = float(np.linalg.norm(proposal.astype(np.float64)-initial.astype(np.float64)))
            reserve = float(2*np.linalg.norm(np.spacing(initial).astype(np.float64)))
            expected_scale = 1. if distance <= common else max(0., common-reserve)/distance
            bound = row['projection']
            assert abs(bound['scale']-expected_scale) < 1e-12 and abs(bound['unconstrained_distance']-distance) < 1e-10
            expected = proposal if bound['scale'] == 1. else (initial.astype(np.float64)+bound['scale']*(proposal.astype(np.float64)-initial)).astype(np.float32)
            assert np.array_equal(actual, expected)
            actual_distance = float(np.linalg.norm(actual.astype(np.float64)-initial.astype(np.float64)))
            assert actual_distance <= common and abs(actual_distance-bound['actual_distance']) < 1e-10
            assert bound['scientific_bound_relaxed'] is False
            current = actual
        final = np.concatenate([state50['model'][name].numpy().reshape(-1) for name in p['trainable_parameter_names'][route]])
        assert np.array_equal(current, final)
        for label, state in [('0', state0), ('50', state50)]:
            record = read(out/('state_'+route+label+'/receipt.json'))
            assert state_hash(state['model']) == record['model_state']
            assert sha(out/('state_'+route+label+'/full_state.pth')) == record['full_state_sha256']
        optimizer = state50['optimizer']
        assert len(optimizer['state']) == len(layout) and state50['route_updates'] == 50
        for i, value in enumerate(optimizer['state'].values()):
            assert int(value['step']) == 50
            assert list(value['exp_avg'].shape) == layout[i]['shape'] == list(value['exp_avg_sq'].shape)
            assert torch.isfinite(value['exp_avg']).all() and torch.isfinite(value['exp_avg_sq']).all()
        outcome = read(out/('outcome_'+route+'.json'))
        checks = {}
        for stage in ['raw', 'png']:
            new = independent_gate(reductions['baseline'][stage], reductions[route+'50'][stage])
            stored = outcome['gates'][stage]
            assert stored['pass'] == new['pass']
            assert [(v['group'], v['metric']) for v in stored['preservation_failures']] == [(v['group'], v['metric']) for v in new['preservation_failures']]
            assert abs(stored['global_structure_gain']-new['global_structure_gain']) < 1e-12
            assert all(abs(stored['source_structure_gains'][s]-new['source_structure_gains'][s]) < 1e-12 for s in new['source_structure_gains'])
            # Denominator propagation affects auditing arithmetic only. Exact
            # scientific pass/failure classification above is always required.
            recorded_groups = read(out/(route+'50')/'metrics.json')['groups'][stage]
            base_recorded = read(out/'baseline/metrics.json')['groups'][stage]
            nr = max(0., base_recorded['degraded']['MSE']-recorded_groups['degraded']['constant_mean_shift_only_MSE'])
            dr = max(base_recorded['degraded']['MSE']-recorded_groups['degraded']['MSE'], 1e-12)
            nd = max(0., reductions['baseline'][stage]['degraded']['MSE']-reductions[route+'50'][stage]['degraded']['constant_mean_shift_only_MSE'])
            dd = max(reductions['baseline'][stage]['degraded']['MSE']-reductions[route+'50'][stage]['degraded']['MSE'], 1e-12)
            propagation = abs(nr-nd)/dr+abs(nd)*abs(dr-dd)/(dr*dd)+1e-12
            assert abs(stored['brightness_gain_fraction']-new['brightness_gain_fraction']) <= propagation
            checks[stage] = new
        assert outcome['numerical_sample_pass'] == all(v['pass'] for v in checks.values())
        assert not outcome['development_model_accepted'] and not outcome['app_promotion']
        trajectories.append(dict(route=route, optimizer_steps_checked=50, projection_formulas_checked=50,
            final_vector_equals_saved_state=True, selected_tensors=len(names), CUDA_gradients_independently_replayed=False))
        outcome_checks.append(dict(route=route, gates=checks))

    historical = np.concatenate([v.numpy().reshape(-1) for name, v in original_provider.items()
                                 if name.startswith(('style_conditioner.', 'spatial_conditioners.'))])
    assert abs(np.linalg.norm(historical.astype(np.float64))-bounds['norms_A_B_historical_conditioning'][2]) < 1e-10
    assert read(out/'fresh_VM_replay.json')['replayed_cases'] == 20
    gallery = receipt.parent/'visual_sheets'
    gallery.mkdir()
    from PIL import ImageDraw
    sheets = []
    # Each sheet shows original, target, baseline, A50 and B50 at exact256 cells.
    for block in p['reference_blocks']:
        canvas = Image.new('RGB', (1280, 1320), '#fafafa')
        draw = ImageDraw.Draw(canvas)
        for i, title in enumerate(['Input', 'TRAIN target', 'Original DGP', 'A decoder', 'B spatial bank']):
            draw.text((i*256+8, 8), title, fill='#111111')
        ref = refs[block['reference']]
        for y, case_id in enumerate(block['case_ids']):
            source_paths = [parent/cases[case_id]['input'], parent/ref['target']]
            images = [pixels(path) for path in source_paths]+[cells[(label, case_id)] for label in ['baseline', 'A50', 'B50']]
            for x, image in enumerate(images):
                canvas.paste(Image.fromarray(image), (x*256, 40+y*256))
        path = gallery/(block['reference']+'.png')
        assert not path.exists()
        canvas.save(path)
        sheets.append(dict(path=str(path), sha256=sha(path), cells=25))
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), paired_RAW_PNG_compositions_checked=checked,
        scalar_replay_atol=1e-13, all17_RAW_all17_PNG_groups_recomputed=True, trajectories=trajectories,
        outcomes=outcome_checks, sheets=sheets, visual_review_pending=True, independent_final_reviewer=False,
        full_corpus_epochs=0, model_qualified=False, neural_calls=0, gradient_queries=0, optimizer_updates=0,
        read_only_tensors_loaded=True, seconds=time.monotonic()-start)
    write(receipt, record)
    print(json.dumps(dict(complete=True, compositions=checked, projection_steps=100, visual_sheets=20, model_qualified=False)), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    a = parser.parse_args()
    p, old = verify_transfer(a.root, a.parent_root, a.protocol_sha)
    audit_saved(a.root.resolve(), a.parent_root.resolve(), p, old, a.receipt.resolve())


if __name__ == '__main__':
    main()
