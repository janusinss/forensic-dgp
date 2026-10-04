"""Rebuild V9 output arithmetic/states and return integrity without inference."""
import argparse
import collections
import importlib.util
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time


def close(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            close(actual[key], expected[key])
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        assert math.isfinite(actual) and math.isclose(actual, expected, rel_tol=3e-6, abs_tol=3e-7), (actual, expected)
    else:
        assert actual == expected, (actual, expected)


def safe_members(members):
    assert len(members) < 10000 and sum(m.size for m in members) < 2 * 1024 ** 3
    names = set()
    for member in members:
        path = PurePosixPath(member.name)
        assert (member.isfile() or member.isdir()) and not path.is_absolute() and '..' not in path.parts
        assert '\\' not in member.name and ':' not in member.name
        assert member.name not in names
        names.add(member.name)


def extract(archive, destination):
    import hashlib
    digest = hashlib.sha256()
    with archive.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert Path(str(archive) + '.sha256').read_text(encoding='ascii').strip().split() == [digest.hexdigest(), archive.name]
    assert not destination.exists(), 'Preserve previous return'
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers(); safe_members(members)
        destination.mkdir(parents=True)
        for member in members:
            if member.isfile():
                target = destination / member.name; target.parent.mkdir(parents=True, exist_ok=True)
                with stream.extractfile(member) as source, target.open('xb') as output:
                    shutil.copyfileobj(source, output)


def audit(root, out, expected_protocol_sha):
    started = time.monotonic()
    spec = importlib.util.spec_from_file_location('pinned_mixed_v9', root / 'cctv_dgp_mixed_v9.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    p = module.verify(root, expected_protocol_sha, weights=False)
    sys.path.insert(0, str(root))
    import numpy as np
    import torch
    from PIL import Image
    from cctv_dgp_pilot import aggregate, exported_pixel_metrics, state_hash
    from models import DGPSynthesizer
    result = module.read(out / 'results.json')
    assert result['complete'] is True and result['protocol_sha256'] == expected_protocol_sha
    assert result['total_optimizer_updates'] == result['training_backward_calls'] == 7820
    assert result['preflight_autograd_calls'] == 1 and 0 < result['elapsed_seconds'] <= 2400
    assert result['buffers_and_teacher_unchanged'] and result['validation_used']
    assert not any(result[k] for k in ('native_used', 'native_reserved_used', 'production_promoted', 'model_improvement_established'))
    for name, pin in result['artifacts_sha256'].items():
        assert module.sha(module.safe_path(out, name)) == pin, name
    execution = module.read(out / 'execution.json')
    assert execution['host'].split('.')[0] == 'forensic-dgp-thesis' and 'L4' in execution['gpu']
    assert execution['source_sha256'] == module.sha(root / 'scripts/train_cctv_dgp_mixed_v9.py') == module.sha(out / 'executed_source.py')
    assert execution['starting_state_hash'] == module.START_STATE and execution['runtime_cap_seconds'] == 2400
    assert not execution['validation_statistics_used_for_loss'] and execution['normalization_frozen']
    assert not execution['amp'] and not execution['ema'] and not execution['production_promoted']
    assert execution['calibration_sha256'] == result['calibration_sha256'] == module.sha(out / 'calibration/weights.json')
    preflight = module.read(out / 'preflight.json')
    assert preflight['passed'] and preflight['autograd_calls'] == 1 and preflight['optimizer_updates'] == 0
    assert preflight['batch_size'] == 10 and preflight['tail_batch_size'] == 5 and preflight['active_gradient_tensors'] > 0
    assert preflight['starting_state_hash'] == module.START_STATE and preflight['buffers_unchanged']
    timing = module.read(out / 'timing_at32.json')
    assert 0 < timing['projected_trainer_seconds'] <= timing['trainer_cap_seconds'] == 2400
    refs = {r['id']: r for r in p['references']}
    cases = {c['id']: c for c in p['training_cases']}
    calibration = module.read(out / 'calibration/weights.json')
    assert calibration['complete'] and calibration['training_only'] and not calibration['validation_statistics_used']
    assert calibration['starting_state_hash'] == module.START_STATE and calibration['formula'] == p['design']['weight_rule']
    assert len(calibration['rows']) == 3905 and len({r['case_id'] for r in calibration['rows']}) == 3905
    grouped = collections.defaultdict(list); raw_calibration_checked = 0
    raw_ids = []
    for row in calibration['rows']:
        case = cases[row['case_id']]; ref = refs[case['reference_id']]
        assert ref['role'] == 'train' and row['reference_id'] == ref['id']
        key = case['source'] + '/' + case['profile']; assert row['group'] == key
        assert math.isfinite(row['MSE_raw']) and row['MSE_raw'] >= 0
        grouped[key].append(row['MSE_raw'])
        if 'raw_float' in row:
            raw = np.load(out / row['raw_float'], allow_pickle=False)
            assert raw.shape == (256, 256, 3) and raw.dtype == np.float32 and np.isfinite(raw).all()
            target = np.asarray(Image.open(root / ref['target']).convert('RGB'), dtype=np.float32) / 255
            mask = np.asarray(Image.open(root / ref['observed'])) > 0
            assert raw.min() >= 0 and raw.max() <= 1
            value = float(np.square(raw - target)[mask].astype(np.float64).mean())
            close(row['MSE_raw'], value)
            raw_ids.append(row['case_id']); raw_calibration_checked += 1
    assert sorted(raw_ids) == calibration['raw_check_case_ids'] and raw_calibration_checked == 20
    expected_raw = sorted(cid for key in sorted(grouped) for cid in sorted(c['id'] for c in cases.values()
                           if c['source'] + '/' + c['profile'] == key)[:2])
    assert sorted(raw_ids) == expected_raw
    means = {k: math.fsum(v) / len(v) for k, v in sorted(grouped.items())}
    close(calibration['group_means'], means)
    overall = math.fsum(means.values()) / 10
    raw_weights = {k: min(4, max(0.25, overall / max(v, 1e-8))) for k, v in means.items()}
    scale = math.fsum(raw_weights.values()) / 10
    weights = {k: v / scale for k, v in raw_weights.items()}
    close(calibration['weights'], weights)
    schedule = module.read(root / 'training_schedule_v9.json')['epochs']
    assert preflight['cases'] == schedule['1'][0]
    traces = [json.loads(line) for line in (out / 'updates.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(traces) == 7820
    for update, trace in enumerate(traces, 1):
        epoch, step = (update - 1) // 391 + 1, (update - 1) % 391 + 1
        assert trace['update'] == update and trace['epoch'] == epoch and trace['step'] == step
        assert trace['cases'] == schedule[str(epoch)][step - 1]
        wanted = [weights[cases[c]['source'] + '/' + cases[c]['profile']] for c in trace['cases']]
        close(trace['weights'], wanted)
        assert len(trace['raw_case_MSE']) == 10 and all(math.isfinite(v) and v >= 0 for v in trace['raw_case_MSE'])
        assert math.isfinite(trace['preclip_norm']) and trace['preclip_norm'] > 0
        assert trace['calibration_sha256'] == result['calibration_sha256']
        close(trace['loss'], math.fsum(v * w for v, w in zip(trace['raw_case_MSE'], wanted)) / 10)
    targets = {}
    validation_refs = {c['reference_id'] for c in p['validation_cases']}
    for rid in validation_refs:
        embed = np.load(out / 'target_embeddings' / (rid + '.npy'), allow_pickle=False)
        assert embed.shape == (512,) and np.isfinite(embed).all() and np.isclose(np.linalg.norm(embed), 1, atol=1e-5)
        targets[rid] = embed
    stages = ['baseline'] + ['epoch' + str(e) for e in [2, 5, 10, 20]]
    summaries = {}; pngs = 0; floats = 0; cosines = 0; input_metrics = 0
    for stage in stages:
        saved = module.read(out / stage / 'metrics.json'); rebuilt = []
        assert saved['complete'] and len(saved['rows']) == 520
        for case, row in zip(p['validation_cases'], saved['rows']):
            assert all(row[k] == v for k, v in case.items())
            ref = refs[case['reference_id']]; assert ref['role'] == 'validation'
            with Image.open(out / row['prediction']) as image:
                assert image.mode == 'RGB' and image.size == (256, 256); rgb = np.asarray(image)
            target = np.asarray(Image.open(root / ref['target']).convert('RGB'))
            support = np.asarray(Image.open(root / ref['observed'])) > 0
            input_rgb = np.asarray(Image.open(root / case['input']).convert('RGB'))
            np.testing.assert_array_equal(rgb[~support], input_rgb[~support])
            values = exported_pixel_metrics(rgb, target, support)
            for key, value in values.items():
                close(row[key], value)
            if case['id'] in p['validation_preview_case_ids']:
                raw = np.load(out / row['raw_float'], allow_pickle=False)
                assert raw.shape == (256, 256, 3) and raw.dtype == np.float32 and np.isfinite(raw).all()
                assert raw.min() >= 0 and raw.max() <= 1
                exported = np.clip(raw * 255, 0, 255).astype(np.uint8); exported[~support] = input_rgb[~support]
                np.testing.assert_array_equal(rgb, exported); floats += 1
            else:
                assert 'raw_float' not in row
            embed = np.load(out / row['embedding'], allow_pickle=False)
            assert embed.shape == (512,) and np.isfinite(embed).all() and np.isclose(np.linalg.norm(embed), 1, atol=1e-5)
            cosine = float(np.clip(embed @ targets[ref['id']], -1, 1)); close(row['ArcFace_observed_fixed'], cosine)
            rebuilt.append({**row, **values, 'ArcFace_observed_fixed': cosine}); pngs += 1; cosines += 1
        close(saved['summary'], aggregate(rebuilt)); summaries[stage] = aggregate(rebuilt)
        image = np.asarray(Image.open(out / saved['preview']).convert('RGB'))
        assert image.shape == (2904, 780, 3)
        by_id = {r['id']: r for r in saved['rows']}
        for i, cid in enumerate(p['validation_preview_case_ids']):
            row = by_id[cid]; y = 24 + i * 288 + 28
            for j, file in enumerate([root / row['input'], out / row['prediction'], root / refs[row['reference_id']]['target']]):
                np.testing.assert_array_equal(image[y:y + 256, 2 + j * 260:258 + j * 260], np.asarray(Image.open(file).convert('RGB')))
        if stage == 'baseline':
            assert len(saved['input_rows']) == 520
            inputs = []
            for case, row in zip(p['validation_cases'], saved['input_rows']):
                assert all(row[k] == v for k, v in case.items())
                ref = refs[case['reference_id']]
                rgb = np.asarray(Image.open(root / case['input']).convert('RGB'))
                target = np.asarray(Image.open(root / ref['target']).convert('RGB'))
                support = np.asarray(Image.open(root / ref['observed'])) > 0
                values = exported_pixel_metrics(rgb, target, support)
                for key, value in values.items():
                    close(row[key], value)
                embed = np.load(out / row['embedding'], allow_pickle=False)
                assert embed.shape == (512,) and np.isfinite(embed).all() and np.isclose(np.linalg.norm(embed), 1, atol=1e-5)
                cosine = float(np.clip(embed @ targets[ref['id']], -1, 1)); close(row['ArcFace_observed_fixed'], cosine)
                inputs.append({**row, **values, 'ArcFace_observed_fixed': cosine}); cosines += 1; input_metrics += 1
            close(saved['input_summary'], aggregate(inputs))
        else:
            assert saved['input_rows'] == [] and saved['input_summary'] is None
        print(f'audited {stage}:520 PNG metrics/embedding cosines elapsed={time.monotonic()-started:.0f}s', flush=True)
    start = torch.load(out / 'checkpoints/baseline.pth', map_location='cpu', weights_only=True)
    assert state_hash(start) == module.START_STATE
    model = DGPSynthesizer().eval(); model.load_state_dict(start, strict=True)
    buffers = {n for n, _ in model.named_buffers()}; parameters = {n for n, _ in model.named_parameters()}
    changes = []
    assert [r['epoch'] for r in result['snapshots']] == [2, 5, 10, 20]
    baseline = summaries['baseline']; best = baseline; selected = 0; selected_source = 'checkpoints/baseline.pth'

    def gate(candidate):
        if set(candidate) != set(baseline) or set(candidate) != set(best):
            return False
        for key, reference in baseline.items():
            row = candidate[key]
            if row['cases'] != reference['cases'] or row['identity_pairs'] != reference['identity_pairs'] or row['identity_pairs'] != row['cases']:
                return False
            for metric in ('MSE', 'SSIM', 'ArcFace_observed_fixed'):
                if row.get(metric) is None or reference.get(metric) is None or not math.isfinite(row[metric]) or not math.isfinite(reference[metric]):
                    return False
                if metric == 'MSE' and (row[metric] < 0 or reference[metric] < 0 or row[metric] > reference[metric] + 1e-12):
                    return False
                if metric != 'MSE' and row[metric] < reference[metric] - 1e-6:
                    return False
        return math.isfinite(best['degraded']['MSE']) and best['degraded']['MSE'] > 0 and candidate['degraded']['MSE'] <= best['degraded']['MSE'] * 10 ** (-0.1 / 10)

    for snapshot in result['snapshots']:
        epoch = snapshot['epoch']; assert snapshot['update'] == epoch * 391
        state = torch.load(out / snapshot['checkpoint'], map_location='cpu', weights_only=True)
        model.load_state_dict(state, strict=True)
        assert state_hash(state) == snapshot['state_hash']
        assert all(torch.equal(state[n], start[n]) for n in buffers)
        changed = sum(not torch.equal(state[n], start[n]) for n in parameters); assert changed > 0
        candidate = summaries['epoch' + str(epoch)]; qualifies = gate(candidate)
        assert snapshot['qualifies'] == qualifies
        if qualifies:
            selected = epoch; selected_source = snapshot['checkpoint']; best = candidate
        assert snapshot['selected_epoch_after'] == selected
        changes.append({'epoch': epoch, 'changed_parameter_tensors': changed})
    selection = module.read(out / 'selection.json')
    assert selection['selected_epoch'] == result['selected_epoch'] == selected
    assert selection['selected_source'] == selected_source
    assert selection['best_sha256'] == module.sha(out / 'best.pth') == module.sha(out / selected_source)
    assert not selection['production_promoted'] and selection['native_visual_review_pending'] and selection['independent_final_review_pending']
    if selected == 0:
        assert state_hash(torch.load(out / 'best.pth', map_location='cpu', weights_only=True)) == module.START_STATE
    return {'complete': True, 'version': 9, 'protocol_sha256': expected_protocol_sha,
        'results_sha256': module.sha(out / 'results.json'), 'validation_pngs_checked': pngs,
        'validation_raw_float_previews_checked': floats, 'input_metrics_checked': input_metrics,
        'embedding_cosines_rebuilt': cosines, 'training_calibration_rows_checked': 3905,
        'training_calibration_raw_MSE_rebuilt': raw_calibration_checked, 'training_loss_weights_rebuilt': weights,
        'update_trace_checked': 7820, 'exposures_checked': 78200, 'checkpoint_changes': changes,
        'selected_epoch': selected, 'selected_source': selected_source,
        'validation_summaries': summaries, 'seconds': time.monotonic() - started,
        'local_model_forwards': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
        'cuda_gradients_recomputed_by_this_audit': False, 'recognizer_forwards_recomputed': False,
        'native_reserved_used': False, 'production_promoted': False, 'model_improvement_established': False,
        'limitation': 'Independent saved-file arithmetic/state/exposure verification. CUDA gradients/recognizer forwards are not replayed;20 calibration raw cases are checked while3905 reported calibration errors derive from pinned VM execution. Useful native outputs and independent final review remain required.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--expected-protocol-sha', required=True)
    parser.add_argument('--results', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--extract-to', type=Path)
    parser.add_argument('--receipt', type=Path, required=True)
    args = parser.parse_args()
    if args.archive:
        assert args.extract_to is not None
        extract(args.archive, args.extract_to)
        args.results = args.extract_to / 'outputs/cctv_dgp_mixed_v9'
    assert args.results is not None
    result = audit(args.root.resolve(), args.results.resolve(), args.expected_protocol_sha)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    with args.receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print({k: v for k, v in result.items() if k not in ('validation_summaries', 'training_loss_weights_rebuilt')}, flush=True)
