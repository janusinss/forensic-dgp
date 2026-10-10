"""Frozen incoming audit: archive, all saved pixels, gates, gradients and full states."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'scripts'))
from cctv_dgp_multiscale_calibration_contract_v1 import NAME, INITIAL, STATE, RECOGNIZER, BUDGETS, WEIGHTS, NORMALIZERS, read, write, sha, verify


def close(a, b, atol=1e-12):
    if isinstance(a, dict):
        assert set(a) == set(b)
        for k in a:
            close(a[k], b[k], atol)
    elif isinstance(a, list):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            close(x, y, atol)
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert np.isclose(a, b, rtol=0, atol=atol), (a, b)
    else:
        assert a == b, (a, b)


def audit(a):
    stamp = time.monotonic(); packet = PROJECT / 'outputs' / NAME
    pin = sha(packet / 'protocol.json'); p = verify(packet, pin)
    assert sha(a.archive) == a.expected_sha and read(a.export)['archive_sha256'] == a.expected_sha
    assert read(a.export)['complete'] and read(a.export)['bytes'] == a.archive.stat().st_size
    returned = PROJECT / 'outputs' / (NAME + '_return')
    if not returned.exists():
        with tarfile.open(a.archive, 'r:gz') as tar:
            members = tar.getmembers(); names = [m.name for m in members]
            assert len(set(names)) == len(names) <= 5000
            assert sum(m.size for m in members) <= BUDGETS['return_uncompressed_bytes'] + 1024**2
            for m in members:
                n = PurePosixPath(m.name)
                assert m.isfile() and not n.is_absolute() and '..' not in n.parts and '\\' not in m.name
                assert ':' not in m.name
            returned.mkdir(); tar.extractall(returned, filter='data')
    manifest = read(returned / 'export_manifest.json'); assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert sha(returned / 'protocol.json') == pin
    for n, d in manifest['files_sha256'].items():
        f = returned / n
        assert f.resolve().is_relative_to(returned.resolve()) and f.is_file() and not f.is_symlink() and sha(f) == d, n
        if n in p['assets_sha256']:
            assert d == p['assets_sha256'][n]
    actual = {f.relative_to(returned).as_posix() for f in returned.rglob('*') if f.is_file()}
    assert actual == set(manifest['files_sha256']) | {'export_manifest.json'}
    out = returned / 'outputs'
    result = read(out / 'results.json'); assert result['complete'] and result['protocol_sha256'] == pin
    assert result['optimizer_updates'] == 12 and result['gradient_queries'] == 280 and result['completed_epochs'] == 0
    assert result['forward_counts'] == {'original': 44, 'candidate': 712, 'recognizer': 400}
    assert result['progress']['backwards'] == 120 and result['progress']['fit_exposures'] == 600
    assert result['progress']['completed_arms'] == [q['id'] for q in p['arms']]
    assert not result['model_qualification'] and not result['app_promotion'] and not result['goal_complete']
    assert result['seconds'] < BUDGETS['worker_seconds'] and int((returned / 'trainer_exit_code.txt').read_text()) == 0
    assert not (out / 'failure.json').exists()
    sys.path.insert(0, str(PROJECT)); sys.path.insert(0, str(packet))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support, capacity, exported_pixel_metrics, detail_metric
    from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups
    from cctv_dgp_pilot import state_hash, buffer_hash
    import torch
    torch.set_num_threads(4)
    initial = torch.load(out / 'initial_candidate.pth', map_location='cpu', weights_only=True)
    assert state_hash(initial) == INITIAL
    # Explicit layout, rather than estimating tensor ownership from a flattened vector.
    grad = read(out / 'gradient_preflight.json'); layout = grad['layout']
    from cctv_dgp_multiscale_calibration_model_v1 import FULL, DEEP
    assert [x['name'] for x in layout] == FULL and len(layout) == 15
    offset = 0
    for row in layout:
        assert row['start'] == offset and row['shape'] == list(initial[row['name']].shape)
        offset += initial[row['name']].numel(); assert row['end'] == offset
    assert offset == 609219 and layout[2]['end'] == 147456
    assert grad['complete'] and grad['queries'] == 280 and grad['signals'] == p['signals']
    assert grad['gradient_sha256'] == sha(out / 'initial_gradients.npy')
    vectors = np.load(out / 'initial_gradients.npy', allow_pickle=False, mmap_mode='r')
    assert vectors.shape == (20, 14, 609219) and vectors.dtype == np.float32
    part_sq = np.zeros(15, dtype=np.float64)
    for batch in range(20):
        record = grad['rows'][batch]
        assert record['case_ids'] == [c['id'] for c in p['cases'][batch*5:batch*5+5]]
        assert np.isfinite(vectors[batch]).all()
        norms = []
        for signal in range(14):
            norms.append([float(np.linalg.norm(vectors[batch, signal, r['start']:r['end']].astype(np.float64))) for r in layout])
            if signal < 4:
                part_sq += np.square(norms[-1])
        close(norms, record['part_L2_by_signal'], 1e-10)
    close(np.sqrt(part_sq).tolist(), grad['all15_improvement_gradient_L2'], 1e-10)
    assert (part_sq > 0).all()
    refs = {r['id']: r for r in p['references']}; stages = ['baseline'] + [q['id'] for q in p['arms']]
    checked_rows = 0; native_rows = 0; metric_results = {}; max_accum_error = 0.; max_parameter_error = 0.

    def pixels(f, mode='RGB'):
        with Image.open(f) as im:
            assert im.size == (256, 256); return np.asarray(im.convert(mode)).copy()

    for stage in stages:
        folder = out / stage; receipt = read(folder / 'metrics.json'); rows = []
        assert receipt['complete'] and receipt['label'] == stage and len(receipt['rows']) == 100
        for begin in range(0, 100, 5):
            cs = p['cases'][begin:begin+5]; ids = [c['id'] for c in cs]
            base, _ = unpack(out / 'baseline/packs' / f'b{begin//5:02d}.npz', ids)
            raw, em = unpack(folder / 'packs' / f'b{begin//5:02d}.npz', ids, baseline=None if stage == 'baseline' else base)
            assert np.allclose(np.linalg.norm(em, axis=2), 1, atol=3e-6, rtol=0)
            for i, c in enumerate(cs):
                r = refs[c['source_person_or_reference']]; camera = pixels(packet / c['input'])
                target = pixels(packet / r['target']); mask = pixels(packet / r['observed'], 'L') > 0
                feature = feature_support(mask, c['landmarks5_canvas_xy']); png = deliver(raw[i], camera, mask)
                assert np.array_equal(png, pixels(folder / 'previews' / (c['id'] + '.png')))
                assert np.array_equal(raw[i][~mask], camera[~mask].astype(np.float32) / np.float32(255))
                mraw, mpng, shift = mean_only(raw[i], base[i], camera, mask)
                rm = pixel_metrics(raw[i], target, mask); pm = exported_pixel_metrics(png, target, mask)
                rm.update({'ArcFace_observed_fixed': float(em[i, 0] @ em[i, 2]),
                    'landmark_high_frequency_MSE': detail_float(raw[i], target, feature),
                    'constant_mean_shift_only_MSE': pixel_metrics(mraw, target, mask)['MSE']})
                pm.update({'ArcFace_observed_fixed': float(em[i, 1] @ em[i, 2]),
                    'landmark_high_frequency_MSE': detail_metric(png, target, feature),
                    'constant_mean_shift_only_MSE': exported_pixel_metrics(mpng, target, mask)['MSE']})
                row = {'id': c['id'], 'source': c['source'], 'profile': c['profile'], 'raw': rm, 'png': pm,
                    'postclip_mean_RGB_shift': shift.tolist(), 'raw_RGB_float32_sha256': hashlib.sha256(raw[i].tobytes()).hexdigest(),
                    'PNG_RGB_sha256': hashlib.sha256(png.tobytes()).hexdigest()}
                close(row, receipt['rows'][begin+i], 2e-12); rows.append(row); checked_rows += 1
        groups = {s: review_groups(rows, s) for s in ['raw', 'png']}; close(groups, receipt['groups'], 2e-12)
        metric_results[stage] = groups
        for c, saved in zip(p['native_development'], receipt['native_unpaired_rows']):
            assert c['id'] == saved['id'] and 'MSE' not in saved and 'PSNR' not in saved and 'SSIM' not in saved
            camera = pixels(packet / c['input']); mask = pixels(packet / c['observed'], 'L') > 0
            raw = np.load(folder / 'native' / (c['id'] + '.npy'), allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
            assert np.array_equal(raw[~mask], camera[~mask].astype(np.float32) / np.float32(255))
            png = deliver(raw, camera, mask); assert np.array_equal(png, pixels(folder / 'native' / (c['id'] + '.png')))
            assert hashlib.sha256(raw.tobytes()).hexdigest() == saved['raw_RGB_float32_sha256']
            assert hashlib.sha256(png.tobytes()).hexdigest() == saved['PNG_RGB_sha256'] and saved['outside_mask_unchanged']
            native_rows += 1
        assert len(receipt['native_unpaired_rows']) == 24
        if stage != 'baseline':
            arm = next(q for q in p['arms'] if q['id'] == stage); gate = read(folder / 'quality_gate.json')
            recalculated = {s: capacity(metric_results['baseline'][s], groups[s], .01) for s in ['raw', 'png']}
            close(recalculated, gate['comparisons'], 2e-12)
            assert gate['sampled_capacity_pass'] == all(g['pass'] for g in recalculated.values())
            state = torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=True)
            assert state['protocol_sha256'] == pin and state['arm'] == arm and state['completed_updates'] == 1
            assert state['completed_epochs'] == 0 and state['fitting_reference_exposures'] == 10 and state['next_schedule_index'] == 10
            assert state['failed_gate_must_not_resume'] and not state['automatic_resume']
            assert state['schedule_sha256'] == pin and state['original_checkpoint_sha256'] == p['original_checkpoint_sha256']
            assert all(k in state for k in ['scheduler', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'])
            names = DEEP if arm['partition'] == 'deep3' else FULL
            assert state['optimizer_parameter_names'] == names
            model = state['model']; assert state_hash(model) == receipt['candidate_state']
            for n in initial:
                if n not in names:
                    assert torch.equal(model[n], initial[n]), n
            steps = read(folder / 'step_receipt.json'); assert steps['actual_optimizer_updates'] == 1 and steps['backwards'] == 10 and steps['fit_exposures'] == 50
            assert [r['begin'] for r in steps['rows']] == p['pools'][arm['pool']]
            assert all(r['barrier'] == 0 for r in steps['rows'])
            accumulated = np.load(folder / 'accumulated_gradient.npy', allow_pickle=False)
            size = 147456 if arm['partition'] == 'deep3' else 609219
            assert accumulated.shape == (size,) and accumulated.dtype == np.float32 and np.isfinite(accumulated).all()
            expected = np.zeros(size, dtype=np.float64)
            for begin in p['pools'][arm['pool']]:
                for signal in range(4):
                    expected += vectors[begin//5, signal, :size].astype(np.float64) * WEIGHTS[signal] / NORMALIZERS[signal] / 10
            error = float(np.max(np.abs(expected - accumulated))); max_accum_error = max(max_accum_error, error)
            assert np.allclose(accumulated, expected, rtol=2e-4, atol=2e-7), ('Initial gradient accumulation differs', stage, error)
            norm = float(np.linalg.norm(accumulated.astype(np.float64))); assert np.isclose(norm, steps['gradient_L2_before_clip'], rtol=2e-6, atol=1e-7)
            clip = min(1., 1. / (steps['gradient_L2_before_clip'] + 1e-6))
            offset = 0; param_groups = state['optimizer']['param_groups']; assert len(param_groups) == 1
            pg = param_groups[0]; assert pg['lr'] == arm['lr'] and pg['betas'] == (.9, .999) and pg['weight_decay'] == 1e-5 and pg['eps'] == 1e-8
            assert len(pg['params']) == len(names) and state['scheduler']['last_epoch'] == 1
            for n, index in zip(names, pg['params']):
                theta = initial[n].numpy(); count = theta.size
                g = accumulated[offset:offset+count].reshape(theta.shape).astype(np.float64) * clip + 1e-5 * theta.astype(np.float64)
                os = state['optimizer']['state'][index]; assert float(os['step']) == 1
                assert np.allclose(os['exp_avg'].numpy(), .1*g, rtol=2e-5, atol=2e-8)
                assert np.allclose(os['exp_avg_sq'].numpy(), .001*g*g, rtol=2e-5, atol=2e-10)
                predicted = theta.astype(np.float64) - arm['lr'] * g / (np.abs(g) + 1e-8)
                weight_error = float(np.max(np.abs(model[n].numpy() - predicted))); max_parameter_error = max(max_parameter_error, weight_error)
                assert np.allclose(model[n].numpy(), predicted, rtol=2e-6, atol=8e-7), (stage, n, weight_error)
                offset += count
            assert offset == size
        print({'audited_snapshot': stage, 'paired_records': checked_rows, 'unpaired_native': native_rows}, flush=True)
        assert time.monotonic() - stamp < 1200, 'Local audit1200s stop; preserve partial evidence'
    assert checked_rows == 1300 and native_rows == 312
    # Independent fresh CPU forwards, two arms / one reference each. No gradients.
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_multiscale_calibration_model_v1 import MultiscaleCalibrationDGP
    original, _ = load_frozen_dgp_restorer(packet / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    assert state_hash(original.net) == STATE
    replay = []
    for arm_index, begin in [(0, 0), (11, 75)]:
        arm = p['arms'][arm_index]; folder = out / arm['id']
        state = torch.load(folder / 'training_state.pt', map_location='cpu', weights_only=True)
        candidate = MultiscaleCalibrationDGP(original.net).eval().requires_grad_(False); candidate.load_state_dict(state['model'])
        cs = p['cases'][begin:begin+5]; cameras = [pixels(packet / c['input']) for c in cs]
        masks = [pixels(packet / refs[c['source_person_or_reference']]['observed'], 'L') > 0 for c in cs]
        x = torch.from_numpy(np.stack(cameras).astype(np.float32) / np.float32(255)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(masks))[:, None]
        with torch.inference_mode():
            value = torch.where(mask, candidate(x), x).permute(0, 2, 3, 1).contiguous().numpy()
        baseline, _ = unpack(out / 'baseline/packs' / f'b{begin//5:02d}.npz', [c['id'] for c in cs])
        raw, _ = unpack(folder / 'packs' / f'b{begin//5:02d}.npz', [c['id'] for c in cs], baseline=baseline)
        error = float(np.max(np.abs(value - raw))); assert error <= 3e-5, ('Fresh CPU replay differs', arm['id'], error)
        replay.append({'arm': arm['id'], 'ids': [c['id'] for c in cs], 'maximum_raw_error': error})
        assert all(v.grad is None for v in candidate.parameters())
    verify(packet, pin)
    write(a.receipt, {'complete': True, 'checker_sha256': sha(Path(__file__)), 'protocol_sha256': pin,
        'archive_sha256': a.expected_sha, 'manifest_files': len(manifest['files_sha256']),
        'paired_raw_and_PNG_records': 1300, 'unpaired_native_records': 312, 'gradient_vectors': 280,
        'full_optimizer_scheduler_RNG_states': 12, 'maximum_gradient_accumulation_error': max_accum_error,
        'maximum_first_Adam_parameter_error': max_parameter_error, 'fresh_CPU_replays': replay,
        'local_optimizer_updates': 0, 'local_gradient_queries': 0, 'native_clean_reference_metrics': False,
        'visual_review_pending': True, 'model_qualification': False, 'goal_complete': False,
        'seconds': time.monotonic() - stamp})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True); parser.add_argument('--export', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    audit(parser.parse_args())
