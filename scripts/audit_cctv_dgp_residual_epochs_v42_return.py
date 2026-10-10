"""Prospective independent return audit. Executes frozen local code, not returned code."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tarfile
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42'
OUT = ROOT/'outputs/cctv_dgp_residual_epochs_vm_v42_return'
sys.path.insert(0, str(BUNDLE))
from cctv_dgp_residual_epochs_v42_contract import (NAME, STEM, BUDGETS, read, write,
    sha, verified_assets, safe_return_members, groups, capacity, exported_pixel_metrics,
    feature_support, detail_metric)
from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def readback(folder, p, initial, started):
    receipt = read(folder/'metrics.json'); assert receipt['complete']
    assert [r['id'] for r in receipt['rows']] == [c['id'] for c in p['cases']]
    rows = []; raw_previews = 0
    for c, reported in zip(p['cases'], receipt['rows']):
        assert time.monotonic()-started < 3600
        cid = c['id']; camera = pixels(BUNDLE/c['input']); target = pixels(BUNDLE/c['target'])
        mask = pixels(BUNDLE/c['observed'], 'L') > 0
        png = pixels(folder/(cid+'.png')); mean_png = pixels(folder/(cid+'_mean_only.png'))
        assert np.array_equal(png[~mask], camera[~mask]) and np.array_equal(mean_png[~mask], camera[~mask])
        truth = np.load(initial/(cid+'_target_embedding.npy'), allow_pickle=False)
        vector = np.load(folder/(cid+'_embedding.npy'), allow_pickle=False)
        raw_vector = np.load(folder/(cid+'_raw_embedding.npy'), allow_pickle=False)
        for a in [truth, vector, raw_vector]:
            assert a.dtype == np.float32 and a.shape == (512,) and np.isfinite(a).all()
            assert abs(float(a@a)-1) < 1e-5
        feature = feature_support(mask, c['landmarks5_canvas_xy'])
        value = exported_pixel_metrics(png, target, mask)
        value.update({'ArcFace_observed_fixed': float(vector@truth),
            'landmark_high_frequency_MSE': detail_metric(png, target, feature),
            'constant_mean_shift_only_MSE': exported_pixel_metrics(mean_png, target, mask)['MSE']})
        for metric, actual in value.items():
            saved = reported['metrics'][metric]
            if actual is None or isinstance(actual, bool): assert actual == saved
            else: assert abs(actual-saved) <= 1e-10, (cid, metric)
        raw_values = reported['raw_metrics']
        assert all(raw_values[m] is None or isinstance(raw_values[m], bool) or np.isfinite(raw_values[m]) for m in raw_values)
        assert abs(raw_values['ArcFace_observed_fixed']-float(raw_vector@truth)) <= 1e-10
        if cid in p['preview_case_ids']:
            a = np.load(folder/(cid+'.npy'), allow_pickle=False)
            base = np.load(initial/(cid+'.npy'), allow_pickle=False)
            assert a.dtype == np.float32 and a.shape == (256, 256, 3)
            assert hashlib.sha256(a.tobytes()).hexdigest() == reported['raw_float32_sha256']
            assert np.array_equal(a[~mask], camera[~mask].astype(np.float32)/np.float32(255))
            assert np.array_equal(deliver(a, camera, mask), png)
            mean_raw, expected_mean, shift = mean_only(a, base, camera, mask)
            assert np.array_equal(expected_mean, mean_png)
            assert np.allclose(shift, reported['postclip_mean_RGB_shift'], rtol=0, atol=1e-12)
            rv = pixel_metrics(a, target, mask)
            rv.update({'ArcFace_observed_fixed': float(raw_vector@truth),
                'landmark_high_frequency_MSE': detail_float(a, target, feature),
                'constant_mean_shift_only_MSE': pixel_metrics(mean_raw, target, mask)['MSE']})
            for metric, actual in rv.items():
                saved = raw_values[metric]
                if actual is None or isinstance(actual, bool): assert actual == saved
                else: assert abs(actual-saved) <= 1e-10
            raw_previews += 1
        rows.append({'id': cid, 'source': c['source'], 'profile': c['profile'], 'metrics': value,
                     'raw': raw_values})
    assert raw_previews == 50
    actual = groups(rows); raw = review_groups(rows, 'raw')
    for values, name in [(actual, 'groups'), (raw, 'raw_groups')]:
        for key, group in values.items():
            assert group['cases'] == receipt[name][key]['cases']
            assert all(abs(v-receipt[name][key][m]) <= 1e-10 for m, v in group.items() if m != 'cases')
    return {'update': receipt['update'], 'groups': actual, 'raw_groups': raw,
        'PNG_cases_checked': 3905, 'raw_previews_checked': 50,
        'all_raw_metric_values_recomputed': False, 'raw_aggregation_checked': True}


def cpu_replay(folder, p, started):
    import torch
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_spatial_decoder_v42 import SpatialDGPCandidateV42
    from cctv_dgp_pilot import FixedObservedIdentity, state_hash, grid112
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    seed = torch.load(BUNDLE/'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
    candidate = SpatialDGPCandidateV42(original, seed)
    checkpoint = torch.load(folder/'spatial_decoder.pth', map_location='cpu', weights_only=True)
    assert set(checkpoint) == {row['name'] for row in p['parameter_layout']}
    for row in p['parameter_layout']:
        value = checkpoint[row['name']]
        assert value.dtype == torch.float32 and list(value.shape) == row['shape'] and torch.isfinite(value).all()
    candidate.decoder.load_state_dict(checkpoint, strict=True)
    candidate.eval().requires_grad_(False)
    identity = FixedObservedIdentity(BUNDLE/'weights/w600k_r50.onnx', 'cpu')
    models = [original.net, candidate.decoder, candidate.reference_decoder, identity]
    before = [state_hash(m) for m in models]
    assert before[0] == p['initial_states']['original'] and before[2] == p['initial_states']['reference_decoder']
    assert before[1] == read(folder/'metrics.json')['decoder_state']
    refs = {r['id']: r for r in p['references']}
    chosen = [c for c in p['cases'] if c['id'] in p['preview_case_ids']]
    count = 0; raw_error = vector_error = 0.; byte_error = 0
    canonical = lambda a: torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None]
    with torch.inference_mode():
        for begin in range(0, 50, 5):
            assert time.monotonic()-started < 3600
            selected = chosen[begin:begin+5]
            cameras = [pixels(BUNDLE/c['input']) for c in selected]
            masks = [pixels(BUNDLE/c['observed'], 'L') > 0 for c in selected]
            x = torch.cat([canonical(a) for a in cameras]); mask = torch.from_numpy(np.stack(masks))[:, None]
            raw = candidate(x, mask).permute(0, 2, 3, 1).numpy().copy()
            exact_raw = []; pngs = []
            for c, a, camera, support in zip(selected, raw, cameras, masks):
                saved = np.load(folder/(c['id']+'.npy'), allow_pickle=False)
                error = float(np.abs(a-saved).max()); assert error <= 1e-5; raw_error = max(raw_error, error)
                png = deliver(a, camera, support); saved_png = pixels(folder/(c['id']+'.png'))
                difference = int(np.abs(png.astype(np.int16)-saved_png.astype(np.int16)).max())
                assert difference <= 1; byte_error = max(byte_error, difference)
                pngs.append(saved_png); exact_raw.append(torch.from_numpy(saved).permute(2, 0, 1)[None])
            grid = torch.from_numpy(np.stack([grid112(refs[c['source_person_or_reference']]['matrix112']) for c in selected]))
            truth_input = torch.cat([canonical(pixels(BUNDLE/c['target'])) for c in selected])
            v = identity.embedding(torch.cat([torch.cat(exact_raw), torch.cat([canonical(a) for a in pngs]), truth_input]),
                torch.cat([mask.float()]*3), torch.cat([grid]*3)).numpy().copy()
            for slot, c in enumerate(selected):
                for a, path in [(v[slot], folder/(c['id']+'_raw_embedding.npy')),
                                (v[slot+5], folder/(c['id']+'_embedding.npy')),
                                (v[slot+10], OUT/'outputs/update0'/(c['id']+'_target_embedding.npy'))]:
                    error = float(np.abs(a-np.load(path, allow_pickle=False)).max())
                    assert error <= 1e-4; vector_error = max(vector_error, error)
                count += 1
    assert before == [state_hash(m) for m in models]
    assert all(not v.requires_grad and v.grad is None for m in models for v in m.parameters())
    state_path = folder/'training_state.pt'
    if state_path.is_file():
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        assert state['completed_updates'] == read(folder/'metrics.json')['update']
        assert state['next_schedule_index'] == state['completed_updates']
        assert state['schedule_sha256'] == p['assets_sha256']['schedule.json']
        assert state['protocol_sha256'] == sha(BUNDLE/'protocol.json')
        assert state_hash(state['decoder']) == before[1]
        assert len(state['optimizer']['param_groups'][0]['params']) == 57
        assert state['scheduler']['last_epoch'] == state['completed_updates']
        assert state['torch_rng'].dtype == torch.uint8 and state['cuda_rng']
        assert state['numpy_rng'][1].shape == (624,) and state['python_rng']
    else:
        assert read(folder/'metrics.json')['update'] == 0
        result = read(OUT/'outputs/failure.json')
        assert result['optimizer_updates'] == 0 and result['optimizer_constructed'] is False
    return {'cases': count, 'maximum_raw_error': raw_error, 'maximum_PNG_byte_error': byte_error,
        'maximum_embedding_error': vector_error, 'states_unchanged': True,
        'full_training_state_keys_verified': state_path.is_file(), 'gradient_queries': 0, 'optimizer_updates': 0}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--expected-bytes', type=int, required=True); a = parser.parse_args()
    started = time.monotonic(); p = read(BUNDLE/'protocol.json'); pin = sha(BUNDLE/'protocol.json')
    verified_assets(BUNDLE, p, pin)
    for name, digest in p['local_sources_sha256'].items(): assert sha(ROOT/name) == digest
    archive = ROOT/'outputs'/(STEM+'-results.tar.gz')
    assert archive.stat().st_size == a.expected_bytes and sha(archive) == a.expected_sha
    assert Path(str(archive)+'.sha256').read_text().strip().split() == [a.expected_sha, archive.name]
    exported = read(ROOT/'outputs'/(STEM+'-export.json'))
    assert exported['complete'] and exported['archive_sha256'] == a.expected_sha and exported['bytes'] == a.expected_bytes
    assert not OUT.exists(), 'Preserve every partial return import'
    with tarfile.open(archive, 'r:gz') as tar:
        members, total = safe_return_members(tar.getmembers(), p); OUT.mkdir()
        for member, name in members:
            destination = OUT/name; assert destination.resolve().is_relative_to(OUT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open('xb') as target:
                for block in iter(lambda: source.read(1024**2), b''): target.write(block)
    mapping = {name: sha(OUT/name) for _, name in members}
    assert mapping['protocol.json'] == pin
    manifest = read(OUT/'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert manifest['files_sha256'] == {k: v for k, v in mapping.items() if k != 'export_manifest.json'}
    write(ROOT/'outputs/cctv_dgp_residual_epochs_v42_return_import.json', {'complete': True,
        'archive_sha256': a.expected_sha, 'bytes': a.expected_bytes, 'members': len(members),
        'uncompressed_bytes': total, 'returned_code_executed': False})
    success = (OUT/'outputs/results.json').is_file()
    assert success != (OUT/'outputs/failure.json').is_file()
    result = read(OUT/('outputs/results.json' if success else 'outputs/failure.json'))
    assert result['protocol_sha256'] == pin and result['native_or_reserved_used'] is False
    assert result['app_promotion'] is False and result['goal_complete'] is False
    assert 0 <= result['optimizer_updates'] <= 3905
    assert result['optimizer_updates'] <= result['backwards'] <= result['optimizer_updates']+1
    assert exported['optimizer_updates'] == result['optimizer_updates'] and exported['training_success_not_implied']
    assert exported['run_results_present'] == success and exported['failure_present'] == (not success)
    outer = read(OUT/'supervisor_receipt.json'); code = int((OUT/'trainer_exit_code.txt').read_text())
    assert outer['worker_exit_code'] == code and (code == 0) == success
    assert outer['cap_seconds'] == BUDGETS['external_seconds'] and outer['kill_grace_seconds'] == 30
    snapshots = []; gates = []; replays = []
    for update in p['snapshots']:
        folder = OUT/('outputs/update'+str(update))
        if not (folder/'metrics.json').is_file(): continue
        current = readback(folder, p, OUT/'outputs/update0', started); snapshots.append(current)
        replays.append(cpu_replay(folder, p, started))
        if update:
            threshold = .1 if update == 3905 else .01
            gate = {'delivered': capacity(snapshots[0]['groups'], current['groups'], threshold),
                    'raw': capacity(snapshots[0]['raw_groups'], current['raw_groups'], threshold)}
            gate['pass'] = gate['delivered']['pass'] and gate['raw']['pass']
            reported = read(OUT/('outputs/capacity_update'+str(update)+'.json'))
            for stage in ['delivered', 'raw']:
                assert gate[stage]['pass'] == reported[stage]['pass']
                assert [v['group']+v['metric'] for v in gate[stage]['preservation_failures']] == \
                       [v['group']+v['metric'] for v in reported[stage]['preservation_failures']]
                assert abs(gate[stage]['relative_feature_gain']-reported[stage]['relative_feature_gain']) <= 1e-10
            assert gate['pass'] == reported['pass']; gates.append({'update': update, **gate})
    if success:
        assert result['optimizer_updates'] == result['backwards'] == 3905 and result['completed_epochs'] == 5
        assert result['new_trained_checkpoint'] and [s['update'] for s in snapshots] == p['snapshots']
        assert all(g['pass'] for g in gates)
        assert result['seconds'] <= BUDGETS['worker_seconds'] and result['fit_seconds'] <= BUDGETS['fit_seconds']
        assert result['peak_allocated_VRAM_bytes'] <= BUDGETS['peak_vram_bytes']
    if (OUT/'outputs/training_steps.jsonl').is_file():
        steps = [json.loads(line) for line in (OUT/'outputs/training_steps.jsonl').read_text().splitlines()]
        assert len(steps) == result['optimizer_updates']
        schedule = read(BUNDLE/'schedule.json')['batches']
        for index, (step, ids) in enumerate(zip(steps, schedule), 1):
            update = step['update']; assert update == index
            assert step['case_ids'] == [p['cases'][i]['id'] for i in ids]
            expected_rate = .0003 if update <= 1562 else .00009
            assert abs(step['learning_rate']-expected_rate) < 1e-15
    for name, cap in [('cache_timing.json', 900), ('timing_update20.json', 6300)]:
        path = OUT/'outputs'/name
        if not path.is_file(): continue
        timing = read(path); assert timing['cap_seconds'] == cap and timing['safety_factor'] == 1.25
        estimate = timing['seconds']+timing.get('remaining_batches', timing.get('remaining_updates'))*float(np.mean(timing['steady_sample_seconds']))*1.25+timing.get('overhead_seconds', 0)
        assert abs(estimate-timing['projected_seconds']) <= 1e-9
        if success: assert estimate <= cap
    for key in ['original', 'reference_decoder', 'recognizer']:
        if result['states_before'] is not None:
            assert result['states_before'][key] == result['states_after'][key] == p['initial_states'][key]
    verified_assets(BUNDLE, p, pin)
    write(ROOT/'outputs/cctv_dgp_residual_epochs_v42_independent_audit.json', {'complete': True,
        'protocol_sha256': pin, 'archive_sha256': a.expected_sha, 'checker_sha256': sha(Path(__file__)),
        'members_verified': len(members), 'snapshots': snapshots, 'gates': gates, 'CPU_replays': replays,
        'five_epochs_finished': success, 'necessary_delivered_capacity_pass': bool(success and all(g['pass'] for g in gates)),
        'all_PNG_pixel_metrics_recomputed': True, 'all_raw_metric_values_recomputed': False,
        'raw_audit_scope': '50 stored raw previews replayed and recomputed; all stored raw aggregates and saved embedding dot products checked',
        'full_raw_or_identity_or_native_quality_proven': False, 'failure_retained': not success,
        'local_gradient_queries': 0, 'local_optimizer_updates': 0, 'VM_connections': 0,
        'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic()-started, 'cap_seconds': 3600})
    print({'complete': True, 'five_epochs_finished': success, 'app_promoted': False}, flush=True)


if __name__ == '__main__': main()
