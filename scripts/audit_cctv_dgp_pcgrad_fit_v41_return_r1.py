"""Prospective saved-PNG capacity audit and frozen CPU replay; no local learning."""
import argparse
import importlib.util
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
import numpy as np
from PIL import Image
from cctv_dgp_spatial_fit_v40_contract import sha, read, write, validate_schedule, feature_support, exported_pixel_metrics, detail_metric, groups, capacity

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_pcgrad_fit_vm_v41'
OUT = ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return'
STEM = 'cctv-dgp-pcgrad-fit-v41'

EXPORT_PREFIX = 'cctv_dgp_spatial_fit_v40_return'  # Exact literal in the immutable V41 exporter.


def safe_members(members, p):
    allowed = {'protocol.json', 'trainer.log', 'trainer_exit_code.txt', 'supervisor_receipt.json', 'export_manifest.json',
               'outputs/results.json', 'outputs/failure.json', 'outputs/cache_receipt.json', 'outputs/cache_timing.json',
               'outputs/cohort_loss_setup.json', 'outputs/timing_update20.json', 'outputs/stopped_spatial_decoder.pth',
               'outputs/capacity_update50.json', 'outputs/capacity_update800.json'}
    allowed.update('outputs/steps/step'+str(i).zfill(4)+s for i in range(1,801) for s in ['.json','.npz'])
    for update in p['snapshots']:
        prefix = 'outputs/update'+str(update)+'/'
        allowed.update([prefix+'metrics.json', prefix+'spatial_decoder.pth'])
        for c in p['cases']:
            allowed.update(prefix+c['id']+suffix for suffix in ['.png', '_mean_only.png', '_embedding.npy'])
            if update == 0: allowed.add(prefix+c['id']+'_target_embedding.npy')
        allowed.update(prefix+cid+'.npy' for cid in p['preview_case_ids'])
    seen = set(); result = []; total = 0
    for m in members:
        assert m.isfile() and not m.issym() and not m.islnk() and m.name.startswith(EXPORT_PREFIX+'/')
        name = m.name[len(EXPORT_PREFIX)+1:]; path = PurePosixPath(name)
        assert not any(v in name for v in ['\\', ':', '\x00']) and not path.is_absolute()
        assert path.parts and all(v not in ['.', '..'] for v in path.parts) and name in allowed and name.lower() not in seen
        assert 0 <= m.size <= 16*1024**2
        seen.add(name.lower()); total += m.size; result.append((m, name))
        assert total <= 3*1024**3 and len(seen) <= 50000
    return result, total


def pixels(path, mode='RGB'):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        return np.asarray(im.convert(mode)).copy()


def readback(folder, p, initial, start):
    receipt = read(folder/'metrics.json'); assert receipt['complete'] and len(receipt['rows']) == 3905
    rows = []; raw_checked = 0; visible_bytes = 0; raw_compositions = {}; targets = {}
    for c, reported in zip(p['cases'], receipt['rows']):
        assert time.monotonic()-start <= 1800
        assert reported['id'] == c['id'] and reported['source'] == c['source'] and reported['profile'] == c['profile']
        key = c['source_person_or_reference']
        if key not in targets:
            targets[key] = (pixels(BUNDLE/c['target']), pixels(BUNDLE/c['observed'], 'L') > 0)
        target, mask = targets[key]; camera = pixels(BUNDLE/c['input'])
        png = pixels(folder/(c['id']+'.png')); mean_png = pixels(folder/(c['id']+'_mean_only.png'))
        assert np.array_equal(png[~mask], camera[~mask]) and np.array_equal(mean_png[~mask], camera[~mask])
        visible_bytes += int(camera[~mask].size)
        vector = np.load(folder/(c['id']+'_embedding.npy'), allow_pickle=False)
        truth = np.load(initial/(c['id']+'_target_embedding.npy'), allow_pickle=False)
        for a in [vector, truth]:
            assert a.dtype == np.float32 and a.shape == (512,) and np.isfinite(a).all() and abs(float(a@a)-1) < 1e-5
        value = exported_pixel_metrics(png, target, mask)
        value.update({'ArcFace_observed_fixed': float(vector@truth), 'landmark_high_frequency_MSE': detail_metric(png, target, feature_support(mask, c['landmarks5_canvas_xy'])),
                      'constant_mean_shift_only_MSE': exported_pixel_metrics(mean_png, target, mask)['MSE']})
        for metric in value:
            actual, saved = value[metric], reported['metrics'][metric]
            if actual is None or isinstance(actual, bool): assert actual == saved
            else: assert abs(actual-saved) <= 1e-10, (c['id'], metric)
        if c['id'] in p['preview_case_ids']:
            raw = np.load(folder/(c['id']+'.npy'), allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
            assert __import__('hashlib').sha256(raw.tobytes()).hexdigest() == reported['raw_float32_sha256']
            expected = np.floor(raw*np.float32(255)).astype(np.uint8); expected[~mask] = camera[~mask]
            assert np.array_equal(png, expected)
            base = np.load(initial/(c['id']+'.npy'), allow_pickle=False)
            shift = (raw-base)[mask].astype(np.float64).mean(0)
            assert np.allclose(shift, reported['postclip_mean_RGB_shift'], rtol=0, atol=1e-12)
            expected_mean = np.floor(np.clip(base.astype(np.float64)+shift, 0, 1).astype(np.float32)*np.float32(255)).astype(np.uint8)
            expected_mean[~mask] = camera[~mask]; assert np.array_equal(mean_png, expected_mean)
            raw_checked += 1
        rows.append({'id': c['id'], 'source': c['source'], 'profile': c['profile'], 'metrics': value})
    actual_groups = groups(rows)
    for key in actual_groups:
        assert actual_groups[key]['cases'] == receipt['groups'][key]['cases']
        for metric in actual_groups[key]:
            if metric != 'cases': assert abs(actual_groups[key][metric]-receipt['groups'][key][metric]) <= 1e-10
    assert raw_checked == 50
    return {'update': receipt['update'], 'groups': actual_groups, 'PNG_cases': 3905, 'mean_only_cases': 3905,
            'saved_vector_pairs': 3905, 'raw_previews_composed_exact': 50, 'outside_support_source_bytes_exact': visible_bytes,
            'all_nonpreview_raw_stages_retained': False}


def CPU_replay(folder, p, start):
    import torch
    sys.path.insert(0, str(BUNDLE)); sys.path.insert(0, str(ROOT))
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_spatial_decoder_v41 import SpatialDGPCandidateV41
    from cctv_dgp_pilot import FixedObservedIdentity, state_hash, grid112
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    seed = torch.load(BUNDLE/'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
    candidate = SpatialDGPCandidateV41(original, seed)
    checkpoint = torch.load(folder/'spatial_decoder.pth', map_location='cpu', weights_only=True)
    assert set(checkpoint) == {row['name'] for row in p['parameter_layout']}
    for row in p['parameter_layout']:
        a = checkpoint[row['name']]; assert a.dtype == torch.float32 and list(a.shape) == row['shape'] and torch.isfinite(a).all()
    candidate.decoder.load_state_dict(checkpoint, strict=True)
    identity = FixedObservedIdentity(BUNDLE/'weights/w600k_r50.onnx', 'cpu')
    before = {'original': state_hash(original.net), 'decoder': state_hash(candidate.decoder), 'reference_decoder': state_hash(candidate.reference_decoder), 'recognizer': state_hash(identity)}
    metrics = read(folder/'metrics.json'); assert before['decoder'] == metrics['decoder_state'] and sha(folder/'spatial_decoder.pth') == metrics['checkpoint_sha256']
    for k in ['original', 'reference_decoder', 'recognizer']: assert before[k] == p['initial_states'][k]
    if metrics['update'] == 0: assert before == p['initial_states']
    refs = {r['id']: r for r in p['references']}; cases = {c['id']: c for c in p['cases']}
    chosen = [cases[cid] for cid in p['preview_case_ids']]; count = 0; raw_err = 0.; byte_err = 0; vector_err = 0.
    def canonical(a): return torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None]
    with torch.inference_mode():
        for begin in range(0, 50, 5):
            assert time.monotonic()-start < 1800
            batch = chosen[begin:begin+5]; cameras = [pixels(BUNDLE/c['input']) for c in batch]
            masks = [pixels(BUNDLE/c['observed'], 'L') > 0 for c in batch]
            x = torch.cat([canonical(a) for a in cameras]); mask = torch.from_numpy(np.stack(masks))[:, None]
            raw = candidate(x, mask).permute(0, 2, 3, 1).numpy().copy()
            pngs = []
            for c, a, camera, support in zip(batch, raw, cameras, masks):
                saved = np.load(folder/(c['id']+'.npy'), allow_pickle=False)
                error = float(np.abs(a-saved).max()); assert error <= 1e-5; raw_err = max(raw_err, error)
                png = np.floor(a*np.float32(255)).astype(np.uint8); png[~support] = camera[~support]
                saved_png = pixels(folder/(c['id']+'.png'))
                error = int(np.abs(png.astype(np.int16)-saved_png.astype(np.int16)).max()); assert error <= 1; byte_err = max(byte_err, error)
                # Replay embeddings of the EXACT saved PNG, separately from CPU quantization.
                pngs.append(saved_png)
            grid = torch.from_numpy(np.stack([grid112(refs[c['source_person_or_reference']]['matrix112']) for c in batch]))
            truth = torch.cat([canonical(pixels(BUNDLE/c['target'])) for c in batch])
            vectors = identity.embedding(torch.cat([canonical(a) for a in pngs]), mask.float(), grid).numpy().copy()
            targets = identity.embedding(truth, mask.float(), grid).numpy().copy()
            for c, vector, target in zip(batch, vectors, targets):
                for a, path in [(vector, folder/(c['id']+'_embedding.npy')), (target, OUT/('outputs/update0/'+c['id']+'_target_embedding.npy'))]:
                    error = float(np.abs(a-np.load(path, allow_pickle=False)).max()); assert error <= 1e-4; vector_err = max(vector_err, error)
                count += 1
    assert before == {'original': state_hash(original.net), 'decoder': state_hash(candidate.decoder), 'reference_decoder': state_hash(candidate.reference_decoder), 'recognizer': state_hash(identity)}
    assert all(not v.requires_grad and v.grad is None for v in candidate.parameters()) and all(v.grad is None for v in identity.parameters())
    return {'cases': count, 'raw_maximum_error': raw_err, 'PNG_maximum_byte_difference': byte_err,
            'embedding_maximum_error': vector_err, 'all_states_unchanged': True, 'local_optimizer_updates': 0, 'local_gradient_calls': 0}



def step_evidence(p, result):
    import torch
    from cctv_dgp_pcgrad_v41_evidence import read_arrays, verify_step
    n = result['optimizer_updates']; logged = result['logged_updates']
    assert n == logged and result['backwards'] == 0
    assert 7*n <= result['component_gradient_queries'] <= 7*(n+1)
    paths = sorted((OUT/'outputs/steps').glob('*.json')) if (OUT/'outputs/steps').exists() else []
    arrays_paths = sorted((OUT/'outputs/steps').glob('*.npz')) if (OUT/'outputs/steps').exists() else []
    assert len(paths) == len(arrays_paths) == n
    seed = torch.load(BUNDLE/'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
    vector = lambda state: np.concatenate([state[r['name']].detach().numpy().reshape(-1) for r in p['parameter_layout']]).copy()
    previous = vector(seed); first = np.zeros(17952, np.float32); second = first.copy(); maximum_error = 0.
    snapshot_values = {0: previous.copy()}; schedule = read(BUNDLE/'schedule.json')['batches']
    for update, (path, proof) in enumerate(zip(paths, arrays_paths), 1):
        assert path.name == 'step'+str(update).zfill(4)+'.json' and proof.with_suffix('.json') == path
        record = read(path); assert record['complete'] and record['proof_sha256'] == sha(proof)
        assert record['case_indices'] == schedule[update-1]
        assert record['case_ids'] == [p['cases'][i]['id'] for i in schedule[update-1]]
        assert len(record['component_means']) == 7 and np.isfinite(record['component_means']).all()
        assert abs(sum(record['component_means'])-record['raw_objective']) <= 2e-6*max(1.,abs(record['raw_objective']))
        arrays = read_arrays(proof); error = verify_step(arrays, record, update, previous, first, second)
        maximum_error = max(maximum_error, error)
        previous, first, second = arrays['after'], arrays['exp_avg'], arrays['exp_avg_sq']
        if update in p['snapshots']: snapshot_values[update] = previous.copy()
    for update, expected in snapshot_values.items():
        path = OUT/('outputs/update'+str(update)+'/spatial_decoder.pth')
        if path.exists(): assert np.array_equal(vector(torch.load(path,map_location='cpu',weights_only=True)), expected)
    stop = OUT/'outputs/stopped_spatial_decoder.pth'
    if stop.exists(): assert np.array_equal(vector(torch.load(stop,map_location='cpu',weights_only=True)), previous)
    return {'complete':True,'all_logged_updates_verified':n,'saved_component_queries':7*n,
        'AdamW_readback_maximum_absolute_parameter_error':maximum_error,'prospective_parameter_arithmetic_tolerance':3e-7,
        'combined_gradient_roundoff_allowance':'Two float32 ULPs plus1e-12',
        'parameter_chain_and_snapshot_tensors_exact':True,'original_loss_weights_gates_and_schedule_retained':True,
        'saved_gradients_not_independently_differentiated':True,'local_gradient_calls':0,'local_optimizer_updates':0}


def audit(digest, size):
    start = time.monotonic(); p = read(BUNDLE/'protocol.json'); pin = sha(BUNDLE/'protocol.json')
    for name, value in p['assets_sha256'].items(): assert sha(BUNDLE/name) == value
    for name, value in p['sources_sha256'].items(): assert sha(ROOT/name) == value
    assert read(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_preparation/independent_packet_audit.json')['complete']
    validate_schedule(p['cases'], read(BUNDLE/'schedule.json')['batches'])
    archive = ROOT/'outputs'/(STEM+'-results.tar.gz'); assert archive.stat().st_size == size and sha(archive) == digest
    assert Path(str(archive)+'.sha256').read_text().strip().split() == [digest, archive.name]
    exported = read(ROOT/'outputs'/(STEM+'-export.json')); assert exported['complete'] and exported['archive_sha256'] == digest and exported['bytes'] == size
    assert not OUT.exists(), 'Preserve prior/partial imports; no automatic repeat'
    with tarfile.open(archive, 'r:gz') as tar:
        members, total = safe_members(tar.getmembers(), p); OUT.mkdir()
        for member, name in members:
            destination = (OUT/name).resolve(); assert destination.is_relative_to(OUT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open('xb') as target:
                for block in iter(lambda: source.read(1024**2), b''): target.write(block)
    mapping = {name: sha(OUT/name) for _, name in members}; assert mapping['protocol.json'] == pin
    manifest = read(OUT/'export_manifest.json'); assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert manifest['files_sha256'] == {k: v for k, v in mapping.items() if k != 'export_manifest.json'}
    write(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_return_import.json', {'complete': True, 'archive_sha256': digest,
          'archive_bytes': size, 'members': len(members), 'uncompressed_bytes': total, 'files_sha256': mapping, 'returned_code_executed': False})
    success = (OUT/'outputs/results.json').is_file(); assert success != (OUT/'outputs/failure.json').is_file()
    r = read(OUT/('outputs/results.json' if success else 'outputs/failure.json'))
    assert r['protocol_sha256'] == pin and r['native_or_reserved_used'] is False and r['app_promotion'] is False and r['goal_complete'] is False
    assert 0 <= r['optimizer_updates'] <= 800 and r['backwards'] == 0
    assert exported['optimizer_updates'] == r['optimizer_updates'] and exported['run_results_present'] == success
    assert exported['training_success_not_implied'] and exported['failure_present'] == (not success)
    outer = read(OUT/'supervisor_receipt.json'); code = int((OUT/'trainer_exit_code.txt').read_text())
    assert outer['complete'] and outer['protocol_sha256'] == pin and outer['worker_exit_code'] == code and (code == 0) == success
    assert outer['cap_seconds'] == 4800 and outer['kill_grace_seconds'] == 30
    learning_evidence = step_evidence(p, r)
    reviewed = []; gates = []; replays = []
    for update in p['snapshots']:
        folder = OUT/('outputs/update'+str(update))
        if not (folder/'metrics.json').is_file(): continue
        checked = readback(folder, p, OUT/'outputs/update0', start); reviewed.append(checked)
        replays.append(CPU_replay(folder, p, start))
        if update:
            gate = capacity(reviewed[0]['groups'], checked['groups'], .01 if update == 50 else .1)
            reported = read(OUT/('outputs/capacity_update'+str(update)+'.json'))
            assert gate['pass'] == reported['pass'] and [v['group']+v['metric'] for v in gate['preservation_failures']] == [v['group']+v['metric'] for v in reported['preservation_failures']]
            assert abs(gate['relative_feature_gain']-reported['relative_feature_gain']) <= 1e-10
            gates.append({'update': update, **gate})
    if success:
        assert r['optimizer_updates'] == r['logged_updates'] == 800 and r['component_gradient_queries'] == 5600 and r['completed_epochs'] == 1
        assert [s['update'] for s in reviewed] == [0, 50, 800] and r['necessary_capacity_pass'] == gates[-1]['pass']
        assert r['seconds'] <= 4500 and r['fit_seconds'] <= 3600 and r['peak_allocated_VRAM_bytes'] <= 20*1024**3
        for k in ['original', 'reference_decoder', 'recognizer']: assert r['states_before'][k] == r['states_after'][k] == p['initial_states'][k]
    for name, cap in [('cache_timing.json', 900), ('timing_update20.json', 3600)]:
        path = OUT/'outputs'/name
        if not path.is_file(): continue
        timing = read(path); assert timing['cap_seconds'] == cap and timing['safety_factor'] == 1.25
        estimate = timing['seconds']+timing['remaining_batches' if name.startswith('cache') else 'remaining_updates']*float(np.mean(timing['steady_sample_seconds']))*1.25+timing.get('overhead_seconds', 0)
        assert abs(estimate-timing['projected_seconds']) <= 1e-9
        if success: assert estimate <= cap
    for name, value in p['assets_sha256'].items(): assert sha(BUNDLE/name) == value
    for name, value in p['sources_sha256'].items(): assert sha(ROOT/name) == value
    assert time.monotonic()-start <= 1800
    write(ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_independent_audit_r1.json', {'complete': True, 'protocol_sha256': pin, 'archive_sha256': digest,
          'checker_sha256': sha(Path(__file__)), 'members_verified': len(members), 'per_update_learning_evidence': learning_evidence, 'snapshots': reviewed, 'gates': gates, 'CPU_replays': replays,
          'all_PNG_metrics_checked': True, 'all_raw_stages_checked': False, 'raw_check_scope': '50 prospective previews per complete snapshot',
          'training_finished800': success, 'necessary_capacity_pass': bool(success and gates[-1]['pass']), 'failure_retained': not success,
          'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0, 'app_promotion': False, 'goal_complete': False,
          'seconds': time.monotonic()-start, 'cap_seconds': 1800})
    print({'complete': True, 'finished800': success, 'capacity_pass': bool(success and gates[-1]['pass']), 'seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--expected-sha', required=True); parser.add_argument('--expected-bytes', type=int, required=True)
    args = parser.parse_args(); audit(args.expected_sha, args.expected_bytes)
