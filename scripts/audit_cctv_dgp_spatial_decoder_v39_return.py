"""Prospective V39 saved-gradient audit; frozen CPU replays, no local derivatives."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_spatial_decoder_vm_v39'
PREP = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_preparation'
OUT = ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_return'
PREFIX = 'cctv_dgp_spatial_decoder_v39_return/'
STEM = 'cctv-dgp-spatial-decoder-v39'
RETURNED_ASSETS = {'scripts/cctv_dgp_spatial_decoder_v39_vm.py', 'cctv_dgp_spatial_decoder_v39.py',
                   'frozen_definitions.py', 'untrained_initial_decoder.pth'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def safe_members(members, p):
    allowed = RETURNED_ASSETS | {'protocol.json', 'export_manifest.json', 'diagnostic.log',
              'diagnostic_exit_code.txt', 'supervisor_receipt.json', 'outputs/results.json',
              'outputs/failure.json', 'outputs/cohort_loss_setup.json', 'outputs/gradient_components.npy',
              'outputs/gradient_summary.json'}
    for c in p['cases']:
        allowed.update('outputs/initial/' + c['id'] + suffix for suffix in
                       ['_original.npy', '_spatial.npy', '_original.png', '_spatial.png', '_target_embedding.npy'])
    allowed.update('outputs/gradients/batch' + str(i) + '.npy' for i in range(10))
    result = []; seen = set(); total = 0
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk(), 'Only regular files'
        assert member.name.startswith(PREFIX) and not any(c in member.name for c in ['\\', ':', '\x00'])
        name = member.name[len(PREFIX):]; parts = PurePosixPath(name).parts
        assert parts and not PurePosixPath(name).is_absolute() and all(v not in ['.', '..'] for v in parts)
        assert name in allowed and name.lower() not in seen, name
        assert 0 <= member.size <= 8 * 1024**2, 'Single member8MiB cap'
        seen.add(name.lower()); total += member.size; result.append((member, name))
        assert total <= p['budgets']['export_uncompressed_bytes'] and len(seen) <= 290
    return result, total


def gradient_summary_check(folder, p, require_complete):
    """Recompute every saved batch, sum, value, norm, Gram and tensor partition."""
    import numpy as np
    summary = read(folder / 'outputs/gradient_summary.json')
    assert summary['complete'] and summary['terms'] == p['terms'] and summary['parameter_layout'] == p['parameter_layout']
    assert len(summary['batches']) == 10
    total = np.zeros((7, 17952), np.float64); values = np.zeros(7, np.float64)
    for i, row in enumerate(summary['batches']):
        path = folder / ('outputs/gradients/batch' + str(i) + '.npy')
        a = np.load(path, allow_pickle=False)
        assert a.dtype == np.float64 and a.shape == (7, 17952) and np.isfinite(a).all()
        assert sha(path) == row['gradient_sha256'] and row['batch'] == i
        assert row['ids'] == [c['id'] for c in p['cases'][5*i:5*(i+1)]]
        assert np.count_nonzero(a[3:]) == 0
        assert np.allclose(np.linalg.norm(a, axis=1), row['norms'], rtol=2e-10, atol=1e-12)
        v = np.asarray(row['values'], np.float64)
        assert v.shape == (7,) and np.isfinite(v).all() and np.count_nonzero(v[3:]) == 0
        total += a; values += v
    saved = np.load(folder / 'outputs/gradient_components.npy', allow_pickle=False)
    assert saved.dtype == np.float64 and saved.shape == total.shape and np.array_equal(saved, total)
    assert sha(folder / 'outputs/gradient_components.npy') == summary['gradient_sha256']
    assert np.array_equal(values, np.asarray(summary['values']))
    assert np.allclose(np.linalg.norm(total, axis=1), summary['component_norms'], rtol=2e-10, atol=1e-12)
    assert np.allclose(total @ total.T, summary['component_gram'], rtol=2e-10, atol=1e-12)
    assert set(summary['partitions']) == {row['name'] for row in p['parameter_layout']}
    zero_names = []
    for row in p['parameter_layout']:
        block = total[:, row['start']:row['end']]
        norms = np.linalg.norm(block, axis=1); gain = float(np.linalg.norm(block[:3].sum(0)))
        reported = summary['partitions'][row['name']]
        assert np.allclose(norms, reported['component_norms'], rtol=2e-10, atol=1e-12)
        assert abs(gain - reported['improvement_gradient_norm']) <= 1e-12
        if gain == 0: zero_names.append(row['name'])
    if require_complete: assert not zero_names, 'Disconnected spatial tensors retained as a failed proof'
    return {'saved_component_queries': 70, 'batch_matrices': 10, 'parameter_tensors': 57,
            'parameter_values': 17952, 'zero_improvement_tensor_names': zero_names,
            'all_initial_preservation_values_and_saved_gradients_exact_zero': True,
            'local_gradient_recomputation': False}


def verify_local_basis(p):
    for name, digest in p['assets_sha256'].items(): assert sha(BUNDLE / name) == digest, name
    for name, digest in p['local_basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    prepared = read(PREP / 'prepared.json'); packet = read(PREP / 'independent_packet_audit.json')
    assert prepared['complete'] and packet['complete'] and packet['protocol_sha256'] == sha(BUNDLE / 'protocol.json')
    assert p['retained_capacity_gates'] == read(ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27/protocol.json')['prospective_gates']


def import_return(p, pin, digest, size):
    archive = ROOT / 'outputs' / (STEM + '-results.tar.gz')
    assert archive.stat().st_size == size and sha(archive) == digest
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [digest, archive.name]
    exported = read(ROOT / 'outputs' / (STEM + '-export.json'))
    assert exported['complete'] and exported['archive_sha256'] == digest and exported['bytes'] == size
    assert exported['training_success_not_implied'] and exported['optimizer_updates'] == 0
    assert not OUT.exists(), 'Preserve prior/partial imports; no automatic repeat'
    with tarfile.open(archive, 'r:gz') as tar:
        members, total = safe_members(tar.getmembers(), p); OUT.mkdir()
        for member, name in members:
            destination = (OUT / name).resolve(); assert destination.is_relative_to(OUT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open('xb') as target:
                for block in iter(lambda: source.read(1024**2), b''): target.write(block)
    mapping = {name: sha(OUT / name) for _, name in members}
    assert mapping['protocol.json'] == pin
    for name in RETURNED_ASSETS: assert mapping[name] == p['assets_sha256'][name], name
    manifest = read(OUT / 'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert manifest['files_sha256'] == {name: value for name, value in mapping.items() if name != 'export_manifest.json'}
    assert manifest['data_and_original_weights_omitted_and_bound_to_protocol']
    receipt = {'complete': True, 'archive_sha256': digest, 'archive_bytes': size, 'members': len(members),
               'uncompressed_bytes': total, 'files_sha256': mapping, 'returned_code_executed': False}
    write(ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_return_import.json', receipt)
    return exported, receipt


def saved_initial_readback(p, success):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import convolve1d
    def pixels(path, mode='RGB'):
        with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()
    z = np.arange(-6, 7, dtype=np.float64); kernel = np.exp(-.5*(z/2)**2); kernel /= kernel.sum()
    def high(rgb):
        y = (rgb.astype(np.float64) * np.asarray([.299, .587, .114])).sum(2)
        return y - convolve1d(convolve1d(y, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
    features = []; checked = 0; spatial = 0; cache_error = 0.; pixel_errors = []
    for c in p['cases']:
        prefix = OUT / 'outputs/initial' / c['id']; path = Path(str(prefix) + '_original.npy')
        if not path.exists():
            assert not success; continue
        a = np.load(path, allow_pickle=False); camera = pixels(BUNDLE / c['input']); target8 = pixels(BUNDLE / c['target'])
        mask = pixels(BUNDLE / c['observed'], 'L') > 0
        assert a.dtype == np.float32 and a.shape == (256, 256, 3) and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
        assert np.array_equal(a[~mask], camera[~mask].astype(np.float32) / np.float32(255))
        png = pixels(Path(str(prefix) + '_original.png'))
        expected = np.floor(a * np.float32(255)).astype(np.uint8); expected[~mask] = camera[~mask]
        assert np.array_equal(png, expected)
        cache = np.load(BUNDLE / c['raw_dgp'], allow_pickle=False)
        err = float(np.abs(cache[mask] - a[mask]).max()); assert err <= p['historical_raw_tolerance']; cache_error = max(cache_error, err)
        truth = np.load(Path(str(prefix) + '_target_embedding.npy'), allow_pickle=False)
        assert truth.dtype == np.float32 and truth.shape == (512,) and np.isfinite(truth).all() and abs(float(truth @ truth) - 1) < 1e-5
        path = Path(str(prefix) + '_spatial.npy')
        if path.exists():
            b = np.load(path, allow_pickle=False)
            assert b.dtype == a.dtype and b.shape == a.shape and np.array_equal(a, b)
            assert np.array_equal(pixels(Path(str(prefix) + '_spatial.png')), expected); spatial += 1
        else: assert not success
        padded = np.pad(mask.astype(np.int64), 6); integral = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        interior = integral[13:, 13:] - integral[:-13, 13:] - integral[13:, :-13] + integral[:-13, :-13] == 169
        patch = np.zeros((256, 256), bool)
        for x, y in np.floor(np.asarray(c['landmarks5_canvas_xy'])).astype(int):
            patch[max(0, y-12):min(256, y+12), max(0, x-12):min(256, x+12)] = True
        patch &= interior; assert patch.any() and interior.any()
        target = target8.astype(np.float32) / np.float32(255)
        delta = high(a) - high(target)
        features.append({'id': c['id'], 'clear': c['profile'] == 'clear', 'feature_MSE': float(np.square(delta)[patch].mean()),
                         'interior_MSE': float(np.square(delta)[interior].mean())})
        pixel_errors.append(float(np.square((a-target)[mask]).astype(np.float64).mean())); checked += 1
    if success: assert checked == spatial == 50
    loss = OUT / 'outputs/cohort_loss_setup.json'; max_readback = 0.
    if loss.exists():
        cohort = read(loss); assert checked == len(cohort['rows']) == 50 and cohort['degraded_cases'] == 40
        for actual, reported in zip(features, cohort['rows'], strict=True):
            assert actual['id'] == reported['id'] and actual['clear'] == reported['clear']
            for key in ['feature_MSE', 'interior_MSE']:
                err = abs(actual[key] - reported[key]); assert err <= 2e-7; max_readback = max(max_readback, err)
        for key, metric in [('feature_normalizer', 'feature_MSE'), ('interior_normalizer', 'interior_MSE')]:
            actual = max(1e-6, float(np.mean([v[metric] for v in features if not v['clear']])))
            assert abs(actual - cohort[key]) <= 2e-7
        summary = OUT / 'outputs/gradient_summary.json'
        if summary.exists():
            values = read(summary)['values']; actual = [
                sum(1.25 * v['feature_MSE'] / cohort['feature_normalizer'] for v in features if not v['clear']) / 50,
                sum(.3125 * v['interior_MSE'] / cohort['interior_normalizer'] for v in features if not v['clear']) / 50,
                sum(.0625 * err / max(err, 1e-5) for v, err in zip(features, pixel_errors) if not v['clear']) / 50,
            ] + [0., 0., 0., 0.]
            assert np.allclose(actual, values, rtol=0, atol=p['CPU_scalar_replay_tolerance'])
    return {'original_cases_readback': checked, 'spatial_initial_pairs_exact': spatial,
            'historical_cache_raw_maximum_error': cache_error, 'raw_filter_metric_maximum_readback_error': max_readback}


def frozen_CPU_replay(p):
    import numpy as np
    from PIL import Image
    import torch
    sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_spatial_decoder_v39 import SpatialDGPCandidateV39
    torch.set_num_threads(4)
    model, _ = load_frozen_dgp_restorer(BUNDLE / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    seed = torch.load(BUNDLE / 'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
    candidate = SpatialDGPCandidateV39(model, seed)
    recognizer = FixedObservedIdentity(BUNDLE / 'weights/w600k_r50.onnx', 'cpu')
    before = {'original': state_hash(model.net), 'decoder': state_hash(candidate.decoder),
              'reference_decoder': state_hash(candidate.reference_decoder), 'recognizer': state_hash(recognizer)}
    assert before == p['expected_states']
    refs = {r['id']: r for r in p['references']}; raw_error = 0.; byte_error = 0; embedding_error = 0.; count = 0
    def pixels(path, mode='RGB'):
        with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()
    def tensor(array):
        return torch.from_numpy(array.astype(np.float32) / np.float32(255)).permute(2, 0, 1)[None]
    with torch.inference_mode():
        for source in sorted({c['source'] for c in p['cases']}):
            rid = next(c['source_person_or_reference'] for c in p['cases'] if c['source'] == source)
            chosen = [c for c in p['cases'] if c['source_person_or_reference'] == rid]; assert len(chosen) == 5
            images = [pixels(BUNDLE / c['input']) for c in chosen]
            masks = [pixels(BUNDLE / c['observed'], 'L') > 0 for c in chosen]
            x = torch.cat([tensor(v) for v in images]); support = torch.from_numpy(np.stack(masks))[:, None]
            parts = candidate.forward_components(x, support)
            assert torch.equal(parts['original_raw'], parts['result']) and torch.count_nonzero(parts['spatial_delta']) == 0
            grid = torch.from_numpy(np.stack([grid112(refs[rid]['matrix112'])] * 5))
            target = torch.cat([tensor(pixels(BUNDLE / c['target'])) for c in chosen])
            truth = recognizer.embedding(target, support.float(), grid).numpy().copy()
            for slot, c in enumerate(chosen):
                raw = parts['result'][slot].permute(1, 2, 0).numpy().copy()
                ref = np.load(OUT / ('outputs/initial/' + c['id'] + '_spatial.npy'), allow_pickle=False)
                err = float(np.abs(raw-ref).max()); assert err <= p['CPU_raw_replay_tolerance']; raw_error = max(raw_error, err)
                png = np.floor(raw*np.float32(255)).astype(np.uint8); png[~masks[slot]] = images[slot][~masks[slot]]
                saved = pixels(OUT / ('outputs/initial/' + c['id'] + '_spatial.png'))
                err = int(np.abs(png.astype(np.int16)-saved.astype(np.int16)).max()); assert err <= p['CPU_PNG_byte_tolerance']; byte_error = max(byte_error, err)
                vector = np.load(OUT / ('outputs/initial/' + c['id'] + '_target_embedding.npy'), allow_pickle=False)
                err = float(np.abs(truth[slot]-vector).max()); assert err <= p['CPU_embedding_absolute_tolerance']; embedding_error = max(embedding_error, err); count += 1
    assert before == {'original': state_hash(model.net), 'decoder': state_hash(candidate.decoder),
                      'reference_decoder': state_hash(candidate.reference_decoder), 'recognizer': state_hash(recognizer)}
    assert all(not v.requires_grad and v.grad is None for v in candidate.parameters())
    assert all(not v.requires_grad and v.grad is None for v in recognizer.parameters())
    return {'cases': count, 'original_DGP_batch_forwards': 2, 'decoder_batch_forwards': 4,
            'recognizer_target_batch_forwards': 2, 'raw_maximum_error': raw_error,
            'PNG_maximum_byte_difference': byte_error, 'embedding_maximum_error': embedding_error,
            'all_states_unchanged': True, 'local_gradient_calls': 0, 'optimizer_updates': 0}


def audit(digest, size):
    start = time.monotonic(); p = read(BUNDLE / 'protocol.json'); pin = sha(BUNDLE / 'protocol.json')
    verify_local_basis(p); exported, imported = import_return(p, pin, digest, size)
    success = (OUT / 'outputs/results.json').is_file(); failure = (OUT / 'outputs/failure.json').is_file()
    assert success != failure
    result = read(OUT / ('outputs/results.json' if success else 'outputs/failure.json'))
    assert result['protocol_sha256'] == pin
    for key in ['optimizer_updates', 'parameter_updates', 'backwards', 'epochs']: assert result[key] == 0
    assert not result['optimizer_constructed'] and not result['new_trained_checkpoint_created']
    assert not result['native_or_reserved_used'] and not result['app_promotion'] and not result['goal_complete']
    assert exported['run_results_present'] == success and exported['failure_present'] == failure
    outer = read(OUT / 'supervisor_receipt.json'); code = int((OUT / 'diagnostic_exit_code.txt').read_text())
    assert outer['complete'] and outer['protocol_sha256'] == pin and outer['diagnostic_exit_code'] == code
    assert outer['cap_seconds'] == 630 and outer['kill_grace_seconds'] == 30
    assert outer['within_external_bound'] == (outer['seconds'] <= 660) and (code == 0) == success
    for key, limit in p['forward_call_limits'].items(): assert 0 <= result[key] <= limit
    assert 0 <= result['component_gradient_calls'] <= 70
    gradients = None
    if (OUT / 'outputs/gradient_summary.json').is_file():
        gradients = gradient_summary_check(OUT, p, success)
    saved = saved_initial_readback(p, success); replay = None
    if success:
        assert result['seconds'] <= 600 and result['peak_allocated_VRAM_bytes'] <= 20*1024**3
        assert result['component_gradient_calls'] == 70 and result['states_before_after'] == p['expected_states']
        assert {k: result[k] for k in p['forward_call_limits']} == p['forward_call_limits']
        assert result['raw_and_PNG_parity_exact_cases'] == 50 and result['all57_improvement_gradients_nonzero']
        assert gradients and not gradients['zero_improvement_tensor_names']
        assert result['summary_sha256'] == sha(OUT / 'outputs/gradient_summary.json')
        replay = frozen_CPU_replay(p)
    verify_local_basis(p)
    assert time.monotonic() - start <= 600
    value = {'complete': True, 'diagnostic_complete': success, 'failure_retained': failure,
             'archive_sha256': digest, 'archive_bytes': size, 'protocol_sha256': pin,
             'checker_sha256': sha(Path(__file__)), 'members_verified': imported['members'],
             'saved_gradient_readback': gradients, 'saved_initial_readback': saved, 'CPU_replay': replay,
             'saved_gradients_audited_without_independent_differentiation': True,
             'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0,
             'training_capacity_pass': False, 'app_promotion': False, 'goal_complete': False,
             'seconds': time.monotonic() - start, 'cap_seconds': 600}
    write(ROOT / 'outputs/cctv_dgp_spatial_decoder_v39_independent_audit.json', value)
    print(json.dumps({'complete': True, 'diagnostic_complete': success, 'failure_retained': failure,
                      'seconds': value['seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--expected-bytes', required=True, type=int); args = parser.parse_args()
    audit(args.expected_sha, args.expected_bytes)
