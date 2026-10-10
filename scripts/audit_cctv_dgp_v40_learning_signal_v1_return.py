"""Prospective independent saved-gradient audit; frozen CPU replay, no derivatives."""
import argparse
import ast
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
from types import MethodType, SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_v40_learning_signal_v1_vm'
STEM = 'cctv-dgp-v40-learning-signal-v1'
BUNDLE = ROOT / 'outputs' / NAME
PREP = ROOT / 'outputs/cctv_dgp_v40_learning_signal_v1_preparation'
OUT = ROOT / 'outputs/cctv_dgp_v40_learning_signal_v1_return'
PREFIX = OUT.name + '/'
RETURNED_ASSETS = {'scripts/cctv_dgp_v40_learning_signal_v1_vm.py', 'cctv_dgp_v40_learning_signal_v1_decoder.py',
                   'frozen_definitions.py', 'untrained_initial_decoder.pth', 'stopped_decoder_input.pth'}


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_basis(p):
    for name, value in p['assets_sha256'].items(): assert sha(BUNDLE/name) == value, name
    for name, value in p['local_basis_sha256'].items(): assert sha(ROOT/name) == value, name
    checked = read(PREP/'independent_packet_audit.json')
    assert checked['complete'] and checked['protocol_sha256'] == sha(BUNDLE/'protocol.json')
    assert p['retained_capacity_gates'] == read(ROOT/'outputs/cctv_dgp_spatial_fit_vm_v40/protocol.json')['retained_capacity_gates']


def safe_members(members, p):
    allowed = RETURNED_ASSETS | {'protocol.json', 'export_manifest.json', 'diagnostic.log', 'diagnostic_exit_code.txt',
        'supervisor_receipt.json', 'outputs/results.json', 'outputs/failure.json', 'outputs/cohort_loss_setup.json', 'outputs/gradient_summary.json'}
    for c in p['cases']:
        cid = c['id']
        allowed.update('outputs/baseline/'+cid+s for s in ['.npy', '.png', '_target_embedding.npy'])
        for state in ['initial', 'stopped50']:
            allowed.update('outputs/'+state+'/'+cid+s for s in ['.npy', '.png', '_delta.npy'])
    for state in ['initial', 'stopped50']:
        for cohort in p['cohorts']:
            allowed.add('outputs/'+state+'/'+cohort+'_gradient_components.npy')
            allowed.update('outputs/'+state+'/gradients/'+cohort+'_batch'+str(i)+'.npy' for i in range(10))
    seen = set(); result = []; total = 0
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk(), 'Regular files only'
        assert member.name.startswith(PREFIX) and not any(c in member.name for c in ['\\', ':', '\x00'])
        name = member.name[len(PREFIX):]; parts = PurePosixPath(name).parts
        assert parts and not PurePosixPath(name).is_absolute() and all(v not in ['.', '..'] for v in parts)
        assert name in allowed and name.lower() not in seen, name
        assert 0 <= member.size <= 8*1024**2
        seen.add(name.lower()); total += member.size; result.append((member, name))
        assert total <= p['budgets']['export_uncompressed_bytes'] and len(seen) <= 970
    return result, total


def import_return(p, pin, digest, size):
    archive = ROOT/'outputs'/(STEM+'-results.tar.gz')
    assert archive.stat().st_size == size and sha(archive) == digest
    assert Path(str(archive)+'.sha256').read_text().strip().split() == [digest, archive.name]
    receipt = read(ROOT/'outputs'/(STEM+'-export.json'))
    assert receipt['complete'] and receipt['archive_sha256'] == digest and receipt['bytes'] == size
    assert receipt['training_success_not_implied'] and receipt['optimizer_updates'] == 0
    assert not OUT.exists(), 'Preserve every previous/partial return'
    with tarfile.open(archive, 'r:gz') as tar:
        members, total = safe_members(tar.getmembers(), p); OUT.mkdir()
        for member, name in members:
            dest = (OUT/name).resolve(); assert dest.is_relative_to(OUT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, dest.open('xb') as target:
                for block in iter(lambda: source.read(1024**2), b''): target.write(block)
    mapping = {name: sha(OUT/name) for _, name in members}; assert mapping['protocol.json'] == pin
    for name in RETURNED_ASSETS: assert mapping[name] == p['assets_sha256'][name]
    manifest = read(OUT/'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert manifest['files_sha256'] == {name: value for name, value in mapping.items() if name != 'export_manifest.json'}
    assert manifest['data_and_original_weights_omitted_and_bound_to_protocol']
    imported = {'complete': True, 'archive_sha256': digest, 'archive_bytes': size, 'members': len(members),
                'uncompressed_bytes': total, 'files_sha256': mapping, 'returned_code_executed': False}
    write(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_return_import.json', imported)
    return receipt, imported


def gradient_readback(p):
    import numpy as np
    summary = read(OUT/'outputs/gradient_summary.json')
    assert summary['complete'] and summary['terms'] == p['terms'] and summary['parameter_layout'] == p['parameter_layout']
    assert set(summary['states']) == {'initial', 'stopped50'}
    delta = np.load(BUNDLE/'saved_parameter_delta.npy', allow_pickle=False)
    assert delta.dtype == np.float64 and delta.shape == (17952,) and np.isfinite(delta).all()
    reports = {}; count = 0
    def close(a, b): assert np.allclose(a, b, rtol=2e-10, atol=1e-12)
    for state in ['initial', 'stopped50']:
        assert set(summary['states'][state]) == set(p['cohorts']); reports[state] = {}
        for cohort, ids in p['cohorts'].items():
            row = summary['states'][state][cohort]; assert len(row['batches']) == 10
            total = np.zeros((7, 17952), np.float64); values = np.zeros(7, np.float64)
            for i, batch in enumerate(row['batches']):
                path = OUT/('outputs/'+state+'/gradients/'+cohort+'_batch'+str(i)+'.npy')
                a = np.load(path, allow_pickle=False)
                assert a.dtype == np.float64 and a.shape == (7, 17952) and np.isfinite(a).all()
                assert sha(path) == batch['gradient_sha256'] and batch['batch'] == i and batch['ids'] == ids[i*5:i*5+5]
                v = np.asarray(batch['values'], np.float64); assert v.shape == (7,) and np.isfinite(v).all()
                per_case = np.asarray([summary['case_term_values'][state+'/'+cid] for cid in batch['ids']], np.float32)
                assert per_case.shape == (5, 7) and np.isfinite(per_case).all()
                assert np.allclose(per_case.mean(0)/np.float32(10), v, rtol=2e-6, atol=1e-9)
                if state == 'initial': assert np.count_nonzero(a[3:]) == np.count_nonzero(v[3:]) == np.count_nonzero(per_case[:, 3:]) == 0
                close(np.linalg.norm(a, axis=1), batch['norms']); total += a; values += v; count += 7
            path = OUT/('outputs/'+state+'/'+cohort+'_gradient_components.npy'); saved = np.load(path, allow_pickle=False)
            assert saved.dtype == np.float64 and saved.shape == total.shape and np.array_equal(saved, total) and sha(path) == row['gradient_sha256']
            assert np.array_equal(values, np.asarray(row['values']))
            norms = np.linalg.norm(total, axis=1); direction = -total.sum(0); dn = np.linalg.norm(direction)
            close(norms, row['component_norms']); close(total @ total.T, row['component_gram'])
            close(total @ delta, row['gradient_dot_actual50_weight_change'])
            products = total @ direction; close(products, row['negative_total_direction_component_derivatives'])
            cosines = [None if n == 0 or dn == 0 else float(d/(n*dn)) for n, d in zip(norms, products)]
            for actual, reported in zip(cosines, row['negative_total_direction_cosines']):
                if actual is None: assert reported is None
                else: close(actual, reported)
            assert set(row['partitions']) == {r['name'] for r in p['parameter_layout']}
            for r in p['parameter_layout']: close(np.linalg.norm(total[:, r['start']:r['end']], axis=1), row['partitions'][r['name']])
            reports[state][cohort] = {'values': row['values'], 'component_norms': row['component_norms'],
                'negative_total_direction_component_derivatives': row['negative_total_direction_component_derivatives'],
                'endpoint_derivatives_only_not_optimizer_trajectory': True}
    assert count == 280 and set(summary['case_term_values']) == {s+'/'+c['id'] for s in ['initial', 'stopped50'] for c in p['cases']}
    return {'component_queries_readback': count, 'tensor_partitions': 57, 'reports': reports,
            'all_initial_preservation_values_and_gradients_exact_zero': True, 'local_gradient_calls': 0}


def saved_composition(p):
    import numpy as np
    from PIL import Image
    def pixels(path, mode='RGB'):
        with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()
    rows = []; compositions = 0
    for c in p['cases']:
        cid = c['id']; x = pixels(BUNDLE/c['input']); mask = pixels(BUNDLE/c['observed'], 'L') > 0
        baseline = np.load(OUT/('outputs/baseline/'+cid+'.npy'), allow_pickle=False)
        assert baseline.dtype == np.float32 and baseline.shape == (256, 256, 3) and np.isfinite(baseline).all()
        assert baseline.min() >= 0 and baseline.max() <= 1 and np.array_equal(baseline[~mask], (x.astype(np.float32)/np.float32(255))[~mask])
        base_png = np.floor(baseline*np.float32(255)).astype(np.uint8); base_png[~mask] = x[~mask]
        assert np.array_equal(base_png, pixels(OUT/('outputs/baseline/'+cid+'.png')))
        target = np.load(OUT/('outputs/baseline/'+cid+'_target_embedding.npy'), allow_pickle=False)
        assert target.dtype == np.float32 and target.shape == (512,) and np.isfinite(target).all() and abs(float(target @ target)-1) < 1e-5
        for state in ['initial', 'stopped50']:
            prefix = OUT/('outputs/'+state+'/'+cid)
            a = np.load(str(prefix)+'.npy', allow_pickle=False); delta = np.load(str(prefix)+'_delta.npy', allow_pickle=False)
            assert a.dtype == delta.dtype == np.float32 and a.shape == delta.shape == baseline.shape and np.isfinite(a).all() and np.isfinite(delta).all()
            assert a.min() >= 0 and a.max() <= 1 and np.abs(delta).max() <= .5
            # Independent float64 readback allows only the prospectively frozen
            # float32 reduction error, not a structure/preservation gate change.
            centered = delta.astype(np.float64)-delta[mask].astype(np.float64).mean(0)
            expected = np.where(mask[..., None], np.clip(baseline.astype(np.float64)+centered, 0, 1), x.astype(np.float32)/np.float32(255))
            assert np.max(np.abs(expected-a)) <= 2e-7
            png = np.floor(a*np.float32(255)).astype(np.uint8); png[~mask] = x[~mask]
            assert np.array_equal(png, pixels(str(prefix)+'.png')) and np.array_equal(png[~mask], x[~mask])
            if state == 'initial': assert np.array_equal(a, baseline) and np.count_nonzero(delta) == 0
            historical = c.get('historical_initial_raw' if state == 'initial' else 'historical_stopped_raw')
            if historical: assert np.max(np.abs(a-np.load(BUNDLE/historical, allow_pickle=False))) <= p['historical_raw_tolerance']
            rows.append({'state': state, 'id': cid, 'raw_correction_RMS': float(np.sqrt(np.square(a.astype(np.float64)-baseline).mean())),
                         'near_tanh_bound_fraction': float(np.mean(np.abs(delta) >= .4995)),
                         'observed_clipped_fraction': float(np.mean((a[mask] == 0) | (a[mask] == 1)))})
            compositions += 1
    assert compositions == 200
    assert read(OUT/'outputs/cohort_loss_setup.json') == read(BUNDLE/'historical_cohort_loss_setup.json')
    return {'raw_compositions': compositions, 'baseline_cases': 100, 'PNG_compositions_exact': 300, 'rows': rows}


def CPU_replay(p):
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    sys.path.insert(0, str(BUNDLE))
    from cctv_dgp_v40_learning_signal_v1_decoder import SpatialDGPCandidateV40LearningSignalV1
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_batchmatched_identity_v26 import objective_terms
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    seed = torch.load(BUNDLE/'untrained_initial_decoder.pth', map_location='cpu', weights_only=True)
    stopped = torch.load(BUNDLE/'stopped_decoder_input.pth', map_location='cpu', weights_only=True)
    candidate = SpatialDGPCandidateV40LearningSignalV1(original, seed)
    identity = FixedObservedIdentity(BUNDLE/'weights/w600k_r50.onnx', 'cpu')
    def states(): return {'original': state_hash(original.net), 'decoder': state_hash(candidate.decoder), 'reference_decoder': state_hash(candidate.reference_decoder), 'recognizer': state_hash(identity)}
    assert states() == p['expected_initial_states']
    tree = ast.parse((BUNDLE/'frozen_definitions.py').read_text()); ns = {'torch': torch, 'F': F}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name != 'require_vm']
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<verified-fixed-functions-only>', 'exec'), ns)
    z = torch.arange(-6, 7, dtype=torch.float32); kernel = torch.exp(-.5*(z/2).square()); kernel /= kernel.sum()
    head = SimpleNamespace(kernel=kernel, reflect_indices=torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1))))
    head.blur = MethodType(ns['blur'], head); head.high = MethodType(ns['high'], head); ns['head'] = head
    old = read(BUNDLE/'historical_cohort_loss_setup.json'); normalizers = tuple(torch.tensor(old[k]) for k in ['feature_normalizer', 'interior_normalizer'])
    lookup = {c['id']: c for c in p['cases']}; refs = {r['id']: r for r in p['references']}
    summary = read(OUT/'outputs/gradient_summary.json'); raw_err = delta_err = vector_err = scalar_err = 0.; byte_err = count = 0
    def pixels(path, mode='RGB'):
        with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()
    def canonical(a): return torch.from_numpy(a.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None]
    with torch.inference_mode():
        for state, weights in [('initial', seed), ('stopped50', stopped)]:
            candidate.decoder.load_state_dict(weights, strict=True)
            expected_states = dict(p['expected_initial_states']); expected_states['decoder'] = p['decoder_states'][state]
            assert states() == expected_states
            for ids in p['CPU_replay_batches']:
                group = [lookup[cid] for cid in ids]; cameras = [pixels(BUNDLE/c['input']) for c in group]
                masks = np.stack([pixels(BUNDLE/c['observed'], 'L') > 0 for c in group]); features = []; interiors = []; valid = []
                for c, mask in zip(group, masks):
                    def erode(radius):
                        size = 2*radius+1; integral = np.pad(np.pad(mask.astype(np.int64), radius).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                        return integral[size:, size:]-integral[:-size, size:]-integral[size:, :-size]+integral[:-size, :-size] == size**2
                    patch = np.zeros((256, 256), bool)
                    for x, y in np.floor(c['landmarks5_canvas_xy']).astype(int): patch[max(0, y-12):min(256, y+12), max(0, x-12):min(256, x+12)] = True
                    features.append(patch & erode(6)); interiors.append(erode(6)); valid.append(erode(3))
                x = torch.cat([canonical(v) for v in cameras]); mask = torch.from_numpy(masks.astype(np.float32))[:, None]
                parts = candidate.forward_components(x, mask); pred = parts['result']
                target = torch.cat([canonical(pixels(BUNDLE/c['target'])) for c in group])
                grid = torch.from_numpy(np.stack([grid112(refs[c['source_person_or_reference']]['matrix112']) for c in group]))
                truth = identity.embedding(target, mask, grid)
                b = {'x': x, 'base': parts['original_raw'], 'target': target, 'mask': mask, 'grid': grid, 'truth': truth,
                     'feature': torch.from_numpy(np.stack(features).astype(np.float32))[:, None],
                     'interior': torch.from_numpy(np.stack(interiors).astype(np.float32))[:, None],
                     'valid7': torch.from_numpy(np.stack(valid).astype(np.float32))[:, None],
                     'degraded_weight': torch.tensor([0. if c['profile'] == 'clear' else 1.25 for c in group]),
                     'clear_weight': torch.tensor([1. if c['profile'] == 'clear' else 0. for c in group])}
                terms = objective_terms(b, pred, identity, ns['mean'], ns['feature_errors'], ns['ssim'], normalizers)
                for slot, c in enumerate(group):
                    cid = c['id']; raw = pred[slot].permute(1, 2, 0).numpy(); delta = parts['spatial_delta'][slot].permute(1, 2, 0).numpy()
                    saved = np.load(OUT/('outputs/'+state+'/'+cid+'.npy'), allow_pickle=False)
                    error = float(np.max(np.abs(raw-saved))); assert error <= p['CPU_raw_replay_tolerance']; raw_err = max(raw_err, error)
                    error = float(np.max(np.abs(delta-np.load(OUT/('outputs/'+state+'/'+cid+'_delta.npy'), allow_pickle=False))))
                    assert error <= p['CPU_raw_replay_tolerance']; delta_err = max(delta_err, error)
                    png = np.floor(raw*np.float32(255)).astype(np.uint8); png[~masks[slot]] = cameras[slot][~masks[slot]]
                    error = int(np.max(np.abs(png.astype(np.int16)-pixels(OUT/('outputs/'+state+'/'+cid+'.png')).astype(np.int16))))
                    assert error <= p['CPU_PNG_byte_tolerance']; byte_err = max(byte_err, error)
                    error = float(np.max(np.abs(truth[slot].numpy()-np.load(OUT/('outputs/baseline/'+cid+'_target_embedding.npy'), allow_pickle=False))))
                    assert error <= p['CPU_embedding_absolute_tolerance']; vector_err = max(vector_err, error)
                    actual = np.asarray([float(terms[k][slot]) for k in p['terms']]); saved_values = summary['case_term_values'][state+'/'+cid]
                    error = float(np.max(np.abs(actual-saved_values))); assert error <= p['CPU_scalar_replay_tolerance']; scalar_err = max(scalar_err, error); count += 1
                assert states() == expected_states
    assert count == 40 and all(not v.requires_grad and v.grad is None for v in candidate.parameters())
    assert all(not v.requires_grad and v.grad is None for v in identity.parameters())
    return {'cases': count, 'four_metadata_batches_at_two_states': True, 'raw_maximum_error': raw_err, 'delta_maximum_error': delta_err,
            'PNG_maximum_byte_difference': byte_err, 'embedding_maximum_error': vector_err, 'scalar_maximum_error': scalar_err,
            'states_unchanged': True, 'local_gradient_calls': 0, 'local_optimizer_updates': 0}


def audit(digest, size):
    start = time.monotonic(); p = read(BUNDLE/'protocol.json'); pin = sha(BUNDLE/'protocol.json'); verify_basis(p)
    exported, imported = import_return(p, pin, digest, size)
    success = (OUT/'outputs/results.json').is_file(); failure = (OUT/'outputs/failure.json').is_file(); assert success != failure
    result = read(OUT/('outputs/results.json' if success else 'outputs/failure.json')); assert result['protocol_sha256'] == pin
    for key in ['optimizer_updates', 'parameter_updates', 'backwards', 'epochs']: assert result[key] == 0
    assert not result['optimizer_constructed'] and not result['new_trained_checkpoint_created']
    assert not result['native_or_reserved_used'] and not result['app_promotion'] and not result['goal_complete']
    assert exported['run_results_present'] == success and exported['failure_present'] == failure
    outer = read(OUT/'supervisor_receipt.json'); code = int((OUT/'diagnostic_exit_code.txt').read_text())
    assert outer['complete'] and outer['protocol_sha256'] == pin and outer['diagnostic_exit_code'] == code
    assert outer['cap_seconds'] == 630 and outer['kill_grace_seconds'] == 30 and outer['within_external_bound'] == (outer['seconds'] <= 660)
    assert (code == 0) == success and 0 <= result['component_gradient_calls'] <= 280
    for key, limit in p['forward_call_limits'].items(): assert 0 <= result[key] <= limit
    gradients = composition = replay = None
    if success:
        assert result['seconds'] <= 600 and result['peak_allocated_VRAM_bytes'] <= 20*1024**3 and result['component_gradient_calls'] == 280
        assert {k: result[k] for k in p['forward_call_limits']} == p['forward_call_limits']
        for state in ['initial', 'stopped50']:
            expected = dict(p['expected_initial_states']); expected['decoder'] = p['decoder_states'][state]
            assert result['endpoint_states_unchanged'][state] == expected
        assert result['summary_sha256'] == sha(OUT/'outputs/gradient_summary.json')
        assert [r['id'] for r in result['historical_parity']] == p['cohorts']['not_yet_optimized']
        assert all(0 <= r[k] <= p['historical_raw_tolerance'] for r in result['historical_parity'] for k in ['initial_error', 'stopped_error'])
        gradients = gradient_readback(p); composition = saved_composition(p); replay = CPU_replay(p)
    verify_basis(p); assert time.monotonic()-start <= 600
    value = {'complete': True, 'diagnostic_complete': success, 'failure_retained': failure, 'protocol_sha256': pin,
             'archive_sha256': digest, 'archive_bytes': size, 'checker_sha256': sha(Path(__file__)), 'members_verified': imported['members'],
             'gradient_readback': gradients, 'saved_composition': composition, 'CPU_replay': replay,
             'saved_gradients_audited_without_independent_differentiation': True, 'local_gradient_calls': 0,
             'local_optimizer_updates': 0, 'VM_calls': 0, 'app_promotion': False, 'training_capacity_pass': False,
             'goal_complete': False, 'seconds': time.monotonic()-start, 'cap_seconds': 600}
    write(ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_independent_audit.json', value)
    print({'complete': True, 'diagnostic_complete': success, 'failure_retained': failure, 'seconds': value['seconds']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--expected-sha', required=True); parser.add_argument('--expected-bytes', required=True, type=int)
    a = parser.parse_args(); audit(a.expected_sha, a.expected_bytes)
