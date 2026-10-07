"""Prospective independent sampling-diagnostic audit; CPU forwards only."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_vm'
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_return'
CLOSED = ROOT / 'outputs/cctv_dgp_profile_batches_v31_return'
V31 = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
ACTIVE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
PIN = 'PROTOCOL_PIN_TO_FILL'
STEM = 'cctv-dgp-feature-fusion-gradient-v1'
PREFIX = 'cctv_dgp_feature_fusion_gradient_v1_return/'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def allowed_members(p):
    allowed = set(p['assets_sha256']) | {'protocol.json', 'export_manifest.json', 'diagnostic.log', 'diagnostic_exit_code.txt', 'supervisor_receipt.json', 'outputs/results.json', 'outputs/failure.json'}
    for state in [0, 50]:
        for cohort in p['cohorts']:
            folder = f'outputs/state{state}_{cohort["name"]}/'
            allowed.update(folder + name for name in ['receipt.json', 'gradient_components.npy'])
            allowed.update(folder + f'batch{i}.npy' for i in range(10))
            for case in cohort['cases']:
                allowed.update(folder + case['id'] + suffix for suffix in ['.png', '.npy', '_embedding.npy'])
                if state == 0: allowed.add(folder + case['id'] + '_target_embedding.npy')
    return allowed


def safe_members(members, allowed):
    result, seen, total = [], set(), 0
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk(), 'Regular files only'
        name = member.name
        assert name.startswith(PREFIX) and '\\' not in name and ':' not in name and '\x00' not in name
        relative = name[len(PREFIX):]
        parts = PurePosixPath(relative).parts
        assert parts and not PurePosixPath(relative).is_absolute() and all(x not in ['.', '..'] for x in parts)
        assert relative in allowed and relative.lower() not in seen, 'Unexpected or duplicate member'
        assert 0 <= member.size <= 64 * 1024**2, 'Per-file64MiB bound'
        seen.add(relative.lower()); total += member.size
        assert total <= 1536 * 1024**2 and len(seen) <= 800, 'Finite return size/count'
        result.append((member, relative))
    return result, total


def verify_basis(p):
    for name, digest in p['assets_sha256'].items(): assert sha(BUNDLE / name) == digest, name
    for name, digest in p['local_basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    for name, digest in p['V31_dependencies_sha256'].items():
        source = V31 / name if name == 'protocol.json' or name in read(V31 / 'protocol.json')['assets_sha256'] else CLOSED / name
        assert sha(source) == digest, name
    for base, key in [(PARENT, 'parent_dependencies_sha256'), (ACTIVE, 'active_decoder_dependencies_sha256'), (MIXED, 'TRAIN_assets_sha256')]:
        for name, digest in p[key].items(): assert sha(base / name) == digest, name


def import_return(p, expected_sha, expected_bytes):
    archive = ROOT / 'outputs' / (STEM + '-results.tar.gz')
    assert archive.stat().st_size == expected_bytes and sha(archive) == expected_sha
    checksum = Path(str(archive) + '.sha256').read_text(encoding='ascii').strip().split()
    assert checksum == [expected_sha, archive.name]
    exported = read(ROOT / 'outputs' / (STEM + '-export.json'))
    assert exported['complete'] and exported['archive_sha256'] == expected_sha and exported['bytes'] == expected_bytes
    assert exported['optimizer_updates'] == 0 and exported['training_success_not_implied']
    assert not OUT.exists(), 'Preserve previous import or partial audit'
    with tarfile.open(archive, 'r:gz') as tar:
        members, total = safe_members(tar.getmembers(), allowed_members(p))
        OUT.mkdir()
        for member, relative in members:
            destination = (OUT / relative).resolve(); assert destination.is_relative_to(OUT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, destination.open('xb') as target:
                for block in iter(lambda: source.read(1024**2), b''): target.write(block)
    hashes = {name: sha(OUT / name) for _, name in members}
    assert hashes['protocol.json'] == PIN
    assert all(hashes.get(name) == digest for name, digest in p['assets_sha256'].items())
    manifest = read(OUT / 'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == PIN
    assert manifest['files_sha256'] == {name: digest for name, digest in hashes.items() if name != 'export_manifest.json'}
    receipt = {'complete': True, 'archive_sha256': expected_sha, 'archive_bytes': expected_bytes, 'members': len(members), 'uncompressed_bytes': total,
               'files_sha256': hashes, 'returned_code_executed': False, 'local_gradient_calls': 0, 'goal_complete': False}
    write(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_return_import.json', receipt)
    return exported, receipt


def gradient_arithmetic(folder, state, cohort, p):
    import numpy as np
    saved = read(folder / 'receipt.json')
    assert saved['complete'] and saved['state'] == state and saved['cohort'] == cohort['name'] and saved['cases'] == 50 and saved['gradient_queries'] == 70
    assert len(saved['batches']) == 10 and len(saved['replay']) == 50
    total = np.zeros((7, 978243), np.float64); values = np.zeros(7, np.float64); batch_norms = []; checked = 0
    for index, row in enumerate(saved['batches']):
        path = folder / f'batch{index}.npy'; assert sha(path) == row['sha256']
        matrix = np.load(path, allow_pickle=False)
        assert matrix.shape == (7, 978243) and matrix.dtype == np.float32 and np.isfinite(matrix).all()
        assert row['batch'] == index and row['ids'] == [c['id'] for c in cohort['cases'][index * 5:index * 5 + 5]]
        value = np.asarray(row['values'], np.float64); assert value.shape == (7,) and np.isfinite(value).all() and (value >= 0).all()
        if state == 0: assert np.count_nonzero(matrix[3:]) == 0 and np.count_nonzero(value[3:]) == 0
        matrix = matrix.astype(np.float64); norms = np.linalg.norm(matrix, axis=1)
        assert np.allclose(norms, row['norms'], rtol=2e-10, atol=1e-11)
        batch_norms.append(float(np.linalg.norm(matrix.sum(0)))); total += matrix; values += value; checked += matrix.size
    aggregate = np.load(folder / 'gradient_components.npy', allow_pickle=False)
    assert aggregate.dtype == np.float64 and aggregate.shape == (7, 978243) and np.isfinite(aggregate).all()
    assert np.array_equal(total, aggregate) and sha(folder / 'gradient_components.npy') == saved['aggregate_sha256']; checked += aggregate.size
    norms = np.linalg.norm(total, axis=1); gram = total @ total.T; denominator = norms[:, None] * norms[None, :]
    cosines = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
    for actual, expected in [(norms, saved['component_norms']), (gram, saved['component_gram']), (cosines, saved['component_cosines'])]: assert np.allclose(actual, expected, rtol=2e-10, atol=1e-11)
    assert np.array_equal(values, np.asarray(saved['component_values'])) and float(values.sum()) == saved['objective']
    improvement, preservation, objective = total[:3].sum(0), total[3:].sum(0), total.sum(0)
    dot_cosine = float(improvement @ preservation) / max(float(np.linalg.norm(improvement) * np.linalg.norm(preservation)), 1e-300)
    report = {'state': state, 'cohort': cohort['name'], 'values_checked': checked, 'component_values': values.tolist(),
              'objective_gradient_norm': float(np.linalg.norm(objective)), 'improvement_gradient_norm': float(np.linalg.norm(improvement)),
              'preservation_gradient_norm': float(np.linalg.norm(preservation)), 'improvement_preservation_cosine': dot_cosine,
              'sampled_batch_gradient_coherence': float(np.linalg.norm(objective)) / max(sum(batch_norms), 1e-300),
              'directional_derivatives_along_negative_original_objective': (-(total @ objective)).tolist(),
              'selected23_improvement_norms': {r['name']: float(np.linalg.norm(improvement[r['start']:r['end']])) for r in p['parameter_layout']},
              'gradient_arrays_sha256': {f.name: sha(f) for f in folder.glob('batch*.npy')}}
    partitions = {}
    for label, names in [('feature_fusion', set(p['fusion_parameter_names'])),
                         ('decoder', set(p['decoder_parameter_names']))]:
        indices = np.concatenate([np.arange(r['start'], r['end']) for r in p['parameter_layout'] if r['name'] in names])
        block = total[:, indices]; ni = np.linalg.norm(block[:3].sum(0)); npres = np.linalg.norm(block[3:].sum(0))
        partitions[label] = {'parameters': len(indices), 'component_norms': np.linalg.norm(block, axis=1).tolist(),
                             'improvement_norm': float(ni), 'preservation_norm': float(npres),
                             'improvement_preservation_cosine': float(block[:3].sum(0) @ block[3:].sum(0)) / max(float(ni * npres), 1e-300),
                             'component_directional_derivatives': (-(block @ block.sum(0))).tolist(),
                             'all_improvement_tensors_nonzero': all(report['selected23_improvement_norms'][n] > 0 for n in names)}
    report['partition_analysis'] = partitions
    return saved, report


def CPU_replay(p, started):
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    for path in [PARENT, ACTIVE, V31]: sys.path.insert(0, str(path))
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_app_input_v28 import canonical_tensor
    from cctv_dgp_mean_centered_decoder_v29 import center_observed_delta
    from cctv_dgp_batchmatched_identity_v26 import objective_terms
    from types import MethodType
    torch.set_num_threads(4)
    spec = importlib.util.spec_from_file_location('pinned_local_V31_CPU_helpers', V31 / 'scripts/cctv_dgp_profile_batches_v31_vm.py')
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    _, definitions, filters = helper.original_functions(PARENT)
    namespace = {'torch': torch, 'F': F}; exec(compile(ast.Module(body=filters, type_ignores=[]), '<pinned-fixed-filter-CPU-only>', 'exec'), namespace)
    class Fixed: pass
    fixed = Fixed(); z = torch.arange(-6, 7, dtype=torch.float32); k = torch.exp(-.5 * (z / 2).square()); fixed.kernel = k / k.sum()
    fixed.reflect_indices = torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1)))
    fixed.blur = MethodType(namespace['blur'], fixed); fixed.high = MethodType(namespace['high'], fixed)
    original, _ = load_frozen_dgp_restorer(PARENT / 'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    candidate, _ = load_frozen_dgp_restorer(CLOSED / 'outputs/update50/dgp_candidate_v31.pth', expected_sha256=p['V31_dependencies_sha256']['outputs/update50/dgp_candidate_v31.pth'], device='cpu')
    recognizer = FixedObservedIdentity(PARENT / 'weights/w600k_r50.onnx', 'cpu')
    model_states = [state_hash(m) for m in [original.net, candidate.net, recognizer]]
    assert model_states == [p['original_DGP_state'], p['stopped50_DGP_state'], p['recognizer_state']]
    namespace.update({'head': fixed, 'identity': recognizer}); exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-original-loss-CPU-only>', 'exec'), namespace)
    normalizers = tuple(torch.tensor(x, dtype=torch.float32) for x in p['normalizers'])
    refs = {r['id']: r for r in read(V31 / 'protocol.json')['training_references']}
    keys = ['x', 'base', 'target', 'mask', 'feature', 'interior', 'valid7', 'grid', 'truth', 'degraded_weight', 'clear_weight']
    worst_raw, worst_png, worst_vector, worst_value = 0., 0, 0., 0.; replayed = 0
    with torch.inference_mode():
        for cohort in p['cohorts']:
            items = []
            for case in cohort['cases']:
                with Image.open(MIXED / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['observed']) as im: mask = np.asarray(im).copy() > 0
                def erode(radius):
                    size = 2 * radius + 1; summed = np.pad(np.pad(mask.astype(np.int64), radius).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
                    return summed[size:, size:] - summed[:-size, size:] - summed[size:, :-size] + summed[:-size, :-size] == size * size
                interior = erode(6); feature = np.zeros((256, 256), bool)
                for point in case['landmarks5_canvas_xy']:
                    xx, yy = np.floor(point).astype(int); feature[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = True
                feature &= interior
                items.append({'case': case, 'camera': camera, 'mask8': mask, 'x': canonical_tensor(camera, 'cpu'), 'target': canonical_tensor(target, 'cpu'),
                              'mask': torch.from_numpy(mask.astype(np.float32))[None, None], 'feature': torch.from_numpy(feature.astype(np.float32))[None, None],
                              'interior': torch.from_numpy(interior.astype(np.float32))[None, None], 'valid7': torch.from_numpy(erode(3).astype(np.float32))[None, None],
                              'grid': torch.from_numpy(grid112(refs[case['source_person_or_reference']]['matrix112']))[None],
                              'degraded_weight': torch.tensor([0. if case['profile'] == 'clear' else 1.25]), 'clear_weight': torch.tensor([1. if case['profile'] == 'clear' else 0.])})
            for begin in range(0, 50, 5):
                batch = items[begin:begin + 5]; x = torch.cat([i['x'] for i in batch]); mask = torch.cat([i['mask'] for i in batch]); grid = torch.cat([i['grid'] for i in batch])
                baseline = torch.where(mask.bool(), original(x), x); truth = recognizer.embedding(torch.cat([i['target'] for i in batch]), mask, grid)
                for index, item in enumerate(batch): item['base'] = baseline[index:index + 1]; item['truth'] = truth[index:index + 1]
            for state in [0, 50]:
                folder = OUT / f'outputs/state{state}_{cohort["name"]}'
                receipt = read(folder / 'receipt.json')
                for begin in range(0, 50, 5):
                    assert time.monotonic() - started < 1800, 'Local forward-only audit1800s cap'
                    batch = items[begin:begin + 5]; b = {key: torch.cat([i[key] for i in batch]) for key in keys}
                    net = original if state == 0 else candidate
                    pred = center_observed_delta(torch.where(b['mask'].bool(), net(b['x']), b['x']), b['base'], b['x'], b['mask'])
                    raw = pred.permute(0, 2, 3, 1).numpy().copy()
                    delivered = []
                    for item, fresh in zip(batch, raw):
                        cid = item['case']['id']; saved = np.load(folder / (cid + '.npy'), allow_pickle=False)
                        assert saved.dtype == np.float32 and saved.shape == (256, 256, 3) and np.isfinite(saved).all() and (saved >= 0).all() and (saved <= 1).all()
                        error = float(np.abs(saved - fresh).max()); assert error <= p['CPU_raw_absolute_tolerance']; worst_raw = max(worst_raw, error)
                        with Image.open(folder / (cid + '.png')) as im: png = np.asarray(im.convert('RGB')).copy()
                        assert np.array_equal(png, np.where(item['mask8'][..., None], np.floor(saved * np.float32(255)), item['camera']).astype(np.uint8))
                        expected = np.where(item['mask8'][..., None], np.floor(fresh * np.float32(255)), item['camera']).astype(np.uint8)
                        byte = int(np.abs(png.astype(int) - expected.astype(int)).max()); assert byte <= p['CPU_PNG_byte_tolerance']; worst_png = max(worst_png, byte)
                        assert np.array_equal(png[~item['mask8']], item['camera'][~item['mask8']]); delivered.append(png); replayed += 1
                    vectors = recognizer.embedding(torch.cat([canonical_tensor(png, 'cpu') for png in delivered]), b['mask'], b['grid']).numpy()
                    for item, vector in zip(batch, vectors):
                        cid = item['case']['id']; expected = np.load(folder / (cid + '_embedding.npy'), allow_pickle=False)
                        truth = np.load(OUT / f'outputs/state0_{cohort["name"]}/{cid}_target_embedding.npy', allow_pickle=False)
                        for a in [expected, truth]: assert a.dtype == np.float32 and a.shape == (512,) and np.isfinite(a).all() and abs(float(a @ a) - 1) < 1e-5
                        error = max(float(np.abs(expected - vector).max()), float(np.abs(truth - item['truth'][0].numpy()).max()))
                        assert error <= p['CPU_vector_absolute_tolerance']; worst_vector = max(worst_vector, error)
                    terms = objective_terms(b, pred, recognizer, namespace['mean'], namespace['feature_errors'], namespace['ssim'], normalizers)
                    values = np.asarray([float(terms[name].mean()) / 10 for name in p['terms']]); reported = np.asarray(receipt['batches'][begin // 5]['values'])
                    error = float(np.abs(values - reported).max()); assert error <= p['CPU_batch_component_value_tolerance']; worst_value = max(worst_value, error)
    assert [state_hash(m) for m in [original.net, candidate.net, recognizer]] == model_states
    assert all(not v.requires_grad and v.grad is None for m in [original, candidate, recognizer] for v in m.parameters())
    return {'cases_at_both_states': replayed, 'raw_maximum_error': worst_raw, 'PNG_maximum_byte_error': worst_png,
            'vector_maximum_error': worst_vector, 'batch_component_value_maximum_error': worst_value,
            'original_DGP_forwards': 40, 'candidate_DGP_forwards': 20, 'recognizer_forwards': 100, 'states_before_after': model_states}


def audit(expected_sha, expected_bytes):
    start = time.monotonic(); assert sha(BUNDLE / 'protocol.json') == PIN
    p = read(BUNDLE / 'protocol.json'); verify_basis(p)
    assert p['optimizer_updates'] == p['backwards'] == p['epochs'] == 0 and p['gradient_queries'] == 280
    assert [c['name'] for c in p['cohorts']] == ['exposed', 'unexposed']
    exported, imported = import_return(p, expected_sha, expected_bytes)
    result_path, failure_path = OUT / 'outputs/results.json', OUT / 'outputs/failure.json'
    assert result_path.exists() != failure_path.exists()
    terminal = read(result_path if result_path.exists() else failure_path)
    assert terminal['protocol_sha256'] == PIN and terminal['optimizer_updates'] == terminal['backwards'] == terminal['epochs'] == 0
    assert not terminal['optimizer_constructed'] and not terminal['new_checkpoint_created'] and not terminal['app_promotion'] and not terminal['goal_complete']
    assert exported['run_results_present'] == result_path.exists() and exported['failure_present'] == failure_path.exists()
    supervision = read(OUT / 'supervisor_receipt.json'); code = int((OUT / 'diagnostic_exit_code.txt').read_text())
    assert supervision['protocol_sha256'] == PIN and supervision['diagnostic_exit_code'] == code and (code == 0) == result_path.exists()
    assert supervision['cap_seconds'] == 900 and supervision['kill_grace_seconds'] == 30 and supervision['within_external_bound'] == (supervision['seconds'] <= 930)
    rows, partial = [], []
    for state in [0, 50]:
        for cohort in p['cohorts']:
            folder = OUT / f'outputs/state{state}_{cohort["name"]}'
            if not folder.exists(): continue
            if not (folder / 'receipt.json').exists():
                assert failure_path.exists(); partial.append({'state': state, 'cohort': cohort['name'], 'files': len(list(folder.iterdir()))}); continue
            saved, actual = gradient_arithmetic(folder, state, cohort, p)
            assert saved['candidate_state'] == (p['original_DGP_state'] if state == 0 else p['stopped50_DGP_state'])
            rows.append(actual)
    replay = None
    if result_path.exists():
        assert terminal['component_gradient_calls'] == 280 and len(rows) == 4 and not partial
        assert terminal['reference_DGP_forwards'] == 20 and terminal['candidate_DGP_forwards'] == 40 and terminal['recognizer_forwards'] == 100
        assert terminal['normalizers'] == p['normalizers'] and terminal['seconds'] <= 600 and terminal['peak_allocated_VRAM_bytes'] <= 20 * 1024**3
        for item in terminal['receipts']: assert sha(OUT / f'outputs/state{item["state"]}_{item["cohort"]}/receipt.json') == item['receipt_sha256']
        replay = CPU_replay(p, start); assert replay['cases_at_both_states'] == 200
    else: assert 0 <= terminal['component_gradient_calls'] <= 280 and not terminal['resume_permitted']
    receipt = {'complete': True, 'protocol_sha256': PIN, 'checker_sha256': sha(Path(__file__)), 'archive_sha256': expected_sha,
               'archive_bytes': expected_bytes, 'members_verified': imported['members'], 'diagnostic_complete': result_path.exists(),
               'VM_failure_retained': failure_path.exists(), 'cohort_gradient_analysis': rows, 'partial_cohorts_retained': partial,
               'CPU_replay': replay, 'new_training_recipe_created': False, 'optimizer_updates': 0, 'local_gradient_calls': 0,
               'local_backward_calls': 0, 'local_optimizer_updates': 0, 'native_or_reserved_used': False,
               'app_promotion': False, 'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic() - start}
    write(ROOT / 'outputs/cctv_dgp_feature_fusion_gradient_v1_independent_audit.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['complete', 'diagnostic_complete', 'VM_failure_retained', 'optimizer_updates', 'seconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--expected-sha', required=True); parser.add_argument('--expected-bytes', required=True, type=int)
    args = parser.parse_args(); audit(args.expected_sha, args.expected_bytes)
