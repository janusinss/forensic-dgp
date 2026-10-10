"""Prospective V37 audit: safe imports, saved arithmetic and frozen CPU replay.

No returned source is executed. Coarse gradients are not replayed locally and
are not asserted to be true derivatives or preservation guarantees.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_delivered_guard_grad_v37_vm'
STEM = 'cctv-dgp-delivered-guard-grad-v37'
BUNDLE = ROOT / 'outputs' / NAME
OUT = ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return'
PREFIX = 'cctv_dgp_delivered_guard_grad_v37_return/'
V36 = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_return'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def group_indices(cases):
    names = sorted({'all', 'clear', 'degraded'} | {c['source'] + '/' + k for c in cases for k in ['all', 'clear', 'degraded', c['profile']]})
    result = {name: [i for i, c in enumerate(cases) if name in {'all', 'clear' if c['profile'] == 'clear' else 'degraded',
        c['source'] + '/all', c['source'] + ('/clear' if c['profile'] == 'clear' else '/degraded'), c['source'] + '/' + c['profile']}]
        for name in names}
    assert len(result) == 17 and all(result.values())
    assert len(result['all']) == 50 and len(result['clear']) == 10 and len(result['degraded']) == 40
    return result


def verify_basis(p):
    for name, digest in p['assets_sha256'].items(): assert sha(BUNDLE / name) == digest, name
    for name, digest in p['local_basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    old = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_vm'
    for name, digest in p['basis_VM_sha256'].items(): assert sha(old / name) == digest, name
    for name, digest in p['V36_original_readback_sha256'].items(): assert sha(V36 / name) == digest, name
    auditor = module('pinned_V36_basis_only_no_probe_execution', ROOT / 'scripts/audit_cctv_dgp_finite_clearance_probe_v36_return.py')
    prior_p = read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_vm/protocol.json')
    basis = auditor.verify_basis(prior_p)
    assert basis['cohorts'] == p['cohorts'] and basis['parameter_layout'] == p['parameter_layout']
    assert read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_independent_audit.json')['complete']
    forward = read(ROOT / 'outputs/cctv_dgp_v36_delivered_metric_path_v1/independent_forward_audit.json')
    assert forward['complete'] and forward['all500_saved_PNG_encodings_exact'] and forward['PNG_only_failures'] == 9
    assert not read(ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1/analysis.json')['jointly_eligible_subset_variants']
    return basis


def safe_members(members, p):
    allowed = set(p['assets_sha256']) | {'protocol.json', 'export_manifest.json', 'diagnostic.log', 'diagnostic_exit_code.txt',
        'supervisor_receipt.json', 'outputs/results.json', 'outputs/failure.json'}
    for cohort in p['cohorts']:
        start = 'outputs/' + cohort['name'] + '/'; allowed.add(start + 'receipt.json')
        for index in range(10): allowed.add(start + f'batch{index}_guard_gradients.npy')
        for case in cohort['cases']:
            allowed.update(start + case['id'] + suffix for suffix in ['.npy', '.png', '_embedding.npy', '_target_embedding.npy'])
    seen, result, size = set(), [], 0
    for member in members:
        assert member.isfile() and not member.issym() and not member.islnk()
        assert member.name.startswith(PREFIX) and not any(c in member.name for c in ['\\', ':', '\x00'])
        name = member.name[len(PREFIX):]; parts = PurePosixPath(name).parts
        assert parts and not PurePosixPath(name).is_absolute() and all(v not in ['.', '..'] for v in parts)
        assert name in allowed and name.lower() not in seen
        assert 0 <= member.size <= 64 * 1024 ** 2
        size += member.size; seen.add(name.lower()); result.append((member, name))
        assert size <= p['budgets']['export_uncompressed_bytes'] and len(seen) <= p['budgets']['return_files_maximum']
    return result, size


def import_return(p, pin, archive_sha, archive_size):
    archive = ROOT / 'outputs' / (STEM + '-results.tar.gz')
    assert archive.stat().st_size == archive_size and sha(archive) == archive_sha
    assert Path(str(archive) + '.sha256').read_text().strip().split() == [archive_sha, archive.name]
    exported = read(ROOT / 'outputs' / (STEM + '-export.json'))
    assert exported['complete'] and exported['archive_sha256'] == archive_sha and exported['bytes'] == archive_size
    assert exported['training_success_not_implied'] and exported['optimizer_updates'] == 0
    assert not OUT.exists(), 'Retain any prior or partial audit'
    with tarfile.open(archive, 'r:gz') as tar:
        items, total = safe_members(tar.getmembers(), p); OUT.mkdir()
        for item, name in items:
            destination = (OUT / name).resolve(); assert destination.is_relative_to(OUT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(item) as source, destination.open('xb') as stream:
                for block in iter(lambda: source.read(1024 ** 2), b''): stream.write(block)
    hashes = {name: sha(OUT / name) for _, name in items}
    assert hashes['protocol.json'] == pin
    assert all(hashes[name] == expected for name, expected in p['assets_sha256'].items())
    manifest = read(OUT / 'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert manifest['files_sha256'] == {n: h for n, h in hashes.items() if n != 'export_manifest.json'}
    imported = {'complete': True, 'archive_sha256': archive_sha, 'archive_bytes': archive_size,
                'files_sha256': hashes, 'members': len(items), 'uncompressed_bytes': total, 'returned_code_executed': False}
    write(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_return_import.json', imported)
    return exported, imported


def CPU_replay(p, basis, started):
    import numpy as np
    from PIL import Image
    import torch
    paths = [ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27', ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28',
             ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2']
    for path in paths: sys.path.insert(0, str(path))
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from cctv_dgp_app_input_v28 import canonical_tensor
    original, _ = load_frozen_dgp_restorer(paths[0] / 'weights/dgp_v2.pth', expected_sha256=basis['original_checkpoint_sha256'], device='cpu')
    identity = FixedObservedIdentity(paths[0] / 'weights/w600k_r50.onnx', 'cpu').eval().requires_grad_(False)
    before = [state_hash(original.net), state_hash(identity)]
    assert before == [basis['original_DGP_state'], basis['recognizer_state']]
    references = {r['id']: r for r in read(paths[2] / 'protocol.json')['training_references']}
    torch.set_num_threads(4)
    counts = {'DGP': 0, 'recognizer': 0}
    for net, key in [(original.net, 'DGP'), (identity.encoder, 'recognizer')]:
        net.register_forward_hook(lambda *_args, key=key: counts.__setitem__(key, counts[key] + 1))
    checked, raw_error, png_error, vector_error = 0, 0., 0, 0.
    with torch.inference_mode():
        for cohort in p['cohorts']:
            label = cohort['name']; folder = OUT / 'outputs' / label
            for begin in range(0, 50, 5):
                group = cohort['cases'][begin:begin + 5]
                if not all(c['id'] in p['CPU_replay_case_ids'][label] for c in group): continue
                assert time.monotonic() - started < p['budgets']['local_audit_seconds']
                items = []
                for case in group:
                    with Image.open(MIXED / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                    with Image.open(MIXED / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
                    with Image.open(MIXED / case['observed']) as im: mask = np.asarray(im).copy() > 0
                    items.append({'case': case, 'camera': camera, 'support': mask,
                        'x': canonical_tensor(camera, 'cpu'), 'target': canonical_tensor(target, 'cpu'),
                        'mask': torch.from_numpy(mask.astype(np.float32))[None, None],
                        'grid': torch.from_numpy(grid112(references[case['source_person_or_reference']]['matrix112']))[None]})
                b = {key: torch.cat([i[key] for i in items]) for key in ['x', 'target', 'mask', 'grid']}
                baseline = torch.where(b['mask'].bool(), original(b['x']), b['x'])
                raw = baseline.permute(0, 2, 3, 1).numpy().copy(); pngs = []
                for item, value in zip(items, raw):
                    cid = item['case']['id']
                    png = np.where(item['support'][..., None], np.floor(value * np.float32(255)), item['camera']).astype(np.uint8)
                    stored = np.load(folder / (cid + '.npy'), allow_pickle=False)
                    with Image.open(folder / (cid + '.png')) as im: saved_png = np.asarray(im.convert('RGB')).copy()
                    raw_error = max(raw_error, float(np.abs(stored - value).max()))
                    png_error = max(png_error, int(np.abs(png.astype(np.int16) - saved_png.astype(np.int16)).max()))
                    pngs.append(canonical_tensor(png, 'cpu')); checked += 1
                vectors = identity.embedding(torch.cat(pngs), b['mask'], b['grid']).numpy().copy()
                truths = identity.embedding(b['target'], b['mask'], b['grid']).numpy().copy()
                for item, vector, truth in zip(items, vectors, truths):
                    cid = item['case']['id']
                    vector_error = max(vector_error, float(np.abs(vector - np.load(folder / (cid + '_embedding.npy'), allow_pickle=False)).max()),
                                       float(np.abs(truth - np.load(folder / (cid + '_target_embedding.npy'), allow_pickle=False)).max()))
    assert checked == 20 and counts == {'DGP': 4, 'recognizer': 8}
    assert raw_error <= 1e-5 and png_error <= 1 and vector_error <= 5e-5
    assert before == [state_hash(original.net), state_hash(identity)]
    assert all(v.grad is None and not v.requires_grad for net in [original, identity] for v in net.parameters())
    return {'complete': True, 'outputs': checked, 'forwards': counts, 'maximum_raw_error': raw_error,
            'maximum_PNG_byte_error': png_error, 'maximum_embedding_error': vector_error, 'state_unchanged': True,
            'local_gradient_calls': 0, 'tolerances_are_replay_only_not_preservation_thresholds': True}


def audit(archive_sha, archive_size):
    started = time.monotonic(); p = read(BUNDLE / 'protocol.json'); pin = sha(BUNDLE / 'protocol.json')
    basis = verify_basis(p); exported, imported = import_return(p, pin, archive_sha, archive_size)
    success, failure = OUT / 'outputs/results.json', OUT / 'outputs/failure.json'
    assert success.exists() != failure.exists()
    result = read(success if success.exists() else failure)
    assert result['protocol_sha256'] == pin
    assert result['optimizer_updates'] == result['parameter_updates'] == result['epochs'] == result['backwards'] == 0
    assert not result['new_checkpoint_created'] and not result['app_promotion'] and not result['goal_complete']
    assert exported['run_results_present'] == success.exists() and exported['failure_present'] == failure.exists()
    supervisor = read(OUT / 'supervisor_receipt.json'); code = int((OUT / 'diagnostic_exit_code.txt').read_text())
    assert supervisor['protocol_sha256'] == pin and supervisor['diagnostic_exit_code'] == code and (code == 0) == success.exists()
    assert supervisor['cap_seconds'] == 930 and supervisor['kill_grace_seconds'] == 30
    assert supervisor['within_external_bound'] == (supervisor['seconds'] <= 960)
    summaries, replay = [], None
    if success.exists():
        import numpy as np
        from PIL import Image
        from scipy.ndimage import minimum_filter
        from skimage.metrics import structural_similarity
        assert result['gradient_queries'] == 300 and result['seconds'] <= 900
        assert all(result[k] == n for k, n in p['forward_call_limits'].items())
        assert result['peak_allocated_VRAM_bytes'] <= p['budgets']['peak_vram_bytes']
        assert not result['true_PNG_derivatives_claimed'] and result['backward_API_calls'] == 0
        assert result['coarse_backward_rule_used_inside_autograd_grad'] and not result['finite_preservation_implied']
        assert result['original_DGP_state'] == basis['original_DGP_state'] and result['recognizer_state'] == basis['recognizer_state']
        for cohort in p['cohorts']:
            label, cases = cohort['name'], cohort['cases']; folder = OUT / 'outputs' / label
            receipt = read(folder / 'receipt.json'); groups = group_indices(cases)
            assert receipt['complete'] and receipt['cohort'] == label and receipt['state'] == 0
            assert receipt['candidate_state'] == basis['original_DGP_state'] and receipt['optimizer_updates'] == 0
            assert receipt['guard_metrics'] == p['guard_metrics'] and receipt['identity_batch_size'] == 5
            assert receipt['original_batch_context_preserved'] and receipt['derivative_is_coarse_estimate_not_true_PNG_gradient']
            assert [r['id'] for r in receipt['rows']] == [c['id'] for c in cases]
            assert len(receipt['batches']) == 10
            vectors = {key: np.zeros((3, 978243), np.float64) for key in groups}
            values = np.empty((50, 3), np.float64); worst_raw, worst_value, queries = 0., 0., 0
            for batch_index, batch in enumerate(receipt['batches']):
                assert time.monotonic() - started < p['budgets']['local_audit_seconds']
                path = folder / f'batch{batch_index}_guard_gradients.npy'
                g = np.load(path, allow_pickle=False, mmap_mode='r')
                assert g.shape == (15, 978243) and g.dtype == np.float32 and np.isfinite(g).all()
                assert batch['batch'] == batch_index and batch['shape'] == list(g.shape) and batch['gradient_dtype'] == 'float32'
                assert batch['gradient_sha256'] == sha(path)
                assert np.allclose(np.linalg.norm(g.astype(np.float64), axis=1), batch['gradient_norms'], rtol=2e-10, atol=1e-11)
                assert batch['rows'] == receipt['rows'][batch_index * 5:batch_index * 5 + 5]
                for slot, row in enumerate(batch['rows']):
                    index = batch_index * 5 + slot; case = cases[index]; cid = case['id']
                    assert row['id'] == cid and row['source'] == case['source'] and row['profile'] == case['profile']
                    def png(path):
                        with Image.open(path) as im: return np.asarray(im.convert('RGB')).copy()
                    raw = np.load(folder / (cid + '.npy'), allow_pickle=False)
                    assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
                    assert float(raw.min()) >= 0 and float(raw.max()) <= 1
                    output = png(folder / (cid + '.png')); camera = png(MIXED / case['input'])
                    target = png(MIXED / case['target']).astype(np.float32) / np.float32(255)
                    with Image.open(MIXED / case['observed']) as im: mask = np.asarray(im).copy() > 0
                    expected = np.where(mask[..., None], np.floor(raw * np.float32(255)), camera).astype(np.uint8)
                    assert np.array_equal(output, expected) and np.array_equal(output[~mask], camera[~mask])
                    baseline = V36 / f'outputs/state0_{label}/before'
                    parity = float(np.abs(raw - np.load(baseline / (cid + '.npy'), allow_pickle=False)).max())
                    png_parity = int(np.abs(output.astype(np.int16) - png(baseline / (cid + '.png')).astype(np.int16)).max())
                    assert parity <= p['same_VM_raw_tolerance'] and parity == row['raw_parity_maximum_error']
                    assert png_parity == row['PNG_parity_maximum_byte_error'] <= p['same_VM_PNG_byte_tolerance']
                    canonical = output.astype(np.float32) / np.float32(255)
                    interior = minimum_filter(mask, size=7, mode='constant', cval=0).astype(bool)
                    _, ssmap = structural_similarity(canonical, target, channel_axis=2, data_range=1., full=True)
                    vector = np.load(folder / (cid + '_embedding.npy'), allow_pickle=False)
                    truth = np.load(folder / (cid + '_target_embedding.npy'), allow_pickle=False)
                    assert vector.shape == truth.shape == (512,) and vector.dtype == truth.dtype == np.float32
                    assert np.isfinite(vector).all() and np.isfinite(truth).all()
                    for embedding, suffix, field in [(vector, '_embedding.npy', 'embedding_parity_maximum_error'),
                                                   (truth, '_target_embedding.npy', 'target_embedding_parity_maximum_error')]:
                        e = float(np.abs(embedding - np.load(baseline / (cid + suffix), allow_pickle=False)).max())
                        assert e == row[field] and e <= p['same_VM_embedding_tolerance']
                    fresh = [float(np.square((canonical - target)[mask]).astype(np.float64).mean()),
                             1 - float(ssmap[interior].astype(np.float64).mean()), 1 - float(vector @ truth)]
                    assert np.allclose(fresh, row['independent_PNG_guard_values'], rtol=0, atol=1e-12)
                    values[index] = row['PNG_guard_values']; error = np.abs(values[index] - fresh)
                    assert bool((error <= np.asarray(p['forward_metric_tolerances'])).all())
                    assert np.allclose(error, row['forward_metric_errors'], rtol=0, atol=1e-12)
                    worst_raw, worst_value = max(worst_raw, parity), max(worst_value, float(error.max()))
                    for key, selection in groups.items():
                        if index in selection: vectors[key] += g[slot * 3:slot * 3 + 3].astype(np.float64) / len(selection)
                    queries += 3
            assert queries == 150
            matrix = np.concatenate([vectors[key] for key in groups]); assert matrix.shape == (51, 978243) and np.isfinite(matrix).all()
            summaries.append({'cohort': label, 'gradient_queries': queries, 'coarse_guard_rows': 51,
                'labels': [{'group': key, 'metric': metric} for key in groups for metric in p['guard_metrics']],
                'group_guard_values': {key: values[indices].mean(0).tolist() for key, indices in groups.items()},
                'group_gradient_norms': np.linalg.norm(matrix, axis=1).tolist(), 'group_gradient_gram': (matrix @ matrix.T).tolist(),
                'maximum_raw_parity_error': worst_raw, 'maximum_PNG_guard_forward_error': worst_value,
                'true_derivative_or_finite_preservation_claimed': False, 'local_gradient_replay_performed': False})
            del matrix, vectors, g
        replay = CPU_replay(p, basis, started)
    else:
        assert 0 <= result['gradient_queries'] <= 300 and not result['resume_permitted']
    assert time.monotonic() - started < p['budgets']['local_audit_seconds']
    verify_basis(p)
    checked = {'complete': True, 'protocol_sha256': pin, 'checker_sha256': sha(Path(__file__)),
               'archive_sha256': archive_sha, 'archive_bytes': archive_size, 'members_verified': imported['members'],
               'diagnostic_complete': success.exists(), 'failure_retained': failure.exists(),
               'coarse_gradient_summaries': summaries, 'fresh_CPU_replay': replay, 'local_gradient_calls': 0,
               'local_optimizer_updates': 0, 'training_capacity_pass': False, 'retained_checkpoints_modified': False,
               'finite_preservation_pass': False, 'app_promotion': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    write(ROOT / 'outputs/cctv_dgp_delivered_guard_grad_v37_independent_audit.json', checked)
    print(json.dumps({k: checked[k] for k in ['complete', 'diagnostic_complete', 'members_verified', 'seconds']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--expected-sha', required=True); parser.add_argument('--expected-bytes', type=int, required=True)
    a = parser.parse_args(); audit(a.expected_sha, a.expected_bytes)
