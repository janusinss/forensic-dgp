"""Prospective V33 audit: independent NumPy arithmetic and frozen CPU forwards only."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile
import time
from types import MethodType

ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_dgp_loss_cone_probe_v33_vm'
STEM = 'cctv-dgp-loss-cone-probe-v33'
BUNDLE = ROOT / 'outputs' / NAME
OUT = ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_return'
OLD = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_vm'
PRIOR = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_return'
CLOSED = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_return'
V32 = ROOT / 'outputs/cctv_dgp_feature_fusion_vm_v32_r2'
PARENT = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
ACTIVE = ROOT / 'outputs/cctv_dgp_active_original_decoder_vm_v28'
MIXED = ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'
PREFIX = 'cctv_dgp_loss_cone_probe_v33_return/'


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream: stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path); value = importlib.util.module_from_spec(spec); spec.loader.exec_module(value); return value


def allowed_members(p):
    allowed = set(p['assets_sha256']) | {'protocol.json', 'export_manifest.json', 'probe.log', 'probe_exit_code.txt', 'supervisor_receipt.json', 'outputs/results.json', 'outputs/failure.json'}
    for state in p['states']:
        for cohort in p['cohorts']:
            folder = f'outputs/state{state}_{cohort["name"]}/'
            allowed.update(folder + name for name in ['theta_before.npy', 'projected_displacement.npy', 'projection.json'])
            for proposal in ['summed', 'restoration']:
                allowed.update(folder + proposal + '/' + name for name in ['receipt.json', 'clipped_gradient.npy', 'theta_after.npy', 'first_moment.npy', 'second_moment.npy'])
            for variant in ['before'] + [v['name'] for v in p['variants']]:
                prefix = folder + variant + '/'; allowed.add(prefix + 'receipt.json')
                if variant != 'before': allowed.add(prefix + 'comparison.json')
                for case in cohort['cases']:
                    allowed.update(prefix + case['id'] + suffix for suffix in ['.npy', '.png', '_embedding.npy', '_raw_embedding.npy'])
                    if variant == 'before': allowed.add(prefix + case['id'] + '_target_embedding.npy')
    return allowed


def safe_members(members, p):
    allowed = allowed_members(p); seen, checked, size = set(), [], 0
    for item in members:
        assert item.isfile() and not item.issym() and not item.islnk(), 'Regular files only'
        assert item.name.startswith(PREFIX) and not any(c in item.name for c in ['\\', ':', '\x00'])
        name = item.name[len(PREFIX):]; parts = PurePosixPath(name).parts
        assert parts and not PurePosixPath(name).is_absolute() and all(c not in ['.', '..'] for c in parts)
        assert name in allowed and name.lower() not in seen, 'Unexpected or duplicate member'
        assert 0 <= item.size <= 16 * 1024**2, 'Per-file16MiB bound'
        size += item.size; seen.add(name.lower())
        assert size <= p['budgets']['export_uncompressed_bytes'] and len(seen) <= p['budgets']['return_files_maximum']
        checked.append((item, name))
    return checked, size


def verify_basis(p):
    for name, digest in p['assets_sha256'].items(): assert sha(BUNDLE / name) == digest, name
    for name, digest in p['local_basis_sha256'].items(): assert sha(ROOT / name) == digest, name
    basis = read(OLD / 'protocol.json'); assert sha(OLD / 'protocol.json') == p['basis_protocol_sha256']
    prior_checker = module('prior_audited_loss_basis', ROOT / 'scripts/audit_cctv_dgp_v32_loss_gradient_v1_return.py')
    prior_checker.verify_basis(basis)
    for name, digest in p['basis_VM_sha256'].items():
        source = OLD / name if name == 'protocol.json' or name in basis['assets_sha256'] else PRIOR / name
        assert sha(source) == digest, name
    return basis


def import_return(p, pin, digest, size):
    archive = ROOT / 'outputs' / (STEM + '-results.tar.gz')
    assert archive.stat().st_size == size and sha(archive) == digest
    assert Path(str(archive) + '.sha256').read_text(encoding='ascii').strip().split() == [digest, archive.name]
    exported = read(ROOT / 'outputs' / (STEM + '-export.json'))
    assert exported['complete'] and exported['archive_sha256'] == digest and exported['bytes'] == size and exported['training_success_not_implied']
    assert exported['committed_trajectory_updates'] == 0 and 0 <= exported['optimizer_updates'] <= 8
    assert not OUT.exists(), 'Preserve any previous import or partial audit'
    with tarfile.open(archive, 'r:gz') as tar:
        items, total = safe_members(tar.getmembers(), p); OUT.mkdir()
        for member, name in items:
            target = (OUT / name).resolve(); assert target.is_relative_to(OUT); target.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, target.open('xb') as destination:
                for block in iter(lambda: source.read(1024**2), b''): destination.write(block)
    hashes = {name: sha(OUT / name) for _, name in items}; assert hashes['protocol.json'] == pin
    assert all(hashes.get(name) == digest for name, digest in p['assets_sha256'].items())
    manifest = read(OUT / 'export_manifest.json'); assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert manifest['files_sha256'] == {name: value for name, value in hashes.items() if name != 'export_manifest.json'}
    receipt = {'complete': True, 'archive_sha256': digest, 'archive_bytes': size, 'files_sha256': hashes, 'members': len(items),
               'uncompressed_bytes': total, 'returned_code_executed': False, 'local_gradient_calls': 0, 'local_optimizer_updates': 0}
    write(ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_return_import.json', receipt)
    return exported, receipt


def projection_primal(g, proposal, reported):
    import numpy as np
    from scipy.optimize import minimize
    norms = np.linalg.norm(g, axis=1); nz = np.flatnonzero(norms > 0); unit = g[nz] / norms[nz, None]
    if not len(nz): assert np.array_equal(proposal, reported); return 0.
    gram, dots = unit @ unit.T, unit @ proposal
    scale = max(np.linalg.norm(proposal), 1e-12); b = dots / scale
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    assert eigenvalues.min() >= -1e-10
    active = eigenvalues > 1e-12
    assert active.any()
    constraint_matrix = eigenvectors[:, active] * np.sqrt(eigenvalues[active])
    back_rotation = eigenvectors[:, active] / np.sqrt(eigenvalues[active])
    # Orthonormal coordinates make the independent primal Hessian the identity.
    # Loss halfspaces and the original discrepancy bound remain unchanged.
    result = minimize(lambda v: .5 * float(v @ v), np.zeros(int(active.sum())), jac=lambda v: v,
                      constraints=[{'type': 'ineq', 'fun': lambda v: b + constraint_matrix @ v,
                                    'jac': lambda v: constraint_matrix}],
                      method='SLSQP', options={'ftol': 1e-15, 'maxiter': 1000})
    assert result.success and (b + constraint_matrix @ result.x).min() >= -1e-8
    independently_projected = proposal + ((back_rotation @ result.x) * scale) @ unit
    error = float(np.linalg.norm(independently_projected - reported)); assert error <= max(2e-7 * scale, 2e-10)
    return error


def proposals(p, state, cohort):
    import numpy as np
    folder = OUT / f'outputs/state{state}_{cohort["name"]}'
    theta = np.load(folder / 'theta_before.npy', allow_pickle=False)
    assert theta.dtype == np.float32 and theta.shape == (978243,) and np.isfinite(theta).all()
    g = np.load(PRIOR / f'outputs/state{state}_{cohort["name"]}/gradient_components.npy', allow_pickle=False)
    assert g.dtype == np.float64 and g.shape == (7, len(theta)) and np.isfinite(g).all()
    deltas, maxima = {}, []
    for name, gradient in [('summed', g.sum(0)), ('restoration', g[:3].sum(0))]:
        proof = folder / name; receipt = read(proof / 'receipt.json')
        assert receipt['proposal'] == name and receipt['fresh_optimizer_step'] == 1 and not receipt['optimizer_moments_from_V32_reconstructed']
        arrays = {key: np.load(proof / (key + '.npy'), allow_pickle=False) for key in ['clipped_gradient', 'theta_after', 'first_moment', 'second_moment']}
        for key, a in arrays.items(): assert a.dtype == np.float32 and a.shape == theta.shape and np.isfinite(a).all() and sha(proof / (key + '.npy')) == receipt['saved_arrays_sha256'][key + '.npy']
        gradient32 = gradient.astype(np.float32); independent_norm = np.linalg.norm(gradient32.astype(np.float64)); norm = receipt['clip_input_norm']
        assert abs(independent_norm - norm) <= max(3e-6 * independent_norm, 1e-8)
        expected_clip = gradient32 * np.float32(min(1., 1. / (norm + 1e-6)))
        assert np.allclose(arrays['clipped_gradient'], expected_clip, rtol=3e-6, atol=1e-10)
        c = arrays['clipped_gradient'].astype(np.float64)
        assert np.allclose(arrays['first_moment'], .1 * c, rtol=3e-6, atol=1e-10)
        assert np.allclose(arrays['second_moment'], .001 * c * c, rtol=3e-6, atol=1e-12)
        expected_theta = theta.astype(np.float64) * (1 - .00003 * .01) - .00003 * c / (np.abs(c) + 1e-8)
        error = float(np.abs(expected_theta - arrays['theta_after']).max()); assert error <= 2e-7
        maxima.append(error); deltas[name] = theta.astype(np.float64) - arrays['theta_after'].astype(np.float64)
    direction = np.load(folder / 'projected_displacement.npy', allow_pickle=False)
    assert direction.dtype == np.float64 and direction.shape == theta.shape and np.isfinite(direction).all()
    projection = read(folder / 'projection.json'); norms = np.linalg.norm(g, axis=1); nz = np.flatnonzero(norms > 0)
    assert projection['nonzero_term_indices'] == nz.tolist()
    unit = g[nz] / norms[nz, None]; multipliers = np.asarray(projection['multipliers'])
    assert np.allclose(direction, deltas['restoration'] + multipliers @ unit, rtol=0, atol=1e-12)
    assert np.allclose(g @ direction, projection['raw_component_dots_after'], rtol=1e-9, atol=1e-11)
    assert (unit @ direction >= -projection['KKT_tolerance']).all() and (multipliers >= 0).all()
    primal_error = projection_primal(g, deltas['restoration'], direction)
    deltas['projected_restoration'] = direction
    return theta, g, deltas, {'AdamW_formula_maximum_error': max(maxima), 'independent_primal_direction_L2_error': primal_error}


def finite_outputs(p, basis, started):
    import numpy as np
    from PIL import Image
    from scipy.ndimage import convolve1d
    metrics = module('pinned_own_probe_metric_readback', ROOT / 'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py')
    cached, checked, reports = {}, 0, []
    for state in p['states']:
        for cohort in p['cohorts']:
            theta, g, deltas, proof = proposals(p, state, cohort); label = cohort['name']; folder = OUT / f'outputs/state{state}_{label}'
            for variant in ['before'] + [v['name'] for v in p['variants']]:
                assert time.monotonic() - started < p['budgets']['local_audit_seconds']
                path = folder / variant; receipt = read(path / 'receipt.json')
                assert receipt['complete'] and receipt['state'] == state and receipt['cohort'] == label and receipt['variant'] == variant and receipt['cases'] == 50
                assert [r['id'] for r in receipt['rows']] == [c['id'] for c in cohort['cases']]
                values, rows, term_maximum = [], [], 0.
                for case, row in zip(cohort['cases'], receipt['rows']):
                    cid = case['id']; raw = np.load(path / (cid + '.npy'), allow_pickle=False)
                    with Image.open(path / (cid + '.png')) as im: png = np.asarray(im.convert('RGB')).copy()
                    with Image.open(MIXED / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                    with Image.open(MIXED / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
                    with Image.open(MIXED / case['observed']) as im: mask = np.asarray(im).copy() > 0
                    interior = metrics.erode(mask, 6); feature = np.zeros((256, 256), bool)
                    for point in case['landmarks5_canvas_xy']:
                        xx, yy = np.floor(point).astype(int); feature[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = True
                    feature &= interior
                    baseline = np.load(OUT / f'outputs/state0_{label}/before/{cid}.npy', allow_pickle=False)
                    base_vector = np.load(OUT / f'outputs/state0_{label}/before/{cid}_raw_embedding.npy', allow_pickle=False)
                    truth = np.load(OUT / f'outputs/state0_{label}/before/{cid}_target_embedding.npy', allow_pickle=False)
                    vector, raw_vector = [np.load(path / (cid + suffix), allow_pickle=False) for suffix in ['_embedding.npy', '_raw_embedding.npy']]
                    for a in [vector, raw_vector, truth, base_vector]: assert a.dtype == np.float32 and a.shape == (512,) and np.isfinite(a).all() and abs(float(a @ a) - 1) < 1e-5
                    actual = metrics.case_metrics(raw, png, target, camera, mask, feature, baseline, vector, raw_vector, truth)
                    assert set(actual) == set(row['metrics'])
                    for key in actual: assert np.allclose(actual[key], row['metrics'][key], rtol=2e-10, atol=1e-11), (cid, key)
                    # Independently assemble every raw loss from saved output and vectors.
                    z = np.arange(-6, 7, dtype=np.float64); kernel = np.exp(-.5 * (z / 2.) ** 2); kernel /= kernel.sum()
                    luma = np.array([.299, .587, .114]); high = lambda a: a - convolve1d(convolve1d(a, kernel, axis=0, mode='reflect'), kernel, axis=1, mode='reflect')
                    delta = high((raw.astype(np.float64) * luma).sum(2)) - high((target.astype(np.float64) / 255 * luma).sum(2))
                    f, inside = float(np.square(delta)[feature].mean()), float(np.square(delta)[interior].mean())
                    base = metrics.case_metrics(baseline, np.where(mask[..., None], np.floor(baseline * np.float32(255)), camera).astype(np.uint8), target, camera, mask, feature, baseline, base_vector, base_vector, truth)
                    denom = max(base['raw_MSE'], 1e-5); degraded = case['profile'] != 'clear'
                    anchor = float(np.square((raw - baseline)[mask]).astype(np.float64).mean())
                    independently_assembled = np.array([1.25 * f / basis['normalizers'][0] if degraded else 0,
                        .3125 * inside / basis['normalizers'][1] if degraded else 0, .0625 * actual['raw_MSE'] / denom if degraded else 0,
                        .05 * anchor / denom if not degraded else 0, 2 * max(0., (actual['raw_MSE'] - base['raw_MSE']) / denom),
                        5 * max(0., base['raw_SSIM'] - actual['raw_SSIM']), 5 * max(0., float(base_vector @ truth) - float(raw_vector @ truth))])
                    error = float(np.abs(independently_assembled - row['raw_terms']).max()); assert error <= p['CPU_component_value_tolerance']; term_maximum = max(term_maximum, error)
                    if variant == 'before':
                        prior = PRIOR / f'outputs/state{state}_{label}'
                        assert np.abs(raw - np.load(prior / (cid + '.npy'), allow_pickle=False)).max() <= p['same_VM_raw_tolerance']
                        with Image.open(prior / (cid + '.png')) as im: old_png = np.asarray(im.convert('RGB')).copy()
                        assert np.abs(png.astype(int) - old_png.astype(int)).max() <= 1
                        assert np.abs(truth - np.load(PRIOR / f'outputs/state0_{label}/{cid}_target_embedding.npy', allow_pickle=False)).max() <= p['CPU_vector_absolute_tolerance']
                    values.append(row['raw_terms']); rows.append({'id': cid, 'source': case['source'], 'profile': case['profile'], 'metrics': actual}); checked += 1
                means = np.asarray(values, np.float64).mean(0)
                assert np.allclose(means, receipt['raw_component_means'], rtol=0, atol=1e-12) and abs(means.sum() - receipt['raw_objective']) <= 1e-12
                assert metrics.groups(rows) == receipt['groups']; cached[(state, label, variant)] = receipt
                if variant != 'before':
                    spec = next(v for v in p['variants'] if v['name'] == variant); comparison = read(path / 'comparison.json')
                    actual_theta = (theta.astype(np.float64) - spec['scale'] * deltas[spec['proposal']]).astype(np.float32)
                    displacement = theta.astype(np.float64) - actual_theta.astype(np.float64)
                    assert np.allclose(-(g @ displacement), comparison['actual_linear_component_derivatives'], rtol=1e-9, atol=1e-11)
                    assert np.allclose(means - cached[(state, label, 'before')]['raw_component_means'], comparison['finite_raw_component_change'], rtol=0, atol=1e-12)
                    original_groups = cached[(0, label, 'before')]['groups']; assert metrics.compare_groups(original_groups, receipt['groups']) == comparison['preservation_against_original']
                    assert not comparison['quality_or_training_capacity_pass'] and not comparison['unconstrained_control_eligible_for_training']
                reports.append({'state': state, 'cohort': label, 'variant': variant, 'raw_loss_assembly_maximum_error': term_maximum,
                                'raw_component_means': receipt['raw_component_means'], 'proposal_proof': proof})
    assert checked == 1400
    return reports


def CPU_replay(p, basis, started):
    import numpy as np
    from PIL import Image
    import torch
    from torch.nn import functional as F
    for path in [PARENT, ACTIVE, V32]: sys.path.insert(0, str(path))
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from cctv_dgp_mean_centered_decoder_v29 import center_observed_delta
    from cctv_dgp_app_input_v28 import canonical_tensor
    from cctv_dgp_batchmatched_identity_v26 import objective_terms
    metrics = module('pinned_probe_erode_only', ROOT / 'scripts/cctv_dgp_loss_cone_probe_v33_metrics.py')
    helper = module('pinned_original_loss_AST_only', V32 / 'scripts/cctv_dgp_feature_fusion_v32_vm.py')
    _, definitions, filters = helper.original_functions(PARENT)
    ns = {'torch': torch, 'F': F}; exec(compile(ast.Module(body=filters, type_ignores=[]), '<pinned-fixed-filter-CPU>', 'exec'), ns)
    class Fixed: pass
    fixed = Fixed(); z = torch.arange(-6, 7, dtype=torch.float32); k = torch.exp(-.5 * (z / 2).square()); fixed.kernel = k / k.sum()
    fixed.reflect_indices = torch.cat((torch.arange(5, -1, -1), torch.arange(256), torch.arange(255, 249, -1)))
    fixed.blur = MethodType(ns['blur'], fixed); fixed.high = MethodType(ns['high'], fixed)
    original, _ = load_frozen_dgp_restorer(PARENT / 'weights/dgp_v2.pth', expected_sha256=basis['original_checkpoint_sha256'], device='cpu')
    candidate, _ = load_frozen_dgp_restorer(PARENT / 'weights/dgp_v2.pth', expected_sha256=basis['original_checkpoint_sha256'], device='cpu')
    identity = FixedObservedIdentity(PARENT / 'weights/w600k_r50.onnx', 'cpu').eval().requires_grad_(False)
    ns.update({'head': fixed, 'identity': identity}); exec(compile(ast.Module(body=definitions, type_ignores=[]), '<pinned-seven-loss-CPU>', 'exec'), ns)
    normalizers = tuple(torch.tensor(v, dtype=torch.float32) for v in basis['normalizers'])
    initial = {name: value.detach().clone() for name, value in candidate.net.state_dict().items()}
    stopped = torch.load(CLOSED / 'outputs/update50/dgp_candidate_v32.pth', map_location='cpu', weights_only=True)
    baseline_states = [state_hash(original.net), state_hash(identity)]; torch.set_num_threads(4)
    assert baseline_states == [basis['original_DGP_state'], basis['recognizer_state']]
    references = {r['id']: r for r in read(V32 / 'protocol.json')['training_references']}
    keys = ['x', 'base', 'target', 'mask', 'feature', 'interior', 'valid7', 'grid', 'truth', 'degraded_weight', 'clear_weight']
    counts = {'original_DGP_forwards': 0, 'candidate_DGP_forwards': 0, 'recognizer_forwards': 0}
    for model, key in [(original.net, 'original_DGP_forwards'), (candidate.net, 'candidate_DGP_forwards'), (identity.encoder, 'recognizer_forwards')]:
        model.register_forward_hook(lambda *_args, key=key: counts.__setitem__(key, counts[key] + 1))
    worst_raw, worst_png, worst_vector, worst_term, checked = 0., 0, 0., 0., 0
    with torch.inference_mode():
        for cohort in p['cohorts']:
            label, items = cohort['name'], []
            for case in cohort['cases']:
                with Image.open(MIXED / case['input']) as im: camera = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['target']) as im: target = np.asarray(im.convert('RGB')).copy()
                with Image.open(MIXED / case['observed']) as im: mask = np.asarray(im).copy() > 0
                interior = metrics.erode(mask, 6); feature = np.zeros((256, 256), bool)
                for point in case['landmarks5_canvas_xy']:
                    xx, yy = np.floor(point).astype(int); feature[max(0, yy - 12):min(256, yy + 12), max(0, xx - 12):min(256, xx + 12)] = True
                feature &= interior
                items.append({'case': case, 'camera': camera, 'mask8': mask, 'x': canonical_tensor(camera, 'cpu'), 'target': canonical_tensor(target, 'cpu'),
                              'mask': torch.from_numpy(mask.astype(np.float32))[None, None], 'interior': torch.from_numpy(interior.astype(np.float32))[None, None],
                              'feature': torch.from_numpy(feature.astype(np.float32))[None, None], 'valid7': torch.from_numpy(metrics.erode(mask, 3).astype(np.float32))[None, None],
                              'grid': torch.from_numpy(grid112(references[case['source_person_or_reference']]['matrix112']))[None],
                              'degraded_weight': torch.tensor([0. if case['profile'] == 'clear' else 1.25]), 'clear_weight': torch.tensor([1. if case['profile'] == 'clear' else 0.])})
            for begin in range(0, 50, 5):
                group = items[begin:begin + 5]; x = torch.cat([i['x'] for i in group]); mask = torch.cat([i['mask'] for i in group]); grid = torch.cat([i['grid'] for i in group])
                baseline = torch.where(mask.bool(), original(x), x); truth = identity.embedding(torch.cat([i['target'] for i in group]), mask, grid)
                for index, item in enumerate(group): item['base'] = baseline[index:index + 1]; item['truth'] = truth[index:index + 1]
            for state, weights in [(0, initial), (50, stopped)]:
                theta, _, deltas, _ = proposals(p, state, cohort); folder = OUT / f'outputs/state{state}_{label}'
                candidate.net.load_state_dict(weights, strict=True)
                selected = dict(candidate.net.named_parameters())
                actual_before = torch.cat([selected[r['name']].reshape(-1) for r in p['parameter_layout']]).numpy()
                assert np.array_equal(actual_before, theta)
                for variant in ['before'] + [v['name'] for v in p['variants']]:
                    assert time.monotonic() - started < p['budgets']['local_audit_seconds']
                    candidate.net.load_state_dict(weights, strict=True)
                    spec = None if variant == 'before' else next(v for v in p['variants'] if v['name'] == variant)
                    if spec:
                        array = (theta.astype(np.float64) - spec['scale'] * deltas[spec['proposal']]).astype(np.float32)
                        for row in p['parameter_layout']: selected[row['name']].copy_(torch.from_numpy(array[row['start']:row['end']].copy()).reshape(row['shape']))
                    path = folder / variant; receipt = read(path / 'receipt.json'); assert state_hash(candidate.net) == receipt['candidate_state']
                    for begin in range(0, 50, 5):
                        group = items[begin:begin + 5]
                        if not all(i['case']['id'] in p['CPU_replay_case_ids'][label] for i in group): continue
                        b = {key: torch.cat([i[key] for i in group]) for key in keys}
                        pred = center_observed_delta(torch.where(b['mask'].bool(), candidate(b['x']), b['x']), b['base'], b['x'], b['mask'])
                        raw = pred.permute(0, 2, 3, 1).numpy().copy(); delivered = []
                        for item, fresh in zip(group, raw):
                            cid = item['case']['id']; saved = np.load(path / (cid + '.npy'), allow_pickle=False)
                            error = float(np.abs(saved - fresh).max()); assert error <= p['CPU_raw_absolute_tolerance']; worst_raw = max(worst_raw, error)
                            with Image.open(path / (cid + '.png')) as im: png = np.asarray(im.convert('RGB')).copy()
                            expected = np.where(item['mask8'][..., None], np.floor(fresh * np.float32(255)), item['camera']).astype(np.uint8)
                            byte = int(np.abs(png.astype(int) - expected.astype(int)).max()); assert byte <= p['CPU_PNG_byte_tolerance']; worst_png = max(worst_png, byte)
                            delivered.append(png); checked += 1
                        vectors = identity.embedding(torch.cat([canonical_tensor(a, 'cpu') for a in delivered]), b['mask'], b['grid']).numpy()
                        raw_vectors = identity.embedding(pred, b['mask'], b['grid']).numpy()
                        for item, vector, raw_vector in zip(group, vectors, raw_vectors):
                            cid = item['case']['id']; truth = np.load(OUT / f'outputs/state0_{label}/before/{cid}_target_embedding.npy', allow_pickle=False)
                            error = max(float(np.abs(vector - np.load(path / (cid + '_embedding.npy'), allow_pickle=False)).max()),
                                        float(np.abs(raw_vector - np.load(path / (cid + '_raw_embedding.npy'), allow_pickle=False)).max()), float(np.abs(truth - item['truth'][0].numpy()).max()))
                            assert error <= p['CPU_vector_absolute_tolerance']; worst_vector = max(worst_vector, error)
                        terms = objective_terms(b, pred, identity, ns['mean'], ns['feature_errors'], ns['ssim'], normalizers)
                        values = torch.stack([terms[name] for name in p['terms']], 1).numpy()
                        expected = np.asarray([r['raw_terms'] for r in receipt['rows'][begin:begin + 5]])
                        error = float(np.abs(values - expected).max()); assert error <= p['CPU_component_value_tolerance']; worst_term = max(worst_term, error)
                    candidate.net.load_state_dict(weights, strict=True)
            candidate.net.load_state_dict(initial, strict=True)
    assert checked == p['CPU_replay_outputs'] == 280
    assert [state_hash(original.net), state_hash(identity)] == baseline_states and state_hash(candidate.net) == basis['original_DGP_state']
    assert all(not v.requires_grad and v.grad is None for model in [original, candidate, identity] for v in model.parameters())
    return {'outputs': checked, **counts, 'raw_maximum_error': worst_raw, 'PNG_maximum_byte_error': worst_png,
            'vector_maximum_error': worst_vector, 'component_value_maximum_error': worst_term, 'all_states_restored': True}


def audit(digest, size):
    started = time.monotonic(); p = read(BUNDLE / 'protocol.json'); pin = sha(BUNDLE / 'protocol.json')
    prep = read(ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_preparation/preparation_receipt.json'); assert pin == prep['protocol_sha256']
    basis = verify_basis(p); exported, imported = import_return(p, pin, digest, size)
    success, failure = OUT / 'outputs/results.json', OUT / 'outputs/failure.json'; assert success.exists() != failure.exists()
    terminal = read(success if success.exists() else failure)
    assert terminal['protocol_sha256'] == pin and terminal['new_gradient_queries'] == terminal['backwards'] == terminal['committed_trajectory_updates'] == terminal['epochs'] == 0
    assert not terminal['new_checkpoint_created'] and not terminal['app_promotion'] and not terminal['goal_complete']
    assert exported['run_results_present'] == success.exists() and exported['failure_present'] == failure.exists() and exported['optimizer_updates'] == terminal['optimizer_updates']
    supervisor = read(OUT / 'supervisor_receipt.json'); code = int((OUT / 'probe_exit_code.txt').read_text())
    assert supervisor['protocol_sha256'] == pin and supervisor['probe_exit_code'] == code and (code == 0) == success.exists()
    assert supervisor['cap_seconds'] == 930 and supervisor['kill_grace_seconds'] == 30 and supervisor['within_external_bound'] == (supervisor['seconds'] <= 960)
    rows, replay = None, None
    if success.exists():
        assert terminal['optimizer_updates'] == 8 and terminal['raw_outputs'] == 1400 and terminal['optimizer_constructed'] and terminal['all_trial_states_reset']
        assert {k: terminal[k] for k in p['forward_call_limits']} == p['forward_call_limits']
        assert terminal['seconds'] <= 900 and terminal['peak_allocated_VRAM_bytes'] <= 20 * 1024**3
        assert terminal['original_DGP_state'] == basis['original_DGP_state'] and terminal['recognizer_state'] == basis['recognizer_state']
        for row in terminal['receipts']: assert sha(OUT / f'outputs/state{row["state"]}_{row["cohort"]}/{row["variant"]}/receipt.json') == row['receipt_sha256']
        rows = finite_outputs(p, basis, started); replay = CPU_replay(p, basis, started)
    else:
        assert 0 <= terminal['optimizer_updates'] <= 8 and not terminal['resume_permitted']
    assert time.monotonic() - started < p['budgets']['local_audit_seconds']
    receipt = {'complete': True, 'protocol_sha256': pin, 'checker_sha256': sha(Path(__file__)), 'archive_sha256': digest,
               'archive_bytes': size, 'members_verified': imported['members'], 'finite_probe_complete': success.exists(),
               'VM_failure_retained': failure.exists(), 'finite_output_arithmetic': rows, 'CPU_replay': replay,
               'local_gradient_calls': 0, 'local_backward_calls': 0, 'local_optimizer_updates': 0,
               'retained_checkpoints_modified': False, 'app_promotion': False, 'training_capacity_pass': False,
               'visual_review_pending': True, 'independent_final_review': False, 'goal_complete': False, 'seconds': time.monotonic() - started}
    write(ROOT / 'outputs/cctv_dgp_loss_cone_probe_v33_independent_audit.json', receipt)
    print(json.dumps({key: receipt[key] for key in ['complete', 'finite_probe_complete', 'VM_failure_retained', 'members_verified', 'seconds']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--expected-sha', required=True); parser.add_argument('--expected-bytes', required=True, type=int)
    args = parser.parse_args(); audit(args.expected_sha, args.expected_bytes)
