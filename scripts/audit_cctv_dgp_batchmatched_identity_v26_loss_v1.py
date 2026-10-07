"""Corrected V26 loss on saved states only; no derivatives or parameter fitting."""
import ast
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_batchmatched_identity_v26 as a


def main():
    import numpy as np
    import torch
    from torch.nn import functional as F
    started = time.monotonic()
    torch.set_num_threads(4)
    bundle = a.BUNDLE
    returned = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'
    out = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_loss_audit_v1'
    assert not out.exists(), 'Preserve diagnostic evidence; no overwrite'
    p = a.verify_bundle(bundle)
    original = a.read(ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json')
    assert original['complete'] and original['complete_snapshot_updates'] == [0, 50]
    assert not original['early_structure_stop']['pass']
    out.mkdir()
    source = bundle / 'scripts/cctv_dgp_batchmatched_identity_v26_vm.py'
    objective = bundle / 'cctv_dgp_batchmatched_identity_v26.py'
    assembly = bundle / 'cctv_dgp_degraded_objective_v24.py'
    cohort_path = returned / 'outputs/cohort_loss_setup.json'
    cohort = a.read(cohort_path)
    plan = {'date': '2026-10-06', 'scope': 'Saved corrected V26 loss scalars; exposed paired photographic TRAIN only',
        'hypothesis': 'Determine whether the saved corrected loss rewards structure and where preservation costs remain; scalar evidence does not establish GPU gradient or optimizer-trajectory causality.',
        'protocol_sha256': a.PIN, 'states': [0, 50], 'case_order': [c['id'] for c in p['cases']],
        'fixed_batches': [list(range(start, start + 5)) for start in range(0, 50, 5)],
        'batch_context': 'Ten fixed five-case batches; baseline/prediction share one concatenated ten-image recognizer call. No grad CPU context, not a GPU derivative rerun.',
        'maximum_recognizer_forwards': 30, 'head_forwards': 0, 'DGP_forwards': 0,
        'seconds_cap': 300, 'compiled_loss_absolute_tolerance': 1e-6,
        'gradient_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0, 'parameter_search': False,
        'native_or_reserved_used': False, 'VM_actions': False, 'app_promotion': False, 'goal_complete': False,
        'runner_sha256': a.sha(Path(__file__))}
    a.write(out / 'plan.json', plan)
    def clock():
        assert time.monotonic() - started < 300, 'Fixed CPU diagnostic cap exceeded'
    sys.path.insert(0, str(bundle))
    from cctv_dgp_pilot import FixedObservedIdentity
    identity = FixedObservedIdentity(bundle / 'weights/w600k_r50.onnx', 'cpu')
    identity_before = a.state_hash(identity.state_dict())
    head = a.make_head(bundle)
    head.load_state_dict(torch.load(returned / 'outputs/update0/head.pth', map_location='cpu', weights_only=True), strict=True)
    head_before = a.state_hash(head.state_dict())
    counts = {'recognizer_forwards': 0, 'saved_prediction_replays': 0}
    def count(*_):
        counts['recognizer_forwards'] += 1
        assert counts['recognizer_forwards'] <= 30
    hook = identity.encoder.register_forward_hook(count)
    current = {}
    class SavedOutput:
        def __call__(self, x, base, mask, features):
            assert features is current['batch']['fpn']
            assert torch.equal(x, current['batch']['x']) and torch.equal(base, current['batch']['base'])
            assert torch.equal(mask, current['batch']['mask'])
            counts['saved_prediction_replays'] += 1
            return current['pred']
        def high(self, value):
            return head.high(value)
    class IdentityCapture:
        def embedding(self, image, mask, grid):
            value = identity.embedding(image, mask, grid)
            current['vectors'] = value
            return value
    def batch(ids):
        assert ids == [0]
        return current['batch']
    namespace = {'torch': torch, 'F': F, 'head': SavedOutput(), 'identity': IdentityCapture(), 'batch': batch}
    namespace['normalizers'] = tuple(torch.tensor(cohort[key], dtype=torch.float32)
                                     for key in ['feature_normalizer', 'interior_normalizer'])
    assembly_tree = ast.parse(assembly.read_text(encoding='utf-8'))
    objective_tree = ast.parse(objective.read_text(encoding='utf-8'))
    nodes = [next(n for n in assembly_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'assemble_terms')]
    nodes += [next(n for n in objective_tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
              for name in ['batchmatched_scores', 'objective_terms']]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<frozen-local-V26-objective-only>', 'exec'), namespace)
    tree = ast.parse(source.read_text(encoding='utf-8'))
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    functions = [next(n for n in run.body if isinstance(n, ast.FunctionDef) and n.name == name)
                 for name in ['mean', 'feature_errors', 'ssim', 'loss']]
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<frozen-local-V26-loss-only>', 'exec'), namespace)
    mean, feature_errors, ssim = [namespace[k] for k in ['mean', 'feature_errors', 'ssim']]
    refs = {r['id']: r for r in p['references']}
    items, truths, rows, batches = [], {}, [], []
    sources = {}
    def bind(path):
        sources[path.relative_to(ROOT).as_posix()] = a.sha(path)
    for path in [source, objective, assembly, cohort_path, bundle / 'protocol.json',
                 bundle / 'cctv_dgp_spatial_features_v25.py', bundle / 'cctv_dgp_pilot.py',
                 bundle / 'weights/w600k_r50.onnx', returned / 'outputs/update0/head.pth',
                 ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json']:
        bind(path)
    def tensor(value):
        return torch.from_numpy(np.asarray(value).copy()).permute(2, 0, 1)[None]
    maximum_target_vector_error = 0.
    maximum_compiled_difference = 0.
    with torch.inference_mode():
        for case in p['cases']:
            clock()
            for path in [bundle / case[k] for k in ['input', 'target', 'observed', 'raw_dgp']]:
                bind(path)
            raw_paths = [returned / ('outputs/update' + str(update)) / (case['id'] + '.npy') for update in [0, 50]]
            for path in raw_paths:
                bind(path)
            saved_truth = returned / 'outputs/update0' / (case['id'] + '_target_embedding.npy')
            bind(saved_truth)
            camera, target8 = a.rgb(bundle / case['input']), a.rgb(bundle / case['target'])
            base = a.raw_rgb(bundle / case['raw_dgp'])
            assert np.array_equal(a.raw_rgb(raw_paths[0]), base)
            mask8 = a.observed(bundle / case['observed'])
            x, target, b = tensor(camera.astype(np.float32) / np.float32(255)), tensor(target8.astype(np.float32) / np.float32(255)), tensor(base)
            mask = torch.from_numpy(mask8.astype(np.float32))[None, None]
            grid = torch.from_numpy(a.fixed_grid(refs[case['source_person_or_reference']]['matrix112']))[None]
            ref = case['source_person_or_reference']
            if ref not in truths:
                truth = identity.embedding(target, mask, grid)
                truths[ref] = {'vector': truth, 'target': target, 'mask': mask, 'grid': grid}
                np.save(out / ('truth_' + ref + '.npy'), truth[0].numpy().copy(), allow_pickle=False)
            else:
                assert torch.equal(target, truths[ref]['target']) and torch.equal(mask, truths[ref]['mask'])
                assert torch.equal(grid, truths[ref]['grid']), 'Cannot reuse target embedding across different geometry'
            truth = truths[ref]['vector']
            difference = float(np.max(np.abs(truth[0].numpy() - a.vector(saved_truth))))
            assert difference <= a.RECOGNIZER_VECTOR_TOLERANCE
            maximum_target_vector_error = max(maximum_target_vector_error, difference)
            items.append({'case': case, 'x': x, 'target': target, 'base': b, 'mask': mask, 'grid': grid, 'truth': truth,
                'feature': torch.from_numpy(a.feature_mask(case, mask8).astype(np.float32))[None, None],
                'interior': torch.from_numpy(a.interior(mask8, 6).astype(np.float32))[None, None],
                'valid7': torch.from_numpy(a.interior(mask8, 3).astype(np.float32))[None, None],
                'degraded_weight': torch.tensor([0. if case['profile'] == 'clear' else 1.25]),
                'clear_weight': torch.tensor([1. if case['profile'] == 'clear' else 0.]),
                'prediction': tensor(a.raw_rgb(raw_paths[1]))})
            rows.append({'id': case['id'], 'source': case['source'], 'reference': ref, 'profile': case['profile'], 'states': {}})
        assert len(truths) == 10 and counts['recognizer_forwards'] == 10
        for update in [0, 50]:
            for index, ids in enumerate(plan['fixed_batches']):
                clock()
                b = {key: torch.cat([items[i][key] for i in ids]) for key in [
                    'x', 'target', 'base', 'mask', 'grid', 'truth', 'feature', 'interior', 'valid7', 'degraded_weight', 'clear_weight']}
                b['base_cosine'] = torch.zeros(5)  # Unused by the corrected same-call objective.
                b['fpn'] = object()
                pred = b['base'] if update == 0 else torch.cat([items[i]['prediction'] for i in ids])
                current.update({'batch': b, 'pred': pred})
                compiled = float(namespace['loss']([0]))
                vectors = current['vectors']
                assert vectors.shape == (10, 512) and not vectors.requires_grad
                vector_name = 'identity_update' + str(update) + '_batch' + str(index).zfill(2) + '.npy'
                np.save(out / vector_name, vectors.numpy().copy(), allow_pickle=False)
                reference = (vectors[:5] * b['truth']).sum(1)
                cosine = (vectors[5:] * b['truth']).sum(1)
                feature, interior = feature_errors(pred, b['target'], b['feature'], b['interior'])
                pixel = mean((pred - b['target']).square(), b['mask'])
                base_pixel = mean((b['base'] - b['target']).square(), b['mask'])
                score, base_score = ssim(pred, b['target'], b['valid7']), ssim(b['base'], b['target'], b['valid7'])
                anchor = mean((pred - b['base']).square(), b['mask'])
                w, clear = b['degraded_weight'], b['clear_weight']
                nf, ni = namespace['normalizers']
                terms = {'degraded_landmark_detail': w * feature / nf,
                    'degraded_observed_detail': w * .25 * interior / ni,
                    'degraded_pixel': w * .05 * pixel / base_pixel.clamp_min(1e-5),
                    'clear_baseline_anchor': clear * .05 * anchor / base_pixel.clamp_min(1e-5),
                    'pixel_regression': 2 * F.relu((pixel - base_pixel) / base_pixel.clamp_min(1e-5)),
                    'SSIM_regression': 5 * F.relu(base_score - score),
                    'ArcFace_regression': 5 * F.relu(reference - cosine)}
                difference = abs(float(sum(terms.values()).mean()) - compiled)
                maximum_compiled_difference = max(maximum_compiled_difference, difference)
                assert difference <= 1e-6
                if update == 0:
                    assert torch.equal(vectors[:5], vectors[5:]) and torch.equal(reference, cosine)
                    assert all(torch.count_nonzero(terms[k]) == 0 for k in [
                        'clear_baseline_anchor', 'pixel_regression', 'SSIM_regression', 'ArcFace_regression'])
                basis = {'raw_feature_MSE': feature, 'raw_interior_MSE': interior, 'raw_pixel_MSE': pixel,
                    'base_raw_pixel_MSE': base_pixel, 'raw_SSIM': score, 'base_raw_SSIM': base_score,
                    'raw_ArcFace': cosine, 'same_call_base_raw_ArcFace': reference, 'baseline_anchor_MSE': anchor}
                for j, i in enumerate(ids):
                    values = {k: float(v[j]) for k, v in terms.items()}
                    rows[i]['states'][str(update)] = {'terms': values, 'objective': sum(values.values()),
                        **{k: float(v[j]) for k, v in basis.items()}, 'identity_batch': index, 'identity_row': j,
                        'identity_vectors': vector_name}
                batches.append({'update': update, 'batch': index, 'ids': [items[i]['case']['id'] for i in ids],
                    'compiled_objective': compiled, 'manual_objective_float32': float(sum(terms.values()).mean()),
                    'absolute_difference': difference, 'identity_vectors': vector_name})
    hook.remove()
    assert counts == {'recognizer_forwards': 30, 'saved_prediction_replays': 20}
    assert a.state_hash(identity.state_dict()) == identity_before and a.state_hash(head.state_dict()) == head_before
    assert all(not value.requires_grad and value.grad is None for value in list(identity.parameters()) + list(head.parameters()))
    groups = {}
    labels = {'all', 'clear', 'degraded'} | {c['source'] + '/' + name for c in p['cases'] for name in ['all', 'clear', 'degraded', c['profile']]}
    for label in sorted(labels):
        selected = [r for r in rows if label in {'all', 'clear' if r['profile'] == 'clear' else 'degraded',
                    r['source'] + '/all', r['source'] + ('/clear' if r['profile'] == 'clear' else '/degraded'), r['source'] + '/' + r['profile']}]
        assert selected
        term_names = list(selected[0]['states']['0']['terms'])
        states = {str(u): {'objective': float(np.mean([r['states'][str(u)]['objective'] for r in selected])),
            'terms': {k: float(np.mean([r['states'][str(u)]['terms'][k] for r in selected])) for k in term_names},
            'active_penalty_cases': {k: sum(r['states'][str(u)]['terms'][k] > 0 for r in selected)
                                     for k in ['pixel_regression', 'SSIM_regression', 'ArcFace_regression']}}
            for u in [0, 50]}
        groups[label] = {'cases': len(selected), 'states': states,
            'objective_reduction': states['0']['objective'] - states['50']['objective'],
            'contribution_to_equal50_case_objective_reduction': len(selected) / 50 * (states['0']['objective'] - states['50']['objective']),
            'term_reductions': {k: states['0']['terms'][k] - states['50']['terms'][k] for k in term_names}}
    assert abs(groups['all']['states']['0']['objective'] - 1.3) <= 1e-6
    artifacts = {path.name: a.sha(path) for path in out.glob('*.npy')}
    result = {'complete': True, 'plan_sha256': a.sha(out / 'plan.json'), 'runner_sha256': a.sha(Path(__file__)),
        'seconds': time.monotonic() - started, 'rows': rows, 'batches': batches, 'groups': groups,
        'source_bindings_sha256': sources, 'artifacts_sha256': artifacts, 'counts': counts,
        'normalizers': {k: cohort[k] for k in ['feature_normalizer', 'interior_normalizer']},
        'maximum_compiled_loss_difference': maximum_compiled_difference,
        'maximum_CPU_saved_target_embedding_difference': maximum_target_vector_error,
        'frozen_recognizer_state': identity_before, 'fixed_filter_head_state': head_before,
        'local_CPU_loss_only': True, 'GPU_gradient_or_optimizer_causality_proven': False,
        'head_forwards': 0, 'DGP_forwards': 0, 'gradient_calls': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'native_or_reserved_used': False, 'VM_actions': False, 'app_promotion': False, 'goal_complete': False}
    a.write(out / 'results.json', result)
    print(json.dumps({'complete': True, 'groups': {k: groups[k] for k in ['all', 'clear', 'degraded']},
                      'counts': counts, 'seconds': result['seconds']}, indent=2))


if __name__ == '__main__':
    main()
