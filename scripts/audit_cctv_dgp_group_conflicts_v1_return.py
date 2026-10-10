"""Prospective group-gradient arithmetic and fresh forward audit; no local autograd."""
import argparse
import copy
import math
import shutil
from pathlib import Path
import sys
import tarfile
import time
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT/'outputs/cctv_dgp_group_conflicts_v1_vm'
OUT = ROOT/'outputs/cctv_dgp_group_conflicts_v1_vm_return'
sys.path.insert(0, str(BUNDLE))
from cctv_dgp_group_conflicts_v1_contract import (
    NAME, STEM, TERMS, SCOPES, FRACTIONS, BUDGETS, STATE, read, write, sha, verified_assets)
from frozen_capacity_contract import exported_pixel_metrics, detail_metric, feature_support, capacity
from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only, review_groups


def pixels(path, mode='RGB'):
    with Image.open(path) as im: return np.asarray(im.convert(mode)).copy()


def exact_structure(saved, fresh, atol=1e-10):
    if isinstance(fresh, dict):
        assert saved.keys() == fresh.keys()
        for k, v in fresh.items(): exact_structure(saved[k], v, atol)
    elif isinstance(fresh, list):
        assert len(saved) == len(fresh)
        for a, b in zip(saved, fresh): exact_structure(a, b, atol)
    elif isinstance(fresh, (float, np.floating)): assert abs(saved-fresh) <= atol
    else: assert saved == fresh



BRIGHTNESS_ARITHMETIC_ROWS = []


def compare_capacity_receipt(saved, fresh, base_saved, proposal_saved, base_fresh, proposal_fresh, label):
    # Stored scientific rows and derived decisions remain exact. No acceptance threshold changes.
    assert saved == capacity(base_saved, proposal_saved, .01)
    assert saved['pass'] == fresh['pass']
    assert [(x['group'], x['metric']) for x in saved['preservation_failures']] == [(x['group'], x['metric']) for x in fresh['preservation_failures']]
    assert (saved['relative_feature_gain'] >= .01) == (fresh['relative_feature_gain'] >= .01)
    assert all(x >= 0 for x in saved['source_feature_gains'].values()) == all(x >= 0 for x in fresh['source_feature_gains'].values())
    assert (saved['brightness_gain_fraction'] > .2) == (fresh['brightness_gain_fraction'] > .2)
    a, b = copy.deepcopy(saved), copy.deepcopy(fresh)
    for row in [a, b]:
        row['brightness_gain_fraction'] = 0.
        for failure in row['preservation_failures']:
            if failure['metric'] == 'brightness_gain_fraction': failure['candidate'] = 0.
    exact_structure(a, b)  # Original arithmetic comparison for every other value.
    bs, bf = base_saved['degraded'], base_fresh['degraded']
    cs, cf = proposal_saved['degraded'], proposal_fresh['degraded']
    input_errors = [abs(bs['MSE']-bf['MSE']), abs(cs['MSE']-cf['MSE']),
                    abs(cs['constant_mean_shift_only_MSE']-cf['constant_mean_shift_only_MSE'])]
    assert max(input_errors) <= 1e-15  # Far smaller than the unchanged scientific MSE allowance.
    ns = max(0., bs['MSE']-cs['constant_mean_shift_only_MSE'])
    nf = max(0., bf['MSE']-cf['constant_mean_shift_only_MSE'])
    ds = max(bs['MSE']-cs['MSE'], 1e-12)
    df = max(bf['MSE']-cf['MSE'], 1e-12)
    vs, vf = saved['brightness_gain_fraction'], fresh['brightness_gain_fraction']
    assert vs == ns/ds and vf == nf/df
    # Bound |ns/ds-nf/df| by numerator/denominator propagation plus binary64 rounding.
    bound = abs(ns-nf)/ds + abs(nf)*abs(df-ds)/(ds*df) + 8*(math.ulp(vs)+math.ulp(vf))
    error = abs(vs-vf)
    assert error <= bound, (label, error, bound)
    BRIGHTNESS_ARITHMETIC_ROWS.append({'comparison': label, 'saved': vs, 'fresh': vf,
        'absolute_difference': error, 'propagation_bound': bound,
        'maximum_input_MSE_difference': max(input_errors), 'scientific_decisions_exact': True})


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--expected-sha', required=True)
    a = parser.parse_args(); started = time.monotonic(); pin = sha(BUNDLE/'protocol.json')
    p = verified_assets(BUNDLE, pin)
    for n, d in p['local_sources_sha256'].items(): assert sha(ROOT/n) == d, n
    archive_path = ROOT/'outputs'/(STEM+'-results.tar.gz')
    assert sha(archive_path) == a.expected_sha
    assert Path(str(archive_path)+'.sha256').read_text().split() == [a.expected_sha, archive_path.name]
    export = read(ROOT/'outputs'/(STEM+'-export.json'))
    assert export['complete'] and export['archive_sha256'] == a.expected_sha and export['bytes'] == archive_path.stat().st_size
    assert not OUT.exists(), 'Preserve any previous return import'
    with tarfile.open(archive_path, 'r:gz') as tar:
        members = tar.getmembers(); seen = set(); total = 0
        for m in members:
            assert m.isfile() and not m.issym() and not m.islnk()
            parts = m.name.split('/'); assert parts[0] == NAME+'_return'
            assert all(part not in ['', '.', '..'] for part in parts) and '\\' not in m.name and ':' not in m.name
            n = '/'.join(parts[1:]); assert n not in seen and n not in p['assets_sha256']; seen.add(n)
            assert n.startswith('outputs/') or n in ['protocol.json', 'export_manifest.json', 'supervisor_receipt.json', 'diagnostic.log', 'diagnostic_exit_code.txt']
            total += m.size; assert total <= BUDGETS['return_uncompressed_bytes']+32*1024**2
        assert len(seen) <= 5000
        assert {'protocol.json', 'export_manifest.json', 'supervisor_receipt.json', 'diagnostic.log', 'diagnostic_exit_code.txt'} <= seen
        OUT.mkdir()
        for m in members:
            destination = OUT/'/'.join(m.name.split('/')[1:])
            assert destination.resolve().is_relative_to(OUT.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(m) as incoming, destination.open('xb') as target:
                shutil.copyfileobj(incoming, target)
        retained = set()
        for path in OUT.rglob('*'):
            assert not path.is_symlink()
            if path.is_file(): retained.add(path.relative_to(OUT).as_posix())
        assert retained == seen, 'Imported members changed after hash-bound transfer'
        assert all((OUT/'/'.join(m.name.split('/')[1:])).stat().st_size == m.size for m in members)
    assert sha(OUT/'protocol.json') == pin
    manifest = read(OUT/'export_manifest.json')
    assert manifest['complete'] and manifest['protocol_sha256'] == pin
    assert set(manifest['files_sha256']) == seen-{'export_manifest.json'}
    for n, d in manifest['files_sha256'].items(): assert sha(OUT/n) == d
    success = (OUT/'outputs/results.json').is_file()
    assert success != (OUT/'outputs/failure.json').is_file()
    result = read(OUT/('outputs/results.json' if success else 'outputs/failure.json'))
    assert result['protocol_sha256'] == pin
    assert result['optimizer_updates'] == result['epochs'] == result['committed_trajectory_updates'] == result['backwards'] == 0
    assert result['optimizer_constructed'] is False and result['new_trained_checkpoint'] is False
    outer = read(OUT/'supervisor_receipt.json')
    assert outer['protocol_sha256'] == pin and outer['cap_seconds'] == BUDGETS['external_seconds'] and outer['kill_grace_seconds'] == 30
    assert int((OUT/'diagnostic_exit_code.txt').read_text()) == outer['worker_exit_code']
    if not success:
        write(ROOT/'outputs/cctv_dgp_group_conflicts_v1_failure_import.json', {
            'complete': True, 'archive_sha256': a.expected_sha, 'members_verified': len(members),
            'protocol_sha256': pin, 'failure': result, 'partial_neural_evidence_not_audited': True,
            'model_qualification': False, 'goal_complete': False})
        print({'complete': True, 'failure_imported': True, 'partial_neural_evidence_not_audited': True}); return
    assert outer['worker_exit_code'] == 0 and result['complete'] and result['gradient_queries'] == 160
    assert result['seconds'] <= BUDGETS['worker_seconds'] and result['trial_seconds'] <= BUDGETS['trial_seconds']
    assert result['peak_allocated_VRAM_bytes'] <= BUDGETS['peak_vram_bytes']
    assert result['states_before'] == result['states_after'] and result['states_before']['original'] == STATE
    assert result['states_before']['candidate'] == STATE and result['states_before']['recognizer'] == p['recognizer_state']
    ntrial = len(result['trial_summaries']); assert ntrial in [0, 3]
    assert result['forward_counts'] == {'original': 20, 'candidate': 30+20*ntrial, 'recognizer': 50+20*ntrial}
    folder = OUT/'outputs'
    with np.load(folder/'group_gradients.npz', allow_pickle=False) as saved:
        assert saved.files == ['gradients']; aggregate = saved['gradients'].copy()
    assert aggregate.dtype == np.float64 and aggregate.shape == (32, 1996035) and np.isfinite(aggregate).all()
    initial = np.load(folder/'initial_parameters.npy', allow_pickle=False)
    assert np.array_equal(initial, np.load(BUNDLE/'evidence/initial_parameters.npy', allow_pickle=False))
    assert initial.dtype == np.float32 and initial.shape == (1996035,) and np.isfinite(initial).all()
    decoder_mask = np.zeros(1996035, bool)
    for desc in p['parameter_layout']:
        if desc['partition'] == 'decoder_control': decoder_mask[desc['start']:desc['end']] = True
    weight_norm = p['initial_decoder_weight_L2']
    assert abs(float(np.linalg.norm(initial.astype(np.float64)[decoder_mask]))-weight_norm) <= 1e-10
    gradient = read(folder/'gradient_receipt.json')
    assert gradient['complete'] and gradient['groups'] == p['group_losses']
    assert gradient['seconds'] <= BUDGETS['gradient_seconds'] and gradient['gradient_queries'] == 160
    assert len(gradient['rows']) == 160 and gradient['aggregate_vectors'] == 32
    assert gradient['individual_query_vectors_exported'] is False and gradient['local_autograd_replay_permitted'] is False
    seen_queries = set(); query_counts = [0]*32
    for row in gradient['rows']:
        idx, batch_index = row['group_index'], row['batch']; assert 0 <= idx < 32 and 0 <= batch_index < 10
        assert (idx, batch_index) not in seen_queries; seen_queries.add((idx, batch_index)); query_counts[idx] += 1
        assert row['group'] == p['group_losses'][idx]
        assert row['case_ids'] == p['cohorts'][0]['case_ids'][5*batch_index:5*batch_index+5]
        assert set(row['case_ids']) & set(row['group']['case_ids'])
        assert len(row['query_vector_sha256']) == 64 and set(row['query_vector_sha256']) <= set('0123456789abcdef')
        assert math.isfinite(row['value']) and math.isfinite(row['query_L2']) and row['query_L2'] >= 0
        assert len(row['parameters']) == 158
        square_sum = 0.
        for desc, value in zip(p['parameter_layout'], row['parameters']):
            assert value['name'] == desc['name'] and isinstance(value['graph_connected'], bool)
            assert 0 <= value['nonzero_values'] <= desc['elements'] and math.isfinite(value['L2']) and value['L2'] >= 0
            assert (value['nonzero_values'] == 0) == (value['L2'] == 0)
            if not value['graph_connected']: assert value['nonzero_values'] == 0
            square_sum += value['L2']**2
        assert abs(math.sqrt(square_sum)-row['query_L2']) <= 1e-10
    assert query_counts == [5]*32
    parity = read(folder/'loss_surrogate_parity.json')
    assert parity['complete'] and parity['passed'] and parity['cases'] == 100 and parity['gradient_queries'] == 0
    assert [r['id'] for r in parity['rows']] == [c['id'] for c in p['cases']]
    maxima = {k: max(r['errors'][k] for r in parity['rows']) for k in ['MSE', 'SSIM', 'structure']}
    assert parity['maximum_errors'] == maxima
    assert maxima['MSE'] <= p['loss_surrogate_parity']['MSE_abs']
    assert maxima['SSIM'] <= p['loss_surrogate_parity']['SSIM_abs']
    assert maxima['structure'] <= p['loss_surrogate_parity']['landmark_structure_abs']
    certificate = read(folder/'direction_certificate.json'); assert certificate['complete'] and certificate['global_infeasibility_proven'] is False
    norms = np.linalg.norm(aggregate, axis=1)
    assert np.allclose(norms, certificate['gradient_norms'], rtol=1e-12, atol=1e-12)
    found = certificate['common_direction_found']; assert isinstance(found, bool)
    direction = None
    if np.all(norms > 0):
        unit = aggregate/norms[:, None]; gram = unit@unit.T
        assert np.allclose(gram, certificate['normalized_Gram'], rtol=0, atol=1e-12)
        weights = np.asarray(certificate['simplex_weights'], dtype=np.float64)
        assert weights.shape == (32,) and np.isfinite(weights).all() and (weights >= 0).all() and abs(weights.sum()-1.) <= 1e-12
        combined = weights@unit; combined_norm = float(np.linalg.norm(combined))
        assert abs(combined_norm-certificate['combined_gradient_norm']) <= 1e-12
        trial_direction = -combined/combined_norm if combined_norm > 0 else np.zeros(1996035, np.float64)
        cosines = -(unit@trial_direction)
        assert np.allclose(cosines, certificate['normalized_descent_cosines'], rtol=0, atol=1e-12)
        assert np.allclose(aggregate@trial_direction, certificate['raw_directional_derivatives'], rtol=1e-12, atol=1e-12)
        assert abs(float(cosines.min())-certificate['minimum_normalized_descent_cosine']) <= 1e-12
        assert certificate['minimum_descent_cosine_requirement'] == p['minimum_normalized_descent_cosine']
        expected_found = bool(certificate['dual_solver_success'] and combined_norm > 0 and cosines.min() >= 1e-7)
        assert found == expected_found and certificate['dual_solver_iterations'] <= 500 and certificate['seconds'] <= 120
        gap = 2.*(float(weights@gram@weights)-float(np.min(gram@weights)))
        assert abs(gap-certificate['duality_gap']) <= 1e-12
        if found:
            direction = np.load(folder/'common_direction.npy', allow_pickle=False)
            assert direction.dtype == np.float64 and direction.shape == (1996035,) and np.isfinite(direction).all()
            assert np.allclose(direction, trial_direction, rtol=0, atol=p['direction_reconstruction_arithmetic_atol'])
            assert abs(float(np.linalg.norm(direction))-1.) <= 1e-12
            assert np.all(-(unit@direction) >= 1e-7)
    else:
        assert not found and certificate['dual_solver_iterations'] == 0
    assert found == result['common_direction_found'] and ntrial == (3 if found else 0)
    if not found: assert not (folder/'common_direction.npy').exists()
    previous = read(folder/'previous_direction_group_predictions.json')
    assert previous['complete'] and previous['groups'] == p['group_losses']
    with np.load(BUNDLE/'evidence/previous_directions.npz', allow_pickle=False) as saved: old_direction = saved['balanced_2'].copy()
    old_cosines = -(aggregate@(old_direction/np.linalg.norm(old_direction)))/np.maximum(norms, 1e-300)
    assert np.allclose(old_cosines, previous['descent_cosines'], rtol=0, atol=1e-12)
    assert np.allclose(aggregate@old_direction, previous['raw_directional_derivatives'], rtol=1e-12, atol=1e-12)
    assert previous['negative_cosine_group_indices'] == np.flatnonzero(old_cosines < 0).tolist()
    labels = ['baseline']+(['common_'+format(f, '.0e').replace('-', 'm') for f in FRACTIONS] if found else [])
    assert result['trials_completed'] == labels[1:] and result['raw_and_PNG_cases'] == 100*len(labels)
    rebuilt = {}; cases_by_id = {c['id']: c for c in p['cases']}
    for label in labels:
        saved = read(folder/label/'metrics.json'); assert saved['complete'] and saved['variant'] == label
        assert [r['id'] for r in saved['rows']] == [c['id'] for c in p['cases']]
        rows = []
        for c, report in zip(p['cases'], saved['rows']):
            assert time.monotonic()-started <= 1500, 'Local audit deadline'
            cid = c['id']; camera = pixels(BUNDLE/c['input']); target = pixels(BUNDLE/c['target']); mask = pixels(BUNDLE/c['observed'], 'L') > 0
            feature = feature_support(mask, c['landmarks5_canvas_xy'])
            raw = np.load(folder/label/(cid+'.npy'), allow_pickle=False)
            base = np.load(folder/'baseline'/(cid+'.npy'), allow_pickle=False)
            png = pixels(folder/label/(cid+'.png')); mean_png = pixels(folder/label/(cid+'_mean_only.png'))
            assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
            assert np.array_equal(raw[~mask], camera[~mask].astype(np.float32)/np.float32(255))
            assert np.array_equal(deliver(raw, camera, mask), png)
            mraw, mpng, shift = mean_only(raw, base, camera, mask); assert np.array_equal(mpng, mean_png)
            vector = np.load(folder/label/(cid+'_embeddings.npy'), allow_pickle=False)
            assert vector.dtype == np.float32 and vector.shape == (3, 512) and np.isfinite(vector).all()
            assert np.allclose(np.sum(vector.astype(np.float64)**2, axis=1), 1, rtol=0, atol=1e-5)
            if label != 'baseline': assert np.array_equal(vector[2], np.load(folder/'baseline'/(cid+'_embeddings.npy'), allow_pickle=False)[2])
            rm = pixel_metrics(raw, target, mask); pm = exported_pixel_metrics(png, target, mask)
            rm.update({'ArcFace_observed_fixed': float(vector[0]@vector[2]), 'landmark_high_frequency_MSE': detail_float(raw, target, feature),
                'constant_mean_shift_only_MSE': pixel_metrics(mraw, target, mask)['MSE']})
            pm.update({'ArcFace_observed_fixed': float(vector[1]@vector[2]), 'landmark_high_frequency_MSE': detail_metric(png, target, feature),
                'constant_mean_shift_only_MSE': exported_pixel_metrics(mpng, target, mask)['MSE']})
            exact_structure(report, {'id': cid, 'source': c['source'], 'profile': c['profile'], 'raw': rm, 'png': pm, 'postclip_mean_RGB_shift': shift.tolist()})
            rows.append({**report, 'raw': rm, 'png': pm})
        groups = {co['name']: {s: review_groups([r for r in rows if r['id'] in co['case_ids']], s) for s in ['raw', 'png']} for co in p['cohorts']}
        # Stored rows must reproduce groups exactly; fresh pixel recomputation uses arithmetic-only tolerance.
        stored_groups = {co['name']: {s: review_groups([r for r in saved['rows'] if r['id'] in co['case_ids']], s) for s in ['raw', 'png']} for co in p['cohorts']}
        assert saved['groups'] == stored_groups; exact_structure(saved['groups'], groups)
        rebuilt[label] = groups
        if label != 'baseline':
            summary = next(r for r in result['trial_summaries'] if r['variant'] == label)
            assert summary['scope'] == 'common' and direction is not None
            assert summary['scale_weight_partition'] == 'decoder_control'
            assert abs(summary['selected_weight_L2']-weight_norm) <= 1e-12 and summary['relative_fraction'] in FRACTIONS
            expected = (initial.astype(np.float64)+summary['relative_fraction']*weight_norm*direction).astype(np.float32)
            actual = np.load(folder/(label+'_parameters.npy'), allow_pickle=False); assert np.array_equal(expected, actual)
            delta = actual.astype(np.float64)-initial.astype(np.float64)
            assert abs(float(np.linalg.norm(delta))-summary['actual_displacement_L2']) <= 1e-12
            dots = aggregate@delta
            assert np.allclose(dots, summary['gradient_dot_actual_displacement'], rtol=1e-12, atol=1e-12)
            assert summary['all_actual_group_derivatives_negative'] == bool(np.all(dots < 0))
            assert abs(float(np.linalg.norm(delta))/weight_norm-summary['relative_displacement_actual']) <= 1e-12
            comparisons = {co['name']: {s: capacity(rebuilt['baseline'][co['name']][s], groups[co['name']][s], .01) for s in ['raw', 'png']} for co in p['cohorts']}
            baseline_saved_groups = read(folder/'baseline/metrics.json')['groups']
            for cohort in p['cohorts']:
                co = cohort['name']
                for stage in ['raw', 'png']:
                    compare_capacity_receipt(summary['comparisons'][co][stage], comparisons[co][stage],
                        baseline_saved_groups[co][stage], saved['groups'][co][stage],
                        rebuilt['baseline'][co][stage], groups[co][stage], label+'/'+co+'/'+stage)

    import torch
    from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
    torch.set_num_threads(4)
    baseline_rows = {r['id']: r for r in read(folder/'baseline/metrics.json')['rows']}
    surrogate_values = {}
    with torch.no_grad():
        for c in p['cases']:
            cid = c['id']; mask = pixels(BUNDLE/c['observed'], 'L') > 0
            feature = feature_support(mask, c['landmarks5_canvas_xy'])
            raw = np.load(folder/'baseline'/(cid+'.npy'), allow_pickle=False); target = pixels(BUNDLE/c['target'])
            to_tensor = lambda a: torch.from_numpy(a.copy()).permute(2, 0, 1)[None]
            losses = pixel_and_structure_losses(to_tensor(raw), to_tensor(target.astype(np.float32)/np.float32(255)),
                torch.from_numpy(mask.astype(np.float32))[None, None], torch.from_numpy(feature.astype(np.float32))[None, None])
            surrogate_values[cid] = {k: float(v[0]) for k, v in losses.items()}
            surrogate_values[cid]['ArcFace_loss'] = 1.-baseline_rows[cid]['raw']['ArcFace_observed_fixed']
            recorded = next(r['errors'] for r in parity['rows'] if r['id'] == cid)
            scientific = baseline_rows[cid]['raw']
            fresh_errors = {'MSE': abs(surrogate_values[cid]['MSE']-scientific['MSE']),
                'SSIM': abs(1.-surrogate_values[cid]['SSIM_loss']-scientific['SSIM']),
                'structure': abs(surrogate_values[cid]['landmark_structure']-scientific['landmark_high_frequency_MSE'])}
            assert abs(fresh_errors['MSE']-recorded['MSE']) <= 1e-12
            assert abs(fresh_errors['SSIM']-recorded['SSIM']) <= 1e-10
            assert abs(fresh_errors['structure']-recorded['structure']) <= 1e-10
    for row in gradient['rows']:
        group = row['group']; ids = [cid for cid in row['case_ids'] if cid in group['case_ids']]
        expected_value = float(np.mean([surrogate_values[cid][group['metric']] for cid in ids]))
        tolerance = 1e-6 if group['metric'] == 'ArcFace_loss' else 1e-10
        assert abs(expected_value-row['value']) <= tolerance

    # Fresh saved-vector CPU inference only: no candidate.assign_trial or gradient API.
    import torch
    from cctv_dgp_pilot import FixedObservedIdentity, grid112, state_hash
    from dgp_frozen_inference_v2 import load_frozen_dgp_restorer
    from cctv_dgp_group_conflicts_v1_candidate import OriginalFeatureProbe
    torch.set_num_threads(4)
    original, _ = load_frozen_dgp_restorer(BUNDLE/'weights/dgp_v2.pth', expected_sha256=p['original_checkpoint_sha256'], device='cpu')
    model = OriginalFeatureProbe(original.net, p['parameter_layout']); identity = FixedObservedIdentity(BUNDLE/'weights/w600k_r50.onnx', 'cpu')
    initial_cpu = model.vector(); assert np.array_equal(initial_cpu, initial)
    frozen_before = (state_hash(original.net), state_hash(identity))
    refs = {r['id']: r for r in p['references']}
    # First reference from each source in each TRAIN cohort:20 cases per variant,200 CPU replays.
    replay_ids = []
    for co in p['cohorts']:
        for source in sorted({c['source'] for c in p['cases']}):
            choices = [cid for cid in co['case_ids'] if cases_by_id[cid]['source'] == source]
            replay_ids.extend(choices[:5])
    worst_raw = 0.; worst_png = 0; worst_vector = 0.
    with torch.no_grad():
        for label in labels:
            values = initial if label == 'baseline' else np.load(folder/(label+'_parameters.npy'), allow_pickle=False)
            for desc, (_, value) in zip(p['parameter_layout'], model.selected):
                value.copy_(torch.from_numpy(values[desc['start']:desc['end']].reshape(desc['shape'])))
            assert state_hash(model.net) == read(folder/label/'metrics.json')['candidate_state']
            for cid in replay_ids:
                assert time.monotonic()-started <= 1500
                c = cases_by_id[cid]; camera = pixels(BUNDLE/c['input']); mask = pixels(BUNDLE/c['observed'], 'L') > 0; target = pixels(BUNDLE/c['target'])
                rgb = lambda v: torch.from_numpy(v.astype(np.float32)/np.float32(255)).permute(2, 0, 1)[None]
                x = rgb(camera); support = torch.from_numpy(mask.astype(np.float32))[None, None]
                base = original(x).detach().clone(); result_cpu = model(x, support, base)[0].permute(1, 2, 0).numpy().copy()
                raw = np.load(folder/label/(cid+'.npy'), allow_pickle=False); png = pixels(folder/label/(cid+'.png'))
                worst_raw = max(worst_raw, float(np.abs(result_cpu-raw).max()))
                worst_png = max(worst_png, int(np.abs(deliver(result_cpu, camera, mask).astype(np.int16)-png.astype(np.int16)).max()))
                grid = torch.from_numpy(grid112(refs[c['source_person_or_reference']]['matrix112']))[None]
                raw_tensor = torch.from_numpy(raw).permute(2, 0, 1)[None]
                vectors = identity.embedding(torch.cat([raw_tensor, rgb(png), rgb(target)]), support.repeat(3, 1, 1, 1), grid.repeat(3, 1, 1, 1)).numpy().copy()
                stored = np.load(folder/label/(cid+'_embeddings.npy'), allow_pickle=False)
                worst_vector = max(worst_vector, float(np.abs(vectors-stored).max()))
            if label != 'baseline':
                for desc, (_, value) in zip(p['parameter_layout'], model.selected): value.copy_(torch.from_numpy(initial[desc['start']:desc['end']].reshape(desc['shape'])))
        assert state_hash(model.net) == state_hash(original.net) == STATE
    tol = p['CPU_replay_tolerances']
    assert worst_raw <= tol['raw_max_abs'] and worst_png <= tol['PNG_byte_max'] and worst_vector <= tol['embedding_max_abs']
    assert frozen_before == (state_hash(original.net), state_hash(identity))
    verified_assets(BUNDLE, pin)
    assert len(BRIGHTNESS_ARITHMETIC_ROWS) == 4*ntrial
    write(ROOT/'outputs/cctv_dgp_group_conflicts_v1_independent_audit.json', {
        'complete': True, 'archive_sha256': a.expected_sha, 'protocol_sha256': pin,
        'checker_sha256': sha(Path(__file__)), 'members_verified': len(members), 'all_saved_raw_and_PNG_metrics_recomputed': True, 'raw_and_PNG_cases': 100*len(labels),
        'all32_saved_group_vectors_and_certificate_arithmetic_verified': True, 'conditional_displacements_verified': ntrial,
        'individual_query_vector_aggregation_independently_recomputed': False, 'common_direction_found': found,
        'gradient_autograd_replay_local': False, 'CPU_replay_cases': len(labels)*len(replay_ids),
        'maximum_raw_error': worst_raw, 'maximum_PNG_byte_error': worst_png, 'maximum_embedding_error': worst_vector,
        'seconds': time.monotonic()-started, 'local_gradient_queries': 0, 'local_optimizer_updates': 0,
        'visual_review_pending': True, 'full_TRAIN_capacity_not_tested': True, 'model_qualification': False, 'goal_complete': False,
        'parent_audit_sha256': sha(BUNDLE/'evidence/parent_independent_audit.json'),
        'brightness_arithmetic_propagation': BRIGHTNESS_ARITHMETIC_ROWS,
        'all_stored_row_comparisons_and_failure_decisions_exact': True, 'comparisons_verified': 4*ntrial,
        'scientific_thresholds_changed': False})
    print({'complete': True, 'diagnostic_evidence_audited': True, 'model_qualification': False})


if __name__ == '__main__': main()
