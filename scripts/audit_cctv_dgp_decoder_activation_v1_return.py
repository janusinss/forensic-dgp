"""Saved activation evidence audit; no learned network or derivative execution.

Tensor loss filters are read-only arithmetic.  Saved embeddings are reused;
neither their neural values nor a selected neural derivative branch are certified
by this checker.  Partial receipts verify retained bytes, never scientific success.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time

from cctv_dgp_decoder_activation_v1_contract import (
    STAGE, STATES, AMPLITUDES, CORRECTED_HELPER_SHA, read, write, sha, array_sha,
    verify_transfer,
)
from cctv_dgp_decoder_activation_v1_math import (
    finite_objective, one_sided_objective, mean_control_right_mse_at_initial,
)

EXPECTED_COUNTS = dict(optimizer_updates=0, gradient_queries=70, backwards=0,
    original_batch_calls=20, prior_feature_batch_calls=20,
    decoder_batch_calls=82, recognizer_batch_calls=82)
NORM_METADATA_BOUND = dict(rtol=2e-6, atol=1e-12)


def verify_counts(counts, success=True):
    assert isinstance(counts, dict) and counts.keys() == EXPECTED_COUNTS.keys()
    assert all(type(value) is int and 0 <= value <= EXPECTED_COUNTS[key]
               for key, value in counts.items())
    assert counts['optimizer_updates'] == counts['backwards'] == 0
    if success:
        assert counts == EXPECTED_COUNTS


def close_tree(actual, saved, *, atol=1e-10, rtol=1e-10):
    """Prospectively bounded arithmetic comparison, never an acceptance slack."""
    import numpy as np
    if isinstance(actual, dict):
        assert isinstance(saved, dict) and actual.keys() == saved.keys()
        return max([close_tree(value, saved[key], atol=atol, rtol=rtol)
                    for key, value in actual.items()] or [0.])
    if isinstance(actual, (bool, str)) or actual is None:
        assert actual == saved
        return 0.
    a, b = np.asarray(actual, np.float64), np.asarray(saved, np.float64)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    difference = float(np.max(np.abs(a-b))) if a.size else 0.
    assert np.all(np.abs(a-b) <= atol+rtol*np.abs(a)), 'Saved scalar arithmetic differs'
    return difference


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode()+b'\0'+str(value.dtype).encode()+b'\0')
        digest.update(str(tuple(value.shape)).encode()+b'\0'+value.numpy().tobytes())
    return digest.hexdigest()


def expected_state_hash(state, layout, vector):
    """Hash substituted saved bytes without assigning a tensor or model."""
    import numpy as np
    slices = {row['name']: row for row in layout}
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if name in slices:
            row = slices[name]
            assert list(tensor.shape) == row['shape'] and str(tensor.dtype) == 'torch.float32'
            value = vector[row['begin']:row['end']].reshape(row['shape'])
        else:
            value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode()+b'\0'+str(tensor.dtype).encode()+b'\0')
        digest.update(str(tuple(tensor.shape)).encode()+b'\0'+np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def bound_module(path, expected, name):
    assert sha(path) == expected, 'Frozen arithmetic helper differs'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == Path(path).resolve()
    return module


def grouped(rows, mode):
    import numpy as np
    groups = {}
    for row in rows:
        source, profile = row['source'], row['profile']
        for key in {'all', 'clear' if profile == 'clear' else 'degraded', source+'/all',
                    source+('/clear' if profile == 'clear' else '/degraded'), source+'/'+profile}:
            groups.setdefault(key, []).append(row[mode])
    assert len(groups) == 17
    metrics = ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE',
               'constant_mean_shift_only_MSE']
    return {key: dict(cases=len(values), **{metric: float(np.mean([v[metric] for v in values]))
            for metric in metrics}) for key, values in sorted(groups.items())}


def unchanged_gate(before, after):
    assert len(before) == len(after) == 17 and before.keys() == after.keys()
    failures = []
    for key in sorted(before):
        assert before[key]['cases'] == after[key]['cases']
        for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
            a, b = after[key][metric], before[key][metric]
            assert math.isfinite(a) and math.isfinite(b)
            if (metric == 'MSE' and a > b+1e-12) or (metric != 'MSE' and a < b-1e-6):
                failures.append(dict(group=key, metric=metric, baseline=b, candidate=a))
    old, new = before['degraded'], after['degraded']
    gain = 1-new['landmark_high_frequency_MSE']/old['landmark_high_frequency_MSE']
    sources = {key: 1-after[key]['landmark_high_frequency_MSE']/before[key]['landmark_high_frequency_MSE']
               for key in before if key.endswith('/degraded')}
    assert len(sources) == 2
    brightness = max(0., old['MSE']-new['constant_mean_shift_only_MSE'])/max(old['MSE']-new['MSE'], 1e-12)
    if brightness > .2:
        failures.append(dict(group='degraded', metric='brightness_gain_fraction', candidate=brightness, maximum=.2))
    return dict(**{'pass': not failures and gain >= .01 and all(v >= .01 for v in sources.values())},
        global_structure_gain=gain, source_structure_gains=sources,
        brightness_gain_fraction=brightness, preservation_failures=failures)


def verify_gradient_matrix(matrix, clipped, direction, measurement, labels, bound, factor_max_ULPs=2):
    import numpy as np
    assert matrix.dtype == clipped.dtype == np.float32
    assert matrix.shape == (35, 609219) and clipped.shape == (609219,)
    assert direction.dtype == np.float64 and direction.shape == (609219,)
    assert np.isfinite(matrix).all() and np.isfinite(clipped).all() and np.isfinite(direction).all()
    assert measurement['scalar_labels'] == labels and len(labels) == 35
    assert measurement['gradients_shape'] == [35, 609219] and measurement['gradients_dtype'] == 'float32'
    assert measurement['gradients_sha256'] == array_sha(matrix)
    assert measurement['clipped_gradient_sha256'] == array_sha(clipped)
    assert measurement['retained_direction_sha256'] == array_sha(direction)
    slopes = matrix.astype(np.float64) @ direction
    close_tree(slopes, measurement['directional_slopes'], atol=1e-12, rtol=1e-12)
    assert bound == NORM_METADATA_BOUND
    norm = float(np.linalg.norm(matrix[-1].astype(np.float64)))
    stored = measurement['full_gradient_norm_before_clip']
    assert math.isfinite(stored) and stored >= 0
    assert abs(norm-stored) <= bound['atol']+bound['rtol']*norm
    factor = measurement['clip_factor']
    assert math.isfinite(factor) and 0 < factor <= 1.
    assert factor_max_ULPs == 2 and float(np.float32(factor)) == factor
    expected_factor = np.minimum(np.float32(1.)/np.float32(np.float32(stored)+np.float32(1e-6)), np.float32(1.))
    ULPs = abs(int(np.float32(factor).view(np.uint32))-int(expected_factor.view(np.uint32)))
    assert ULPs <= factor_max_ULPs, 'Clipping-factor metadata exceeds prospectively frozen ULP bound'
    assert np.array_equal(clipped, (matrix[-1]*np.float32(factor)).astype(np.float32))
    return slopes, dict(float64_full_gradient_L2=norm, saved_float32_norm=stored,
        absolute_norm_difference=abs(norm-stored), prospectively_frozen_norm_metadata_bound=bound,
        clipping_factor_numpy_float32_reference=float(expected_factor), clipping_factor_metadata_ULPs=ULPs,
        clipping_factor_metadata_max_ULPs=factor_max_ULPs, clipped_float32_bytes_exact=True)


def finite_comparisons(records, measurement, amplitudes=AMPLITUDES, states=STATES):
    """Only the two queried references have a selected-branch linear model."""
    assert amplitudes == AMPLITUDES and states == STATES
    reference = measurement['reference']
    initial = records[(states[0], reference)]
    slope = measurement['directional_slopes'][-1]
    one_sided = measurement['one_sided']['objective_directional']
    assert math.isfinite(slope) and math.isfinite(one_sided)
    rows = []
    for state, amplitude in zip(states, amplitudes):
        actual = records[(state, reference)]
        change = actual['objective']-initial['objective']
        per_profile = {}
        for key, profile_slopes in measurement['current_slopes'].items():
            begin = 1 if key == 'landmark_structure' else 0
            per_profile[key] = [dict(profile_index=i, actual_change=actual['current'][key][i]-initial['current'][key][i],
                selected_branch_linear_change=amplitude*profile_slopes[i],
                actual_minus_linear=actual['current'][key][i]-initial['current'][key][i]-amplitude*profile_slopes[i]) for i in range(begin,5)]
        rows.append(dict(reference=reference, source=measurement['source'], state=state,
            retained_amplitude=amplitude, actual_batch5_objective=actual['objective'],
            initial_batch5_objective=initial['objective'], actual_objective_change=change,
            gradient_pass_batch5_objective=measurement['objective'],
            gradient_vs_snapshot_initial_objective_offset=measurement['objective']-initial['objective'],
            actual_minus_gradient_base_selected_prediction=actual['objective']-(measurement['objective']+amplitude*slope),
            actual_minus_gradient_base_one_sided_prediction=actual['objective']-(measurement['objective']+amplitude*one_sided),
            per_profile_metrics=per_profile, clear_structure_derivative_not_queried=True,
            selected_autograd_linear_change=amplitude*slope,
            affine_residual_one_sided_linear_change=amplitude*one_sided,
            actual_minus_selected_linear=change-amplitude*slope,
            actual_minus_one_sided_linear=change-amplitude*one_sided,
            finite_network_execution_authoritative=True, selected_neural_branch_certified=False,
            literal_float32_derivative_certified=False, PNG_derivative_certified=False,
            new_amplitude_or_feasibility_search=False, model_qualified=False))
    return rows


def partial_evidence(root, out, p, receipt):
    """Preserve a stopped/malformed tail, with no fabricated exact run counts."""
    def optional(path):
        if not path.is_file():
            return None
        try:
            return read(path)
        except (ValueError, UnicodeError):
            return dict(opaque_partial_JSON=True, sha256=sha(path), bytes=path.stat().st_size)
    terminal, failure = optional(root/'run_exit.json'), optional(out/'failure.json')
    assert failure is not None or (isinstance(terminal, dict) and type(terminal.get('exit_code')) is int
                                  and terminal['exit_code'] != 0), 'No authoritative stopped-worker evidence'
    if terminal and not terminal.get('opaque_partial_JSON'):
        assert terminal['flag'] == '--run' and type(terminal['exit_code']) is int
        assert terminal['exit_code'] != 0, 'Successful worker cannot omit its complete result'
    last, malformed = None, 0
    journal = out/'progress.jsonl'
    if journal.is_file():
        for line in journal.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if isinstance(value, dict) and isinstance(value.get('counts'), dict):
                verify_counts(value['counts'], success=False)
                last = value
    write(receipt, dict(complete=True, partial_failure_evidence_only=True,
        protocol_sha256=sha(root/'protocol.json'), failure=failure, terminal_worker=terminal,
        last_known_progress=last, malformed_progress_lines_retained=malformed,
        run_gradient_queries=None, exact_run_counts_known=False, no_missing_result_implies_success=True,
        saved_files={f.relative_to(root).as_posix(): dict(bytes=f.stat().st_size, sha256=sha(f))
                     for f in sorted(out.rglob('*')) if f.is_file()},
        complete_run_or_derivative_validity_verified=False, model_qualified=False,
        no_local_model_construction_or_weight_assignment=True,
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0))


def audit_saved(root, repair, source, parent, replay, p, rp, old, data, receipt):
    import numpy as np
    import torch  # Saved deserialization and fixed loss arithmetic only.
    from PIL import Image
    root, repair, source, parent, replay, receipt = [Path(x).resolve() for x in (root,repair,source,parent,replay,receipt)]
    out, control = root/STAGE, repair/'outputs/decoder_hinge_repair_v1'
    assert not receipt.exists(), 'Retain every earlier audit'
    began = time.monotonic()
    def clock():
        assert time.monotonic()-began <= p['budgets']['saved_audit_seconds'], 'Finite saved-audit stop'
    assert p['gradient_norm_metadata_bound'] == NORM_METADATA_BOUND
    assert p['clip_factor_metadata_max_ULPs'] == 2
    result_path = out/'run_results.json'
    try:
        result = read(result_path) if result_path.is_file() else None
    except (ValueError, UnicodeError):
        result = None
    if result is None or not result.get('complete'):
        partial_evidence(root, out, p, receipt)
        return
    verify_counts(result['counts'])
    assert result['state_labels'] == STATES and result['derivative_reference_ids'] == p['derivative_reference_ids']
    for key, value in dict(exact_retained_RAW_PNG_cases=400, batch_five_objective_reference_states=80,
        full_parameter_gradients=70, parameters_per_gradient=609219, recognizer_batch=5,
        new_parameter_proposals=0, new_amplitudes=0, full_corpus_epochs=0).items():
        assert result[key] == value
    for key in ['historical_batch15_separate','initial_full_state_and_RNG_restored','frozen_models_unchanged']:
        assert result[key] is True
    for key in ['model_qualified','new_incumbent_selection','automatic_follow_on','failed_recipe_resume_allowed']:
        assert result[key] is False
    assert 0 < result['seconds'] <= p['budgets']['worker_seconds']
    assert not (out/'failure.json').exists(), 'A complete result cannot conceal a failure'
    environment = read(out/'environment.json')
    assert environment['GPU'] == 'NVIDIA L4' and environment['host'].split('.')[0] == 'forensic-dgp-thesis'
    assert environment['frozen_states'] == dict(original=old['original_state'], recognizer=old['recognizer_state'],
                                               provider=old['feature_provider_initial_state'])
    assert environment['corrected_helper_sha256'] == CORRECTED_HELPER_SHA
    assert environment['optimizer_present'] is environment['historical_training_called'] is False
    assert environment['new_parameter_proposals'] == environment['new_amplitudes'] == 0
    assert environment['peak_includes_dependency_loading'] is True
    cache = read(out/'cache_receipt.json')
    source_cache = read(source/'outputs/spatial_readout_v1/cache_receipt.json')['rows']
    assert cache['complete'] and cache['exact_original_and_features'] and len(cache['rows']) == 20
    assert 0 < cache['seconds'] <= p['budgets']['cache_seconds']
    for row,inherited in zip(cache['rows'],source_cache):
        assert all(row[key] == inherited[key] for key in ['reference','source','maps','features'])
    storage = read(out/'storage_start.json')
    assert storage['free_bytes'] >= p['budgets']['minimum_free_bytes']
    assert storage['MemAvailable_bytes'] >= p['budgets']['available_RAM_before_import_bytes']
    assert storage['output_cap_bytes'] == p['budgets']['output_byte_cap']
    assert storage['emergency_metadata_bytes'] == p['budgets']['emergency_metadata_bytes']
    assert storage['projected_output_bytes'] <= storage['output_cap_bytes']
    assert storage['free_bytes'] >= 2*storage['projected_output_bytes']+p['budgets']['disk_reserve_bytes']
    assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) <= p['budgets']['output_byte_cap']
    for filename,field,cap in [('cache_projection.json','projected_seconds','cache_seconds'),
                               ('state_timing_projection.json','projected_worker_seconds','worker_seconds'),
                               ('gradient_timing_projection.json','projected_worker_seconds','worker_seconds')]:
        projection = read(out/filename)
        assert math.isfinite(projection[field]) and 0 < projection[field] <= p['budgets'][cap]
    gradient_projection = read(out/'gradient_timing_projection.json')
    assert gradient_projection['projected_gradient_seconds'] <= p['budgets']['gradient_seconds']
    losses = bound_module(parent/'cctv_dgp_group_conflicts_v1_losses.py',
        p['parent_required_sha256']['cctv_dgp_group_conflicts_v1_losses.py'], 'activation_saved_loss_math')
    guard = bound_module(repair/'cctv_dgp_bank_hinge_boundary_v1_loss.py', CORRECTED_HELPER_SHA, 'activation_saved_guard_math')
    metrics = bound_module(parent/'frozen_raw_metrics.py', p['parent_required_sha256']['frozen_raw_metrics.py'], 'activation_saved_raw_metrics')
    capacity = bound_module(parent/'frozen_capacity_contract.py', p['parent_required_sha256']['frozen_capacity_contract.py'], 'activation_saved_capacity')
    packed = bound_module(parent/'cctv_dgp_head4_lossless_v1.py', p['parent_required_sha256']['cctv_dgp_head4_lossless_v1.py'], 'activation_saved_lossless')
    torch.set_num_threads(4)
    def pixels(path, mode='RGB'):
        with Image.open(path) as image:
            assert image.size == (256,256)
            return np.asarray(image.convert(mode)).copy()
    refs, cases = {r['id']:r for r in data['references']}, {c['id']:c for c in data['cases']}
    initial = np.load(control/'initial_vector.npy', allow_pickle=False)
    first = np.load(control/'fit_u001_vector.npy', allow_pickle=False)
    direction = np.load(out/'retained_direction.npy', allow_pickle=False)
    assert initial.dtype == first.dtype == np.float32 and initial.shape == first.shape == (609219,)
    assert direction.dtype == np.float64 and np.array_equal(direction, first.astype(np.float64)-initial.astype(np.float64))
    direction_record = read(out/'retained_direction.json')
    assert direction_record['direction_sha256'] == array_sha(direction) and direction_record['dtype'] == 'float64'
    assert direction_record['elements'] == 609219 and direction_record['new_parameter_proposal'] is False
    assert direction_record['norm_L2'] == float(np.linalg.norm(direction)) <= p['common_radius_L2']
    records, inputs, gradient_targets, gate_rows, max_arithmetic, exact_outputs = {}, {}, {}, {}, 0., 0
    source_full = torch.load(source/'outputs/spatial_readout_v1/state_A0/full_state.pth', map_location='cpu', weights_only=True)
    source_state = source_full['model']; del source_full
    source_receipt = read(source/'outputs/spatial_readout_v1/state_A0/receipt.json')
    assert source_receipt['model_state'] == state_hash(source_state)
    layout = read(source/'outputs/spatial_readout_v1/A_parameter_layout.json')
    assert [row['name'] for row in layout] == p['trainable_parameter_names'] and len(layout) == 15
    assert layout[-1]['end'] == 609219
    def nchw(array):
        return torch.from_numpy(array).permute(0,3,1,2).contiguous()
    for label in STATES:
        clock(); folder = out/label
        saved = read(folder/'objective_components.json')
        assert saved['complete'] and saved['recognizer_batch'] == 5 and saved['historical_batch15_separate']
        assert saved['model_qualified'] is False and len(saved['rows']) == 20
        inherited = control/label/'objective_components.json'
        assert sha(folder/'retained_batch15_objective_components.json') == sha(inherited) == saved['historical_components_sha256']
        vector = np.load(control/(label+'_vector.npy'), allow_pickle=False)
        assert saved['state_vector_sha256'] == array_sha(vector)
        assert saved['model_state'] == expected_state_hash(source_state, layout, vector)
        inherited_metrics = read(control/label/'metrics.json')
        inherited_rows = {row['case']:row for row in inherited_metrics['rows']}
        assert set(inherited_rows) == set(p['case_ids'])
        scientific_rows = []
        for index, block in enumerate(p['reference_blocks']):
            clock(); reference = refs[block['reference']]
            baseline, ev = packed.unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz", block['case_ids'])
            with np.load(folder/'packs'/f'r{index:02d}.npz', allow_pickle=False) as archive:
                assert set(archive.files) == {'ids','raw','embedding_batch5','target_embeddings'}
                assert archive['ids'].tolist() == block['case_ids']
                raw, embedding, target_vector = [archive[k].copy() for k in ['raw','embedding_batch5','target_embeddings']]
            assert raw.dtype == embedding.dtype == target_vector.dtype == np.float32
            assert raw.shape == (5,256,256,3) and embedding.shape == target_vector.shape == (5,512)
            assert np.isfinite(raw).all() and np.isfinite(embedding).all() and 0 <= raw.min() <= raw.max() <= 1
            assert np.array_equal(target_vector,ev[:,2])
            with np.load(control/label/'packs'/f'r{index:02d}.npz', allow_pickle=False) as archive:
                assert np.array_equal(raw,archive['raw']) and archive['ids'].tolist() == block['case_ids']
                er, ep = archive['embeddings_raw'].copy(), archive['embeddings_png'].copy()
            mask = pixels(parent/reference['observed'],'L') > 0
            target8 = pixels(parent/reference['target'])
            cameras = [pixels(parent/cases[cid]['input']) for cid in block['case_ids']]
            supports = [capacity.feature_support(mask,cases[cid]['landmarks5_canvas_xy']) for cid in block['case_ids']]
            target = np.stack([target8]*5).astype(np.float32)/np.float32(255)
            observed = np.stack([mask]*5).astype(np.float32)[:,None]
            support = np.stack(supports).astype(np.float32)[:,None]
            with torch.no_grad():
                pred, base, truth = nchw(raw), nchw(baseline), nchw(target)
                mt, ft = torch.from_numpy(observed), torch.from_numpy(support)
                terms, old_terms = losses.pixel_and_structure_losses(pred,truth,mt,ft), losses.pixel_and_structure_losses(base,truth,mt,ft)
                old_arc = 1-(ev[:,0].astype(np.float64)*ev[:,2].astype(np.float64)).sum(1)
                arc = 1-(embedding.astype(np.float64)*target_vector.astype(np.float64)).sum(1)
                current = {k:v.tolist() for k,v in terms.items()} | {'ArcFace_loss':arc.tolist()}
                old_values = {k:old_terms[k].tolist() for k in ['MSE','SSIM_loss']} | {'ArcFace_loss':old_arc.tolist()}
                _, photo = guard.brightness_guard(pred,base,truth,mt,old['normalizers'][0],10.,.2)
            mean_shift = (((raw-baseline).astype(np.float64)*mask[None,:,:,None]).sum((1,2))/mask.sum()).tolist()
            actual = finite_objective(current,old_values,old['normalizers'],old['active_anchor_normalizers'],
                photometric=photo,reconstruction_weights=old['reconstruction_weights'])
            row = saved['rows'][index]
            assert row['reference'] == block['reference'] and row['source'] == block['source'] and row['case_ids'] == block['case_ids']
            assert row['raw_control_exact'] and row['floor_PNG_control_exact'] and row['recognizer_batch'] == 5
            assert row['raw_sha256'] == array_sha(raw) and row['embedding_sha256'] == array_sha(embedding)
            for key, value in [('current',current),('baseline',old_values),('channel_mean_shift',mean_shift),('per_image_photometric',photo)]:
                max_arithmetic = max(max_arithmetic,close_tree(value,row[key]))
            finite_saved = finite_objective(row['current'],row['baseline'],old['normalizers'],old['active_anchor_normalizers'],
                photometric=row['per_image_photometric'],reconstruction_weights=old['reconstruction_weights'])
            close_tree(finite_saved,row['finite_arithmetic'],atol=0,rtol=0)
            assert abs(finite_saved['objective']-row['objective']) <= p['objective_replay_max_abs']
            assert row['finite_minus_torch_objective'] == finite_saved['objective']-row['objective']
            max_arithmetic = max(max_arithmetic,close_tree(actual,finite_saved))
            for key in ['normalized_weighted_RGB_SSIM_recognizer_structure','regression_barriers_RGB_SSIM_recognizer','clear_blur_RGB_anchors','photometric_guard']:
                close_tree(actual[key],row[key])
            close_tree(actual['regression_barriers_RGB_SSIM_recognizer'],row['regression_barriers_RGB_SSIM_recognizer'])
            close_tree(sum(row['regression_barriers_RGB_SSIM_recognizer']),row['barrier'])
            close_tree(sum(row['clear_blur_RGB_anchors']),row['anchors'])
            close_tree([np.mean(current['MSE']),np.mean(current['SSIM_loss']),np.mean(current['ArcFace_loss']),np.mean(current['landmark_structure'][1:])],row['raw_terms'])
            records[(label,block['reference'])] = row
            if label == 'initial' and index in p['derivative_reference_indices']:
                inputs[block['reference']] = (baseline.transpose(0,3,1,2).copy(),target.transpose(0,3,1,2).copy(),observed.copy())
                gradient_targets[block['reference']] = target_vector.copy()
            delivered = []
            for slot,cid in enumerate(block['case_ids']):
                camera = cameras[slot]
                png = pixels(folder/'png'/(cid+'.png'))
                assert np.array_equal(png,pixels(control/label/'png'/(cid+'.png')))
                assert np.array_equal(png,metrics.deliver(raw[slot],camera,mask))
                assert np.array_equal(raw[slot][~mask],camera[~mask].astype(np.float32)/np.float32(255))
                assert np.array_equal(png[~mask],camera[~mask])
                delivered.append(png); exact_outputs += 1
                mean,mean_png,_ = metrics.mean_only(raw[slot],baseline[slot],camera,mask)
                rm, pm = metrics.pixel_metrics(raw[slot],target8,mask), capacity.exported_pixel_metrics(png,target8,mask)
                for values,vectors,retained_slot in [(rm,er,0),(pm,ep,1)]:
                    a,t = (ev[slot,retained_slot],ev[slot,2]) if label == 'initial' else (vectors[slot],vectors[10+slot])
                    values['ArcFace_observed_fixed'] = float(a@t)
                rm.update(landmark_high_frequency_MSE=metrics.detail_float(raw[slot],target8,supports[slot]),constant_mean_shift_only_MSE=metrics.pixel_metrics(mean,target8,mask)['MSE'])
                pm.update(landmark_high_frequency_MSE=capacity.detail_metric(png,target8,supports[slot]),constant_mean_shift_only_MSE=capacity.exported_pixel_metrics(mean_png,target8,mask)['MSE'])
                for mode, value in [('raw',rm),('png',pm)]:
                    for key, scalar in value.items():
                        close_tree(scalar,inherited_rows[cid][mode][key],atol=1e-13,rtol=0)
                scientific_rows.append(dict(case=cid,source=cases[cid]['source'],profile=cases[cid]['profile'],raw=rm,png=pm))
            assert row['delivered_uint8_sha256'] == array_sha(np.stack(delivered))
        gate_rows[label] = {mode:grouped(scientific_rows,mode) for mode in ['raw','png']}
        close_tree(gate_rows[label],inherited_metrics['groups'],atol=1e-13,rtol=0)
    assert exact_outputs == 400 and len(records) == 80
    # Gate evidence remains the exact historical RAW/PNG outputs and inherited
    # recognizer measurements, never the newly batch-five objective embeddings.
    baseline_groups = read(control/'baseline/metrics.json')['groups']
    outcomes = {label:{mode:unchanged_gate(baseline_groups[mode],read(control/label/'metrics.json')['groups'][mode])
               for mode in ['raw','png']} for label in STATES}
    for label in STATES[1:]:
        inherited_outcome = read(control/(label+'_outcome.json'))
        assert inherited_outcome['model_qualified'] is False
        close_tree(outcomes[label],inherited_outcome['gates'],atol=1e-12,rtol=1e-12)
    gradients, comparisons = [], []
    for ordinal,index in enumerate(p['derivative_reference_indices']):
        clock(); block = p['reference_blocks'][index]; folder = out/'gradients'/f'r{index:02d}'
        measured = read(folder/'measurement.json')
        assert measured['complete'] and measured['reference'] == block['reference'] and measured['source'] == block['source']
        assert measured['case_ids'] == block['case_ids'] and measured['recognizer_batch'] == 5
        assert measured['selected_decoder_branch_only'] and measured['literal_float32_step_derivative_claim'] is False
        assert measured['optimizer_updates'] == measured['backwards'] == 0
        matrix = np.load(folder/'gradients.npy',allow_pickle=False)
        clipped = np.load(folder/'clipped_full_gradient.npy',allow_pickle=False)
        slopes, clipping = verify_gradient_matrix(matrix,clipped,direction,measured,p['gradient_layout'],p['gradient_norm_metadata_bound'],p['clip_factor_metadata_max_ULPs'])
        initial_record = records[('initial',block['reference'])]
        assert measured['embedding_batch5_file'] == 'embedding_batch5.npy'
        assert measured['embedding_batch5_file_sha256'] == sha(folder/'embedding_batch5.npy')
        embedding = np.load(folder/'embedding_batch5.npy',allow_pickle=False)
        assert embedding.dtype == np.float32 and embedding.shape == (5,512) and np.isfinite(embedding).all()
        assert measured['embedding_batch5_array_sha256'] == measured['embedding_sha256'] == array_sha(embedding)
        expected_current = dict(initial_record['current'])
        expected_current['ArcFace_loss'] = (1-(embedding.astype(np.float64)*gradient_targets[block['reference']].astype(np.float64)).sum(1)).tolist()
        close_tree(expected_current,measured['current'])
        for key in ['baseline','channel_mean_shift','per_image_photometric']:
            close_tree(initial_record[key],measured[key],atol=0,rtol=0)
        gradient_finite = finite_objective(measured['current'],measured['baseline'],old['normalizers'],old['active_anchor_normalizers'],
            photometric=measured['per_image_photometric'],reconstruction_weights=old['reconstruction_weights'])
        assert abs(gradient_finite['objective']-measured['objective']) <= p['objective_replay_max_abs']
        values = ([measured['current']['MSE'][i] for i in range(5)]+[measured['current']['SSIM_loss'][i] for i in range(5)]
            +[measured['current']['ArcFace_loss'][i] for i in range(5)]+measured['current']['landmark_structure'][1:]
            +[v for row in measured['channel_mean_shift'] for v in row]+[measured['objective']])
        close_tree(values,measured['scalar_values'],atol=0,rtol=0)
        current_slopes = dict(MSE=slopes[:5].tolist(),SSIM_loss=slopes[5:10].tolist(),ArcFace_loss=slopes[10:15].tolist(),landmark_structure=[0.]+slopes[15:19].tolist())
        channel_slopes = slopes[19:34].reshape(5,3)
        close_tree(current_slopes,measured['current_slopes'],atol=1e-12,rtol=1e-12)
        close_tree(channel_slopes,measured['channel_mean_slopes'],atol=1e-12,rtol=1e-12)
        assert measured['unqueried_clear_structure_slope_placeholder'] == 0.
        right = mean_control_right_mse_at_initial(*inputs[block['reference']],channel_slopes)
        close_tree(right,measured['mean_control_right_derivative'])
        photo_slopes = dict(actual_MSE=current_slopes['MSE'],mean_only_MSE=right['mean_only_MSE_slopes'])
        close_tree(photo_slopes,measured['photometric_slopes'])
        one_sided = one_sided_objective(measured['current'],measured['baseline'],current_slopes,
            old['normalizers'],old['active_anchor_normalizers'],photometric=measured['per_image_photometric'],
            photometric_slopes=photo_slopes,mean_control_derivative_kind='idealized_pixel_clip',reconstruction_weights=old['reconstruction_weights'])
        close_tree(one_sided,measured['one_sided'])
        if ordinal == 0:
            with np.load(control/'trajectory/update_001.npz',allow_pickle=False) as archive:
                retained = archive['clipped_gradient'].copy()
            a,z = clipped.astype(np.float64),retained.astype(np.float64)
            maximum,l2 = float(np.max(np.abs(a-z))),float(np.linalg.norm(a-z))
            cosine = float(a@z/(np.linalg.norm(a)*np.linalg.norm(z)))
            bound = p['clipped_gradient_replay_bound']
            assert bound == dict(max_abs=2e-7,cosine_min=.999999)
            expected = dict(max_abs=maximum,difference_L2=l2,cosine=cosine,fresh_clipped_L2=float(np.linalg.norm(a)),retained_clipped_L2=float(np.linalg.norm(z)),
                bound=bound,pass_=maximum<=bound['max_abs'] and cosine>=bound['cosine_min'],full_gradient_norm_before_clip=measured['full_gradient_norm_before_clip'],clip_factor=measured['clip_factor'],actual_fitting_reference=True,quality_gate_relaxed=False)
            assert expected['pass_']
            close_tree(expected,measured['clipped_gradient_replay'],atol=1e-12,rtol=1e-12)
            close_tree(expected,read(folder/'clipped_gradient_replay.json'),atol=1e-12,rtol=1e-12)
        else:
            assert measured['clipped_gradient_replay'] is None and not (folder/'clipped_gradient_replay.json').exists()
        comparisons += finite_comparisons(records,measured)
        gradients.append(dict(reference=block['reference'],source=block['source'],matrix_SHA256=sha(folder/'gradients.npy'),
            shape=list(matrix.shape),scalars=35,clipping=clipping,selected_autograd_objective_slope=float(slopes[-1]),affine_residual_one_sided=one_sided))
    assert len(gradients) == 2 and len(comparisons) == 8
    saved_states = []
    for label in ['before','after']:
        clock(); folder = out/('state_'+label); saved = torch.load(folder/'full_state.pth',map_location='cpu',weights_only=True)
        state_receipt = read(folder/'receipt.json')
        assert state_receipt['full_state_sha256'] == sha(folder/'full_state.pth')
        assert saved['protocol_sha256'] == sha(root/'protocol.json') and saved['optimizer'] is None
        assert saved['optimizer_updates'] == 0 and saved['exact_future_resume_guaranteed'] is saved['failed_recipe_resume_allowed'] is False
        verify_counts(saved['counts'],success=label=='after')
        if label == 'before':
            assert saved['counts'] == dict(optimizer_updates=0,gradient_queries=0,backwards=0,
                original_batch_calls=20,prior_feature_batch_calls=20,decoder_batch_calls=0,recognizer_batch_calls=0)
        assert saved['model'].keys() == source_state.keys() and state_hash(saved['model']) == source_receipt['model_state']
        assert all(torch.equal(value,source_state[key]) for key,value in saved['model'].items())
        assert state_receipt['model_state'] == source_receipt['model_state'] and state_receipt['buffer_state'] == source_receipt['buffer_state']
        assert state_receipt['optimizer_present'] is False
        assert saved['CPU_rng'].dtype == torch.uint8 and saved['CPU_rng'].ndim == 1
        assert saved['CUDA_rng'] and all(v.dtype==torch.uint8 and v.ndim==1 for v in saved['CUDA_rng'])
        assert state_receipt['CPU_rng_sha256'] == array_sha(saved['CPU_rng'].numpy())
        assert state_receipt['CUDA_rng_sha256'] == [array_sha(v.numpy()) for v in saved['CUDA_rng']]
        saved_states.append((saved,state_receipt))
    before,after = saved_states
    assert torch.equal(before[0]['CPU_rng'],after[0]['CPU_rng'])
    assert len(before[0]['CUDA_rng']) == len(after[0]['CUDA_rng']) and all(torch.equal(a,b) for a,b in zip(before[0]['CUDA_rng'],after[0]['CUDA_rng']))
    restoration = read(out/'restoration.json')
    for key in ['complete','initial_model_restored_exact','frozen_models_unchanged','CPU_CUDA_rng_restored_exact','RNG_unchanged_before_restore']:
        assert restoration[key] is True
    for key in ['model_state','buffer_state','CPU_rng_sha256','CUDA_rng_sha256']:
        assert restoration[key] == after[1][key]
    clock()
    files = {f.relative_to(root).as_posix():dict(bytes=f.stat().st_size,sha256=sha(f)) for f in sorted(out.rglob('*')) if f.is_file()}
    record = dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(root/'protocol.json'),
        checker_sha256=sha(__file__),exact_retained_RAW_and_floor_PNG_cases=exact_outputs,
        batch5_objective_rows_checked=len(records),batch5_saved_embedding_packs=80,historical_batch15_files_exact=4,
        full_parameter_gradient_matrices=gradients,finite_vs_linear_comparisons=comparisons,
        derivative_references_measured=p['derivative_reference_ids'],remaining_reference_derivatives_not_measured=18,
        all80_actual_batch5_objective_rows=[dict(state=s,reference=r,objective=v['objective'],current=v['current'],baseline=v['baseline'],photometric=v['per_image_photometric']) for (s,r),v in records.items()],
        maximum_fixed_arithmetic_error=max_arithmetic,fixed_arithmetic_tolerance=dict(rtol=1e-10,atol=1e-10),
        all17_RAW_and17_PNG_original_gates_retained=outcomes,gate_embeddings='Inherited audited batch15 RAW/PNG embeddings; batch5 objective embeddings are separate.',
        saved_model_and_CPU_CUDA_RNG_before_after_exact=True,source_A0_all_model_tensor_bytes_exact=True,
        returned_run_counts=result['counts'],saved_files=files,
        protocol_bound_dependencies={key:dict(files=len(p[key]),bindings=p[key]) for key in
            ['assets_sha256','repair_required_sha256','source_required_sha256','parent_required_sha256','replay_required_sha256']},
        scope='Exposed paired photographic TRAIN only; no native CCTV, final identities, completion or new model selection.',
        saved_embeddings_reused_no_independent_recognizer_forward=True,CUDA_gradients_not_independently_replayed=True,
        selected_neural_branch_certified=False,literal_float32_derivative_certified=False,PNG_derivative_certified=False,
        no_new_visual_review_claim=True,outputs_exact_to_already_reviewed_controls=True,independent_final_reviewer=False,
        no_local_model_construction_or_weight_assignment=True,model_qualified=False,new_incumbent_selection=False,
        local_neural_calls=0,local_gradient_queries=0,local_optimizer_updates=0,seconds=time.monotonic()-began)
    write(receipt,record)
    print(json.dumps(dict(complete=True,exact_controls=400,batch5_rows=80,gradient_queries_audited=70,new_local_gradient_queries=0,model_qualified=False)),flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root','repair-root','source-root','parent-root','replay-root','receipt']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True)
    args = parser.parse_args()
    p,rp,old,data = verify_transfer(args.root,args.repair_root,args.source_root,args.parent_root,args.replay_root,args.protocol_sha)
    audit_saved(args.root,args.repair_root,args.source_root,args.parent_root,args.replay_root,p,rp,old,data,args.receipt)


if __name__ == '__main__':
    main()
