"""Prospective saved-data audit of one fixed decoder feasibility experiment.

No solver, learned forward, autograd, optimizer or model assignment is called.
Exact endpoint certificates concern the saved selected-branch linear surrogate;
the independent RAW/PNG checks remain the finite scientific evidence.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import math
from pathlib import Path
import time

from cctv_dgp_decoder_feasibility_v1_contract import (
    STAGE, STATES, AMPLITUDES, CORRECTED_HELPER_SHA,
    read, write, sha, array_sha, verify_transfer,
)

EXPECTED_COUNTS = dict(optimizer_updates=0, gradient_queries=0, backwards=0,
    original_batch_calls=20, prior_feature_batch_calls=20,
    decoder_batch_calls=80, recognizer_batch_calls=240)
FIXED_ARITHMETIC_BOUND = dict(atol=1e-10, rtol=1e-10)
METRIC_REPLAY_BOUND = dict(atol=1e-13, rtol=0.)


def compare(actual, saved, *, atol=0., rtol=0., path='record'):
    """Strict containers/types, with bounds only on floating arithmetic leaves."""
    if type(actual) is dict:
        assert type(saved) is dict and actual.keys() == saved.keys(), path+' keys'
        return max([compare(v, saved[k], atol=atol, rtol=rtol, path=path+'.'+k)
                    for k, v in actual.items()] or [0.])
    if type(actual) in (list, tuple):
        assert type(saved) is list and len(actual) == len(saved), path+' sequence'
        return max([compare(a, b, atol=atol, rtol=rtol, path=path+'['+str(i)+']')
                    for i, (a, b) in enumerate(zip(actual, saved))] or [0.])
    if actual is None or type(actual) in (bool, str, int):
        assert type(saved) is type(actual) and actual == saved, path+' exact typed leaf'
        return 0.
    assert type(actual) is float and type(saved) is float, path+' float type'
    assert math.isfinite(actual) and math.isfinite(saved), path+' finite'
    error = abs(actual-saved)
    assert error <= atol+rtol*abs(actual), path+' floating arithmetic'
    return error


def verify_counts(counts, complete=False, no_direction=False):
    assert type(counts) is dict and counts.keys() == EXPECTED_COUNTS.keys()
    assert all(type(v) is int and 0 <= v <= EXPECTED_COUNTS[k] for k, v in counts.items())
    assert counts['optimizer_updates'] == counts['gradient_queries'] == counts['backwards'] == 0
    if complete:
        assert counts == ({k: 0 for k in EXPECTED_COUNTS} if no_direction else EXPECTED_COUNTS)


def bound_module(path, expected, name):
    assert sha(path) == expected, 'Bound read-only helper differs: '+str(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == Path(path).resolve()
    return module


def state_hash(state, layout=None, vector=None):
    """Hash saved tensors or saved-vector substitution, never assign a model."""
    import numpy as np
    slices = {} if layout is None else {r['name']: r for r in layout}
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


def grouped(rows, mode):
    import numpy as np
    result = {}
    for row in rows:
        source, profile = row['source'], row['profile']
        for key in {'all', 'clear' if profile == 'clear' else 'degraded', source+'/all',
                    source+('/clear' if profile == 'clear' else '/degraded'), source+'/'+profile}:
            result.setdefault(key, []).append(row[mode])
    names = ['MSE', 'SSIM', 'ArcFace_observed_fixed',
             'landmark_high_frequency_MSE', 'constant_mean_shift_only_MSE']
    assert len(result) == 17
    return {key: dict(cases=len(values), **{name: float(np.mean([v[name] for v in values]))
            for name in names}) for key, values in sorted(result.items())}


def scientific_gate(before, after, thresholds):
    """Literal retained 17-group gate, including its exact metadata schema."""
    assert set(before) == set(after) and len(before) == 17
    failures = []
    for key in sorted(before):
        a, b = after[key], before[key]
        assert a['cases'] == b['cases']
        for metric in ['MSE', 'SSIM', 'ArcFace_observed_fixed']:
            x, y = a[metric], b[metric]
            assert math.isfinite(x) and math.isfinite(y)
            if (x > y+1e-12) if metric == 'MSE' else (x < y-1e-6):
                failures.append(dict(group=key, metric=metric, baseline=y, candidate=x))
    old, new = before['degraded'], after['degraded']
    assert old['landmark_high_frequency_MSE'] > 0
    gain = 1-new['landmark_high_frequency_MSE']/old['landmark_high_frequency_MSE']
    sources = {k: 1-after[k]['landmark_high_frequency_MSE']/before[k]['landmark_high_frequency_MSE']
               for k in before if k.endswith('/degraded')}
    assert len(sources) == 2 and math.isfinite(gain) and all(math.isfinite(v) for v in sources.values())
    brightness = max(0., old['MSE']-new['constant_mean_shift_only_MSE'])/max(old['MSE']-new['MSE'], 1e-12)
    if brightness > .2:
        failures.append(dict(group='degraded', metric='brightness_gain_fraction', candidate=brightness, maximum=.2))
    return dict(**{'pass': not failures and gain >= .01 and all(v >= .01 for v in sources.values())},
        global_structure_gain=gain, source_structure_gains=sources, brightness_gain_fraction=brightness,
        preservation_failures=failures, scientific_thresholds=thresholds,
        numerical_sample_gate_only=True, model_qualified=False, automatic_extra_epochs=False)


def partial_evidence(root, out, p, receipt):
    """Opaque/interrupted artifacts are retained without inventing final counts."""
    def optional(path):
        if not path.is_file():
            return None
        try:
            return read(path)
        except (ValueError, UnicodeError):
            return dict(opaque_partial_JSON=True, sha256=sha(path), bytes=path.stat().st_size)
    terminal, failure = optional(root/'run_exit.json'), optional(out/'failure.json')
    assert failure is not None or (type(terminal) is dict and type(terminal.get('exit_code')) is int
                                  and terminal['exit_code'] != 0), 'Need authoritative stopped-worker evidence'
    if terminal and not terminal.get('opaque_partial_JSON'):
        assert terminal['flag'] == '--run' and type(terminal['exit_code']) is int and terminal['exit_code'] != 0
    last, malformed = None, 0
    journal = out/'progress.jsonl'
    if journal.is_file():
        for line in journal.read_text(encoding='utf-8', errors='replace').splitlines():
            try:
                import json
                row = json.loads(line)
            except ValueError:
                malformed += 1
                continue
            if type(row) is dict and type(row.get('counts')) is dict:
                verify_counts(row['counts'])
                last = row
    write(receipt, dict(complete=True, partial_failure_evidence_only=True,
        protocol_sha256=sha(root/'protocol.json'), failure=failure, terminal_worker=terminal,
        last_known_progress=last, malformed_progress_lines_retained=malformed,
        exact_final_counts_known=False, final_counts=None, model_qualified=False,
        full_saved_scientific_audit_complete=False, solver_infeasibility_certified=False,
        saved_files={f.relative_to(root).as_posix(): dict(bytes=f.stat().st_size, sha256=sha(f))
                     for f in sorted(out.rglob('*')) if f.is_file()},
        local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0))


def verify_saved_proposal(root, activation, source, parent, p, data, certificate, clock):
    """Replay only proposed bytes/certificates already retained by the worker."""
    import numpy as np
    from PIL import Image
    out = root/STAGE
    proposed = 'placements' in certificate
    proposal_path = out/'solver_proposal_receipt.json'
    if not proposed:
        assert not proposal_path.exists() and not (out/'selected_direction.npy').exists()
        assert not any((out/('proposed_'+label+'_vector.npy')).exists() for label in STATES[1:])
        return None
    exact = bound_module(root/'cctv_dgp_decoder_feasibility_v1_math.py', p['assets_sha256']['cctv_dgp_decoder_feasibility_v1_math.py'], 'feasibility_saved_exact_certificate')
    packed = bound_module(parent/'cctv_dgp_head4_lossless_v1.py', p['parent_required_sha256']['cctv_dgp_head4_lossless_v1.py'], 'feasibility_certificate_packs')
    initial = np.load(source/'outputs/spatial_readout_v1/A_initial_vector.npy',allow_pickle=False)
    direction = np.load(out/'selected_direction.npy',allow_pickle=False)
    assert initial.dtype == np.float32 and initial.shape == (609219,) and np.isfinite(initial).all()
    assert direction.dtype == np.float64 and direction.shape == initial.shape and np.isfinite(direction).all()
    assert certificate['ideal_direction_sha256'] == array_sha(direction)
    proposal = read(proposal_path)
    assert proposal['complete'] is proposal['certificate_pending'] is True
    assert proposal['neural_assignments'] == 0 and proposal['model_qualified'] is False
    assert proposal['direction_sha256'] == array_sha(direction)
    compare(read(out/'solver_receipt.json'),proposal['solver'])
    refs = {r['id']:r for r in data['references']}
    gradients, structure, photo = [], [], []
    for index in p['derivative_reference_indices']:
        folder = activation/'outputs/decoder_activation_v1/gradients'/f'r{index:02d}'
        gradients.append(np.load(folder/'gradients.npy',allow_pickle=False))
        structure.append(read(folder/'measurement.json')['scalar_values'][15:19])
        block = p['reference_blocks'][index]; ref = refs[block['reference']]
        baseline,_ = packed.unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz",block['case_ids'])
        with Image.open(parent/ref['target']) as image:
            assert image.size == (256,256)
            target8 = np.asarray(image.convert('RGB')).copy()
        with Image.open(parent/ref['observed']) as image:
            assert image.size == (256,256)
            mask = np.asarray(image.convert('L')).copy() > 0
        target = np.stack([target8]*5).astype(np.float32)/np.float32(255)
        photo.append(exact.exact_photo_coefficients(baseline.transpose(0,3,1,2).copy(),
            target.transpose(0,3,1,2).copy(),np.stack([mask]*5).astype(np.float32)[:,None]))
    expected_coefficients = [[[[exact.rational_record(v) for v in pair] for pair in profile] for profile in ref] for ref in photo]
    compare(expected_coefficients,certificate['exact_photo_coefficients'])
    gradients = np.asarray(gradients)
    vectors, passed = {}, []
    assert set(certificate['placements']) == set(STATES[1:])
    for label,amplitude in zip(STATES[1:],AMPLITUDES[1:]):
        clock()
        vector = np.load(out/('proposed_'+label+'_vector.npy'),allow_pickle=False)
        assert vector.dtype == np.float32 and vector.shape == initial.shape and np.isfinite(vector).all()
        assert np.array_equal(vector,(initial.astype(np.float64)+amplitude*direction).astype(np.float32))
        assert proposal['placed_vector_sha256'][label] == array_sha(vector)
        checked = exact.certify_placement(initial,vector,gradients,structure,photo,p['common_radius_L2'],
            p['solver']['full_minimum_predicted_structure_gain'] if label == 'full' else 0.,clock)
        compare(checked,certificate['placements'][label])
        vectors[label] = vector; passed.append(checked['admitted'])
    assert certificate['admitted'] is all(passed)
    assert certificate['actual_finite_RAW_PNG_checks_still_required'] is True
    assert certificate['full_decoder_infeasibility_certified'] is False
    return dict(initial=initial,direction=direction,vectors=vectors)


def audit_saved(root, activation, repair, source, parent, replay, p, ap, rp, old, data, receipt):
    import numpy as np
    root, activation, repair, source, parent, replay, receipt = [Path(x).resolve()
        for x in (root, activation, repair, source, parent, replay, receipt)]
    out = root/STAGE
    assert not receipt.exists(), 'Retain every earlier saved audit'
    began = time.monotonic()
    def clock():
        assert time.monotonic()-began <= p['budgets']['saved_audit_seconds'], 'Finite saved-audit stop'
    path = out/'run_results.json'
    try:
        result = read(path) if path.is_file() else None
    except (ValueError, UnicodeError):
        result = None
    if type(result) is not dict or result.get('complete') is not True:
        partial_evidence(root, out, p, receipt)
        return
    assert not (out/'failure.json').exists(), 'A complete run cannot conceal a failure'
    no_direction = result['stop_reason'] == 'no_certified_direction'
    assert no_direction or result['stop_reason'] == 'three_fixed_placements_complete'
    verify_counts(result['counts'], complete=True, no_direction=no_direction)
    assert type(result['seconds']) is float and 0 < result['seconds'] <= p['budgets']['worker_seconds']
    for key in ['model_qualified', 'new_incumbent_selection', 'automatic_follow_on']:
        assert result[key] is False
    assert type(result['full_corpus_epochs']) is int and result['full_corpus_epochs'] == 0
    assert result['state_labels'] == ([] if no_direction else STATES)
    assert result['certified_direction'] is (not no_direction)
    assert type(result['new_parameter_placements']) is int and result['new_parameter_placements'] == (0 if no_direction else 3)
    storage = read(out/'storage_start.json')
    assert storage['free_bytes'] >= p['budgets']['minimum_free_bytes']
    assert storage['MemAvailable_bytes'] >= p['budgets']['available_RAM_before_import_bytes']
    assert storage['output_cap_bytes'] == p['budgets']['output_byte_cap']
    assert storage['emergency_metadata_bytes'] == p['budgets']['emergency_metadata_bytes']
    assert storage['projected_output_bytes'] <= storage['output_cap_bytes']
    assert storage['free_bytes'] >= 2*storage['projected_output_bytes']+p['budgets']['disk_reserve_bytes']
    assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) <= p['budgets']['output_byte_cap']
    solver = read(out/'solver_receipt.json')
    assert solver['complete'] is True and solver['solver_calls'] == 1 and solver['method'] == 'highs-ds'
    assert type(solver['status']) is int and type(solver['success']) is bool
    assert type(solver['iterations']) is int and 0 <= solver['iterations'] <= p['solver']['maxiter']
    assert type(solver['seconds']) is float and 0 < solver['seconds'] <= p['budgets']['solver_seconds']
    assert solver['solver_success_is_certificate'] is solver['full_decoder_infeasibility_certified'] is False
    expected_options = {k: p['solver'][k] for k in ['time_limit','maxiter','primal_feasibility_tolerance','dual_feasibility_tolerance','presolve']}
    compare(expected_options, solver['options'])
    certificate = read(out/'direction_certificate.json')
    assert certificate['complete'] is True and certificate['admitted'] is (not no_direction)
    assert certificate['model_qualified'] is False
    solver_terminal = read(out/'solver_terminal_receipt.json')
    assert solver_terminal['complete'] is solver_terminal['certificate_pending'] is True
    assert solver_terminal['neural_assignments'] == 0 and solver_terminal['model_qualified'] is False
    compare(solver,solver_terminal['solver'])
    verified_proposal = verify_saved_proposal(root,activation,source,parent,p,data,certificate,clock)
    if no_direction:
        marker = read(out/'no_certified_direction.json')
        assert marker == dict(complete=True, no_neural_placements=True, model_qualified=False, no_automatic_follow_on=True)
        assert read(out/'no_model_state_created.json') == dict(complete=True, neural_models_constructed=0,
            parameter_assignments=0, model_state_restore_not_needed=True, source_checkpoints_unchanged=True)
        assert not any((out/name).exists() for name in STATES+['state_before','state_after','environment.json','cache_receipt.json'])
        write(receipt, dict(complete=True, protocol_sha256=sha(root/'protocol.json'), checker_sha256=sha(__file__),
            terminal_scope='No candidate admitted to neural comparison', returned_run_counts=result['counts'],
            no_neural_placement_confirmed=True, saved_scientific_outputs=0, solver_receipt=solver,
            rejected_certificate=certificate, strict_homogeneous_subset_only=True,
            rejected_candidate_bytes_and_exact_certificate_checked=verified_proposal is not None,
            original_slack_or_whole_decoder_infeasibility_certified=False,
            full_saved_scientific_audit_complete=False, model_qualified=False,
            local_neural_calls=0, local_gradient_queries=0, local_optimizer_updates=0,
            seconds=time.monotonic()-began))
        return
    _audit_admitted(root, activation, repair, source, parent, replay, p, ap, rp, old, data,
                    out, result, certificate, solver, receipt, clock, began, verified_proposal)


def _audit_admitted(root, activation, repair, source, parent, replay, p, ap, rp, old, data,
                    out, result, certificate, solver, receipt, clock, began, verified_proposal):
    import numpy as np
    import torch  # Saved weights-only deserialization and fixed loss filters only.
    from PIL import Image
    torch.set_num_threads(4)
    losses = bound_module(parent/'cctv_dgp_group_conflicts_v1_losses.py', p['parent_required_sha256']['cctv_dgp_group_conflicts_v1_losses.py'], 'feasibility_saved_losses')
    guard = bound_module(repair/'cctv_dgp_bank_hinge_boundary_v1_loss.py', CORRECTED_HELPER_SHA, 'feasibility_saved_guard')
    metrics = bound_module(parent/'frozen_raw_metrics.py', p['parent_required_sha256']['frozen_raw_metrics.py'], 'feasibility_saved_metrics')
    capacity = bound_module(parent/'frozen_capacity_contract.py', p['parent_required_sha256']['frozen_capacity_contract.py'], 'feasibility_saved_capacity')
    packed = bound_module(parent/'cctv_dgp_head4_lossless_v1.py', p['parent_required_sha256']['cctv_dgp_head4_lossless_v1.py'], 'feasibility_saved_packs')
    activation_math = bound_module(activation/'cctv_dgp_decoder_activation_v1_math.py', p['activation_required_sha256']['cctv_dgp_decoder_activation_v1_math.py'], 'feasibility_saved_objective')
    def pixels(path, mode='RGB'):
        with Image.open(path) as image:
            assert image.size == (256,256)
            return np.asarray(image.convert(mode)).copy()
    def nchw(value):
        return torch.from_numpy(value.transpose(0,3,1,2).copy())
    refs, cases = {r['id']:r for r in data['references']}, {c['id']:c for c in data['cases']}
    layout = read(source/'outputs/spatial_readout_v1/A_parameter_layout.json')
    assert [r['name'] for r in layout] == p['trainable_parameter_names']
    assert verified_proposal is not None
    initial = verified_proposal['initial']
    assert np.array_equal(initial, np.load(out/'initial_vector.npy', allow_pickle=False))
    vectors = {'initial': initial}
    for label in STATES[1:]:
        vector = np.load(out/(label+'_vector.npy'), allow_pickle=False)
        assert np.array_equal(vector,verified_proposal['vectors'][label])
        vectors[label] = vector
    admission = read(out/'direction_admission.json')
    assert admission == dict(complete=True, certified=True, one_fixed_comparison=True,
        fixed_placements=STATES[1:], quality_failure_does_not_authorize_continuation=True, model_qualified=False)
    environment = read(out/'environment.json')
    assert environment['GPU'] == 'NVIDIA L4' and environment['host'].split('.')[0] == 'forensic-dgp-thesis'
    assert environment['frozen_states'] == dict(original=old['original_state'], recognizer=old['recognizer_state'], provider=old['feature_provider_initial_state'])
    assert environment['corrected_helper_sha256'] == CORRECTED_HELPER_SHA
    assert environment['optimizer_present'] is environment['historical_training_called'] is False
    assert environment['new_gradient_queries'] == 0 and environment['peak_includes_dependency_loading'] is True
    cache = read(out/'cache_receipt.json')
    retained_cache = read(source/'outputs/spatial_readout_v1/cache_receipt.json')['rows']
    assert cache['complete'] is cache['exact_original_and_features'] is True and len(cache['rows']) == 20
    assert type(cache['seconds']) is float and 0 < cache['seconds'] <= p['budgets']['cache_seconds']
    for row, retained in zip(cache['rows'], retained_cache):
        for key in ['reference','source','maps','features']:
            compare(retained[key], row[key])
    source_receipt = read(source/'outputs/spatial_readout_v1/state_A0/receipt.json')
    source_state = torch.load(source/'outputs/spatial_readout_v1/state_A0/full_state.pth', map_location='cpu', weights_only=True)['model']
    assert state_hash(source_state) == source_receipt['model_state']
    groups, records, max_arithmetic, outputs = {}, [], 0., 0
    for label in STATES:
        clock(); folder = out/label
        installed = read(out/(label+'_state.json'))
        expected_hash = state_hash(source_state, layout, vectors[label])
        assert installed['model_state'] == expected_hash and installed['buffer_state'] == source_receipt['buffer_state']
        assert installed['vector_sha256'] == array_sha(vectors[label]) and installed['installed_vector_exact'] is True
        assert installed['gradients_enabled'] is installed['model_qualified'] is False
        assert installed['optimizer_updates'] == installed['new_gradient_queries'] == 0
        saved = read(folder/'objective_components.json')
        assert saved['schema'] == 'fixed_batch5_objective_v1' and saved['complete'] is True and len(saved['rows']) == 20
        assert saved['recognizer_batch'] == 5 and saved['scientific_RAW_PNG_embedding_batch'] == 15 and saved['model_qualified'] is False
        measurement = read(folder/'metrics.json')
        assert len(measurement['rows']) == 100 and [r['case'] for r in measurement['rows']] == p['case_ids']
        assert type(measurement['seconds']) is float and 0 < measurement['seconds'] <= p['budgets']['snapshot_seconds']
        scientific_rows = []
        for index, block in enumerate(p['reference_blocks']):
            clock(); ref = refs[block['reference']]
            baseline, ev = packed.unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz", block['case_ids'])
            with np.load(folder/'packs'/f'r{index:02d}.npz', allow_pickle=False) as archive:
                assert set(archive.files) == {'ids','raw','embedding_batch5','embeddings_raw','embeddings_png','target_embeddings'}
                assert archive['ids'].tolist() == block['case_ids']
                raw, embedding, er, ep, targets = [archive[k].copy() for k in ['raw','embedding_batch5','embeddings_raw','embeddings_png','target_embeddings']]
            for value, shape in [(raw,(5,256,256,3)),(embedding,(5,512)),(er,(15,512)),(ep,(15,512)),(targets,(5,512))]:
                assert value.dtype == np.float32 and value.shape == shape and np.isfinite(value).all()
            assert raw.min() >= 0 and raw.max() <= 1 and np.array_equal(targets, ev[:,2])
            for vectors15, slot in [(er,0),(ep,1)]:
                np.testing.assert_allclose(vectors15[5:10], ev[:,slot], rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(vectors15[10:15], targets, rtol=1e-6, atol=1e-6)
            if label == 'initial':
                with np.load(activation/'outputs/decoder_activation_v1/initial/packs'/f'r{index:02d}.npz', allow_pickle=False) as control:
                    assert np.array_equal(raw,control['raw']) and np.array_equal(embedding,control['embedding_batch5'])
            mask, target8 = pixels(parent/ref['observed'],'L') > 0, pixels(parent/ref['target'])
            cameras = [pixels(parent/cases[cid]['input']) for cid in block['case_ids']]
            support = [capacity.feature_support(mask,cases[cid]['landmarks5_canvas_xy']) for cid in block['case_ids']]
            target = np.stack([target8]*5).astype(np.float32)/np.float32(255)
            observed = np.stack([mask]*5).astype(np.float32)[:,None]
            with torch.no_grad():
                pred, base, truth = nchw(raw), nchw(baseline), nchw(target)
                mt, ft = torch.from_numpy(observed), torch.from_numpy(np.stack(support).astype(np.float32)[:,None])
                terms = losses.pixel_and_structure_losses(pred,truth,mt,ft)
                old_terms = losses.pixel_and_structure_losses(base,truth,mt,ft)
                current = {k:v.tolist() for k,v in terms.items()} | {'ArcFace_loss':(1-(embedding.astype(np.float64)*targets.astype(np.float64)).sum(1)).tolist()}
                old_values = {k:old_terms[k].tolist() for k in ['MSE','SSIM_loss']} | {'ArcFace_loss':(1-(ev[:,0].astype(np.float64)*ev[:,2].astype(np.float64)).sum(1)).tolist()}
                _, photo_detail = guard.brightness_guard(pred,base,truth,mt,old['normalizers'][0],10.,.2)
            row = saved['rows'][index]
            assert row['reference'] == block['reference'] and row['source'] == block['source'] and row['case_ids'] == block['case_ids']
            assert row['recognizer_batch'] == 5 and row['model_qualified'] is False
            assert row['raw_sha256'] == array_sha(raw) and row['embedding_sha256'] == array_sha(embedding)
            mean_shift = (((raw-baseline).astype(np.float64)*mask[None,:,:,None]).sum((1,2))/mask.sum()).tolist()
            for key, value in [('current',current),('baseline',old_values),('channel_mean_shift',mean_shift),('per_image_photometric',photo_detail)]:
                max_arithmetic = max(max_arithmetic,compare(value,row[key],**FIXED_ARITHMETIC_BOUND))
            finite = activation_math.finite_objective(row['current'],row['baseline'],old['normalizers'],old['active_anchor_normalizers'],
                photometric=row['per_image_photometric'], reconstruction_weights=old['reconstruction_weights'])
            assert abs(finite['objective']-row['objective']) <= p['objective_replay_max_abs']
            for key in ['normalized_weighted_RGB_SSIM_recognizer_structure','regression_barriers_RGB_SSIM_recognizer','clear_blur_RGB_anchors','photometric_guard']:
                max_arithmetic = max(max_arithmetic,compare(finite[key],row[key],**FIXED_ARITHMETIC_BOUND))
            compare(sum(row['regression_barriers_RGB_SSIM_recognizer']),row['barrier'],**FIXED_ARITHMETIC_BOUND)
            compare(sum(row['clear_blur_RGB_anchors']),row['anchors'],**FIXED_ARITHMETIC_BOUND)
            compare([float(np.mean(current['MSE'])),float(np.mean(current['SSIM_loss'])),float(np.mean(current['ArcFace_loss'])),float(np.mean(current['landmark_structure'][1:]))],row['raw_terms'],**FIXED_ARITHMETIC_BOUND)
            delivered = []
            for slot,cid in enumerate(block['case_ids']):
                png = pixels(folder/'png'/(cid+'.png'))
                assert np.array_equal(png, metrics.deliver(raw[slot],cameras[slot],mask))
                assert np.array_equal(raw[slot][~mask],cameras[slot][~mask].astype(np.float32)/np.float32(255))
                assert np.array_equal(png[~mask],cameras[slot][~mask])
                if label == 'initial':
                    assert np.array_equal(png,pixels(activation/'outputs/decoder_activation_v1/initial/png'/(cid+'.png')))
                mean, mean_png, _ = metrics.mean_only(raw[slot],baseline[slot],cameras[slot],mask)
                rm, pm = metrics.pixel_metrics(raw[slot],target8,mask), capacity.exported_pixel_metrics(png,target8,mask)
                for values, vectors15, retained_slot in [(rm,er,0),(pm,ep,1)]:
                    a,t = (ev[slot,retained_slot],ev[slot,2]) if label == 'initial' else (vectors15[slot],vectors15[10+slot])
                    values['ArcFace_observed_fixed'] = float(a@t)
                rm.update(landmark_high_frequency_MSE=metrics.detail_float(raw[slot],target8,support[slot]),constant_mean_shift_only_MSE=metrics.pixel_metrics(mean,target8,mask)['MSE'])
                pm.update(landmark_high_frequency_MSE=capacity.detail_metric(png,target8,support[slot]),constant_mean_shift_only_MSE=capacity.exported_pixel_metrics(mean_png,target8,mask)['MSE'])
                actual = dict(case=cid,source=cases[cid]['source'],profile=cases[cid]['profile'],raw=rm,png=pm)
                compare(actual,measurement['rows'][index*5+slot],**METRIC_REPLAY_BOUND)
                scientific_rows.append(actual); delivered.append(png); outputs += 1
            assert row['delivered_uint8_sha256'] == array_sha(np.stack(delivered))
            records.append(dict(state=label,reference=block['reference'],objective=row['objective'],current=row['current'],photometric=row['per_image_photometric']))
        groups[label] = {mode:grouped(scientific_rows,mode) for mode in ['raw','png']}
        compare(groups[label],measurement['groups'],**METRIC_REPLAY_BOUND)
    assert outputs == 400 and len(records) == 80
    outcomes = {}
    for label in STATES:
        saved = read(out/(label+'_outcome.json'))
        assert saved['label'] == label and saved['diagnostic_fixed_placement'] is saved['quality_does_not_authorize_follow_on'] is True and saved['model_qualified'] is False
        outcomes[label] = {mode: scientific_gate(groups['initial'][mode],groups[label][mode],old['scientific_thresholds']) for mode in ['raw','png']}
        compare(outcomes[label],saved['gates'],atol=1e-12,rtol=1e-12)
    saved_states = []
    for label in ['before','quarter','half','full','after']:
        clock(); folder = out/('state_'+label)
        saved = torch.load(folder/'full_state.pth',map_location='cpu',weights_only=True)
        proof = read(folder/'receipt.json')
        vector = initial if label in ['before','after'] else vectors[label]
        assert saved['model'].keys() == source_state.keys() and state_hash(saved['model']) == state_hash(source_state,layout,vector)
        for key,value in source_state.items():
            if key not in p['trainable_parameter_names']:
                assert torch.equal(saved['model'][key],value)
        assert saved['optimizer'] is None and saved['optimizer_updates'] == 0 and saved['protocol_sha256'] == sha(root/'protocol.json')
        assert saved['exact_future_resume_guaranteed'] is saved['failed_recipe_resume_allowed'] is saved['model_qualified'] is False
        assert saved['fixed_direction_diagnostic_only'] is True
        verify_counts(saved['counts'])
        n = {'before':0,'quarter':40,'half':60,'full':80,'after':80}[label]
        expected_counts = dict(EXPECTED_COUNTS,decoder_batch_calls=n,recognizer_batch_calls=3*n)
        assert saved['counts'] == expected_counts
        assert proof['full_state_sha256'] == sha(folder/'full_state.pth')
        assert proof['model_state'] == state_hash(saved['model']) and proof['buffer_state'] == source_receipt['buffer_state']
        assert proof['selected_vector_sha256'] == array_sha(vector) and proof['optimizer_present'] is proof['model_qualified'] is False
        assert saved['CPU_rng'].dtype == torch.uint8 and saved['CPU_rng'].ndim == 1
        assert saved['CUDA_rng'] and all(v.dtype == torch.uint8 and v.ndim == 1 for v in saved['CUDA_rng'])
        assert proof['CPU_rng_sha256'] == array_sha(saved['CPU_rng'].numpy())
        assert proof['CUDA_rng_sha256'] == [array_sha(v.numpy()) for v in saved['CUDA_rng']]
        if saved_states:
            assert torch.equal(saved_states[0][0]['CPU_rng'],saved['CPU_rng'])
            assert len(saved_states[0][0]['CUDA_rng']) == len(saved['CUDA_rng']) and all(torch.equal(a,b) for a,b in zip(saved_states[0][0]['CUDA_rng'],saved['CUDA_rng']))
        saved_states.append((saved,proof))
    restoration = read(out/'restoration.json')
    for key in ['complete','initial_model_restored_exact','frozen_models_unchanged','CPU_CUDA_rng_restored_exact','RNG_unchanged_before_restore']:
        assert restoration[key] is True
    for key in ['model_state','buffer_state','CPU_rng_sha256','CUDA_rng_sha256']:
        assert restoration[key] == saved_states[-1][1][key]
    assert result['original_and_frozen_models_unchanged'] is result['initial_model_and_RNG_restored_exact'] is True
    clock()
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__),protocol_sha256=sha(root/'protocol.json'),
        full_saved_scientific_audit_complete=True, returned_run_counts=result['counts'],
        exact_saved_RAW_and_floor_PNG_cases=400, batch5_objective_rows=80,
        independently_recomputed_all17_RAW_and17_PNG_groups=outcomes,
        exact_saved_gradient_surrogate_certificates_rechecked=True,
        missing18_reference_derivatives_replaced_only_by_finite_validation=True,
        maximum_fixed_arithmetic_replay_error=max_arithmetic,
        fixed_arithmetic_bound=FIXED_ARITHMETIC_BOUND,metric_replay_bound=METRIC_REPLAY_BOUND,
        all80_batch5_objectives=records, saved_full_state_count=5,
        source_A0_fixed_tensor_bytes_unchanged=True, diagnostic_CPU_CUDA_RNG_exact=True,
        historical_source_RNG_equality_not_required=True,
        selected_neural_branch_certified=False,literal_float32_derivative_certified=False,
        saved_embeddings_not_independently_neurally_replayed=True,
        no_new_visual_review_claim=True,independent_final_reviewer=False,
        model_qualified=False,new_incumbent_selection=False,automatic_follow_on=False,
        no_local_model_construction_or_parameter_assignment=True,
        local_neural_calls=0,local_gradient_queries=0,local_optimizer_updates=0,
        seconds=time.monotonic()-began))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root','activation-root','repair-root','source-root','parent-root','replay-root','receipt']:
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True)
    a = parser.parse_args()
    p,ap,rp,old,data = verify_transfer(a.root,a.activation_root,a.repair_root,a.source_root,a.parent_root,a.replay_root,a.protocol_sha)
    audit_saved(a.root,a.activation_root,a.repair_root,a.source_root,a.parent_root,a.replay_root,p,ap,rp,old,data,a.receipt)


if __name__ == '__main__':
    main()
