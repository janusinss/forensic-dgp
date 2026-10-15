"""Prospective CPU saved-evidence audit; no model construction or autograd."""
import argparse
import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time


def verify_corrected_guard_binding(source):
    """Reject a regression to the legacy derivative in the actual objective."""
    tree = ast.parse(source)
    imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
               and node.module == 'cctv_dgp_bank_hinge_boundary_v1_loss']
    aliases = [item.asname or item.name for node in imports for item in node.names if item.name == 'brightness_guard']
    objectives = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == 'objective']
    assert len(objectives) == 1
    calls = [node for node in ast.walk(objectives[0]) if isinstance(node, ast.Call)]
    if aliases:
        assert len(aliases) == 1
        assert sum(isinstance(node.func, ast.Name) and node.func.id == aliases[0] for node in calls) == 1
    else:
        bindings = [node for node in ast.walk(tree) if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == 'file_module'
                    and len(node.value.args) == 3
                    and isinstance(node.value.args[0], ast.BinOp) and isinstance(node.value.args[0].right, ast.Constant)
                    and node.value.args[0].right.value == 'cctv_dgp_bank_hinge_boundary_v1_loss.py'
                    and isinstance(node.value.args[2], ast.Name) and node.value.args[2].id == 'CORRECTED_HELPER_SHA']
        assert len(bindings) == 1 and isinstance(bindings[0].targets[0], ast.Name)
        alias = bindings[0].targets[0].id
        guards = [node.func for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == 'brightness_guard']
        assert len(guards) == 1 and isinstance(guards[0].value, ast.Name) and guards[0].value.id == alias
    legacy_aliases = [item.asname or item.name for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        and node.module == 'cctv_dgp_bank_photometric_guard_v1_loss' for item in node.names if item.name == 'brightness_guard']
    assert not any(isinstance(node.func, ast.Name) and node.func.id in legacy_aliases for node in calls)


def continuation_allowed(gates):
    """Continuation is distinct from the unchanged one-percent quality gate."""
    if set(gates) != {'raw', 'png'}:
        return False
    for gate in gates.values():
        values = [gate['global_structure_gain'], *gate['source_structure_gains'].values()]
        if len(gate['source_structure_gains']) != 2 or not all(math.isfinite(v) and v > 0 for v in values):
            return False
        if gate['preservation_failures']:
            return False
        if not math.isfinite(gate['brightness_gain_fraction']) or gate['brightness_gain_fraction'] > .2:
            return False
    return True


def verify_guard_boundary(row, legacy_gradient, corrected_gradient):
    import numpy as np
    a, b = legacy_gradient, corrected_gradient
    assert a.dtype == b.dtype == np.float32 and a.shape == b.shape
    assert np.isfinite(a).all() and np.isfinite(b).all()
    assert row['old_penalty'] == row['new_penalty'] == 0.
    assert row['old_metadata'] == row['new_metadata']
    assert np.count_nonzero(b) == 0, 'Inactive corrected guard must have exactly zero derivative'


def verify_probe_vector(initial, first, probe, fraction, radius):
    import numpy as np
    assert fraction in [1/16, 1/4]
    assert all(v.dtype == np.float32 and v.ndim == 1 and v.shape == initial.shape and np.isfinite(v).all()
               for v in [initial, first, probe])
    expected = (initial.astype(np.float64)+fraction*(first.astype(np.float64)-initial.astype(np.float64))).astype(np.float32)
    assert np.array_equal(probe, expected), 'Probe must use the corrected full first-step direction'
    assert float(np.linalg.norm(probe.astype(np.float64)-initial.astype(np.float64))) <= radius


def verify_projection_journal(initial, proposal, actual, row, radius):
    import numpy as np
    assert all(v.dtype == np.float32 and v.ndim == 1 and v.shape == initial.shape and np.isfinite(v).all()
               for v in [initial, proposal, actual])
    delta = proposal.astype(np.float64)-initial.astype(np.float64)
    distance = float(np.linalg.norm(delta))
    reserve = float(2*np.linalg.norm(np.spacing(initial).astype(np.float64)))
    scale = 1. if distance <= radius else max(0., radius-reserve)/distance
    expected = proposal if scale == 1. else (initial.astype(np.float64)+scale*delta).astype(np.float32)
    assert np.array_equal(actual, expected)
    actual_distance = float(np.linalg.norm(actual.astype(np.float64)-initial.astype(np.float64)))
    assert actual_distance <= radius
    assert row['radius'] == radius and abs(row['scale']-scale) < 1e-12
    assert abs(row['unconstrained_distance']-distance) < 1e-10
    assert abs(row['actual_distance']-actual_distance) < 1e-10
    assert row['float32_roundoff_reserve'] == reserve
    assert row['bound_reference'] == 'original selected vector, never previous step'
    assert row['scientific_bound_relaxed'] is False


def verify_stop_sequence(snapshots, updates, reason, maximum=50):
    schedule = [1, 10, 20, 30, 40, 50]
    assert maximum == 50 and 1 <= updates <= maximum and snapshots
    expected = [u for u in schedule if u <= updates]
    assert [r['update'] for r in snapshots] == expected
    failed = [r['update'] for r in snapshots if not continuation_allowed(r['gates'])]
    if failed:
        assert failed == [updates] and reason == 'quality_stop'
    else:
        assert updates == maximum and reason == 'maximum_updates'


def _check_gate(stored, computed, before_stored, after_stored, before, after):
    assert stored['pass'] == computed['pass']
    assert [(f['group'], f['metric']) for f in stored['preservation_failures']] == [
        (f['group'], f['metric']) for f in computed['preservation_failures']]
    assert abs(stored['global_structure_gain']-computed['global_structure_gain']) < 1e-12
    assert stored['source_structure_gains'].keys() == computed['source_structure_gains'].keys()
    assert all(abs(stored['source_structure_gains'][s]-computed['source_structure_gains'][s]) < 1e-12
               for s in computed['source_structure_gains'])
    bs, ass = before_stored['degraded'], after_stored['degraded']
    b, a = before['degraded'], after['degraded']
    nr = max(0., bs['MSE']-ass['constant_mean_shift_only_MSE']); dr = max(bs['MSE']-ass['MSE'], 1e-12)
    nd = max(0., b['MSE']-a['constant_mean_shift_only_MSE']); dd = max(b['MSE']-a['MSE'], 1e-12)
    propagation = abs(nr-nd)/dr+abs(nd)*abs(dr-dd)/(dr*dd)+1e-12
    assert abs(stored['brightness_gain_fraction']-computed['brightness_gain_fraction']) <= propagation


def _numeric_scalar_checks(saved, actual):
    for key, value in actual.items():
        other = saved[key]
        if value is None or isinstance(value, bool):
            assert value == other
        else:
            assert math.isfinite(value) and math.isfinite(other) and abs(value-other) <= 1e-13, key


def _read_json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def saved_Adam_arithmetic(current, gradient, moments, squares, index):
    """Read-only numerical reconstruction from a saved journal, never an optimizer."""
    import numpy as np
    decay_gradient = (gradient+np.float32(1e-5)*current).astype(np.float32)
    moments = (moments+np.float32(.1)*(decay_gradient-moments)).astype(np.float32)
    squares = (np.float32(.999)*squares+np.float32(.001)*decay_gradient*decay_gradient).astype(np.float32)
    denominator = (np.sqrt(squares)/np.float32(math.sqrt(1-.999**index))+np.float32(1e-8)).astype(np.float32)
    proposal = (current-np.float32(.0001/(1-.9**index))*moments/denominator).astype(np.float32)
    return proposal, moments, squares


def fixed_objective_arrays(raw, baseline, target8, mask8, supports, retained_embeddings, er, initial, old):
    """Fixed loss arithmetic on saved pixels; never a fresh model forward."""
    import numpy as np
    import torch
    from cctv_dgp_group_conflicts_v1_losses import pixel_and_structure_losses
    from cctv_dgp_bank_hinge_boundary_v1_loss import brightness_guard
    def as_tensor(value):
        return torch.from_numpy(value).permute(0, 3, 1, 2).contiguous()
    with torch.no_grad():
        pred, base = as_tensor(raw), as_tensor(baseline)
        target = as_tensor(np.stack([target8]*5).astype(np.float32)/np.float32(255))
        mask = torch.from_numpy(np.stack([mask8]*5).astype(np.float32))[:, None]
        feature = torch.from_numpy(np.stack(supports).astype(np.float32))[:, None]
        old_terms = pixel_and_structure_losses(base, target, mask, feature)
        terms = pixel_and_structure_losses(pred, target, mask, feature)
        target_vector = torch.from_numpy(retained_embeddings[:, 2]).double()
        old_arc = 1-(torch.from_numpy(retained_embeddings[:, 0]).double()*target_vector).sum(1)
        arc = old_arc if initial else 1-(torch.from_numpy(er[:5]).double()*target_vector).sum(1)
        normalizers = old['normalizers']
        averaged = [terms['MSE'].mean(), terms['SSIM_loss'].mean(), arc.mean(), terms['landmark_structure'][1:].mean()]
        components = [float(w*v/n) for w, v, n in zip(old['reconstruction_weights'], averaged, normalizers)]
        barriers = [float(5*torch.relu(terms['MSE']-old_terms['MSE']-1e-12).mean()/normalizers[0]),
                    float(5*torch.relu(terms['SSIM_loss']-old_terms['SSIM_loss']-1e-6).mean()/normalizers[1]),
                    float(5*torch.relu(arc-old_arc-1e-6).mean()/normalizers[2])]
        anchors = [float(terms['MSE'][0]/old['active_anchor_normalizers'][0]),
                   float(terms['MSE'][1]/old['active_anchor_normalizers'][1])]
        photo, detail = brightness_guard(pred, base, target, mask, normalizers[0], 10., .2)
        assert not photo.requires_grad and photo.grad_fn is None
        return dict(objective=sum(components)+sum(barriers)+sum(anchors)+float(photo),
            normalized_weighted_RGB_SSIM_recognizer_structure=components,
            regression_barriers_RGB_SSIM_recognizer=barriers, clear_blur_RGB_anchors=anchors,
            photometric_guard=float(photo), per_image_photometric=detail,
            raw_RGB_MSE=terms['MSE'].tolist(), raw_SSIM_loss=terms['SSIM_loss'].tolist(),
            raw_recognizer_loss=arc.tolist(), raw_structure_loss=terms['landmark_structure'].tolist())


def _compare_fixed_arithmetic(actual, saved):
    """CPU/CUDA reduction tolerance, unrelated to the output quality gate."""
    import numpy as np
    maximum = 0.
    for key, value in actual.items():
        other = saved[key]
        if isinstance(value, dict):
            maximum = max(maximum, _compare_fixed_arithmetic(value, other))
        elif isinstance(value, bool):
            assert value == other
        else:
            a, b = np.asarray(value, np.float64), np.asarray(other, np.float64)
            assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
            maximum = max(maximum, float(np.max(np.abs(a-b))))
            np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-10)
    return maximum


def audit_saved(root, source, parent, replay, p, old, data, receipt):
    import numpy as np
    import torch  # Only read-only saved-state deserialization; never constructs a model.
    from PIL import Image, ImageDraw
    from cctv_dgp_decoder_hinge_repair_v1_contract import read, write, sha, array_sha
    from cctv_dgp_spatial_path_replay_v1_contract import expected_state_hash
    from audit_cctv_dgp_spatial_path_replay_v1_return import recompute_metrics
    from audit_cctv_dgp_spatial_readout_v1_return import grouped, independent_gate, state_hash
    sys.path.insert(1, str(source)); sys.path.insert(1, str(parent)); sys.path.insert(1, str(root))
    from frozen_capacity_contract import feature_support
    from cctv_dgp_head4_lossless_v1 import unpack
    import cctv_dgp_bank_hinge_boundary_v1_loss as fixed_guard
    import cctv_dgp_group_conflicts_v1_losses as fixed_losses
    assert sha(fixed_guard.__file__) == p['corrected_helper_sha256']
    assert sha(fixed_losses.__file__) == p['parent_required_sha256']['cctv_dgp_group_conflicts_v1_losses.py']
    torch.set_num_threads(4)
    start = time.monotonic(); out = root/'outputs/decoder_hinge_repair_v1'
    assert not receipt.exists(), 'Retain the previous independent audit'
    verify_corrected_guard_binding((root/'scripts/cctv_dgp_decoder_hinge_repair_v1_vm.py').read_text(encoding='utf-8'))
    def clock():
        assert time.monotonic()-start <= p['budgets']['saved_audit_seconds'], 'Finite saved-audit stop'
    if not (out/'run_results.json').is_file():
        assert (out/'failure.json').is_file()
        failure = read(out/'failure.json')
        write(receipt, dict(complete=True, partial_failure_evidence_only=True, failure=failure,
            saved_files={path.relative_to(root).as_posix(): sha(path) for path in out.rglob('*') if path.is_file()},
            no_missing_result_implies_success=True, model_qualified=False, neural_calls=0,
            optimizer_updates=0, gradient_queries=0, visual_review_pending=True))
        return
    result = read(out/'run_results.json')
    assert result['complete'] and result['model_qualified'] is False and result['probe_admission_used'] is False
    assert result['noautomaticresume'] and result['full_corpus_epochs'] == 0
    assert result['failed_recipe_resume_allowed'] is False and result['automatic_extra_epochs'] is False
    assert result['learning_rate'] == .0001 and result['learning_rate_changed'] is False
    assert result['derivative_boundary_is_only_intended_objective_change'] and result['original_provider_recognizer_unchanged']
    updates = result['training_updates']
    assert 1 <= updates <= 50 and result['counts']['optimizer_updates'] == updates
    assert result['counts']['backwards'] == updates and result['counts']['gradient_queries'] == 42
    assert set(result['counts']) >= {'optimizer_updates', 'backwards', 'gradient_queries'}
    for key, value in result['counts'].items():
        maximum = p['budgets'].get('maximum_'+key, p['budgets'].get(key))
        if maximum is not None: assert value <= maximum
    labels = result['snapshot_labels']
    fit_labels = ['fit_u'+str(u).zfill(3) for u in [1, 10, 20, 30, 40, 50] if u <= updates]
    assert labels == ['baseline', 'initial', 'legacy_A1', 'fit_u001', 'probe_1_16', 'probe_1_4']+fit_labels[1:]
    assert result['counts'] == dict(optimizer_updates=updates, backwards=updates, gradient_queries=42,
        original_batch_calls=20, prior_feature_batch_calls=20,
        decoder_batch_calls=(len(labels)-1)*20+2+updates,
        recognizer_batch_calls=len(labels)*40+2+updates)
    environment = read(out/'environment.json')
    assert environment['GPU'] == 'NVIDIA L4' and environment['host'].split('.')[0] == 'forensic-dgp-thesis'
    assert environment['original_state'] == old['original_state'] and environment['recognizer_state'] == old['recognizer_state']
    assert environment['provider_state'] == old['feature_provider_initial_state']
    assert environment['legacy_helper_sha256'] == p['legacy_helper_sha256']
    assert environment['corrected_helper_sha256'] == p['corrected_helper_sha256']
    assert environment['learning_rate'] == .0001 and not environment['historical_training_called']
    refs, cases = {r['id']: r for r in data['references']}, {c['id']: c for c in data['cases']}
    reductions, saved_metrics, cells = {}, {}, {}
    compositions = controls = 0; component_rows = []; component_error = 0.
    def pixels(path, mode='RGB'):
        with Image.open(path) as image:
            assert image.size == (256, 256)
            return np.asarray(image.convert(mode)).copy()
    for label in labels:
        clock(); saved = read(out/label/'metrics.json'); saved_metrics[label] = saved
        component_receipt = read(out/label/'objective_components.json')
        assert component_receipt['diagnostic_embedding_batch15_not_fresh_training_batch5']
        assert component_receipt['initial_cached_embedding_controls'] == (label in ['baseline', 'initial'])
        assert not component_receipt['gradient_or_model_quality_claim']
        assert len(component_receipt['rows']) == 20
        recorded = {row['case']: row for row in saved['rows']}
        assert set(recorded) == set(p['case_ids'])
        rows = []
        initial = label in ['baseline', 'initial']
        for index, block in enumerate(p['reference_blocks']):
            clock(); reference = refs[block['reference']]
            retained, ev = unpack(parent/'outputs/bank_comparison_v1/baseline/packs'/f"b{block['begin']//5:03d}.npz", block['case_ids'])
            with np.load(out/label/'packs'/f'r{index:02d}.npz', allow_pickle=False) as pack:
                assert set(pack.files) == {'ids', 'raw', 'embeddings_raw', 'embeddings_png', 'target_embeddings'}
                assert pack['ids'].tolist() == block['case_ids']
                raw, er, ep, tv = [pack[key].copy() for key in ['raw', 'embeddings_raw', 'embeddings_png', 'target_embeddings']]
            assert raw.shape == (5, 256, 256, 3) and raw.dtype == np.float32
            assert np.isfinite(raw).all() and raw.min() >= 0 and raw.max() <= 1
            assert np.array_equal(tv, ev[:, 2])
            for v, slot in [(er, 0), (ep, 1)]:
                assert v.shape == (15, 512) and v.dtype == np.float32 and np.isfinite(v).all()
                np.testing.assert_allclose(v[5:10], ev[:, slot], rtol=1e-6, atol=1e-6)
                np.testing.assert_allclose(v[10:15], tv, rtol=1e-6, atol=1e-6)
            control_folder = source/'outputs/spatial_readout_v1/baseline' if initial else replay/'outputs/spatial_path_replay_v1/A_u001' if label == 'legacy_A1' else None
            if control_folder is not None:
                with np.load(control_folder/'packs'/f'r{index:02d}.npz', allow_pickle=False) as pack:
                    assert np.array_equal(raw, pack['raw'])
            target8 = pixels(parent/reference['target']); mask8 = pixels(parent/reference['observed'], 'L') > 0
            supports = [feature_support(mask8, cases[cid]['landmarks5_canvas_xy']) for cid in block['case_ids']]
            component = fixed_objective_arrays(raw, retained, target8, mask8, supports, ev, er, initial, old)
            saved_component = component_receipt['rows'][index]
            assert saved_component['reference'] == block['reference'] and saved_component['source'] == block['source']
            component_error = max(component_error, _compare_fixed_arithmetic(component, saved_component))
            component_rows.append(dict(snapshot=label, reference=block['reference'], source=block['source'], **component))
            for slot, cid in enumerate(block['case_ids']):
                c = cases[cid]; camera = pixels(parent/c['input']); target = pixels(parent/reference['target'])
                mask = pixels(parent/reference['observed'], 'L') > 0
                support = feature_support(mask, c['landmarks5_canvas_xy'])
                png = pixels(out/label/'png'/(cid+'.png'))
                rm, pm = recompute_metrics(raw[slot], png, camera, target, mask, support, retained[slot], ev, er, ep, slot, initial)
                if control_folder is not None:
                    assert np.array_equal(png, pixels(control_folder/'png'/(cid+'.png'))); controls += 1
                row = dict(case=cid, source=c['source'], profile=c['profile'], raw=rm, png=pm)
                assert all(row[k] == recorded[cid][k] for k in ['case', 'source', 'profile'])
                for mode in ['raw', 'png']: _numeric_scalar_checks(recorded[cid][mode], row[mode])
                rows.append(row); cells[(label, cid)] = png; compositions += 1
        reductions[label] = {mode: grouped(rows, mode) for mode in ['raw', 'png']}
        for mode in ['raw', 'png']:
            for group, metrics in reductions[label][mode].items(): _numeric_scalar_checks(saved['groups'][mode][group], metrics)
    assert controls == 300 and compositions == len(labels)*100
    outcomes = []
    for label in labels:
        if label == 'baseline': continue
        gates = {mode: independent_gate(reductions['baseline'][mode], reductions[label][mode]) for mode in ['raw', 'png']}
        if label.startswith(('fit_u', 'probe_')):
            stored = read(out/(label+'_outcome.json'))
            for mode in ['raw', 'png']:
                _check_gate(stored['gates'][mode], gates[mode], saved_metrics['baseline']['groups'][mode],
                            saved_metrics[label]['groups'][mode], reductions['baseline'][mode], reductions[label][mode])
            assert stored['continuation_eligible'] == continuation_allowed(gates)
            assert not stored['model_qualified']
        outcomes.append(dict(label=label, gates=gates, continuation_eligible=continuation_allowed(gates)))
    fit_outcomes = [dict(update=int(row['label'][-3:]), gates=row['gates']) for row in outcomes if row['label'].startswith('fit_u')]
    verify_stop_sequence(fit_outcomes, updates, result['stop_reason'])
    assert result['training_stopped'] == (result['stop_reason'] == 'quality_stop')
    assert result['fit_outcomes'] == [read(out/(label+'_outcome.json')) for label in fit_labels]
    endpoint_pass = updates == 50 and all(value['pass'] for value in fit_outcomes[-1]['gates'].values())
    assert result['numerical_endpoint_pass'] == endpoint_pass
    if result['training_stopped']:
        stop = read(out/'quality_stop.json')
        assert stop['snapshot'] == fit_labels[-1] and stop['outcome'] == result['fit_outcomes'][-1]
        assert stop['failed_recipe_resume_allowed'] is False and stop['automatic_extra_updates'] == 0
    boundary = read(out/'guard_boundary/receipt.json')
    assert len(boundary['rows']) == len(p['reference_blocks']) == 20
    assert boundary['complete'] and boundary['all20_scalar_metadata_exact'] and boundary['all20_corrected_zero_gradient']
    assert boundary['gradient_queries'] == 40 and boundary['optimizer_updates'] == 0
    boundary_checked = []
    for index, (row, block) in enumerate(zip(boundary['rows'], p['reference_blocks'])):
        clock(); assert row['reference'] == block['reference'] and row['case_ids'] == block['case_ids']
        path = out/'guard_boundary'/f'r{index:02d}.npz'
        with np.load(path, allow_pickle=False) as archive:
            assert set(archive.files) == {'ids', 'legacy_gradient', 'corrected_gradient'}
            assert archive['ids'].tolist() == block['case_ids']
            legacy, corrected = archive['legacy_gradient'].copy(), archive['corrected_gradient'].copy()
        assert legacy.shape == corrected.shape == (5, 3, 256, 256)
        verify_guard_boundary(dict(old_penalty=row['legacy_scalar'], new_penalty=row['corrected_scalar'],
            old_metadata=row['legacy_metadata'], new_metadata=row['corrected_metadata']), legacy, corrected)
        assert row['legacy_gradient_sha256'] == array_sha(legacy) and row['corrected_gradient_sha256'] == array_sha(corrected)
        assert abs(float(np.linalg.norm(legacy.astype(np.float64)))-row['legacy_gradient_L2']) < 1e-10
        assert row['corrected_gradient_L2'] == 0.
        boundary_checked.append(dict(reference=block['reference'], legacy_gradient_nonzero=bool(np.count_nonzero(legacy)), corrected_gradient_exact_zero=True))
    names = old['trainable_parameter_names']['A']
    assert len(names) == 15
    layout = read(source/'outputs/spatial_readout_v1/A_parameter_layout.json')
    source_state = torch.load(source/'outputs/spatial_readout_v1/state_A0/full_state.pth', map_location='cpu', weights_only=True)['model']
    initial = np.load(source/'outputs/spatial_readout_v1/A_initial_vector.npy', allow_pickle=False)
    fixed = state_hash({key: value for key, value in source_state.items() if key not in names})
    buffers = read(source/'outputs/spatial_readout_v1/state_A0/receipt.json')['buffer_state']
    radius = read(source/'outputs/spatial_readout_v1/distance_bound.json')['common_radius']
    assert radius == p['common_radius_L2']
    first = np.load(out/'fit_u001_vector.npy', allow_pickle=False)
    for label in [v for v in labels if v != 'baseline']:
        vector = np.load(out/(label+'_vector.npy'), allow_pickle=False)
        record = read(out/(label+'_state.json'))
        assert vector.dtype == np.float32 and vector.shape == initial.shape and np.isfinite(vector).all()
        assert np.linalg.norm(vector.astype(np.float64)-initial.astype(np.float64)) <= radius
        if label == 'initial': assert np.array_equal(vector, initial)
        if label == 'legacy_A1':
            with np.load(source/'outputs/spatial_readout_v1/trajectory_A/update_001.npz', allow_pickle=False) as archive:
                assert np.array_equal(vector, archive['actual'])
        if label.startswith('probe_'):
            verify_probe_vector(initial, first, vector, 1/16 if label == 'probe_1_16' else 1/4, radius)
        assert record['vector_sha256'] == array_sha(vector) and record['installed_vector_exact']
        assert record['model_state'] == expected_state_hash(source_state, names, layout, vector)
        assert record['fixed_state'] == fixed and record['buffer_state'] == buffers
        expected_steps = int(label[-3:]) if label.startswith('fit_u') else 1 if label.startswith('probe_') else 0
        assert record['optimizer_updates'] == expected_steps and record['gradients_enabled'] is False
    trace = _read_json_lines(out/'trajectory/trace.jsonl')
    assert len(trace) == updates
    fit_trace = read(out/'fit_trace.json')
    assert fit_trace['rows'] == trace and fit_trace['updates'] == updates
    assert fit_trace['training_seconds_excluding_snapshots'] <= fit_trace['cap_seconds'] == 600
    current = initial.copy(); moments = np.zeros_like(current); squares = np.zeros_like(current)
    max_proposal_error = max_moment_error = 0.
    snapshots = {}
    for index, row in enumerate(trace, 1):
        clock(); assert row['update'] == index and row['cache_index'] == p['schedule'][index-1]
        assert row['reference'] == p['reference_blocks'][row['cache_index']]['reference']
        with np.load(out/'trajectory'/f'update_{index:03d}.npz', allow_pickle=False) as archive:
            assert set(archive.files) == {'clipped_gradient', 'unconstrained', 'actual'}
            gradient, proposal, actual = [archive[key].copy() for key in ['clipped_gradient', 'unconstrained', 'actual']]
        assert gradient.dtype == np.float32 and gradient.shape == initial.shape and np.isfinite(gradient).all()
        assert np.linalg.norm(gradient.astype(np.float64)) <= 1.00001
        assert math.isfinite(row['gradient_norm_before_clip']) and row['gradient_norm_before_clip'] >= 0
        factor = min(1., 1./(row['gradient_norm_before_clip']+1e-6))
        assert abs(row['clip_factor']-factor) < 1e-12
        assert abs(float(np.linalg.norm(gradient.astype(np.float64)))-row['gradient_norm_before_clip']*factor) <= 2e-5
        verify_projection_journal(initial, proposal, actual, row['projection'], radius)
        # Independent float32 Adam arithmetic. This tolerance audits implementation
        # rounding only; output preservation and projection bytes remain exact.
        expected_proposal, moments, squares = saved_Adam_arithmetic(current, gradient, moments, squares, index)
        error = float(np.max(np.abs(expected_proposal.astype(np.float64)-proposal.astype(np.float64))))
        max_proposal_error = max(max_proposal_error, error)
        np.testing.assert_allclose(proposal, expected_proposal, rtol=2e-6, atol=3e-7)
        current = actual
        if index in [int(label[-3:]) for label in fit_labels]: snapshots[index] = (current.copy(), moments.copy(), squares.copy())
    for label in ['initial']+fit_labels:
        clock(); state_path = out/('state_'+label)/'full_state.pth'
        state = torch.load(state_path, map_location='cpu', weights_only=True)
        state_receipt = read(state_path.parent/'receipt.json')
        step = 0 if label == 'initial' else int(label[-3:])
        assert state['updates'] == step and state_receipt['updates'] == step
        assert state['protocol_sha256'] == sha(root/'protocol.json')
        assert state['counts']['optimizer_updates'] == step and state['counts']['backwards'] == step
        assert state['failed_recipe_resume_allowed'] is False and state['exact_future_resume_guaranteed'] is False
        assert state_receipt['full_state_sha256'] == sha(state_path)
        assert state_receipt['model_state'] == state_hash(state['model'])
        assert state_receipt['buffer_state'] == buffers
        assert state_hash({key: value for key, value in state['model'].items() if key not in names}) == fixed
        vector = np.concatenate([state['model'][name].numpy().reshape(-1) for name in names])
        expected = initial if step == 0 else snapshots[step][0]
        assert np.array_equal(vector, expected)
        assert np.array_equal(vector, np.load(out/(label+'_vector.npy'), allow_pickle=False))
        assert state['CPU_rng'].dtype == torch.uint8 and state['CPU_rng'].ndim == 1
        assert state['CUDA_rng'] and all(v.dtype == torch.uint8 and v.ndim == 1 for v in state['CUDA_rng'])
        if step == 1:
            restoration = read(out/'probe_restoration.json')
            for key in ['complete', 'RNG_unchanged_before_restore', 'CPU_CUDA_rng_restored_exact',
                        'corrected_first_model_restored_exact', 'optimizer_shadow_unchanged']:
                assert restoration[key] is True
            assert restoration['CPU_rng_sha256'] == array_sha(state['CPU_rng'].numpy())
            assert restoration['CUDA_rng_sha256'] == [array_sha(value.numpy()) for value in state['CUDA_rng']]
            assert restoration['model_state'] == state_receipt['model_state']
        optimizer = state['optimizer']; assert len(optimizer['param_groups']) == 1
        group = optimizer['param_groups'][0]
        assert group['lr'] == .0001 and group['betas'] == (.9, .999) and group['eps'] == 1e-8
        assert group['weight_decay'] == 1e-5 and group['foreach'] is False and group['fused'] is False
        assert len(group['params']) == len(names)
        if step == 0:
            assert optimizer['state'] == {}
            shadow = np.concatenate([value.numpy().reshape(-1) for value in state['shadow']])
            assert np.array_equal(shadow, vector)
            continue
        assert len(optimizer['state']) == len(names)
        m, v = snapshots[step][1:]
        offset = 0
        for serial, (identifier, item) in enumerate(zip(group['params'], layout)):
            saved = optimizer['state'][identifier]; count = math.prod(item['shape'])
            assert int(saved['step']) == step
            for key, expected_flat in [('exp_avg', m), ('exp_avg_sq', v)]:
                value = saved[key].numpy(); assert list(value.shape) == item['shape'] and np.isfinite(value).all()
                expected_array = expected_flat[offset:offset+count].reshape(item['shape'])
                max_moment_error = max(max_moment_error, float(np.max(np.abs(value.astype(np.float64)-expected_array.astype(np.float64)))))
                np.testing.assert_allclose(value, expected_array, rtol=2e-5, atol=3e-8)
            offset += count
        assert offset == initial.size
        shadow = np.concatenate([value.numpy().reshape(-1) for value in state['shadow']])
        assert np.array_equal(shadow, vector), 'Shadow must reset to projected original-ball vector'
    preflight = read(out/'preflight.json')
    assert len(preflight['rows']) == 2
    norm_sum = np.zeros(15, np.float64)
    for row in preflight['rows']:
        assert row['names'] == names and len(row['norms']) == 15 and all(math.isfinite(v) and v >= 0 for v in row['norms'])
        norm_sum += np.asarray(row['norms'], np.float64)
    assert (norm_sum > 0).all() and np.array_equal(norm_sum, np.asarray(preflight['summed_norms'], np.float64))
    gallery = receipt.parent/'visual_sheets'; assert not gallery.exists(); gallery.mkdir()
    titles = ['Input', 'TRAIN target']+labels
    assert len(titles) <= 13
    sheets = []; bindings = []
    for block in p['reference_blocks']:
        clock(); reference = refs[block['reference']]
        sheet = Image.new('RGB', (256*len(titles), 1320), '#fafafa'); draw = ImageDraw.Draw(sheet)
        for column, title in enumerate(titles): draw.text((256*column+8, 8), title, fill='#111111')
        for row, cid in enumerate(block['case_ids']):
            values = [pixels(parent/cases[cid]['input']), pixels(parent/reference['target'])]+[cells[(label, cid)] for label in labels]
            for column, value in enumerate(values):
                sheet.paste(Image.fromarray(value), (256*column, 40+256*row))
                bindings.append(dict(sheet=block['reference']+'.png', case=cid, column=column, pixels_sha256=array_sha(value)))
        path = gallery/(block['reference']+'.png'); sheet.save(path)
        sheets.append(dict(path=str(path), sha256=sha(path), cells=5*len(titles), reference=block['reference']))
    clock()
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), compositions_checked=compositions,
        exact_original_initial_legacy_control_cases=controls, all17_RAW_all17_PNG_groups_recomputed=True,
        outcomes=outcomes, training_updates_audited=updates, first_scheduled_stop_logic_verified=True,
        guard_boundary=boundary_checked, full_saved_states=1+len(fit_labels),
        saved_Adam_rounding_maximum_proposal_error=max_proposal_error, saved_Adam_rounding_maximum_moment_error=max_moment_error,
        all_projection_vectors_byte_exact=True, no_local_model_construction_or_weight_assignment=True,
        saved_embeddings_reused_no_independent_recognizer_forward=True, CUDA_derivatives_not_independently_replayed=True,
        fixed_loss_arithmetic_rows=component_rows, maximum_CPU_CUDA_fixed_component_error=component_error,
        fixed_component_tolerance_not_scientific_gate=1e-10,
        sheets=sheets, cell_bindings=bindings, visual_review_pending=True, independent_final_reviewer=False,
        model_qualified=False, full_corpus_epochs=0, neural_calls=0, gradient_queries=0, optimizer_updates=0,
        seconds=time.monotonic()-start)
    write(receipt, record)
    print(json.dumps(dict(complete=True, compositions=compositions, training_updates_audited=updates,
        visual_sheets=len(sheets), new_local_optimizer_updates=0, model_qualified=False)), flush=True)


def main():
    from cctv_dgp_decoder_hinge_repair_v1_contract import verify_transfer
    parser = argparse.ArgumentParser()
    for name in ['root', 'source-root', 'parent-root', 'replay-root', 'receipt']:
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    args = parser.parse_args()
    p, old, data = verify_transfer(args.root, args.source_root, args.parent_root, args.protocol_sha, replay=args.replay_root)
    audit_saved(args.root.resolve(), args.source_root.resolve(), args.parent_root.resolve(), args.replay_root.resolve(),
                p, old, data, args.receipt.resolve())


if __name__ == '__main__': main()
