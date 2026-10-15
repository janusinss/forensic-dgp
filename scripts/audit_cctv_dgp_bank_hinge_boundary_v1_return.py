"""Audit raw replay and sampled gradient arithmetic without local autograd."""
import ast
from datetime import datetime, timezone
import hashlib
import sys
import time

import numpy as np
from PIL import Image
from scipy.ndimage import uniform_filter, convolve1d
import torch

from import_cctv_dgp_bank_hinge_boundary_v1_return import ROOT, OUT, RETURN, ANALYSIS, PARENT, PIN, read, sha, write


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def image(path, mode='RGB'):
    with Image.open(path) as value:
        assert value.size == (256, 256)
        return np.asarray(value.convert(mode)).copy()


def losses(actual, target8, observed, feature):
    target32 = target8.astype(np.float32) / np.float32(255)
    mse = float(np.square(actual - target32)[observed].astype(np.float64).mean())
    x, y = actual.astype(np.float64), target32.astype(np.float64)
    avg = lambda a: uniform_filter(a, size=(7, 7, 1), mode='constant')[3:-3, 3:-3]
    ux, uy = avg(x), avg(y)
    vx, vy = (avg(x*x)-ux*ux)*(49/48), (avg(y*y)-uy*uy)*(49/48)
    cov = (avg(x*y)-ux*uy)*(49/48)
    interior = uniform_filter(observed.astype(np.float64), size=7, mode='constant')[3:-3, 3:-3] > 1-1e-12
    ss = ((2*ux*uy+.01**2)*(2*cov+.03**2))/((ux*ux+uy*uy+.01**2)*(vx+vy+.03**2))
    ssim_loss = 1-float(ss[interior].mean())
    z = np.arange(-6, 7, dtype=np.float64)
    kernel = np.exp(-.5*(z/2)**2)
    kernel /= kernel.sum()
    def high(a):
        luma = (a * np.array([.299, .587, .114])).sum(2)
        return luma-convolve1d(convolve1d(luma, kernel, axis=1, mode='constant'), kernel, axis=0, mode='constant')
    detail = float(np.square(high(x)-high(target8.astype(np.float64)/255))[feature].mean())
    return np.array([mse, ssim_loss, detail], dtype=np.float64)


def state_hash(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous()
        digest.update(name.encode()+b'\0'+str(a.dtype).encode()+b'\0')
        digest.update(str(tuple(a.shape)).encode()+b'\0'+a.numpy().tobytes())
    return digest.hexdigest()


def partitions(names):
    result = dict(all=list(range(len(names))),
        decoder=[i for i, n in enumerate(names) if n.startswith('current.')],
        style=[i for i, n in enumerate(names) if n.startswith('style_conditioner.')],
        spatial=[i for i, n in enumerate(names) if n.startswith('spatial_conditioners.')])
    result.update({'fusion'+s:[i for i, n in enumerate(names) if n.startswith('fusions.'+s+'.')]
                   for s in ['16', '32', '64', '128', '256']})
    assert sorted(i for label, indices in result.items() if label != 'all' for i in indices) == list(range(len(names)))
    return result


def check_initial_inactive_guard(rows):
    initial = [row for row in rows if row['state'] == 'initial']
    assert len(initial) == 20
    for row in rows:
        assert row['scalar_penalty_exactly_matches_retained_guard']
        assert row['inactive_at_zero_hinge_derivative_declared']
    for row in initial:
        assert row['component_values'][6] == 0, 'Initial scalar guard should be zero'
        for group in row['groups'].values():
            matrix = np.asarray(group['Gram'])
            assert np.count_nonzero(matrix[6]) == np.count_nonzero(matrix[:,6]) == 0, 'Inactive hinge has a nonzero gradient'
            assert group['Adam_direction_dot_components'][6] == 0
    return len(initial)


def main():
    start = time.monotonic()
    receipt = ANALYSIS / 'independent_audit.json'
    assert not receipt.exists()
    imported = read(ANALYSIS / 'import.json')
    assert imported['complete'] and imported['protocol_sha256'] == PIN
    assert sha(RETURN / 'protocol.json') == PIN
    p, old = read(RETURN / 'protocol.json'), read(PARENT / 'protocol.json')
    manifest = read(RETURN / 'export_manifest.json')
    assert manifest['complete_diagnostic'] and not manifest['failure_present']
    assert {f.relative_to(RETURN).as_posix() for f in RETURN.rglob('*') if f.is_file()} == set(manifest['files']) | {'export_manifest.json'}
    for name, digest in manifest['files'].items():
        assert sha(RETURN / name) == digest
    for name, digest in p['parent_required_sha256'].items():
        assert sha(PARENT / name) == digest
    out = RETURN / 'outputs/loss_alignment_v1'
    run = read(out / 'run_results.json')
    assert run['complete'] and not (out / 'failure.json').exists()
    counts = dict(B=80, recognizer=40, original=0, gradient_queries=360, backwards=0, optimizer_updates=0)
    assert run['counts'] == counts and not run['model_qualified']
    assert run['inactive_initial_guard_zero_gradient_all20_verified'] and run['scalar_penalty_parity_all40_verified']
    check_initial_inactive_guard(run['rows'])
    assert run['state_and_buffer_bytes_unchanged'] and not run['new_checkpoint_created']
    assert run['all40_frozen_raw_controls_retained'] and run['input_layout_matches_parent']
    assert run['paired_TRAIN_cases'] == 100 and run['states_measured'] == 2
    assert run['native_pixels_used'] == run['final_identity_pixels'] == 0
    assert run['sampled_full_gradient_reference_batches'] == 4
    assert run['full_gradient_audit_scope_is_sampled_not_all40'] and run['raw_PNG_model_quality_not_requalified']
    assert 0 < run['seconds'] < p['worker_seconds'] and 0 < run['peak_VRAM'] <= p['peak_VRAM_bytes']
    assert (RETURN / 'review_exit_code.txt').read_text().strip() == '0'
    assert not list(RETURN.rglob('*.pth'))
    environment = read(out / 'environment.json')
    assert environment['GPU'] == 'NVIDIA L4' and environment['host'] == 'forensic-dgp-thesis'
    assert environment['manual_tmux'] and not environment['optimizer_constructed']
    assert environment['failed_state_used_for_diagnosis_only']
    selected = [i for i in range(0, 3905, 5) if old['cases'][i]['id'] in p['case_ids']]
    assert len(selected) == 20
    refs = {r['id']: r for r in old['references']}
    rows = run['rows']
    assert [(r['state'], r['begin']) for r in rows] == [(s, b) for s in p['states'] for b in selected]
    names = p['learning_names']
    assert run['learning_names'] == names and len(names) == 37
    components = p['component_names']
    assert run['component_names'] == components and len(components) == 9
    groups = partitions(names)
    constants = ast.parse((PARENT / 'cctv_dgp_bank_comparison_v1_contract.py').read_text())
    assignments = {n.targets[0].id:n.value for n in constants.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    normalizers, anchors = [np.array(ast.literal_eval(assignments[k]), dtype=np.float64) for k in ['NORMALIZERS', 'ANCHORS']]
    weights = np.array(old['reconstruction_weights'], dtype=np.float64)
    assert p['preserved_scientific_thresholds'] == old['scientific_thresholds']
    assert p['bit_exact_control_check_retained'] and not p['failed_run_resume_allowed'] and not p['app_promotion']
    torch.set_grad_enabled(False)
    torch.set_num_threads(4)
    initial = torch.load(PARENT / 'initializers/initial_B.pth', map_location='cpu', weights_only=True)
    stored = torch.load(PARENT / 'outputs/bank_comparison_v1/state_B50/full_state.pth', map_location='cpu', weights_only=True)
    assert state_hash(initial) == p['model_states']['initial']
    assert state_hash(stored['model']) == p['model_states']['joint50']
    assert stored['updates'] == 50 and not stored['manual_failed_resume_allowed']
    assert len(stored['optimizer']['param_groups']) == 1
    optimizer = stored['optimizer']['param_groups'][0]
    assert optimizer['betas'] == (.9, .999) and optimizer['eps'] == 1e-8 and optimizer['weight_decay'] == 1e-5
    assert len(optimizer['params']) == len(names)
    # Only retained pure storage/mask helpers are used; never import VM workers.
    sys.path.insert(0, str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support
    errors, sample_checks = [], []
    base_cache = {}
    for serial, row in enumerate(rows):
        state, begin = row['state'], row['begin']
        assert row == read(out / f'{state}_b{begin//5:03d}.json')
        block = old['cases'][begin:begin+5]
        ids = [c['id'] for c in block]
        assert row['case_ids'] == ids and row['reference'] == block[0]['source_person_or_reference']
        assert row['source'] == block[0]['source'] and row['model_state'] == p['model_states'][state]
        assert row['gradient_queries'] == 9 and row['Adam_direction_is_hypothetical_single_reference_not_actual_accumulated_update']
        replay = read(out / f'forward_replay/{state}_b{begin//5:03d}.json')
        raw_path = out / replay['raw_file']
        raw = np.load(raw_path, allow_pickle=False)
        assert raw.dtype == np.float32 and raw.shape == (5, 256, 256, 3) and np.isfinite(raw).all()
        if begin not in base_cache:
            base_cache[begin] = unpack(PARENT / f'outputs/bank_comparison_v1/baseline/packs/b{begin//5:03d}.npz', ids)
        baseline, base_emb = base_cache[begin]
        if state == 'initial':
            expected, embeddings = baseline, base_emb
        else:
            expected, embeddings = unpack(PARENT / f'outputs/bank_comparison_v1/B50/packs/b{begin//5:03d}.npz', ids, baseline=baseline)
        assert np.array_equal(raw, expected)
        assert replay['actual_RGB_float32_sha256'] == replay['expected_RGB_float32_sha256'] == array_sha(raw)
        assert replay['raw_file_sha256'] == sha(raw_path)
        assert replay['maximum_absolute_error'] == replay['changed_values'] == 0 and replay['exact_raw_replay']
        assert replay['gradient_queries_before_batch'] == serial*9 and replay['case_ids'] == ids
        assert replay['input_stride'] == replay['target_stride'] == [196608, 65536, 256, 1]
        assert replay['input_NCHW_contiguous'] and replay['target_NCHW_contiguous']
        assert 0 <= row['gradient_forward_maximum_error'] <= p['gradient_vs_frozen_raw_parity_tolerance']
        # CPython 3.10 and newer compensated float summation differ by last bits.
        # This derived scalar check is separate from exact raw replay and gates.
        assert np.isclose(row['objective'], sum(row['component_values'][:7]), rtol=0, atol=2e-14)
        cameras, actual_terms, previous_terms, mean_mse = [], [], [], []
        for index, case in enumerate(block):
            ref = refs[case['source_person_or_reference']]
            camera = image(PARENT / case['input'])
            target = image(PARENT / ref['target'])
            mask = image(PARENT / ref['observed'], 'L') > 0
            feature = feature_support(mask, case['landmarks5_canvas_xy'])
            actual_terms.append(losses(raw[index], target, mask, feature))
            previous_terms.append(losses(baseline[index], target, mask, feature))
            shift = (raw[index]-baseline[index])[mask].astype(np.float64).mean(0)
            control = np.clip(baseline[index].astype(np.float64)+shift, 0, 1).astype(np.float32)
            control[~mask] = baseline[index][~mask]
            mean_mse.append(float(np.square(control-target.astype(np.float32)/np.float32(255))[mask].astype(np.float64).mean()))
            assert np.array_equal(raw[index][~mask], (camera.astype(np.float32)/np.float32(255))[~mask])
            cameras.append(camera)
        input_float = np.stack(cameras).astype(np.float32)/np.float32(255)
        assert replay['input_RGB_float32_sha256'] == array_sha(input_float)
        actual_terms, previous_terms = np.array(actual_terms), np.array(previous_terms)
        arc = 1-(embeddings[:, 0].astype(np.float64)*embeddings[:, 2].astype(np.float64)).sum(1)
        old_arc = 1-(base_emb[:, 0].astype(np.float64)*base_emb[:, 2].astype(np.float64)).sum(1)
        mean_gain = np.maximum(previous_terms[:, 0]-mean_mse, 0)
        actual_gain = np.maximum(previous_terms[:, 0]-actual_terms[:, 0], 0)
        excess = np.maximum(mean_gain-.2*actual_gain, 0)
        photo = 10*excess[1:].mean()/normalizers[0]
        expected_values = list(weights*np.array([actual_terms[:, 0].mean(), actual_terms[:, 1].mean(), arc.mean(), actual_terms[1:, 2].mean()])/normalizers)
        expected_values += [actual_terms[0, 0]/anchors[0]+actual_terms[1, 0]/anchors[1],
            5*(np.maximum(actual_terms[:, 0]-previous_terms[:, 0]-1e-12, 0).mean()/normalizers[0]
               +np.maximum(actual_terms[:, 1]-previous_terms[:, 1]-1e-6, 0).mean()/normalizers[1]
               +np.maximum(arc-old_arc-1e-6, 0).mean()/normalizers[2]), photo,
            actual_terms[0, 0]/anchors[0], actual_terms[1:, 0].mean()/normalizers[0]]
        component_error = float(np.abs(np.array(expected_values)-row['component_values']).max())
        assert np.allclose(expected_values, row['component_values'], rtol=2e-5, atol=2e-5), (state, begin, expected_values, row['component_values'])
        errors.append(component_error)
        for label, indices in groups.items():
            group = row['groups'][label]
            assert group['parameter_names'] == [names[i] for i in indices]
            gram = np.array(group['Gram'], dtype=np.float64)
            dots = np.array(group['Adam_direction_dot_components'], dtype=np.float64)
            assert gram.shape == (9, 9) and dots.shape == (9,) and np.isfinite(gram).all() and np.isfinite(dots).all()
            assert np.allclose(gram, gram.T, rtol=0, atol=1e-12)
            assert np.linalg.eigvalsh(gram).min() >= -1e-10*max(1, float(np.diag(gram).max()))
            assert group['direction_squared_norm'] >= 0
        all_group = row['groups']['all']
        for field in ['Gram', 'Adam_direction_dot_components', 'direction_squared_norm']:
            assert np.allclose(all_group[field], sum(np.asarray(row['groups'][label][field]) for label in groups if label != 'all'), rtol=1e-9, atol=1e-10)
        gradient_norm = float(np.sqrt(max(0, np.array(all_group['Gram'])[:7, :7].sum())))
        assert np.isclose(gradient_norm, row['total_gradient_norm'], rtol=2e-7, atol=1e-9)
        assert np.isclose(min(1, 1/(row['total_gradient_norm']+1e-6)), row['clip_factor'], rtol=1e-7, atol=1e-8)

        sample = row['sampled_gradient_file']
        assert bool(sample) == (begin in p['sample_gradient_begins'])
        if sample:
            model = initial if state == 'initial' else stored['model']
            with np.load(out / sample, allow_pickle=False) as archive:
                keys = {f'g{j:02d}_p{i:02d}' for j in range(9) for i in range(37)} | {f'd_p{i:02d}' for i in range(37)}
                assert set(archive.files) == keys
                gradients = [[archive[f'g{j:02d}_p{i:02d}'] for i in range(37)] for j in range(9)]
                directions = [archive[f'd_p{i:02d}'] for i in range(37)]
            if state == 'initial':
                assert all(np.count_nonzero(a) == 0 for a in gradients[6]), 'Saved inactive guard gradient is nonzero'
            assert all(a.dtype == np.float32 and a.shape == tuple(model[names[i]].shape) and np.isfinite(a).all()
                       for values in gradients+[directions] for i, a in enumerate(values))
            total = [sum((values[i] for values in gradients[:7])) for i in range(37)]
            norm = float(np.sqrt(sum(np.square(g.astype(np.float64)).sum() for g in total)))
            factor = np.float32(min(1, 1/(norm+1e-6)))
            assert np.isclose(norm, row['total_gradient_norm'], rtol=1e-10, atol=1e-10)
            assert np.isclose(factor, row['clip_factor'], rtol=0, atol=1e-8)
            max_direction_error = 0.0
            for i, (g, actual) in enumerate(zip(total, directions)):
                effective = g*factor+model[names[i]].numpy()*np.float32(1e-5)
                if state == 'initial':
                    moment, variance, step = np.zeros_like(effective), np.zeros_like(effective), 1
                else:
                    state_i = stored['optimizer']['state'][optimizer['params'][i]]
                    assert int(state_i['step'].item()) == 50
                    moment, variance, step = state_i['exp_avg'].numpy(), state_i['exp_avg_sq'].numpy(), 51
                next_m = np.float32(.9)*moment+np.float32(.1)*effective
                next_v = np.float32(.999)*variance+np.float32(.001)*np.square(effective)
                expected_direction = -(next_m/np.float32(1-.9**step))/(np.sqrt(next_v/np.float32(1-.999**step))+np.float32(1e-8))
                assert np.allclose(actual, expected_direction, rtol=2e-5, atol=2e-5), ('sampled direction differs', state, begin, names[i])
                max_direction_error = max(max_direction_error, float(np.abs(actual-expected_direction).max()))
            for label, indices in groups.items():
                group = row['groups'][label]
                matrix = np.array([[sum(float(np.sum(gradients[j][i].astype(np.float64)*gradients[k][i].astype(np.float64))) for i in indices) for k in range(9)] for j in range(9)])
                dots = [sum(float(np.sum(values[i].astype(np.float64)*directions[i].astype(np.float64))) for i in indices) for values in gradients]
                length = sum(float(np.square(directions[i].astype(np.float64)).sum()) for i in indices)
                assert np.allclose(matrix, group['Gram'], rtol=1e-9, atol=1e-11)
                assert np.allclose(dots, group['Adam_direction_dot_components'], rtol=1e-9, atol=1e-11)
                assert np.isclose(length, group['direction_squared_norm'], rtol=1e-10, atol=1e-10)
            sample_checks.append(dict(state=state, begin=begin, full_parameter_gradients=37*9,
                parameter_direction_arrays=37, groups_checked=9, maximum_direction_array_error=max_direction_error))
            del gradients, directions, total
        if serial % 10 == 0:
            print(dict(rows_checked=serial+1, of=40, sampled_gradient_batches=len(sample_checks)), flush=True)
    assert len(sample_checks) == 4
    printed = [ast.literal_eval(line) for line in (RETURN / 'review.log').read_text().splitlines()]
    assert len(printed) == 40
    for serial, (printed_row, row) in enumerate(zip(printed, rows)):
        assert printed_row['state'] == row['state'] and printed_row['reference_batches'] == serial % 20+1
        assert printed_row['gradient_queries'] == (serial+1)*9 and printed_row['of'] == 20
        assert 0 < printed_row['seconds'] <= run['seconds']
    projection = read(out / 'timing_projection.json')
    assert projection['measured_reference_batches'] == 2 and projection['safety_factor'] == 1.5
    assert np.isclose(projection['projection_seconds'], max(r['seconds'] for r in rows[:2])*40*1.5, rtol=0, atol=1e-8)
    assert projection['projection_seconds'] <= projection['cap_seconds'] == p['measurement_seconds']
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / name) == digest for name, digest in protected.items())
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        import_sha256=sha(ANALYSIS / 'import.json'), protocol_sha256=PIN,
        run_results_sha256=sha(out / 'run_results.json'), parent_assets_checked=5852, returned_namespace_files_checked=len(manifest['files'])+1,
        exact_frozen_L4_raw_controls_checked=40, raw_case_replays_checked=200,
        independently_recomputed_scalar_component_rows=40, maximum_scalar_component_error=max(errors),
        scalar_and_sampled_array_tolerances_are_numerical_not_scientific=True,
        all40_summary_partition_algebra_and_PSD_checked=True,
        sampled_full_gradient_batches_checked=4, sampled_arrays_each=370, sampled_gradient_checks=sample_checks,
        sampled_Adam_direction_and_Gram_recomputed_in_NumPy=True,
        gradient_derivatives_not_freshly_replayed_or_independently_verified_for_all40=True,
        original_initial_and_joint50_state_hashes_checked=True, counts_recomputed=counts,
        VM_seconds=run['seconds'], VM_peak_VRAM=run['peak_VRAM'], measured_timing_projection=projection,
        inactive_initial_guard_zero_gradient_all20_independently_checked=True, scalar_penalty_definition_retained=True, exact_control_layout_repair_confirmed_on_L4=True, diagnostic_complete=True,
        zero_optimizer_updates_and_no_checkpoint=True, protected_bindings_unchanged=True,
        no_local_neural_calls_gradients_or_optimizer=True, no_agent_VM_connection_or_launch=True,
        new_raw_PNG_model_quality_or_native_qualification=False, final_identities_unused=True,
        all_seven_completion_families_pending=True, independent_final_review_pending=True,
        model_qualified=False, app_promotion=False, goal_complete=False, seconds=time.monotonic()-start))
    print(dict(complete=True, raw_replay_cases=200, scalar_rows=40, independently_checked_gradient_batches=4,
               VM_updates=0, model_qualified=False), flush=True)


if __name__ == '__main__':
    main()
