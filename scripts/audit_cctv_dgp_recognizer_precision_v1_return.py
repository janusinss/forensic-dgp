"""Replay returned precision gradients, scores and packed activations; no neural calls."""
from datetime import datetime, timezone
import hashlib
import numpy as np

from import_cctv_dgp_recognizer_precision_v1_return import ANALYSIS, BANK, INPUT, OUT, PIN, RETURN, ROOT, SOURCE, read, sha, write
from cctv_dgp_recognizer_precision_v1_contract import decode_sign_map, project
from cctv_dgp_RGB_gradient_parity_v1 import report as parity_report
from cctv_dgp_conditioning_response_v1_contract import response_receipt


def audit_phase(folder, p, old, block, precision, label):
    stem = f"b{block['begin']//5:03d}"; tag = stem + '_' + label
    record = read(folder / (tag + '.json'))
    assert record['precision'] == precision and record['begin'] == block['begin'] and record['label'] == label
    cases = old['cases'][block['begin']:block['begin'] + 5]
    assert [r['id'] for r in record['rows']] == block['case_ids']
    dtype = np.float32 if precision == 'FP32' else np.float64
    with np.load(folder / (tag + '_embeddings.npz'), allow_pickle=False) as f:
        assert f.files == ['raw']; v = f['raw'].copy()
    assert v.dtype == dtype and v.shape == (15, 512) and np.isfinite(v).all()
    assert np.allclose(np.linalg.norm(v, axis=1), 1, rtol=0, atol=1e-6 if precision == 'FP32' else 1e-12)
    actual = 'initial' if label == 'repeat' else label
    if precision == 'FP32':
        with np.load(SOURCE / 'outputs/conditioning_response_v1' / actual / (stem + '_embeddings.npz'), allow_pickle=False) as f:
            np.testing.assert_allclose(v, f['raw'], rtol=1e-6, atol=1e-6)
    gradients = None
    before = np.load(SOURCE / 'outputs/conditioning_response_v1/initial' / (stem + '.npy'), allow_pickle=False)
    if label in ['initial', 'repeat']:
        gradients = np.load(folder / (tag + '_RGB_gradients.npy'), allow_pickle=False)
        assert gradients.dtype == dtype and gradients.shape == (5, 256, 256, 3) and np.isfinite(gradients).all()
    for i, (r, c) in enumerate(zip(record['rows'], cases)):
        assert c['role'] == 'train' and {k: r[k] for k in ['id', 'source', 'profile']} == {k: c[k] for k in ['id', 'source', 'profile']}
        fresh = 1 - float(v[i].astype(np.float64) @ v[10 + i].astype(np.float64))
        assert abs(r['loss'] - fresh) <= 1e-13
        if gradients is not None:
            assert hashlib.sha256(gradients[i].tobytes()).hexdigest() == r['gradient_bytes_sha256']
            assert np.isclose(np.linalg.norm(gradients[i].astype(np.float64)), r['gradient_norm'], rtol=1e-10, atol=1e-12)
            assert r['cross_case_gradient_maximum'] == 0 and set(r['projections']) == {'plus02', 'minus02'}
            for endpoint in ['plus02', 'minus02']:
                after = np.load(SOURCE / 'outputs/conditioning_response_v1' / endpoint / (stem + '.npy'), allow_pickle=False)
                assert np.isclose(project(gradients[i], before[i], after[i]), r['projections'][endpoint], rtol=1e-10, atol=1e-13)
    assert record['activations'] and record['encoder_floating_outputs']
    expected_dtype = 'torch.float32' if precision == 'FP32' else 'torch.float64'
    assert all(r['dtype'] == expected_dtype for r in record['encoder_floating_outputs'])
    maps = {}
    for name, r in record['activations'].items():
        assert r['dtype'] == expected_dtype and 'RELU' in r['module_type'].upper()
        path = folder / r['path']; assert path.parent == folder and sha(path) == r['packed_sha256']
        packed = np.load(path, allow_pickle=False)
        values = decode_sign_map(packed, r['shape'])
        assert r['shape'][0] == 5 and r['units'] == values.size
        assert r['positive'] == int((values > 0).sum()) and r['zero'] == int((values == 0).sum())
        assert r['negative'] == int((values < 0).sum()) and r['units'] == r['positive'] + r['zero'] + r['negative']
        maps[name] = values
    return record, v, gradients, maps


def main():
    imported = read(ANALYSIS / 'import.json'); p = read(RETURN / 'protocol.json'); old = read(BANK / 'protocol.json')
    assert imported['complete'] and sha(RETURN / 'protocol.json') == PIN == imported['protocol_sha256']
    manifest = read(RETURN / 'export_manifest.json')
    for folder, hashes in [(RETURN, manifest['files']), (INPUT, p['input_required_sha256']),
                           (SOURCE, p['response_required_sha256']), (BANK, p['bank_required_sha256'])]:
        assert all(sha(folder / n) == h for n, h in hashes.items())
    out = RETURN / 'outputs/recognizer_precision_v1'
    run = read(out / 'run_results.json') if (out / 'run_results.json').exists() else None
    failure = read(out / 'failure.json') if (out / 'failure.json').exists() else None
    exit_code = int((RETURN / 'review_exit_code.txt').read_text().strip())
    assert manifest['complete_diagnostic'] == bool(run) and manifest['failure_present'] == bool(failure)
    assert (exit_code == 0) == bool(run)
    phases = {}; comparisons = []; transitions = []; parity = []
    for precision in ['FP32', 'FP64']:
        for block in p['reference_blocks']:
            stem = f"b{block['begin']//5:03d}"
            for label in p['labels']:
                if (out / precision / (stem + '_' + label + '.json')).exists():
                    phases[(precision, block['begin'], label)] = audit_phase(out / precision, p, old, block, precision, label)
            if all((precision, block['begin'], l) in phases for l in p['labels']):
                base, vector, g, masks = phases[(precision, block['begin'], 'initial')]
                repeated, _, repeated_g, repeated_masks = phases[(precision, block['begin'], 'repeat')]
                for i, row in enumerate(base['rows']):
                    parity.append(dict(precision=precision, id=row['id'], **parity_report(repeated_g[i].reshape(-1), g[i].reshape(-1))))
                    if precision == 'FP32':
                        prior = np.load(INPUT / 'outputs/appearance_input_response_v1/repeat00' / (stem + '_RGB_gradients.npy'), allow_pickle=False)
                        parity.append(dict(precision='FP32_vs_prior_input', id=row['id'], **parity_report(g[i].reshape(-1), prior[i].reshape(-1))))
                    endpoints = {l: phases[(precision, block['begin'], l)][0]['rows'][i]['loss'] for l in ['plus02', 'minus02']}
                    magnitude = p['retained_smallest_magnitude']
                    receipt = response_receipt(row['loss'], endpoints['plus02'], endpoints['minus02'], magnitude,
                        row['projections']['plus02'], row['projections']['minus02'], abs(row['loss'] - repeated['rows'][i]['loss']))
                    comparisons.append(dict(precision=precision, id=row['id'], source=row['source'], profile=row['profile'],
                        full_network_precision_response=receipt, scientific_FP32_score_and_quality_gates_not_replaced=True))
                for label in ['plus02', 'minus02', 'repeat']:
                    changed = phases[(precision, block['begin'], label)][3]
                    assert set(changed) == set(masks)
                    for name, initial in masks.items():
                        assert changed[name].shape == initial.shape
                        transitions.append(dict(precision=precision, begin=block['begin'], label=label, module=name,
                            units=int(initial.size), changed_sign_units=int(np.count_nonzero(changed[name] != initial)),
                            per_case_changed_sign_units=[int(np.count_nonzero(changed[name][i] != initial[i])) for i in range(5)]))
                # Baseline and target remain the same in every phase of a given precision.
                for label in p['labels']:
                    np.testing.assert_array_equal(phases[(precision, block['begin'], label)][1][5:], vector[5:])
    if run:
        assert not failure and len(phases) == 16 and len(comparisons) == 20 and len(parity) == 30
        env = read(out / 'environment.json')
        assert env['GPU'] == 'NVIDIA L4' and env['torch'] == p['expected_torch'] and env['manual_tmux']
        assert env['host'].split('.')[0] == 'forensic-dgp-thesis'
        assert env['states']['FP32'] == env['FP64_state_roundtrip_to_original_FP32'] == p['recognizer_state']
        assert run['states'] == env['states'] and run['buffers'] == env['buffers']
        assert run['FP64_state_roundtrip_to_original_FP32'] == p['recognizer_state']
        assert all(env[k] is False for k in ['optimizer_constructed', 'matmul_tf32', 'cudnn_tf32', 'cudnn_benchmark']) and env['cudnn_deterministic']
        assert run['counts'] == dict(recognizer=16, gradient_queries=40, restorer=0, optimizer_updates=0, accepted_updates=0, learned_weight_updates=0)
        assert run['seconds'] < 360 and run['peak_VRAM'] <= p['peak_VRAM_bytes'] and run['peak_RSS'] <= p['peak_RSS_bytes']
        assert run['native_pixels_used'] == run['final_identity_pixels'] == run['full_epochs'] == 0 and not run['model_qualified']
        with np.load(out / 'initial_RNG.npz', allow_pickle=False) as a, np.load(out / 'restored_RNG.npz', allow_pickle=False) as b:
            assert set(a.files) == set(b.files) and all(np.array_equal(a[n], b[n]) for n in a.files)
        timing = read(out / 'FP64_timing_projection.json')
        assert timing['cap_seconds'] == 300 and timing['projected_seconds'] == timing['first_phase_seconds'] * 8 * 1.5 <= 300
    else:
        assert failure or exit_code != 0
    protected = read(ANALYSIS / 'local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(ANALYSIS / ('independent_audit.json' if run else 'failure_audit.json'), dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), protocol_sha256=PIN, full_diagnostic_returned=bool(run), failure=failure,
        phases_replayed=len(phases), precision_responses=comparisons, activation_transitions=transitions,
        original_bound_parity_reports=parity, all_original_parity_checks_pass=all(r['original_parity_check_pass'] for r in parity) if parity else None,
        frozen_neural_state_and_cross_case_zero_are_VM_receipts_not_local_neural_reexecution=True,
        scientific_FP32_and_PNG_scores_and_gates_unchanged=True, no_new_face_images_or_visual_sheets=True,
        local_neural_or_training_calls=0, model_qualified=False, more_epochs_qualified=False, goal_complete=False))
    print(dict(complete=True, diagnostic_completed=bool(run), phases=len(phases), precision_comparisons=len(comparisons), model_qualified=False))


if __name__ == '__main__':
    main()
