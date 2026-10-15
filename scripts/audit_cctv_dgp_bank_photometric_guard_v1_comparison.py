"""Compare the returned brightness-loss trial with audited B r4 saved evidence.

This performs array/state arithmetic only. It never imports a VM trainer or
performs neural forwards, gradients, backwards, updates or gate changes.
"""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_return'
CURRENT = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_return'
ANALYSIS = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_analysis'
OLD_ANALYSIS = ROOT/'outputs/cctv_dgp_bank_comparison_v1_r4_B_analysis'
PIN = 'f64154bd602070edaea61ccbf960ec3028e545c6540a4421ec41f9af21128a12'
OLD_PIN = '8c906e8b1a3b9534872313414321a2c780560d9ea5ab50fec3e34232813cad85'


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def aggregate(rows, stage):
    assert rows
    return {key: float(np.mean([row[stage][key] for row in rows])) for key in
            ['MSE', 'SSIM', 'ArcFace_observed_fixed', 'landmark_high_frequency_MSE']}


def difference(before, after):
    return dict(MSE_change=after['MSE']-before['MSE'],
                SSIM_change=after['SSIM']-before['SSIM'],
                observed_recognizer_similarity_change=after['ArcFace_observed_fixed']-before['ArcFace_observed_fixed'],
                relative_landmark_detail_gain=1-after['landmark_high_frequency_MSE']/before['landmark_high_frequency_MSE'])


def main():
    started = time.monotonic()
    receipt = ANALYSIS/'matched_guard_comparison.json'
    assert not receipt.exists()
    torch.set_num_threads(4)
    bindings = {}

    def bind(path):
        bindings[str(path.resolve())] = sha(path)

    independent = []
    protocols = []
    for root, analysis, pin in [(OLD, OLD_ANALYSIS, OLD_PIN), (CURRENT, ANALYSIS, PIN)]:
        assert sha(root/'protocol.json') == pin
        p = read(root/'protocol.json')
        assert p['selected_routes'] == ['B']
        protocols.append(p)
        for name in ['import.json', 'execution_audit.json', 'independent_audit.json']:
            path = analysis/name
            record = read(path)
            assert record['complete'] and record['protocol_sha256'] == pin
            bind(path)
        audit = read(analysis/'independent_audit.json')
        assert audit['optimizer_updates_in_VM'] == 50
        assert audit['saved_paired_raw_PNG_cases_checked'] == 7910
        assert audit['native_compositions_checked'] == 72
        assert len(audit['limited_CPU_replays']) == 2
        assert audit['CPU_CUDA_prior_fixtures_rechecked'] == 6
        assert audit['original_thresholds_and_gate_decisions_unchanged_required']
        assert audit['local_optimizer_updates'] == audit['local_gradient_queries'] == audit['local_backwards'] == 0
        assert sha(root/'export_manifest.json') == audit['manifest_sha256']
        bind(root/'protocol.json')
        bind(root/'export_manifest.json')
        independent.append(audit)
    before, p = protocols
    common = ['cases', 'references', 'schedule', 'native_development', 'preview_case_ids',
              'canonical_target_RGB_sha256', 'original_checkpoint_sha256', 'original_state',
              'recognizer_state', 'standalone_prior_sha256', 'mean_style_sha256', 'scientific_thresholds',
              'normalizers', 'active_anchor_normalizers', 'reconstruction_weights',
              'regression_barrier_coefficient', 'active_clear_blur_anchor_coefficients', 'optimizer',
              'maximum_updates_per_route', 'full_epochs_per_route', 'fitting_reference_exposures_per_route',
              'fitting_image_exposures_per_route', 'reserved_final_pixels', 'budgets', 'visual_review_plan']
    for name in common:
        assert before[name] == p[name], name
    assert p['reserved_final_pixels'] == 0
    for name, expected in before['assets_sha256'].items():
        if name.startswith(('data/', 'weights/', 'initializers/')):
            assert expected == p['assets_sha256'][name]
            assert sha(OLD/name) == sha(CURRENT/name) == expected
    transfer_path = ROOT/'outputs/cctv_dgp_bank_photometric_guard_v1_transfer_audit.json'
    transfer = read(transfer_path)
    assert transfer['complete'] and transfer['protocol_sha256'] == PIN
    assert transfer['model_computation_AST_unchanged']
    assert transfer['initializer_weights_frozen_assets_optimizer_data_schedule_gates_caps_unchanged']
    bind(transfer_path)
    sys.path.insert(0, str(CURRENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_raw_metrics import mean_only
    from PIL import Image

    def image(path, mode='RGB'):
        with Image.open(path) as im:
            assert im.size == (256, 256)
            return np.asarray(im.convert(mode)).copy()

    def mse(actual, target, mask):
        # Preserve the recorded float32 error/square and float64 reduction.
        # No SSIM is needed for hinge arithmetic; the full audit checks SSIM.
        error = actual-target.astype(np.float32)/np.float32(255)
        return float(np.square(error[mask]).astype(np.float64).mean())

    refs = {r['id']: r for r in p['references']}
    out = CURRENT/'outputs/bank_comparison_v1'
    old_out = OLD/'outputs/bank_comparison_v1'
    metrics = {}
    for label, path in [('baseline', out/'baseline/metrics.json'), ('B4', old_out/'B50/metrics.json'),
                        ('guard', out/'B50/metrics.json'), ('disabled', out/'B50_bank_disabled/metrics.json')]:
        bind(path)
        metrics[label] = read(path)
    maps = {label: {r['id']: r for r in data['rows']} for label, data in metrics.items()}
    hinge_rows = []
    exact_packs = 0
    for begin in range(0, 3905, 5):
        cases = p['cases'][begin:begin+5]
        ids = [c['id'] for c in cases]
        packed = [unpack(folder/'baseline/packs'/f'b{begin//5:03d}.npz', ids)
                  for folder in [old_out, out]]
        assert np.array_equal(packed[0][0], packed[1][0])
        assert np.array_equal(packed[0][1], packed[1][1])
        exact_packs += 1
        baseline = packed[1][0]
        candidates = {label: unpack(folder/'B50/packs'/f'b{begin//5:03d}.npz', ids, baseline=baseline)[0]
                      for label, folder in [('B4', old_out), ('guard', out)]}
        for i, case in enumerate(cases):
            if case['profile'] == 'clear':
                continue
            camera = image(CURRENT/case['input'])
            ref = refs[case['source_person_or_reference']]
            target = image(CURRENT/ref['target'])
            mask = image(CURRENT/ref['observed'], 'L') > 0
            base_mse = mse(baseline[i], target, mask)
            assert np.isclose(base_mse, maps['baseline'][case['id']]['raw']['MSE'], rtol=0, atol=1e-13)
            row = dict(id=case['id'], source=case['source'], profile=case['profile'])
            for label, arrays in candidates.items():
                shifted, _, _ = mean_only(arrays[i], baseline[i], camera, mask)
                candidate_mse = mse(arrays[i], target, mask)
                mean_mse = mse(shifted, target, mask)
                assert np.isclose(candidate_mse, maps[label][case['id']]['raw']['MSE'], rtol=0, atol=1e-13)
                assert np.isclose(mean_mse, maps[label][case['id']]['raw']['constant_mean_shift_only_MSE'], rtol=0, atol=1e-13)
                mean_gain = max(0, base_mse-mean_mse)
                total_gain = max(0, base_mse-candidate_mse)
                excess = max(0, mean_gain-.2*total_gain)
                row[label] = dict(positive_mean_only_gain=mean_gain, positive_total_gain=total_gain,
                                  positive_guard_excess=excess, per_image_hinge_active=excess > 0)
            hinge_rows.append(row)
        if exact_packs % 200 == 0:
            print(dict(matched_baseline_packs=exact_packs, of=781), flush=True)
    assert len(hinge_rows) == 3124
    for case in p['native_development']:
        paths = [folder/'baseline/native'/(case['id']+'.npy') for folder in [old_out, out]]
        assert np.array_equal(np.load(paths[0], allow_pickle=False), np.load(paths[1], allow_pickle=False))
        assert sha(paths[0].with_suffix('.png')) == sha(paths[1].with_suffix('.png'))
    tables = {}
    for stage in ['raw', 'png']:
        base_groups = metrics['baseline']['groups'][stage]
        tables[stage] = {
            group: {label: difference(base_groups[group], metrics[label]['groups'][stage][group])
                    for label in ['B4', 'guard']} for group in base_groups}
    exposed = {p['cases'][i]['source_person_or_reference'] for batch in p['schedule'] for i in batch}
    assert len(exposed) == 250
    partitions = {}
    for name, selected_flag in [('exposed_TRAIN', True), ('unexposed_same_TRAIN_pool', False)]:
        ids = [c['id'] for c in p['cases'] if c['profile'] != 'clear' and
               (c['source_person_or_reference'] in exposed) == selected_flag]
        partitions[name] = dict(cases=len(ids), references=len(ids)//4,
            paired_TRAIN_not_independent_DEV=True, stages={stage: {
                label: difference(aggregate([maps['baseline'][i] for i in ids], stage),
                                  aggregate([maps[label][i] for i in ids], stage))
                for label in ['B4', 'guard']} for stage in ['raw', 'png']})
    guard_summary = {}
    for profile in ['degraded'] + p['photometric_guard']['profiles']:
        selected = [r for r in hinge_rows if profile == 'degraded' or r['profile'] == profile]
        guard_summary[profile] = {label: dict(cases=len(selected),
            active_hinges=sum(r[label]['per_image_hinge_active'] for r in selected),
            mean_positive_excess=float(np.mean([r[label]['positive_guard_excess'] for r in selected])),
            positive_mean_gain=float(np.mean([r[label]['positive_mean_only_gain'] for r in selected])),
            positive_total_gain=float(np.mean([r[label]['positive_total_gain'] for r in selected])))
            for label in ['B4', 'guard']}

    initial_path = CURRENT/'initializers/initial_B.pth'
    state_path = out/'state_B50/full_state.pth'
    conditioning_path = out/'B_conditioning_after_update1.json'
    for path in [initial_path, state_path, conditioning_path]:
        bind(path)
    initial = torch.load(initial_path, map_location='cpu', weights_only=True)
    state = torch.load(state_path, map_location='cpu', weights_only=True)
    gradients = read(conditioning_path)
    names = set(gradients['names'])
    assert len(names) == p['routes']['B']['tensors'] == 37 and gradients['all_nonzero']
    assert all(np.isfinite(v) and v > 0 for v in gradients['summed_norms'])
    moved = []
    frozen = 0
    assert set(initial) == set(state['model'])
    for name, old in initial.items():
        new = state['model'][name]
        assert old.shape == new.shape and old.dtype == new.dtype
        assert torch.isfinite(old).all() and torch.isfinite(new).all()
        if name not in names:
            assert torch.equal(old, new), name
            frozen += 1
            continue
        moved.append(dict(name=name, partition='current_reconstruction' if name.startswith('current.') else name.split('.')[0],
            elements=old.numel(), change_L2_norm=float((new.double()-old.double()).norm()),
            maximum_absolute_change=float((new.double()-old.double()).abs().max()),
            bitwise_changed=not torch.equal(old, new)))
    assert all(r['bitwise_changed'] for r in moved)
    history_path = out/'history_B.json'
    bind(history_path)
    history = read(history_path)['rows']
    assert len(history) == 50
    assert all(np.isfinite(r['photometric_guard']) and r['photometric_guard'] >= 0 for r in history)
    comparison_records = []
    for row in history:
        assert row['objective'] >= row['photometric_guard']-1e-12
        comparison_records.append(dict(update=row['update'], contribution=row['photometric_guard'],
            fraction_of_objective=row['photometric_guard']/row['objective'] if row['objective'] else 0))
    for name, expected in read(ANALYSIS/'local_protected_before.json')['files_sha256'].items():
        assert sha(ROOT/name) == expected, name
    for path, expected in bindings.items():
        assert sha(path) == expected, path
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        protocol_A_sha256=OLD_PIN, protocol_B_sha256=PIN,
        comparison_labels=['rejected_B_r4', 'brightness_guard_v1'], compared_receipt_bindings=bindings,
        matched_protocol_fields=common, unchanged_baseline_RGB_embedding_packs=exact_packs,
        unchanged_native_baseline_cases=24, optimizer_updates_each=50, references_each=250,
        full_epochs_each=0, initializer_architecture_and_frozen_prior_unchanged=True,
        original_gate_decisions={label: a['metric_and_gate_decisions'] for label, a in
                                zip(['B4', 'guard'], independent)},
        source_profile_changes=tables, exposure_partitions=partitions,
        final_snapshot_per_image_guard_math=guard_summary,
        final_guard_math_does_not_reconstruct_per_update_losses=True,
        history_guard_contributions=comparison_records,
        nonzero_guard_updates=sum(r['photometric_guard'] > 0 for r in history),
        first10_mean_objective=float(np.mean([r['objective'] for r in history[:10]])),
        last10_mean_objective=float(np.mean([r['objective'] for r in history[-10:]])),
        different_batches_do_not_prove_plateau_or_convergence=True,
        changed_trainable_tensors=moved, exact_frozen_tensors=frozen,
        changed_tensor_counts=dict(Counter(r['partition'] for r in moved)),
        baseline_comparison_not_a_new_incumbent=True, final_identities_used=False,
        local_model_forwards=0, local_gradients=0, local_optimizer_updates=0,
        quality_qualified=False, app_promotion=False, goal_complete=False,
        seconds=time.monotonic()-started)
    write(receipt, record)
    print(dict(complete=True, matched_baselines=exact_packs, changed_tensors=len(moved),
               frozen_tensors=frozen, local_training_calls=0), flush=True)


if __name__ == '__main__':
    main()
