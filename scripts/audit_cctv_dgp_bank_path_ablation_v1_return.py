"""Independently recompute frozen saved outputs; never import the VM worker."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

from import_cctv_dgp_bank_path_ablation_v1_return import ROOT, RETURN, PARENT, ANALYSIS, PIN, read, sha, write
from audit_cctv_dgp_bank_comparison_return_v1_portable import equal_metrics, grouped, gate


def image(path, mode='RGB'):
    with Image.open(path) as im:
        assert im.size == (256, 256)
        return np.asarray(im.convert(mode)).copy()


def main():
    start = time.monotonic()
    receipt = ANALYSIS/'independent_audit.json'
    assert not receipt.exists()
    imported = read(ANALYSIS/'import.json')
    assert imported['complete'] and imported['protocol_sha256'] == PIN
    assert sha(RETURN/'protocol.json') == PIN
    p, old = read(RETURN/'protocol.json'), read(PARENT/'protocol.json')
    manifest = read(RETURN/'export_manifest.json')
    assert {f.relative_to(RETURN).as_posix() for f in RETURN.rglob('*') if f.is_file()} == set(manifest['files']) | {'export_manifest.json'}
    for name, expected in manifest['files'].items():
        assert sha(RETURN/name) == expected
    for name, expected in p['parent_required_sha256'].items():
        assert sha(PARENT/name) == expected
    out = RETURN/'outputs/path_ablation_v1'
    run = read(out/'run_results.json')
    assert manifest['complete_diagnostic'] and not manifest['failure_present'] and not (out/'failure.json').exists()
    assert run['complete'] and run['frozen_inference_only'] and run['diagnostic_not_model_qualification']
    assert not run['local_app_promotion'] and run['full_epochs'] == 0
    assert run['model_state_unchanged'] == p['rejected_B50_model_state']
    assert 0 < run['seconds'] < p['worker_seconds'] and 0 < run['peak_VRAM'] <= p['peak_VRAM_bytes']
    assert sum(f.stat().st_size for f in out.rglob('*') if f.is_file()) <= p['output_byte_cap']
    assert (RETURN/'review_exit_code.txt').read_text().strip() == '0'
    environment = read(out/'environment.json')
    assert environment['GPU'] == 'NVIDIA L4' and environment['host'].split('.')[0] == 'forensic-dgp-thesis'
    assert environment['manual_tmux'] and environment['no_new_checkpoint']
    assert environment['frozen_rejected_state_for_diagnosis_only']
    assert not list(RETURN.rglob('*.pth'))
    parity = read(out/'parity.json')
    assert parity['complete'] and parity['gradient_queries'] == 0
    assert [r['begin'] for r in parity['rows']] == p['parity_begins']
    assert len(parity['rows']) == 2 and len({r['source'] for r in parity['rows']}) == 2
    assert all(r['maximum_error'] == 0 and r['cases'] == 5 for r in parity['rows'])
    printed = [ast.literal_eval(line) for line in (RETURN/'review.log').read_text().splitlines()]
    assert [r['variant'] for r in printed] == p['variants']
    assert all(r['paired_cases'] == 100 and r['native_cases'] == 24 and 0 < r['seconds'] < p['inference_seconds'] for r in printed)
    assert all(b['seconds'] > a['seconds'] for a, b in zip(printed, printed[1:]))
    expected_counts = dict(B=7*(20+24)+4, recognizer=7*20, original=0,
                           gradient_queries=0, backwards=0, optimizer_updates=0)
    assert run['counts'] == expected_counts
    assert run['paired_variant_compositions'] == 700 and run['native_variant_compositions'] == 168
    # Only hash-verified, pure storage/measurement helpers are imported.
    sys.path.insert(0, str(PARENT))
    from cctv_dgp_head4_lossless_v1 import unpack
    from frozen_capacity_contract import feature_support, exported_pixel_metrics, detail_metric
    from frozen_raw_metrics import pixel_metrics, detail_float, deliver, mean_only
    helpers = ['cctv_dgp_head4_lossless_v1.py', 'frozen_capacity_contract.py', 'frozen_raw_metrics.py']
    helper_bindings = {name: sha(PARENT/name) for name in helpers}
    for name, expected in helper_bindings.items():
        assert expected == p['parent_required_sha256'][name]
    refs = {r['id']: r for r in old['references']}
    selected = [begin for begin in range(0, 3905, 5) if old['cases'][begin]['id'] in p['case_ids']]
    assert len(selected) == 20
    ids = [c['id'] for begin in selected for c in old['cases'][begin:begin+5]]
    assert set(ids) == set(p['case_ids']) and len(ids) == 100
    parent_out = PARENT/'outputs/bank_comparison_v1'
    base_rows = {r['id']: r for r in read(parent_out/'baseline/metrics.json')['rows']}
    cached, baseline_rows = {}, []
    for begin in selected:
        block = old['cases'][begin:begin+5]; block_ids = [c['id'] for c in block]
        base, base_emb = unpack(parent_out/'baseline/packs'/f'b{begin//5:03d}.npz', block_ids)
        data = []
        for i, case in enumerate(block):
            ref = refs[case['source_person_or_reference']]
            camera, target = image(PARENT/case['input']), image(PARENT/ref['target'])
            assert hashlib.sha256(target.tobytes()).hexdigest() == old['canonical_target_RGB_sha256'][ref['id']]
            observed = image(PARENT/ref['observed'], 'L') > 0
            support = feature_support(observed, case['landmarks5_canvas_xy'])
            png = deliver(base[i], camera, observed)
            rm, pm = pixel_metrics(base[i], target, observed), exported_pixel_metrics(png, target, observed)
            rm.update(ArcFace_observed_fixed=float(base_emb[i, 0]@base_emb[i, 2]),
                      landmark_high_frequency_MSE=detail_float(base[i], target, support), constant_mean_shift_only_MSE=rm['MSE'])
            pm.update(ArcFace_observed_fixed=float(base_emb[i, 1]@base_emb[i, 2]),
                      landmark_high_frequency_MSE=detail_metric(png, target, support), constant_mean_shift_only_MSE=pm['MSE'])
            equal_metrics(rm, base_rows[case['id']]['raw']); equal_metrics(pm, base_rows[case['id']]['png'])
            baseline_rows.append(dict(id=case['id'], source=case['source'], profile=case['profile'], raw=rm, png=pm))
            data.append((camera, target, observed, support))
        cached[begin] = (base, base_emb, data)
    baseline_groups = {stage: grouped(baseline_rows, stage) for stage in ['raw', 'png']}
    saved, summaries, native_deltas = {}, {}, {}
    replay_cases, replay_native, composition_count = 0, 0, 0
    for variant in p['variants']:
        folder = out/variant
        metrics = read(folder/'metrics.json')
        assert metrics['complete'] and metrics['variant'] == variant and metrics['model_state'] == p['rejected_B50_model_state']
        assert not metrics['model_qualified'] and metrics['full_3905_gate_not_tested']
        assert metrics['source_and_profile_metrics_TRAIN_only']
        assert [r['id'] for r in metrics['rows']] == ids
        assert {f.name for f in (folder/'packs').iterdir()} == {f'b{begin//5:03d}.npz' for begin in selected}
        assert {f.name for f in (folder/'previews').iterdir()} == {id_+'.png' for id_ in ids}
        assert {f.name for f in (folder/'native').iterdir()} == {c['id']+ext for c in old['native_development'] for ext in ['.npy', '.png']}
        rows = []
        recorded = {r['id']: r for r in metrics['rows']}
        for begin in selected:
            block = old['cases'][begin:begin+5]; block_ids = [c['id'] for c in block]
            base, base_emb, data = cached[begin]
            raw, emb = unpack(folder/'packs'/f'b{begin//5:03d}.npz', block_ids)
            assert np.allclose(np.linalg.norm(emb, axis=2), 1, rtol=0, atol=1e-5)
            assert np.array_equal(emb[:, 2], base_emb[:, 2])
            if variant in ['enabled', 'disabled']:
                label = 'B50' if variant == 'enabled' else 'B50_bank_disabled'
                prev, prev_em = unpack(parent_out/label/'packs'/f'b{begin//5:03d}.npz', block_ids, baseline=base)
                assert np.array_equal(prev, raw) and np.array_equal(prev_em, emb)
                replay_cases += 5
            for i, case in enumerate(block):
                camera, target, observed, support = data[i]; a = raw[i]
                assert np.array_equal(a[~observed], (camera.astype(np.float32)/np.float32(255))[~observed])
                png = deliver(a, camera, observed)
                assert np.array_equal(png, image(folder/'previews'/(case['id']+'.png')))
                mean_raw, mean_png, shift = mean_only(a, base[i], camera, observed)
                rm, pm = pixel_metrics(a, target, observed), exported_pixel_metrics(png, target, observed)
                rm.update(ArcFace_observed_fixed=float(emb[i, 0]@emb[i, 2]), landmark_high_frequency_MSE=detail_float(a, target, support),
                          constant_mean_shift_only_MSE=pixel_metrics(mean_raw, target, observed)['MSE'])
                pm.update(ArcFace_observed_fixed=float(emb[i, 1]@emb[i, 2]), landmark_high_frequency_MSE=detail_metric(png, target, support),
                          constant_mean_shift_only_MSE=exported_pixel_metrics(mean_png, target, observed)['MSE'])
                row = recorded[case['id']]
                assert row['source'] == case['source'] and row['profile'] == case['profile']
                equal_metrics(rm, row['raw']); equal_metrics(pm, row['png'])
                assert np.allclose(shift, row['postclip_mean_RGB_shift'], rtol=0, atol=1e-14)
                assert hashlib.sha256(a.tobytes()).hexdigest() == row['raw_RGB_float32_sha256']
                assert hashlib.sha256(png.tobytes()).hexdigest() == row['PNG_RGB_sha256']
                rows.append(dict(id=case['id'], source=case['source'], profile=case['profile'], raw=rm, png=pm))
                composition_count += 1
        rebuilt = {stage: grouped(rows, stage) for stage in ['raw', 'png']}
        for stage in rebuilt:
            assert set(rebuilt[stage]) == set(metrics['groups'][stage])
            for key, value in rebuilt[stage].items():
                equal_metrics(value, metrics['groups'][stage][key])
        assert [r['id'] for r in metrics['native_unpaired_rows']] == p['native_case_ids']
        native_deltas[variant] = []
        for case, row in zip(old['native_development'], metrics['native_unpaired_rows']):
            camera, mask = image(PARENT/case['input']), image(PARENT/case['observed'], 'L') > 0
            raw = np.load(folder/'native'/(case['id']+'.npy'), allow_pickle=False)
            assert raw.dtype == np.float32 and raw.shape == (256, 256, 3) and np.isfinite(raw).all()
            assert 0 <= raw.min() <= raw.max() <= 1
            assert np.array_equal(raw[~mask], (camera.astype(np.float32)/np.float32(255))[~mask])
            png = deliver(raw, camera, mask)
            assert np.array_equal(png, image(folder/'native'/(case['id']+'.png')))
            assert row['unpaired_no_clean_metrics']
            assert hashlib.sha256(raw.tobytes()).hexdigest() == row['raw_RGB_float32_sha256']
            assert hashlib.sha256(png.tobytes()).hexdigest() == row['PNG_RGB_sha256']
            if variant in ['enabled', 'disabled']:
                label = 'B50' if variant == 'enabled' else 'B50_bank_disabled'
                assert np.array_equal(raw, np.load(parent_out/label/'native'/(case['id']+'.npy'), allow_pickle=False))
                assert np.array_equal(png, image(parent_out/label/'native'/(case['id']+'.png')))
                replay_native += 1
            enabled = np.load(out/'enabled/native'/(case['id']+'.npy'), allow_pickle=False)
            delta = (raw-enabled)[mask].astype(np.float64)
            native_deltas[variant].append(dict(id=case['id'], mean_RGB_difference_from_enabled=delta.mean(0).tolist(),
                RMS_difference_from_enabled=float(np.sqrt(np.square(delta).mean())), unpaired=True, quality_metric=False))
        summaries[variant] = {stage: dict(gate(baseline_groups[stage], rebuilt[stage]),
                                             only_100_TRAIN_preview_cases=True, model_qualified=False) for stage in rebuilt}
        saved[variant] = rebuilt
        print(dict(variant=variant, paired_checked=len(rows), native_checked=24,
                   raw_gain_percent=100*summaries[variant]['raw']['gain'], raw_failures=len(summaries[variant]['raw']['failures'])), flush=True)
    assert composition_count == 700 and replay_cases == 200 and replay_native == 48
    exposed = {old['cases'][i]['source_person_or_reference'] for batch in old['schedule'] for i in batch}
    reference_ids = {old['cases'][begin]['source_person_or_reference'] for begin in selected}
    for name, expected in manifest['files'].items():
        assert sha(RETURN/name) == expected
    for name, expected in read(ANALYSIS/'local_protected_before.json')['files_sha256'].items():
        assert sha(ROOT/name) == expected
    write(receipt, dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(__file__),
        import_sha256=sha(ANALYSIS/'import.json'), protocol_sha256=PIN, export_manifest_sha256=sha(RETURN/'export_manifest.json'),
        helper_sha256=helper_bindings, original_thresholds_unchanged=True, saved_paired_raw_PNG_compositions_checked=700,
        independently_recomputed_baseline_cases=100, native_compositions_checked=168,
        exact_parent_paired_raw_embedding_replays=200, exact_parent_native_raw_PNG_replays=48,
        VM_counts_recomputed=expected_counts, VM_seconds=run['seconds'], VM_peak_VRAM=run['peak_VRAM'],
        zero_all_readout_parity_receipt=parity, run_results_sha256=sha(out/'run_results.json'),
        diagnostic_finished=True, new_checkpoint_created=False, model_state_unchanged=p['rejected_B50_model_state'],
        baseline_groups=baseline_groups, variant_groups=saved, partial_preview_gate_diagnostics=summaries,
        native_variant_differences=native_deltas, paired_TRAIN_references=20, exposed_TRAIN_references=len(reference_ids&exposed),
        unexposed_TRAIN_is_not_DEV_or_final=True, native_DEV_unpaired_cases=24, final_identity_pixels_used=0,
        full_3905_quality_gate_tested=False, ablations_are_not_independently_trained_controls=True,
        local_neural_forwards=0, local_gradients=0, local_backwards=0, local_optimizer_updates=0,
        limited_frozen_CPU_replay_pending=True, visual_review_pending=True, independent_final_review_pending=True,
        model_qualified=False, app_promotion=False, all_seven_completion_families_pending=True,
        goal_complete=False, seconds=time.monotonic()-start))
    print(dict(complete=True, paired_compositions=700, native_compositions=168, model_qualified=False), flush=True)


if __name__ == '__main__':
    main()
