"""Saved-only audit of the ten completed cases from an operator-interrupted run.

No inference/HTTP/browser/VM. Preserve the68-case protocol and all failure bytes.
"""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / 'scripts/audit_codeformer_spatial_dev_local_v1_r2.py'
HELPER_SHA = '619a7863031ede9d1dcb78fa0bd116014e514cfc7ee1e5fb3d4c7155bc91fb2b'
RELEASE = 'outputs/codeformer_spatial_dev_review_v1/interactive_release_r1.json'
RELEASE_SHA = '1651e8385b105cd2e532ecd6931ccb4e2595a29347e4b117fc81f11a8b93eda2'
RUN = ROOT / 'outputs/codeformer_spatial_dev_review_v1/run_v1_r2'
OUT = ROOT / 'outputs/codeformer_spatial_dev_review_v1/independent_partial_saved_output_audit_v1.json'


def main():
    started = time.monotonic()
    assert not OUT.exists(), 'Preserve existing partial audit'
    assert hashlib.sha256(HELPER.read_bytes()).hexdigest() == HELPER_SHA
    spec = importlib.util.spec_from_file_location('independent_spatial_saved_audit_r2', HELPER)
    a = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(a)
    need = a.need
    release = a.unique_json(a.checked_bytes(RELEASE, RELEASE_SHA))
    comparison = a.unique_json(a.bound(release['running_receipt_not_mutated']))
    cohort = a.unique_json(a.checked_bytes(a.COHORT, a.COHORT_SHA))
    manifest = a.unique_json(a.checked_bytes(a.MODEL_MANIFEST, a.MODEL_MANIFEST_SHA))
    need(comparison['complete'] is False and len(comparison['cases']) == 10,
         'Exact interrupted prefix of10 completed cases required')
    need(release['planned_cases'] == 68 and release['complete_saved_cases'] == 10 and
         release['own_saved_successful_responses'] == 40 and release['no_automatic_resume'] is True,
         'Corrected interruption scope differs')
    cases = cohort['cases'][:10]
    need([r['id'] for r in comparison['cases']] == [r['id'] for r in cases], 'Prefix membership differs')
    need(release['pending_case_ids'] == [r['id'] for r in cohort['cases'][10:]], '58pending membership differs')
    need(len(cohort['cases']) == 68 and len(release['pending_case_ids']) == 58 and
         sum(r.get('appearance_flag', False) for r in cohort['cases']) == 9,
         'Frozen68 and flagged9 must remain intact')
    need(all(r['cohort'] == 'filter_DEV' for r in cases), 'Completed prefix unexpectedly includes native')
    controls = cohort['native_input_only_controls'] + cohort['filter_insufficient_controls']
    need(len(controls) == 40 and all(r['neural_calls_allowed'] == 0 for r in controls) and
         not any((RUN / r['id']).exists() for r in controls), 'Input-only refusal controls changed')
    a.bound(comparison['worker'])
    a.bound(comparison['app_source'])
    need(comparison['app_source']['sha256'] == a.APP_SHA and
         comparison['transport_version'] == 'single-arm-persistent-exact-body-r2', 'Worker/app binding differs')
    for arm in ('spatial_u', 'spatial_s'):
        a.bound(manifest['arms'][arm]['export'])
    results, differences = [], []
    transport_files, case_files = [], []
    for row, saved in zip(cases, comparison['cases']):
        folder = RUN / row['id']
        need({p.name for p in folder.iterdir()} == a.EXPECTED_CASE_FILES, 'Exact10 saved case files required')
        need(len(saved['files']) == 10 and {p['path'] for p in saved['files']} ==
             {p.relative_to(ROOT).as_posix() for p in folder.iterdir()}, 'Case file manifest differs')
        for binding in saved['files']:
            a.bound(binding)
            case_files.append(binding)
        for key in ('id', 'source', 'profile', 'evidence', 'split', 'reference_id'):
            need(saved[key] == row[key], 'Completed-case provenance differs')
        need(saved['input_review_retained'] == row['input_review'] and saved['native_PSNR_SSIM'] is None and
             saved['appearance_flag'] is row.get('appearance_flag', False) and saved['no_optimization'] is True,
             'Frozen input-only labels or evidence scope differs')
        need(row['target'] is not None and row['evidence'] == 'paired_synthetic_fixed_filter_stress',
             'Only paired filter prefix expected')
        upload = a.bound(row['upload_source'])
        canvas = a.rgb(a.bound(row['input']))
        support = a.support_png(a.bound(row['support']))
        target = a.rgb(a.bound(row['target'])).astype(np.float32) / np.float32(255)
        recreated, observed, geometry = a.prepare_independent(a.upload_rgb(upload))
        need(np.array_equal(canvas, recreated) and np.array_equal(support, observed) and
             geometry == row['geometry'], 'Independent original/canvas/support geometry differs')
        meta = a.unique_json(a.checked_bytes(folder / 'app-response-metadata.json'))
        transport_files.extend(a.verify_transport(RUN, row, meta, upload, canvas, support))
        need(meta['model_qualified'] is False and meta['main_app_default_changed'] is False and
             meta['inference_only'] is True and meta['device'] == 'cpu' and
             meta['operator_input_review_is_not_a_validated_automatic_threshold'] is True,
             'Experimental inference scope differs')
        arrays = {}
        own = {'resize_control': a.metrics(canvas.astype(np.float32) / np.float32(255), target, support)}
        with zipfile.ZipFile(io.BytesIO(a.checked_bytes(folder / 'app-comparison.zip'))) as z:
            need(len(z.namelist()) == 12 and set(z.namelist()) == a.EXPECTED_ZIP and z.testzip() is None,
                 'Exact12-member merged ZIP CRC/schema differs')
            need(a.unique_json(z.read('processing.json')) == meta, 'Merged processing differs')
            for arm in a.ARMS:
                raw_data, png_data = z.read(arm + '-raw.npy'), z.read(arm + '.png')
                need(raw_data == a.checked_bytes(folder / (arm + '-raw.npy')) and
                     png_data == a.checked_bytes(folder / (arm + '.png')), 'Merged/unpacked bytes differ')
                raw, image = a.raw_array(raw_data), a.rgb(png_data)
                need(np.array_equal(np.floor(raw * np.float32(255)).astype(np.uint8), image), 'PNG floor differs')
                need(np.array_equal(raw[~support], (canvas.astype(np.float32) / np.float32(255))[~support]) and
                     np.array_equal(image[~support], canvas[~support]), 'Protected outside-support pixels changed')
                declaration = meta['outputs'][arm]
                need(a.digest(raw.tobytes()) == declaration['raw_rgb_sha256'] and
                     a.digest(image.tobytes()) == declaration['PNG_RGB8_sha256'] and
                     a.digest(png_data) == declaration['PNG_file_sha256'], 'RAW/PNG hashes differ')
                p = declaration['metadata']
                need(p['arm'] == arm and p['fidelity'] == 1 and p['adain'] is True and
                     p['internal_resolution'] == 512 and p['output_resolution'] == 256 and
                     p['alignment'] == 'none' and p['display_processing'] == 'none' and
                     p['model_qualified'] is False and p['main_app_default_changed'] is False,
                     'Runtime/model qualification fields differ')
                if arm == 'original':
                    need(p['weights_sha256'] == a.ORIGINAL_WEIGHTS_SHA and
                         p['model_tensor_sha256'] == a.ORIGINAL_TENSOR_SHA, 'Original benchmark differs')
                elif arm == 'r2':
                    need(p['weights_sha256'] == a.R2_WEIGHTS_SHA and p['model_tensor_sha256'] == a.R2_TENSOR_SHA,
                         'R2 model differs')
                else:
                    expected = manifest['arms'][arm]
                    need(p['weights_sha256'] == expected['export']['sha256'] and
                         p['adapter_tensor_sha256'] == expected['adapter_tensor_sha256'] and
                         p['source_checkpoint_sha256'] == expected['source_checkpoint_sha256'] and
                         p['base_tensor_sha256'] == a.R2_TENSOR_SHA and p['source_training_updates'] == 96 and
                         p['additional_continuation_updates'] == 72 and p['all515_R2_tensors_frozen'] is True,
                         'Spatial adapter model/base/update provenance differs')
                arrays[arm] = {'RAW': raw, 'PNG': image.astype(np.float32) / np.float32(255)}
                own[arm] = {view: a.metrics(arrays[arm][view], target, support) for view in ('RAW', 'PNG')}
        for arm in a.ARMS:
            own[arm]['diagnostic_distance_to_R2'] = {view: float(np.abs(
                arrays[arm][view][support].astype(np.float64) -
                arrays['r2'][view][support].astype(np.float64)).mean(dtype=np.float64)) for view in ('RAW', 'PNG')}
        a.compare_numeric(saved['target_metrics'], own, row['id'], differences)
        results.append({'id': row['id'], 'reference_id': row['reference_id'], 'profile': row['profile'],
            'input_review': row['input_review'], 'appearance_flag': row.get('appearance_flag', False),
            'evidence': row['evidence'], 'target_metrics': own, 'geometry_RAW_PNG_origin_joins_pass': True})
    need(len(transport_files) == 160 and len(case_files) == 100, 'Exact partial payload count differs')
    degraded = [r for r in results if r['profile'] != 'clear']
    clear = [r for r in results if r['profile'] == 'clear']
    need(len(degraded) == 8 and len(clear) == 2 and
         {r['reference_id'] for r in degraded} == {'va_ffhq_00383'}, 'Descriptive group scope differs')
    summaries = {name: {arm: {view: a.scalar_summary([r['target_metrics'][arm][view] for r in rows])
                             for view in ('RAW', 'PNG')} for arm in a.ARMS}
                 for name, rows in [('degraded8_one_photo00383', degraded), ('clear2_photos00383_01093', clear)]}
    expected_mae = {'original': .02977987601299533, 'r2': .04849834399978044,
                    'spatial_u': .045698894714987524, 'spatial_s': .04817757532417962}
    expected_ssim = {'original': .8074822832100863, 'r2': .784620205729249,
                     'spatial_u': .7886876075039553, 'spatial_s': .7850957314941518}
    for arm in a.ARMS:
        mean = summaries['degraded8_one_photo00383'][arm]['PNG']
        a.compare_numeric(expected_mae[arm], mean['mae']['mean'], 'root_descriptive_MAE/' + arm, differences)
        a.compare_numeric(expected_ssim[arm], mean['ssim']['mean'], 'root_descriptive_SSIM/' + arm, differences)
    feedback_path = ROOT / 'outputs/codeformer_spatial_dev_review_v1/user_visual_feedback_v1.json'
    feedback = a.unique_json(a.checked_bytes(feedback_path))
    need(feedback['U96_S96_user_visible_upgrade_accepted'] is False and feedback['restoration_accepted'] is False,
         'Retained user acceptance status differs')
    need('torch' not in sys.modules, 'Saved-only audit imported Torch')
    result = {'format': 'codeformer-spatial-partial-development-saved-output-audit-v1',
        'complete': True, 'audit_pass': True, 'scope': 'Artifact audit of10savedcases only; interrupted68case experiment remains incomplete',
        'auditor': a.pin(Path(__file__)), 'independent_metric_helper': a.pin(HELPER),
        'corrected_release': a.pin(ROOT / RELEASE), 'comparison': a.pin(RUN / 'comparison.json'),
        'cohort': a.pin(ROOT / a.COHORT), 'user_feedback': a.pin(feedback_path),
        'counts': {'planned_cases': 68, 'saved_completed_cases_audited': 10, 'pending_cases': 58,
            'saved_single_arm_responses': 40, 'original_server_6member_ZIPs': 40,
            'client_merged_12member_ZIPs': 10, 'RAW_outputs': 40, 'PNG_outputs': 40,
            'verified_transport_files': 160, 'verified_case_files': 100, 'paired_degraded_observations': 8,
            'paired_degraded_reference_photographs': 1, 'clear_controls': 2, 'native_completed_cases': 0,
            'flagged08025_completed_cases': 0, 'flagged08025_pending_cases': 9,
            'native_input_only_controls': 16, 'filter_insufficient_controls': 24,
            'additional_R2_completed_undelivered_forward_not_saved_case': 1,
            'prior_V1_R1_completed_undelivered_forwards_not_saved_results': 5},
        'interruption_is_not_quality_stop_or_full_run_completion': True,
        'only_exact_launcher_PID28996_certified_directly_stopped': True,
        'server_PID32392_kept_live_by_release_record': True,
        'pending_case_ids': release['pending_case_ids'], 'native_results_inferred': False,
        'independent_metric_max_absolute_difference': max(differences),
        'arithmetic_tolerance_only_not_quality_threshold_change': {'absolute': a.ABS_TOLERANCE, 'relative': a.REL_TOLERANCE},
        'paired_descriptive_summaries': summaries,
        'degraded8_pairwise_PNG_MAE_counts': {arm: {
            'lower_than_R2': sum(r['target_metrics'][arm]['PNG']['mae'] < r['target_metrics']['r2']['PNG']['mae'] for r in degraded),
            'higher_than_original': sum(r['target_metrics'][arm]['PNG']['mae'] > r['target_metrics']['original']['PNG']['mae'] for r in degraded)}
            for arm in ('spatial_u', 'spatial_s')},
        'cases': results, 'model_qualified': False, 'restoration_accepted': False,
        'user_visible_upgrade_accepted': False, 'quality_gates_changed': False,
        'full_development_comparison_complete': False, 'independent_FINAL_review': False,
        'goal_complete': False, 'original68_case_manifest_preserved': True,
        'audit_neural_HTTP_gradient_optimizer_VM_browser_calls': 0, 'torch_imported': False,
        'seconds': time.monotonic() - started}
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    print(json.dumps({'complete': True, 'audit_pass': True, 'counts': result['counts'],
        'PNGdegraded8': {arm: {k: summaries['degraded8_one_photo00383'][arm]['PNG'][k]['mean']
                              for k in ('mae', 'ssim')} for arm in a.ARMS},
        'max_metric_difference': max(differences), 'seconds': result['seconds'],
        'receipt': str(OUT), 'sha256': hashlib.sha256(OUT.read_bytes()).hexdigest()}, indent=2), flush=True)


if __name__ == '__main__':
    main()
