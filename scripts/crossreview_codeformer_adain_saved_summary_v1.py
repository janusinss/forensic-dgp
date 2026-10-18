"""Recalculate summary groups/deltas and verify every rendered source cell, CPU only."""
from collections import Counter
import hashlib
import math
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_adain_gradient_isolation_v1.return_archive import read, sha, require, write_new


def near(a, b): require(math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-14), 'Summary numeric readback differs')


def main():
    review = ROOT / 'outputs/codeformer_adain_gradient_isolation_v1_return_review'
    metricroot = review / 'metrics_v1'; summarypath = review / 'scientific_readback_v1.json'
    reportpath = review / 'SCIENTIFIC_READBACK_V1.md'
    summary = read(summarypath); rows = read(metricroot / 'metrics.json')['rows']
    require(summary['complete'] and not summary['model_qualified'] and not summary['new_training_authorized'], 'Summary status differs')
    require(summary['source_sha256'] == sha(ROOT / 'scripts/summarize_codeformer_adain_saved_pixels_v1.py'), 'Summary source differs')
    for name, digest in summary['metrics_files_sha256'].items(): require(sha(metricroot / name) == digest, 'Summary input changed')
    require(summary['prior_binding_sha256'] == sha(review / 'prior_comparison_binding_v1.json'), 'Prior binding differs')
    grouped = {}
    for row in rows:
        key = tuple(row[name] for name in ('state', 'view', 'source', 'evidence', 'profile'))
        grouped.setdefault(key, []).append(row)
    require(len(grouped) == len(summary['full_source_profile_metrics']), 'Metric group census differs')
    group_checks = 0; metric_valid_counts = []
    for group in summary['full_source_profile_metrics']:
        key = tuple(group[name] for name in ('state', 'view', 'source', 'evidence', 'profile'))
        selected = grouped[key]; require(group['cases'] == len(selected), 'Metric group size differs')
        valid = {}
        for name, expected in group.items():
            if not name.startswith('mean_'): continue
            field = name[5:]; values = [row[field] for row in selected if row[field] is not None]
            valid[field] = len(values)
            if expected is None: require(not values, 'Native group has invented reference mean')
            else: require(bool(values), 'Missing available paired metric'); near(math.fsum(values) / len(values), expected)
        metric_valid_counts.append({'group': dict(zip(('state', 'view', 'source', 'evidence', 'profile'), key)),
                                    'total_cases': len(selected), 'available_per_metric': valid})
        group_checks += 1
    delta_checks = 0
    for group in summary['full_source_profile_deltas']:
        lhs, rhs = group['comparison'].split('_minus_')
        key = tuple(group[name] for name in ('view', 'source', 'evidence', 'profile'))
        left, right = grouped[(lhs, *key)], grouped[(rhs, *key)]
        other = {row['id']: row for row in right}; require({row['id'] for row in left} == set(other), 'Delta case namespace differs')
        require(group['cases'] == len(left), 'Delta group count differs')
        for name, expected in group.items():
            if not name.startswith('mean_delta_'): continue
            field = name[len('mean_delta_'):]
            if expected is None:
                require(all(row[field] is None and other[row['id']][field] is None for row in left), 'Native delta is invented')
                continue
            deltas = [row[field] - other[row['id']][field] for row in left
                      if row[field] is not None and other[row['id']][field] is not None]
            near(math.fsum(deltas) / len(deltas), expected)
            require(group['cases_lower_' + field] == sum(value < 0 for value in deltas) and
                group['cases_higher_' + field] == sum(value > 0 for value in deltas), 'Delta sign count differs')
        delta_checks += 1
    native_counts = Counter(row['source'] for row in rows if row['evidence'] == 'native_unpaired')
    require(dict(native_counts) == summary['native_sources'] == {'ChokePoint': 144, 'QMUL-SurvFace': 144}, 'Native metric row count differs')
    clear = summary['clear_failed_case_ids']; require(clear['isolated_update50'] == clear['retained_FP32_update50'] and len(clear['isolated_update50']) == 3,
        'Same three clear-failure claim differs')
    text = reportpath.read_text(encoding='utf-8')
    for row in summary['guard_summary']:
        line = '| %s | %s | %s | %s | %.9f | %.9f | %d/%d | %s |' % (row['state'], row['control'], row['view'], row['metric'],
            row['mean_delta'], row['worst_delta'], len(row['case_failures']), row['cases'], row['worst_id'])
        require(line in text, 'Markdown guard table differs')
    hq_lines = 0
    for row in summary['full_source_profile_deltas']:
        if row['source'] != 'FFHQ validation': continue
        require(row['cases'] == 53, 'HQ cohort must use all53 controls')
        label = row['comparison'].replace('isolated_update50', 'AdaIN50').replace('retained_FP32_update50', 'FP32_50').replace('original_update0', 'original0')
        line = '| %s | %s | %s | %.9f | %.9f | %.9f | %.9f | %.9f | %d/%d |' % (label, row['view'], row['profile'],
            row['mean_delta_mae'], row['mean_delta_ssim'], row['mean_delta_luminance_bias'], row['mean_delta_coarse_rgb_mae'],
            row['mean_delta_gradient_vector_mae'], row['cases_higher_ssim'], row['cases_lower_ssim'])
        require(line in text, 'Markdown HQ table differs'); hq_lines += 1
    require(hq_lines == 36 and 'The24 QMUL-SurvFace and24 ChokePoint' in text and
        'This readback does not claim personal visual review or user acceptance.' in text,
        'Prose scope differs')
    sheetroot = review / 'visual_sheets'; sheetmanifest = read(sheetroot / 'manifest.json')
    require(sheetmanifest['complete'] and len(sheetmanifest['pages']) == 55 and
        sheetmanifest['source_visual_manifest_sha256'] == sha(metricroot / 'visual_manifest.json') and
        sheetmanifest['source_metrics_audit_sha256'] == sha(metricroot / 'metrics_audit.json'), 'Sheet evidence chain differs')
    source_hashes, bindings, all_cases = {}, 0, set()
    for page in sheetmanifest['pages']:
        path = sheetroot / page['file']; require(sha(path) == page['sha256'], 'Rendered sheet changed')
        require(page['tile_pixels'] == 256 and page['resampling'] == 'none', 'Sheet changed display scale')
        with Image.open(path) as sheet:
            require(list(sheet.size) == page['size'], 'Sheet dimensions differ')
            for binding in page['bindings']:
                source = Path(binding['source_path']); digest = binding['source_sha256']
                if str(source) not in source_hashes:
                    require(sha(source) == digest, 'Rendered source changed'); source_hashes[str(source)] = digest
                else: require(source_hashes[str(source)] == digest, 'Source digest contradiction')
                with Image.open(source) as image:
                    require(image.mode == 'RGB' and image.size == (256, 256), 'Source is not exact RGB256')
                    pixels = np.asarray(image)
                require(hashlib.sha256(pixels.tobytes()).hexdigest() == binding['source_pixel_sha256'], 'Source RGB digest differs')
                box = binding['pixel_box']; require(box[2] - box[0] == box[3] - box[1] == 256, 'Source tile geometry changed')
                require(np.array_equal(np.asarray(sheet.crop(box).convert('RGB')), pixels), 'Rendered tile resampled or changed')
                bindings += 1
        if page['kind'] == 'full_DEV_cohort': all_cases.update(page['case_ids'])
    require(len(all_cases) == 406, 'Planned full cohort is incomplete')
    result = {'complete': True, 'approved_summary_evidence_consistency': True,
        'summary_sha256': sha(summarypath), 'markdown_sha256': sha(reportpath),
        'source_profile_groups_recalculated': group_checks, 'source_profile_delta_groups_recalculated': delta_checks,
        'available_metric_denominators': metric_valid_counts,
        'older_paired_photometry_only_full_observed_canvas_available_cases': True,
        'HQ53case_profile_table_rows_recalculated': hq_lines, 'native_sources_field_means_metric_rows_not_unique_cases': True,
        'native_unique_cases_per_source': 24, 'native_metric_rows_per_source': 144,
        'visual_sheet_manifest_sha256': sha(sheetroot / 'manifest.json'), 'sheets_all_hash_verified': 55,
        'all406_planned_cases_bound': True, 'source_cells_exact_256_no_resampling': bindings,
        'no_personal_visual_quality_verdict': True, 'outputs_and_thresholds_unchanged': True,
        'linked_independent_state_pixel_decomposition_readback_sha256': sha(review / 'independent_scientific_readback_v1.json'),
        'source_sha256': sha(__file__), 'prior_local_checker_failure_sha256': sha(review / 'summary_crosschecker_attempt1_failure.json'),
        'neural_gradient_optimizer_VM_calls': 0,
        'model_qualified': False, 'training_authorized': False}
    write_new(review / 'independent_summary_render_readback_v1.json', result)
    print({key: result[key] for key in ('complete', 'source_profile_groups_recalculated',
        'source_profile_delta_groups_recalculated', 'source_cells_exact_256_no_resampling')})


if __name__ == '__main__': main()
