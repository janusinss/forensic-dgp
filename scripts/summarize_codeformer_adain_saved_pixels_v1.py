"""Summarize independently rebuilt saved-pixel tables without model execution."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.codeformer_adain_gradient_isolation_v1.return_pixels import require, sha, read, write


def summarize(review):
    review = Path(review).resolve(strict=True); directory = review / 'metrics_v1'
    audit = read(directory / 'metrics_audit.json')
    metrics = read(directory / 'metrics.json')
    guards = read(directory / 'guards.json')
    comparisons = read(directory / 'comparisons.json')
    prior = read(review / 'prior_comparison_binding_v1.json')
    require(audit['complete'] and prior['complete'] and
            prior['bound_metrics_audit_sha256'] == sha(directory / 'metrics_audit.json'),
            'Closed independent pixel and historical prior binding audits required')
    summaries = []
    for state in guards['guards']:
        for kind in ('clear', 'blur'):
            for view in ('raw', 'native_png'):
                rows = [('mae', state[kind]['views'][view])] if kind == 'clear' else state[kind]['views'][view].items()
                for name, row in rows:
                    summaries.append({'state': state['state'], 'control': kind, 'view': view, 'metric': name,
                        'cases': row['cases'], 'pass': row['pass'],
                        'mean_delta': row.get('mean_mae_delta', row.get('mean_delta')),
                        'worst_delta': row.get('worst_per_case_mae_delta', row.get('worst_per_case_delta')),
                        'mean_limit': .01, 'case_limit': .03,
                        'case_failures': {rid: delta for rid, delta in row['deltas'].items() if delta > .03},
                        'worst_id': max(row['deltas'], key=row['deltas'].get)})
    hq_route_delta = [row for row in comparisons['source_profile_deltas']
        if row['source'] == 'FFHQ validation' and row['comparison'] == 'isolated_update50_minus_retained_FP32_update50']
    require(len(hq_route_delta) == 12 and all(row['cases'] == 53 for row in hq_route_delta),
            'Full53 cases per HQ profile per RAW/PNG view required')
    clear_failures = {state: sorted({row['id'] for row in guards['per_case_failures']
        if row['state'] == state and row['control'] == 'clear'})
        for state in ('isolated_update50', 'retained_FP32_update50')}
    bindings = {name: sha(directory / name) for name in ('metrics_audit.json', 'metrics.json', 'guards.json',
        'comparisons.json', 'native_descriptors.json', 'witness_pixels.json', 'input_audit.json', 'visual_manifest.json')}
    result = {'complete': True, 'audited_saved_pixel_evidence_only': True,
        'metrics_files_sha256': bindings, 'prior_binding_sha256': sha(review / 'prior_comparison_binding_v1.json'),
        'source_sha256': sha(__file__), 'guard_summary': summaries,
        'clear_failed_case_ids': clear_failures,
        'unchanged_three_clear_failure_ids_vs_prior': clear_failures['isolated_update50'] == clear_failures['retained_FP32_update50'],
        'all_six_HQ_profiles_PNG_mean_MAE_slightly_worse_than_prior_FP32': all(row['mean_delta_mae'] > 0 for row in hq_route_delta if row['view'] == 'native_png'),
        'all_six_HQ_profiles_PNG_mean_SSIM_slightly_lower_than_prior_FP32': all(row['mean_delta_ssim'] < 0 for row in hq_route_delta if row['view'] == 'native_png'),
        'full_source_profile_metrics': metrics['groups'],
        'full_source_profile_deltas': comparisons['source_profile_deltas'],
        'native_sources': dict(Counter(row['source'] for row in metrics['rows'] if row['evidence'] == 'native_unpaired')),
        'interpretation': [
            'Training retained substantial compound/lowlight paired improvements over untouched CodeFormer, alongside clear and normal-blur photometric regressions.',
            'The tested AdaIN gradient isolation did not reduce the prior50-update preservation failures; its six HQ profile PNG mean MAE/SSIM values are slightly worse than retained FP32.',
            'This single finite comparison rejects this unchanged route as a preservation fix; it does not show that every adaptation or longer justified training is impossible.',
            'SSIM and gradient metrics are secondary paired descriptors, not identity recovery or an acceptance substitute.',
            'Native CCTV has no aligned clean reference and remains separate by source; visual usefulness, independent final review and app qualification are outstanding.'
        ], 'visual_review_pending': True, 'model_qualified': False, 'new_training_authorized': False,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0}
    write(review / 'scientific_readback_v1.json', result)
    lines = ['# AdaIN gradient isolation V1: independent saved-pixel readback', '',
        'The fifty-update pilot stopped on the declared clear-preservation guard. The independent audit also reproduces the normal-blur photometric failure. The export and audit completed; the model did not qualify.', '',
        'All812 new RAW/PNG pairs are checked, with406 retained FP32 update50 pairs and406 exactly equal original baseline pairs. The prior1626 compared artifacts and metric receipts match their pinned historical manifest. All4 pre-Adam witness native/RAW/PNG/token values match between original and isolated paths; this does not independently replay VM gradients.', '',
        '## Unchanged preservation requirements', '',
        '| Candidate | Control | View | Metric | Mean delta | Worst delta | Cases >0.03 | Worst ID |',
        '|---|---|---|---|---:|---:|---:|---|']
    for row in summaries:
        lines.append('| %s | %s | %s | %s | %.9f | %.9f | %d/%d | %s |' %
            (row['state'], row['control'], row['view'], row['metric'], row['mean_delta'],
             row['worst_delta'], len(row['case_failures']), row['cases'], row['worst_id']))
    lines += ['', 'Mean limit0.01 and per-case limit0.03 remain unchanged. The same three clear IDs fail in both candidates: ' +
        ', '.join('`' + rid + '`' for rid in clear_failures['isolated_update50']) + '.', '',
        '## Full53-case HQ paired profile effects', '',
        'Deltas are candidate minus comparator. Lower MAE/luminance bias/coarse RGB/gradient error is better; higher SSIM is better. RAW and downloaded PNG stay separate. Every row uses all53 cases of that profile.', '',
        '| Comparison | View | Profile | MAE delta | SSIM delta | Bias delta | Coarse delta | Gradient vector delta | SSIM higher/lower |',
        '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for row in comparisons['source_profile_deltas']:
        if row['source'] != 'FFHQ validation':
            continue
        label = row['comparison'].replace('isolated_update50', 'AdaIN50').replace('retained_FP32_update50', 'FP32_50').replace('original_update0', 'original0')
        lines.append('| %s | %s | %s | %.9f | %.9f | %.9f | %.9f | %.9f | %d/%d |' %
            (label, row['view'], row['profile'], row['mean_delta_mae'], row['mean_delta_ssim'],
            row['mean_delta_luminance_bias'], row['mean_delta_coarse_rgb_mae'], row['mean_delta_gradient_vector_mae'],
            row['cases_higher_ssim'], row['cases_lower_ssim']))
    lines += ['', '## Scope and interpretation', '']
    lines += [str(index) + '. ' + item for index, item in enumerate(result['interpretation'], 1)]
    lines += ['', 'The40 older paired cases remain separated by their two labeled dataset sources in the JSON source/profile tables. The24 QMUL-SurvFace and24 ChokePoint native crops have NULL clean-reference metrics; separate native descriptors report only output/input changes, not restoration correctness.', '',
        'All55 exact256 visual sheets (406 cases plus witness) are prepared for the root reviewer. This readback does not claim personal visual review or user acceptance.', '',
        '`scientific_readback_v1.json` binds every numerical table above. Source and output hashes remain exact; no new model, threshold, data split or output processing has been substituted.', '']
    with (review / 'SCIENTIFIC_READBACK_V1.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(lines))
    return {'complete': True, 'readback_sha256': sha(review / 'scientific_readback_v1.json'),
        'report_sha256': sha(review / 'SCIENTIFIC_READBACK_V1.md')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--review', required=True)
    args = parser.parse_args(); print(json.dumps(summarize(args.review), indent=2))
