"""Record actual original-size review and saved arithmetic; no model modification."""
import json
from pathlib import Path

import audit_cctv_dgp_feature_skips_v27 as a

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_skips_v27_saved_output_review'
NOTES = {
    'tr_ffhq_00084': 'Clear glasses/eye boundaries, nostrils and open lips retain baseline softness. Blur and compound do not resolve glasses/eyes or mouth; lowlight is diffuse, and motion retains coarse facial arrangement. V27 has subtle tonal changes without convincing added boundaries across any row. Visible glasses and hair are retained.',
    'tr_ffhq_00323': 'Clear eyes, nose, closed mouth and wrinkles retain baseline smoothing. All degraded rows keep broad boundaries, with motion retaining more coarse structure. V27 adds no convincing eye/nose/mouth/outline detail; slight tonal changes do not establish recovered structure.',
    'tr_ffhq_00178': 'Clear eyes, nose and smile remain baseline-soft. Degraded eyes and teeth remain unresolved, and compound remains very diffuse. Motion preserves the coarse smile but has no convincing V27 improvement in eyes, mouth or outline. Hair and overall arrangement remain visible.',
    'tr_ffhq_00616': 'Clear eye and smile/tooth boundaries remain softened. Blur, lowlight and compound retain diffuse eye/nose/mouth structure; motion has the clearest coarse smile. V27 does not convincingly separate new facial boundaries. Cap, hair, earrings and visible hand remain present.',
    'tr_ffhq_01210': 'The mildly turned face retains baseline-soft glasses, eyes, nose, mouth and teeth. Strong degradation leaves eye/glasses boundaries especially diffuse. V27 shows small tonal differences without convincing added structure in any profile; visible clear glasses and hair remain.',
    'tr_asian_00048': 'Clear eyes, nose, lips and facial hair remain softened. Blur, lowlight and compound retain broad facial boundaries; motion retains the coarse moustache/mouth arrangement. V27 changes tone slightly without convincing added eye/nose/mouth/outline structure.',
    'tr_asian_00133': 'Clear eyes, nose and open mouth match the baseline structure by inspection, with subtle tonal changes. Degraded lip/eye/nose boundaries remain broad in all four profiles. V27 does not convincingly recover new structure or finer hair detail.',
    'tr_asian_00176': 'Clear eyes, small mouth and cheek outline retain baseline softness. Blur/lowlight/compound remain diffuse; motion keeps the coarse eye/mouth arrangement. V27 has subtle tone changes without convincing structural gain. Visible hair and clothing are retained.',
    'tr_asian_00180': 'Clear mildly turned eye/nose arrangement, open mouth and cheek outline stay baseline-soft. Motion retains coarse expression; stronger degradation retains diffuse boundaries. V27 adds no convincing visible structure across the five profiles.',
    'tr_asian_00196': 'Clear eyes, nose, smile, cheek outline and visible hand remain baseline-soft. Degraded eyes and mouth remain smeared, particularly compound. V27 changes tone slightly without convincing whole-face gain. The hand stays present; this is not a covering-removal test.',
}


def main():
    preparation = a.read(OUT / 'preparation.json')
    audit_path = ROOT / 'outputs/cctv_dgp_feature_skips_v27_independent_audit.json'
    audit = a.read(audit_path)
    a.require(audit['complete'] and a.sha(audit_path) == preparation['audit_sha256'] and
              preparation['exact_source_cells_verified'] == 200, 'Audited exact review preparation required')
    a.require(set(NOTES) == {sheet['reference'] for sheet in preparation['sheets']}, 'Every actually viewed sheet required')
    rows = []
    for sheet in preparation['sheets']:
        a.require(a.sha(OUT / sheet['file']) == preparation['sheet_sha256'][sheet['file']], 'Viewed sheet changed')
        for case in sheet['cases']:
            measured = next(row for row in preparation['rows'] if row['id'] == case)
            rows.append({'id': case, 'reference': sheet['reference'], 'sheet': sheet['file'],
                         'profile': measured['profile'], 'source': measured['source'],
                         'regions_reviewed': ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance'],
                         'convincing_whole_face_structure_gain': False, 'note': NOTES[sheet['reference']]})
    a.require(len(rows) == 50 and len({row['id'] for row in rows}) == 50, 'Every50-case visual review required')
    review = {'complete': True, 'date': '2026-10-06', 'reviewer': 'Codex primary assistant',
              'method': 'Viewed all ten actual 1072x1516 sheets at original detail, comparing each unenhanced 256x256 input, own-DGP baseline, V27 stopped50 and paired photographic TRAIN target cell.',
              'cases_reviewed': 50, 'exact_source_cells': 200, 'rows': rows,
              'sheet_sha256': preparation['sheet_sha256'], 'preparation_sha256': a.sha(OUT / 'preparation.json'),
              'runner_sha256': a.sha(Path(__file__)),
              'conclusion': 'Small delivered pixel/tonal changes, with no convincing added whole-face structure across the reviewed50 training cases. All50 PNGs differ; this is not a byte-identity claim.',
              'source_labels_are_not_ethnicity': True, 'native_or_reserved_used': False,
              'independent_final_review': False, 'app_promotion': False, 'goal_complete': False}
    a.write(OUT / 'visual_review.json', review)
    returned = ROOT / 'outputs/cctv_dgp_feature_skips_v27_return/outputs'
    baseline = a.read(returned / 'update0/metrics.json')
    candidate = a.read(returned / 'update50/metrics.json')
    arithmetic = a.capacity(baseline['groups'], candidate['groups'])
    a.require(arithmetic['degraded_feature_MSE_relative_gain'] == audit['early_structure_stop']['relative_feature_error_gain'], 'Early gain differs')
    previous = {}
    for label, folder in (
        ('V25', 'cctv_dgp_spatial_features_v25_diagnostic'),
        ('V26', 'cctv_dgp_batchmatched_identity_v26_diagnostic'),
    ):
        path = ROOT / 'outputs' / folder / 'results.json'
        group = a.read(path)['groups']['degraded']
        previous[label] = {'source_sha256': a.sha(path),
                           'delivered_structure_gain_percent': 100 * (1 - group['update50_PNG_feature_MSE'] / group['baseline_PNG_feature_MSE']),
                           'median_raw_correction_byte_units': 255 * group['median_saved_correction_RMS']}
    previous['V27'] = {'source_sha256': a.sha(OUT / 'preparation.json'),
                       'delivered_structure_gain_percent': 100 * arithmetic['degraded_feature_MSE_relative_gain'],
                       'median_raw_correction_byte_units': preparation['groups']['degraded']['median_raw_correction_byte_units']}
    source_groups = {name: 100 * (1 - candidate['groups'][name]['landmark_high_frequency_MSE'] / group['landmark_high_frequency_MSE'])
                     for name, group in baseline['groups'].items()}
    result = {'complete': True, 'scope': 'Saved stopped50 photographic TRAIN metrics only; final800 never ran',
              'audit_sha256': a.sha(audit_path), 'baseline_metrics_sha256': a.sha(returned / 'update0/metrics.json'),
              'stopped_metrics_sha256': a.sha(returned / 'update50/metrics.json'),
              'capacity_arithmetic_at_stopped50': arithmetic, 'group_structure_gain_percent': source_groups,
              'comparison': previous, 'neural_or_gradient_calls': 0, 'new_recipe_created': False,
              'model_modifications_stopped_for_architecture_discussion': True,
              'invalid_assumption': 'These finite small-head changes on frozen own-DGP pixels/features would provide enough whole-face structure under the retained data/objective/constraints.',
              'causal_limit': 'Three failed finite recipes do not prove all heads incapable, do not identify the unique optimization bottleneck and do not justify weakening the gates.',
              'independent_final_review': False, 'app_promotion': False, 'goal_complete': False}
    a.write(OUT / 'stopped_capacity_arithmetic.json', result)
    print(json.dumps({'complete': True, 'cases_reviewed': 50, 'comparison': previous,
                      'model_modifications_stopped_for_architecture_discussion': True}, indent=2))


if __name__ == '__main__':
    main()
