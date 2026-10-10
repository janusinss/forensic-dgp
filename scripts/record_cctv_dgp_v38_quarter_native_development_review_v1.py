"""Actual observations from the six previously displayed native comparison pages."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v38_quarter_native_development_v1'
NOTES = [
    'Downward eyes, closed lips and hair remain coarse. The quarter output is nearly indistinguishable from original DGP; eyelid and mouth detail stay weak.',
    'The input has readable eye and lip boundaries. Original and quarter DGP both soften these; no convincing incremental clarity is visible.',
    'The previously useful crop retains the broad eye/nose/mouth arrangement. The quarter output stays close to original DGP with coarse fine features.',
    'The clearer crop retains readable broad structure. Quarter and original DGP are very similar and do not add convincing eye or mouth definition.',
    'Small coarse eyes, nose and lips remain present. Quarter DGP retains softness and shows no convincing new clarity.',
    'Readable eyes, nose and closed lips survive, with finer appearance softened relative to the input. Quarter and original DGP remain close.',
    'Ordinary clear glasses are retained. Fine eye/frame boundaries and the nose remain blurred in quarter DGP, much as in original DGP.',
    'Clear glasses and face arrangement remain present. Quarter and original DGP are similar; the visible frame and lip detail are still soft.',
    'Clear glasses remain, but frame and eye boundaries merge under the soft DGP rendering. Quarter DGP does not clarify the lips or nose convincingly.',
    'The clearer glasses and face shape survive. The quarter output remains very close to original DGP with no convincing new fine structure.',
    'Dark coarse eyes, nose and lips remain present. The quarter output remains diffuse around gaze and mouth boundaries and close to original DGP.',
    'Broad expression and facial arrangement remain present. Quarter DGP softens visible fine features and adds no convincing clarity over original DGP.',
    'The mildly turned clear-glasses face retains its broad outline. Fine gaze and mouth structure remain weak; quarter DGP stays close to original DGP.',
    'Clear glasses, face outline and hair survive. Fine eyelids and lips remain soft, without a convincing quarter-step improvement.',
    'Side hair and ordinary clear glasses are preserved. The quarter output has broad gaze and a soft mouth, much like original DGP.',
    'Glasses, nose and mouth remain present. Eye and nose contours in quarter DGP are similar to original DGP, without convincing new clarity.',
    'Visible lens reflections and clear glasses are retained. Coarse eyes and mouth stay soft and nearly unchanged from original DGP.',
    'Clearer frame and face arrangement survive. Quarter DGP softens fine eye/lip structure and adds no convincing new definition.',
    'Broad eyes, nose and closed lips remain present. Quarter and original DGP stay close, with diffuse fine boundaries.',
    'The input has readable eyebrows and mouth outline. Quarter DGP softens these and does not establish a convincing clarity gain.',
    'Visible eyewear highlights are retained in this restoration comparison. Quarter DGP stays soft around eyes and mouth and close to original DGP.',
    'Visible open-mouth and eye structure remain broadly present. Quarter DGP keeps the soft original rendering without newly clear fine parts.',
    'Clear frame, nose and lips survive broadly. The quarter output remains similar to original DGP with weak fine eyelid and lip structure.',
    'Clear frame and face outline are retained. The quarter output stays soft around lips and nose, without a convincing incremental gain.',
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    results = read(OUT / 'results.json')
    audit = read(OUT / 'saved_output_audit.json')
    assert results['complete'] and audit['complete'] and audit['all144_sheet_cells_exact']
    assert len(results['rows']) == len(NOTES) == 24 and len(results['sheets']) == 6
    rows = []
    for row, note in zip(results['rows'], NOTES):
        rows.append({'id': row['id'], 'auto_selected': row['auto_selected'],
                     'input_only_usable': True, 'input_usability_reclassified_by_output': False,
                     'regions_reviewed': ['eyes', 'nose', 'mouth', 'face outline', 'visible appearance'],
                     'note': note, 'convincing_incremental_clarity_over_original_DGP': False,
                     'useful_structure_gain_over_resize_demonstrated': False})
    value = {'complete': True, 'date': '2026-10-08', 'reviewer': 'Codex primary assistant',
             'method': 'Actually viewed all six 1604x1220 pages at original image detail, including every native face and all144 unchanged256x256 comparison cells.',
             'cases_reviewed': 24, 'comparison_cells_reviewed': 144, 'rows': rows,
             'results_sha256': sha(OUT / 'results.json'),
             'saved_output_audit_sha256': sha(OUT / 'saved_output_audit.json'),
             'sheet_sha256': {s['file']: sha(OUT / s['file']) for s in results['sheets']},
             'recorder_sha256': sha(Path(__file__)),
             'conclusion': 'The fixed quarter step preserves broad face arrangement and is visually very similar to retained DGP. This native cohort does not demonstrate convincing added clarity. CodeFormer has visibly clearer plausible estimates in these comparisons and remains a declared pretrained baseline. Numerical TRAIN preservation is insufficient for native/app qualification.',
             'input_only_usable_count_retained': 24, 'user_previously_useful_case': 'choke_dev_02_t033',
             'all_visible_regions_still_require_clearer_structure': True,
             'no_visible_information_failure_claim_from_model_softness': True,
             'native_evidence_unpaired': True, 'PSNR': None, 'SSIM': None, 'identity_accuracy': None,
             'captured_country_or_ethnicity_inferred': False, 'zamboanga_validation': False,
             'reserved_final_used': False, 'independent_final_review': False,
             'restoration_qualified': False, 'app_promotion': False, 'goal_complete': False}
    with (OUT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'native_faces_reviewed': 24,
                      'comparison_cells_reviewed': 144, 'native_app_qualified': False}))


if __name__ == '__main__':
    main()
