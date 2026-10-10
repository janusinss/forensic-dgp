"""Actual review of the ten prospective preview references, not all104 references."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v38_quarter_paired_development_v1'
NOTES = {
    'va_asian_00209': 'Broad eye/nose/mouth arrangement and hair survive. Clear and motion crops are more readable; blur and compound retain weak nostrils/lips. Quarter and original DGP are visually very close.',
    'va_asian_00222': 'Clear wrinkles, eyes and smile remain broadly present but are softened. Blur/lowlight/compound lack fine eye and tooth structure; motion retains more coarse expression. Quarter remains close to original.',
    'va_asian_00302': 'Broad smiling expression and face outline survive. Visible eye and open-mouth detail stay diffuse under blur/compound. Quarter is nearly unchanged from original DGP.',
    'va_asian_00563': 'Clear eyes, nose and closed lips retain broad arrangement. Degraded eye shape and lip definition remain weak; original and quarter outputs are very similar. The black padding is retained.',
    'va_asian_00700': 'Clear eye and mouth shapes remain present. Severe blur/lowlight/compound stay diffuse around eyelids, nostrils and lips. Quarter adds no convincing clarity over original DGP.',
    'va_ffhq_00383': 'Visible hair decoration, face outline and clear mouth survive. Motion retains coarse eye/nose structure, while blur/compound remain very soft. Quarter remains close to original DGP.',
    'va_ffhq_01093': 'Clear eyes, nostrils, smiling mouth and visible teeth remain broadly present but softened. Motion retains more expression than blur/compound; fine eyelids/teeth stay weak. Quarter and original are very close.',
    'va_ffhq_08025': 'Visible cosmetics, hair/headwear and broad facial geometry remain. Clear fine appearance is softened and severe degradation leaves weak eye/nose/lip structure. Quarter does not provide a convincing new clarity gain.',
    'va_ffhq_09056': 'Ordinary clear red glasses are retained on the clear crop; severe blur and compound leave frame/eye and lip boundaries weak. Quarter remains close to original without convincingly clarifying those features.',
    'va_ffhq_10401': 'Hair, broad gaze and closed-mouth arrangement survive. Clear facial detail is softened; blur/lowlight/compound remain diffuse. Quarter adds no convincing clarity over original DGP.',
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = read(OUT / 'plan.json'); r = read(OUT / 'results.json'); audit = read(OUT / 'saved_output_audit.json')
    assert r['complete'] and audit['complete'] and audit['all200_preview_cells_exact']
    assert set(NOTES) == set(p['preview_reference_ids']) and len(NOTES) == len(r['sheets']) == 10
    rows = []
    for rid in p['preview_reference_ids']:
        chosen = [row for row in r['rows'] if row['reference_id'] == rid]
        assert len(chosen) == 5
        for row in chosen:
            rows.append({'id': row['id'], 'source': row['source'], 'profile': row['profile'],
                         'regions_reviewed': ['eyes', 'nose', 'mouth', 'face outline', 'visible appearance'],
                         'reference_note': NOTES[rid], 'convincing_incremental_clarity_over_original_DGP': False,
                         'input_criteria_changed': False})
    value = {'complete': True, 'date': '2026-10-08', 'reviewer': 'Codex primary assistant',
             'method': 'Actually viewed all ten prospective1072x1516 pages at original detail;50 cases with256x256 input/original/quarter/target cells.',
             'cases_reviewed': 50, 'references_reviewed': 10, 'image_cells_reviewed': 200,
             'total_numeric_cases': 520, 'all520_images_visually_reviewed': False,
             'visual_selection_frozen_before_outputs': True, 'rows': rows,
             'sheet_sha256': {s['file']: sha(OUT / s['file']) for s in r['sheets']},
             'results_sha256': sha(OUT / 'results.json'), 'saved_output_audit_sha256': sha(OUT / 'saved_output_audit.json'),
             'recorder_sha256': sha(Path(__file__)),
             'conclusion': 'The fixed quarter step shows no convincing incremental whole-face clarity in these50 previews. All520 numeric cases include a retained source/profile ArcFace regression. It is rejected for adoption without selecting another scale from development outcomes.',
             'photographic_paired_development_only': True, 'native_CCTV': False,
             'ethnicity_inferred': False, 'zamboanga_validation': False, 'reserved_final_used': False,
             'independent_final_review': False, 'restoration_qualified': False,
             'app_promotion': False, 'goal_complete': False}
    with (OUT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'actual_cases_reviewed': 50, 'all520_visually_reviewed': False}))


if __name__ == '__main__':
    main()
