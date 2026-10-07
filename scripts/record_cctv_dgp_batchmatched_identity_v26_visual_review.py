"""Record the completed V26 original-cell review; no final human acceptance."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_batchmatched_identity_v26 as a


def main():
    folder = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_diagnostic'
    result = a.read(folder / 'results.json')
    check = a.read(folder / 'independent_saved_diagnostic_audit.json')
    assert check['complete'] and check['exact_review_cells'] == 200
    assert check['results_sha256'] == a.sha(folder / 'results.json')
    notes = {
        'tr_ffhq_00084': 'Clear eye/glasses boundaries and open lips remain softer than the input/target. In blur, lowlight and compound, eyes, glasses, nose and lips remain broad or smeared. Motion retains coarse features. V26 adds no discernible gain in any row; outline, visible hair and colour match baseline.',
        'tr_ffhq_00323': 'Clear eye boundaries, wrinkles, nose and closed mouth retain baseline smoothing. Blur, lowlight and compound have diffuse eye/nose/mouth boundaries and outline. Motion retains coarse facial features. No discernible V26 advantage in any row or visible hair/colour.',
        'tr_ffhq_00178': 'Clear eyes, nose and smile/teeth are softened by baseline. Degraded eyes and mouth stay broad and teeth unresolved; motion retains more of the coarse smile. V26 adds no discernible structure to eyes, nose, mouth, outline or hairline across the five rows.',
        'tr_ffhq_00616': 'Clear eye and tooth detail remain softened. Degraded eye, nose, smile and outline boundaries stay broad, especially compound. Motion is more legible. V26 matches baseline by inspection, including visible hair, cap, clothing and colour.',
        'tr_ffhq_01210': 'The mildly turned clear face retains softened glasses, eyes, nose and open mouth. Blur/compound glasses and eye boundaries are weak, and lowlight remains dark and diffuse. No discernible V26 gain in those features, mouth, outline or visible hair/appearance; clear glasses are retained.',
        'tr_asian_00048': 'Clear eyes, nose, mouth and visible facial hair retain baseline softness. Motion has the clearest coarse degraded arrangement; other degraded rows remain diffuse. V26 has no discernible advantage in any row, including face outline, hair and colour.',
        'tr_asian_00133': 'The clear open mouth, eyes, broad nose and hairline match baseline by inspection. Degraded nose/mouth edges and eye/hairline detail remain soft; motion retains coarse arrangement. V26 adds no discernible gain to any visible feature or overall appearance.',
        'tr_asian_00176': 'Clear eyes, small closed mouth, nose and cheek outline stay baseline-soft. Blur, lowlight and compound retain diffuse boundaries; motion retains the coarse arrangement. V26 shows no discernible gain across any row, with visible clothing, hair and colour matching baseline.',
        'tr_asian_00180': 'The mildly turned eye/nose arrangement, open mouth and cheek outline retain baseline softness. Motion preserves coarse expression; stronger degradation keeps the eye and mouth boundaries diffuse. No discernible V26 whole-face advantage or change to visible appearance.',
        'tr_asian_00196': 'Clear eyes, nose, smile, outline and the visible hand remain baseline-soft. Strong degradation retains smeared features; motion is more legible. V26 adds no discernible gain in any row. The hand is retained; this is not a covering-removal test.'
    }
    profile_notes = {
        'clear': 'No discernible added structure; baseline softness remains.',
        'blur_lr24': 'No discernible eye, nose, mouth or outline gain over baseline.',
        'lowlight_lr32': 'Dark and soft boundaries remain; no discernible added structure.',
        'motion_lr48': 'Coarse features remain where readable; no discernible V26 gain.',
        'compound_lr24': 'Strongly diffuse facial boundaries remain; no discernible V26 gain.'
    }
    rows = []
    for sheet in result['sheets']:
        assert sheet['reference'] in notes
        assert a.sha(folder / sheet['file']) == result['artifacts_sha256'][sheet['file']]
        for case in sheet['cases']:
            measured = next(row for row in result['rows'] if row['id'] == case)
            rows.append({'id': case, 'sheet': sheet['file'], 'reference': sheet['reference'],
                'source': measured['source'], 'profile': measured['profile'],
                'regions_reviewed': ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance'],
                'convincing_visible_structure_gain': False,
                'case_note': profile_notes[measured['profile']], 'reference_note': notes[sheet['reference']]})
    assert len(rows) == 50 and {r['id'] for r in rows} == {r['id'] for r in result['rows']}
    review = {'complete': True, 'date': '2026-10-06', 'reviewer': 'Codex primary assistant',
        'method': 'Viewed all ten original 1072x1516 sheets at original detail this turn, comparing every 256x256 input, own-DGP baseline, V26 stopped50 and paired TRAINING target cell. No display enhancement.',
        'cases_reviewed': 50, 'exact_cells': 200, 'regions_required_together': rows[0]['regions_reviewed'],
        'rows': rows, 'reference_notes_recorded_from_actual_V26_images': True,
        'measured_change_limit': 'All fifty PNGs change; degraded changes reach at most one byte level and clear changes at most two. No convincing structure is discernible at original size. This is a visual verdict, not an identical-byte claim.',
        'conclusion': 'No convincing visible whole-face structural improvement over frozen own-DGP baseline in any of the fifty reviewed training cases.',
        'diagnostic_sha256': a.sha(folder / 'results.json'),
        'saved_array_check_sha256': a.sha(folder / 'independent_saved_diagnostic_audit.json'),
        'sheets_sha256': result['artifacts_sha256'], 'runner_sha256': a.sha(Path(__file__)),
        'scope': 'Exposed paired photographic TRAINING data only. This review does not prove native generalization, hidden identity or independent final acceptance and cannot relabel a usable CCTV input as insufficient.',
        'source_labels_are_not_ethnicity': True, 'native_or_reserved_used': False,
        'independent_final_review': False, 'app_promotion': False, 'goal_complete': False}
    a.write(folder / 'visual_review.json', review)
    print(json.dumps({'complete': True, 'cases_reviewed': 50, 'convincing_structure_gain_cases': 0,
        'independent_final_review': False}, indent=2))


if __name__ == '__main__':
    main()
