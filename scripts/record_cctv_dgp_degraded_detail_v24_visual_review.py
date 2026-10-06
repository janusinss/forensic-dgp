"""Record the assistant's completed original-cell review, not final human approval."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_cctv_dgp_degraded_detail_v24 as a


def main():
    folder = ROOT / 'outputs/cctv_dgp_degraded_detail_v24_diagnostic'
    result = a.read(folder / 'results.json')
    check = a.read(folder / 'independent_saved_diagnostic_audit.json')
    assert check['complete'] and check['exact_review_cells'] == 200
    assert check['results_sha256'] == a.sha(folder / 'results.json')
    notes = {
        'tr_ffhq_00084': 'Clear glasses, eyes, nose and open mouth remain softened by baseline DGP. The stopped head adds no discernible feature or outline improvement across the four degraded rows; glasses remain weak in the strongest degradation.',
        'tr_ffhq_00323': 'Eye boundaries, nose and closed mouth remain diffuse in blur/lowlight/compound. Motion retains more coarse structure, with no discernible stopped-head advantage. Face outline, hair and colour match the baseline by inspection.',
        'tr_ffhq_00178': 'Eyes, nose, mouth/teeth and hairline stay visibly softer than the paired target. The stopped head does not resolve the degraded features or change the visible outline relative to baseline.',
        'tr_ffhq_00616': 'Clear eye and mouth detail is softened by baseline. Degraded eye, nose, smile and outline remain broad; visible hair and graduation cap match baseline. No discernible stopped-head structural improvement.',
        'tr_ffhq_01210': 'Clear glasses, eyes, nose, mouth and outline remain baseline-soft. Degraded glasses/eye boundaries are weak; no discernible improvement of those features, mouth or visible hair/colour from the stopped head.',
        'tr_asian_00048': 'Eyes, nose, mouth, visible facial hair and face outline match the baseline in all five rows. Motion is more legible than blur/compound, without a discernible stopped-head gain.',
        'tr_asian_00133': 'The open-mouth shape and coarse eye/nose arrangement remain visible, but degraded edges and hairline stay diffuse. No discernible stopped-head improvement of eyes, nose, mouth, outline or overall appearance.',
        'tr_asian_00176': 'Eyes, small mouth and broad nose/cheek outline remain baseline-soft. Clothing and visible hair/colour match baseline; no discernible stopped-head structure gain in any row.',
        'tr_asian_00180': 'Mildly turned eye/nose arrangement, open mouth and cheek outline stay baseline-soft. The stopped head does not resolve their degraded boundaries or improve overall visible appearance.',
        'tr_asian_00196': 'Eyes, nose, smile, face outline and visible hand/hair match the baseline by inspection. Strong degradation retains smearing; no discernible stopped-head gain. This is no covering-removal experiment.'
    }
    profile_notes = {
        'clear': 'No discernible new structure; original detail softened by baseline remains soft.',
        'blur_lr24': 'No discernible gain of eye, nose, mouth or outline detail over baseline.',
        'lowlight_lr32': 'Dark/soft feature boundaries remain; no discernible stopped-head gain.',
        'motion_lr48': 'Coarse facial features remain legible where present, with no discernible stopped-head gain.',
        'compound_lr24': 'Strongly smeared structure remains; no discernible stopped-head gain.'
    }
    rows = []
    for sheet in result['sheets']:
        assert sheet['reference'] in notes and a.sha(folder / sheet['file']) == result['artifacts_sha256'][sheet['file']]
        for case in sheet['cases']:
            measured = next(row for row in result['rows'] if row['id'] == case)
            rows.append({'id': case, 'sheet': sheet['file'], 'reference': sheet['reference'],
                'source': measured['source'], 'profile': measured['profile'],
                'regions_reviewed': ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance'],
                'convincing_visible_structure_gain': False,
                'case_note': profile_notes[measured['profile']], 'reference_note': notes[sheet['reference']]})
    assert len(rows) == 50 and {row['id'] for row in rows} == {row['id'] for row in result['rows']}
    review = {'complete': True, 'date': '2026-10-06', 'reviewer': 'Codex primary assistant',
        'method': 'Viewed all ten original 1072x1516 sheets at original detail, comparing every 256x256 input, own-DGP baseline, stopped50 and paired TRAINING target cell. No display enhancement.',
        'cases_reviewed': 50, 'exact_cells': 200, 'regions_required_together': list(rows[0]['regions_reviewed']),
        'rows': rows, 'conclusion': 'No convincing visible whole-face structural improvement over frozen own-DGP baseline in any of the 50 reviewed training cases.',
        'diagnostic_sha256': a.sha(folder / 'results.json'), 'saved_array_check_sha256': a.sha(folder / 'independent_saved_diagnostic_audit.json'),
        'sheets_sha256': result['artifacts_sha256'], 'runner_sha256': a.sha(Path(__file__)),
        'scope': 'Exposed paired photographic TRAINING data only. This review cannot classify new native inputs or prove generalization, hidden identity or independent final acceptance.',
        'source_labels_are_not_ethnicity': True, 'native_or_reserved_used': False,
        'independent_final_review': False, 'app_promotion': False, 'goal_complete': False}
    a.write(folder / 'visual_review.json', review)
    print(json.dumps({'complete': True, 'cases_reviewed': 50, 'convincing_structure_gain_cases': 0,
        'independent_final_review': False}, indent=2))


if __name__ == '__main__':
    main()
