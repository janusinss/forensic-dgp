"""Actual whole-face observations from all six native V29 comparison sheets."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v29_native_development_v1'
NOTES = [
    'Downward eyes, nose and closed lips remain broadly present. V29 stays soft around eyelids and mouth, with no convincing structure gain over resize.',
    'Eyes, nose, lips and outline remain readable; V29 has modest boundary changes relative to original DGP, without a convincing gain over the clearer input.',
    'The previously useful02_t033 retains eye/nose/mouth arrangement. V29 is slightly more defined than original DGP but remains diffuse; finer visible eye and lip structure still needs improvement.',
    'Eyes, nose and mouth remain readable, with modest local contrast changes. Fine eye and skin structure stay smoothed relative to resize.',
    'Broad eyes, nose, mouth and outline survive; V29 makes only small boundary changes and stays soft around eye and lip contours.',
    'Readable eyes and nose remain, with a soft closed-mouth boundary. V29 does not establish useful added structure over the observed input.',
    'Clear glasses, broad eyes, nose and lips remain present; V29 has no convincing clarity gain and glasses/fine eye structure remain smoothed.',
    'Clear glasses and facial arrangement survive, with modest eye/mouth contrast changes; finer glasses and skin detail remain soft.',
    'Clear glasses, eyes and lips remain broadly present. V29 still softens nose and mouth structure compared with resizing.',
    'Clear glasses, eye/nose/lip arrangement and hair survive; V29 differs little from original DGP and retains smoothing of fine appearance.',
    'Dark coarse eyes, nose and lips remain present; V29 stays diffuse, especially around gaze and mouth boundaries.',
    'Eyes, nose, lips and outline remain readable. V29 is only modestly changed from original DGP, without a convincing added structure gain.',
    'Mildly turned face and clear glasses remain present; eyes and mouth are still diffuse, with only small V29 contrast changes.',
    'Clear glasses and broad face arrangement remain readable; V29 still smooths fine eyelids and mouth detail relative to the observed crop.',
    'Motion-soft clear-glasses face retains its broad outline and mouth. V29 remains diffuse and does not clarify the visible glasses/eyes convincingly.',
    'Clear glasses, nose and mouth remain present. V29 is modestly changed from original DGP and does not produce a convincing new clarity gain.',
    'Glasses and broad eyes/nose/mouth remain visible, with lens highlights retained. V29 remains soft around fine eyes and lip boundaries.',
    'Clear glasses, eye/nose/lip arrangement and outline survive; small V29 contrast changes do not establish added useful structure.',
    'Broad eyes, nose and closed lips remain readable. V29 stays diffuse around fine gaze and mouth detail compared with the input.',
    'Eyes, nose, closed lips and outline remain present; V29 differs only modestly from original DGP and keeps fine appearance smoothed.',
    'Glasses, broad eyes/nose/mouth and facial hair remain present. V29 stays diffuse around eye detail and facial-hair pattern.',
    'Glasses, open mouth and facial hair remain readable; V29 retains smoothing with no convincing structural clarity gain over resize.',
    'Clear glasses, nose, lips and outline survive; V29 remains soft in eyelids and mouth, with small changes from original DGP.',
    'Clear glasses and broad eye/nose/lip arrangement remain present. V29 still smooths visible fine structure and adds no convincing clarity gain.',
]


def sha(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    p, r, audit = [read(OUT / n) for n in ['plan.json', 'results.json', 'saved_output_audit.json']]
    assert audit['complete'] and audit['all144_sheet_cells_exact'] and len(r['rows']) == len(NOTES) == 24
    rows = [{'id': row['id'], 'auto_selected': row['auto_selected'],
             'input_only_usable': True, 'input_usability_reclassified_by_output': False,
             'regions_reviewed': ['eyes', 'nose', 'mouth', 'face outline', 'visible appearance'],
             'note': note, 'useful_structure_gain_over_resize_demonstrated': False}
            for row, note in zip(r['rows'], NOTES)]
    value = {'complete': True, 'date': '2026-10-06', 'reviewer': 'Codex primary assistant',
             'method': 'Actually viewed all six1604x1220 comparison sheets at original image detail, reviewing every24 native face and all144 unchanged256x256 cells.',
             'cases_reviewed': 24, 'rows': rows, 'results_sha256': sha(OUT / 'results.json'),
             'saved_output_audit_sha256': sha(OUT / 'saved_output_audit.json'),
             'sheet_sha256': {s['file']: sha(OUT / s['file']) for s in r['sheets']},
             'recorder_sha256': sha(Path(__file__)),
             'conclusion': 'V29 retains broad native face arrangement and makes modest changes from original DGP, but no convincing useful structural clarity gain over resizing is demonstrated across this cohort. CodeFormer produces visibly clearer plausible estimates under its declared common-input path and remains a pretrained comparison baseline. TRAIN capacity success is insufficient for native/app qualification.',
             'input_only_usable_count_retained': 24, 'user_previously_useful_case': 'choke_dev_02_t033',
             'all_visible_regions_still_require_clearer_structure': True,
             'no_visible_information_failure_claim_from_model_softness': True,
             'native_evidence_unpaired': True, 'PSNR': None, 'SSIM': None, 'identity_accuracy': None,
             'captured_country_or_ethnicity_inferred': False, 'zamboanga_validation': False,
             'reserved_final_used': False, 'independent_final_review': False,
             'restoration_qualified': False, 'app_promotion': False, 'goal_complete': False}
    with (OUT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'native_faces_reviewed': 24, 'native_app_qualified': False}))


if __name__ == '__main__': main()
