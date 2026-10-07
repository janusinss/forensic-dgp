"""Record actual original-detail inspection of all10 sheets and all50 fixed cases."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_feature_fusion_v32_r2_failure_review_v1'
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
OBSERVATIONS = {
    'tr_ffhq_00084': [
        'Clear-glasses frames, eye placement, nostrils and open lip shape remain. Cheek/jaw arrangement is retained. Hair, skin, lens and tooth detail are softened; R2 is close to V31.',
        'Eyes and glasses merge into diffuse brow-level shading. Nose and pink mouth are broad soft shapes within the retained face silhouette. R2 supplies no distinct eyelid, frame or lip clarity.',
        'Dim eye/glasses shading remains without readable lids. Nose and mouth contours are weak, while cheek and hair outline remain. R2 retains the low-light appearance without useful added fine structure.',
        'Eye/glasses placement and nose direction are more readable than the blur row, but soft. Open lips and cheek/jaw arrangement remain. Hair and teeth have no convincing extra definition over V31.',
        'Only vague eye shading, a broad nose and diffuse mouth remain. Hair and face silhouette are present, but fine eye, lip and outline cues are insufficient. A clearer crop is needed.',
    ],
    'tr_ffhq_00323': [
        'Eyes, brows, nose and slight closed smile remain placed correctly within broad cheek/chin geometry. Pale hair, jacket and skin are softened. R2 shows little incremental definition over V31.',
        'Eyes are dark diffuse shapes, nose broad and mouth a weak line. Head, cheeks and collar remain. Smoothing removes block boundaries without restoring fine facial definition.',
        'Dim eyes and brow placement remain, while nose and smile are low contrast. Head silhouette and collar are retained. Fine eyelids, nostrils and lip edges remain unresolved.',
        'Eye placement, nose direction and slight smile remain more readable than in blur. Cheek/chin geometry survives, with softened hair and skin. R2 is close to the retained/V31 output.',
        'Dark eye patches survive without lids; nose and smile are indistinct. Broad head and collar silhouette remain. Useful fine structure needs a clearer crop.',
    ],
    'tr_ffhq_00178': [
        'Large eye positions, small nose and open smile remain. Rounded cheeks and chin follow the source. Hair and tooth detail are softened; no substantial R2 clarity advantage appears.',
        'Eyes become dark patches; nose and mouth stay diffuse. Rounded face and short hair outline remain. R2 smoothing does not define eyelids, nostrils or individual teeth.',
        'Dark eye placement, faint nose and dim smile survive within the round face outline. Hair and low-light tone are retained. Fine mouth and eyelid boundaries stay weak.',
        'Broad eyes, nose and smile remain readable, with soft teeth and eyelids. Rounded cheeks and hair silhouette are retained. R2 adds little visible definition over V31.',
        'Dark eyes and a diffuse smile band remain, but nose is poorly defined. Only broad face/hair geometry survives. A clearer crop is needed for useful fine features.',
    ],
    'tr_ffhq_00616': [
        'Eye placement, nose and wide tooth-bearing smile remain readable. Cheek/chin geometry, cap, hair, earrings and adjacent hand remain. R2 has softened skin and little extra clarity over V31.',
        'Eyes and nose are faint diffuse marks; smile is a soft pale band. Broad face, cap and hair outline survive. Fine lids, nostrils and tooth boundaries are not recovered.',
        'Dim eyes, weak nose and soft smile remain inside the face silhouette. Dark cap and hair are retained. R2 does not provide distinct lip, eyelid or hair detail.',
        'Eyes, nose direction and broad smile are readable with some softness. Cheek/chin geometry, cap and hair survive. The broad structure is usable, but R2 has no convincing added fine clarity.',
        'Eyes and nose are indistinct and smile is a faint patch. Cap, face and hair silhouette remain. The compound input needs a clearer crop for useful facial structure.',
    ],
    'tr_ffhq_01210': [
        'Clear glasses, eye placement, nose and open smile survive. Cheek/chin geometry and long hair remain. Frames, teeth and skin are softened; R2 closely follows V31.',
        'Glasses and eyes merge into a diffuse upper-face band. Nose and open mouth are broad soft forms, with retained hair/face silhouette. R2 does not recover clear frame or lip edges.',
        'Dim glasses/eye shading, weak nose and soft mouth remain. Broad head and dark hair are retained. Fine eyelid, nostril and tooth cues stay poorly delineated.',
        'Broad clear-glasses band, nose direction and smile remain readable. Cheeks, jaw and long hair follow the input. R2 offers no distinct added frame, lip or hair-strand detail.',
        'Eyes/glasses, nose and mouth are barely delineated. Broad cheek/chin and hair outline survive, but the compound input requires a clearer crop for useful finer structure.',
    ],
    'tr_asian_00048': [
        'Eye placement, nose direction and closed lip line remain. Cheek/jaw proportions, short hair and facial hair survive. R2 preserves broad appearance with smoothed fine texture.',
        'Eyes are diffuse marks, nose a broad bright form and mouth/facial hair a soft line. Jaw and head outline remain. R2 supplies little extra eyelid, nostril or moustache definition.',
        'Dim eye positions, nose shadow and closed mouth remain softly. Cheek/jaw placement and facial-hair appearance survive. R2 does not establish added low-light fine structure.',
        'Eye placement, nose/nostril locations and lip line remain broad but readable. Jaw and facial hair survive. The image is smoother without distinct R2 fine-detail improvement.',
        'Eye details are vague, nose highly diffuse and mouth/moustache merge. Only broad cheek, jaw and hair outline remain. A clearer crop is needed.',
    ],
    'tr_asian_00133': [
        'Narrow eyes, small nose and open smile remain. Rounded cheeks/chin and fringe placement survive. R2 retains the broad expression with softened fine skin detail.',
        'Eyes are narrow dark bands; nose and open mouth remain soft. Rounded cheek and fringe silhouette survive. Lids, nostrils and lip boundaries do not gain convincing clarity.',
        'Dark eye positions, low-contrast nose and dim open mouth remain. Round face and hair outline survive. R2 closely follows V31 without useful extra fine definition.',
        'Narrow eye placement, nose and open lip shape remain readable but soft. Rounded cheek/chin geometry and fringe survive. Added facial definition over V31 is weak.',
        'Eyes are broad dark strips, nose poorly delineated and mouth a diffuse horizontal opening. Round face silhouette survives with degraded tone. A clearer crop is needed.',
    ],
    'tr_asian_00176': [
        'Large dark eye placement, small nose and pursed mouth remain. Rounded cheeks/chin, cap edge and bright collar survive. Skin and eyelid boundaries remain soft.',
        'Eyes become diffuse dark dots, nose a faint central form and mouth a soft curved line. Round face and collar geometry remain. R2 has little extra fine structural definition.',
        'Dim eye dots, weak nose and pursed mouth survive. Cheek/chin, cap and coloured collar remain. Fine lids, nostrils and lips are not distinctly recovered.',
        'Eye placement, small nose and lip curve remain broad and readable. Rounded cheeks/chin and collar survive. R2 is close to V31, with no distinct finer boundary gain.',
        'Dark eye marks and weak mouth survive while nose is indistinct. Round face and collar silhouette remain under degraded colour. A clearer crop is needed for useful finer features.',
    ],
    'tr_asian_00180': [
        'Broad upward/sideways eye direction, nose rim and open mouth remain. Rounded cheek/chin geometry, hair and red garment survive. R2 is softer than the paired target and close to V31.',
        'Eyes are dark soft spots; nose and mouth are broad shapes. Rounded face and red garment remain. Precise gaze, nostril and lip-corner definition are not recovered.',
        'Dim dark eye placement, weak nose and open mouth survive softly. Cheek/chin outline and garment remain. R2 has no convincing extra low-light facial clarity.',
        'Eye locations, nose rim and open lip shape remain readable. Rounded face, hair and garment survive. Fine gaze and mouth corners remain soft, with little R2/V31 difference.',
        'Eyes are indistinct dots, nose poorly defined and mouth a diffuse dark opening. Only broad rounded head and garment silhouette survive. A clearer crop is needed.',
    ],
    'tr_asian_00196': [
        'Narrow eyes/brows, nose and smiling mouth remain. Jaw/cheeks and the hand beside the face survive. R2 preserves broad appearance with softened skin and tooth detail.',
        'Eyes/brows and nose are diffuse; smile a weak curved line. Broad cheeks/jaw and adjacent hand placement remain. Smoothing does not restore useful fine eye or lip definition.',
        'Dark eye/brow placement, faint nose and dim smile remain. Broad head and side hand survive. R2 has little added nostril, lip or eyelid clarity.',
        'Narrow eyes/brows, nose direction and smile remain readable but soft. Cheeks/jaw and adjacent hand survive. R2 is close to V31 without convincing whole-face extra clarity.',
        'Eyes and nose are indistinct and mouth barely readable. Broad cheek and side-hand silhouette remain. A clearer crop is needed for useful finer structure.',
    ],
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    prepared, plan, checked = map(read, [OUT / 'preparation.json', OUT / 'plan.json', OUT / 'independent_preparation_audit.json'])
    assert checked['complete'] and checked['preparation_sha256'] == sha(OUT / 'preparation.json')
    assert checked['exact_256_cells_verified'] == 250 and not checked['preservation_regressions_at50']
    assert not prepared['early_stop']['pass']
    rows, sheets = [], []
    for sheet in prepared['sheets']:
        reference = sheet['reference']
        assert len(OBSERVATIONS[reference]) == len(sheet['cases']) == 5
        assert sha(OUT / sheet['file']) == prepared['sheet_sha256'][sheet['file']]
        sheets.append({'file': sheet['file'], 'sha256': sha(OUT / sheet['file']),
                       'actually_viewed': True, 'requested_detail': 'original', 'dimensions': [1336, 1516]})
        for cid, profile, observation in zip(sheet['cases'], PROFILES, OBSERVATIONS[reference], strict=True):
            assert cid.endswith('_' + profile)
            rows.append({'id': cid, 'reference': reference, 'profile': profile,
                         'regions_inspected': plan['regions'], 'observation': observation,
                         'V32_r2_vs_V31': 'Small changes; no convincing incremental whole-face clarity gain established.',
                         'paired_target_is_TRAIN_only': True,
                         'clearer_crop_needed_for_useful_fine_structure': profile == 'compound_lr24',
                         'input_only_eligibility_unchanged': True})
    assert [r['id'] for r in rows] == plan['case_ids'] and len(rows) == 50 and len(sheets) == 10
    receipt = {'complete': True, 'recorded_UTC': datetime.now(timezone.utc).isoformat(),
               'recorder_sha256': sha(Path(__file__)), 'preparation_sha256': sha(OUT / 'preparation.json'),
               'independent_preparation_audit_sha256': sha(OUT / 'independent_preparation_audit.json'),
               'cases_reviewed': 50, 'comparison_cells_reviewed': 250, 'rows': rows, 'sheets': sheets,
               'reviewer': 'Implementing Codex assistant, development TRAIN review, not independent human final review.',
               'useful_incremental_whole_face_gain_established': False, 'early_numerical_failure_waived': False,
               'native_or_reserved_used': False, 'identity_or_ethnicity_claim': False, 'Zamboanga_performance_claim': False,
               'neural_or_gradient_calls': 0, 'optimizer_updates': 0, 'app_promotion': False,
               'independent_final_review': False, 'goal_complete': False}
    with (OUT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({'complete': True, 'cases_reviewed': 50, 'comparison_cells_reviewed': 250,
                      'useful_incremental_whole_face_gain_established': False}))


if __name__ == '__main__':
    main()
