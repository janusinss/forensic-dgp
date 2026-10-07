"""Record the primary agent's completed original-resolution review of all50 TRAIN rows."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'outputs/cctv_dgp_broader_mean_v30_failure_review_v1'
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
REGIONS = ['eyes', 'nose', 'mouth', 'face outline', 'overall visible appearance']

# One observed row for every profile, in the actual sheet order. These are
# visual impressions of the displayed images, not an identity correctness score.
OBSERVATIONS = {
    'tr_ffhq_00084': [
        'Clear glasses, eyes, nose, open lips, hair and earring remain broadly present; skin and eye boundaries are already soft in the original DGP.',
        'Eyes and glasses frames are unreadable; nose and lip contours are diffuse. V30 adds little visible definition.',
        'Nostrils and lips remain coarse and soft; eyes and glasses frames are unresolved. Small contrast changes do not provide full facial clarity.',
        'Broad nose and lip arrangement is readable, but eyes and frames remain vague. Face outline and hair remain soft.',
        'The dark face is very diffuse; eyes, nose and mouth lack convincing detail. V30 does not provide useful whole-face clarity.',
    ],
    'tr_ffhq_00323': [
        'Eyes, nose, closed-mouth smile and face outline remain broadly present; wrinkles are softened by the original DGP.',
        'Coarse dark eyes and fuzzy lip contours remain. Nose and outline are soft; the V30 change is small.',
        'Coarse eye contrast and mouth contour are present, with a soft nose and outline. V30 does not resolve fine expression detail.',
        'Eye, nose, nasolabial and lip arrangement is readable at a coarse level. V30 adds little definition across the face.',
        'Eyes, nose and mouth remain diffuse and close to the original DGP. Face outline is broad, without a convincing clarity gain.',
    ],
    'tr_ffhq_00178': [
        'Wide eyes, smile, teeth, hair and outline are broadly retained; the original DGP is smoother than the input.',
        'Eyes, nose and smile remain coarse; tooth and lip shapes are diffuse. V30 is close to the original DGP.',
        'Coarse eyes and smile are readable, but precise eye and tooth details are absent. Nose and face outline remain soft.',
        'Smile contrast and mouth edge are present, but tooth alignment and eye details remain unresolved. Broad appearance is retained.',
        'Dark coarse eyes remain; fine smile and tooth details are absent. Nose and outline are soft, with little extra V30 clarity.',
    ],
    'tr_ffhq_00616': [
        'Smile, eyes, nose, cap, hair, earrings and visible hand are broadly retained; tooth edges are soft.',
        'Eyes, smile and nose are diffuse. Cap, hair and broad outline remain; V30 adds little definition.',
        'Coarse eyes and bright teeth remain soft; fine lip and eye details are unreadable. Nose and outline are also coarse.',
        'The broad smile and facial arrangement are readable; fine tooth and lip patterns remain uncertain. Cap and visible hand remain present.',
        'Dark eyes and mouth are diffuse and the nose has little definition. Outline and accessories remain broad; V30 changes are minor.',
    ],
    'tr_ffhq_01210': [
        'Mild turn, glasses, nose, teeth, hair and outline are broadly retained; the original DGP is smooth.',
        'Eyes, glasses and teeth are unreadable. Nose, lips and face outline remain very soft; V30 is close to baseline.',
        'Coarse nose and lips remain, with faint frames and unresolved eyes and teeth. Broad pose and outline are retained.',
        'Glasses frames are faintly recognizable, but eye and tooth details are diffuse. Nose and lips remain coarse.',
        'Eyes and mouth are very soft. The broad pose remains, but V30 provides no convincing structural advantage.',
    ],
    'tr_asian_00048': [
        'Eyes, nose, closed lips, moustache, hair and outline are broadly present; fine moustache and skin detail are softened.',
        'Nose and lips are coarse while eyes and facial hair are diffuse. V30 adds little definition to the soft outline.',
        'Coarse eyes, nose and mouth remain with a small contrast change. Facial hair and outline stay soft.',
        'Coarse eye, nose and lip arrangement is readable, but finer detail is soft. Hair and outline remain broadly present.',
        'Eyes, mouth and outline are dark and diffuse; nose boundaries are soft. V30 changes are minimal.',
    ],
    'tr_asian_00133': [
        'Narrow dark eyes, nose, open mouth, hair and outline are present; the paired photographic target itself has limited resolution.',
        'Horizontal eyes and open mouth are coarsely readable. Nose and lips are diffuse and precise mouth shape differs from the paired target.',
        'Coarse eyes and mouth are dark; finer lips and teeth are unreadable. Nose and outline remain soft with only small changes.',
        'Coarse eyes, nose and mouth remain, but finer expression is uncertain. The broad outline remains visible.',
        'The open mouth collapses into a dark line, with very soft nose and eye features. V30 does not resolve the expression.',
    ],
    'tr_asian_00176': [
        'Large asymmetric dark eyes, nose, closed lips, cheeks and clothing outline remain broadly present; the original DGP is smooth.',
        'Eyes are dark spots; nose, lips and face outline remain soft. V30 adds little visible detail.',
        'Coarse eyes and mouth line remain with a hazy nose. The outline is broad and soft.',
        'Coarse eyes, nose and mouth arrangement is readable, while gaze and eye details remain uncertain. Broad appearance remains.',
        'Diffuse eye clusters, nose and mouth remain, with a soft outline close to the original DGP.',
    ],
    'tr_asian_00180': [
        'Mildly turned eyes, nose, open mouth and outline remain broadly present; skin detail is smooth.',
        'Coarse eyes and open mouth are readable, but nose and lip edges are diffuse. V30 is close to the original DGP.',
        'Eyes are dark spots and the mouth is an open dark shape. The nose is very soft and the outline remains broad.',
        'Coarse eyes, nose, open mouth and outline are retained, but finer gaze and lip details remain diffuse.',
        'Eyes and mouth remain dark shapes; broad pose is present, with very soft nose and lip contours. V30 changes are minimal.',
    ],
    'tr_asian_00196': [
        'Smile, eyes, nose, hair, outline and visible hand remain broadly present; the original DGP is already soft.',
        'Eyes, nose, smile and hand are readable at a coarse level; fine mouth and eye details remain diffuse.',
        'Coarse eyes, nose and smile remain very soft. Outline and visible hand are retained without convincing finer definition.',
        'This is the only fixed preview case exposed by update50. Broad smile, eyes, nose and hand remain; the small V30 change does not qualify as useful whole-face clarity.',
        'Dark eyes, nose and lips are diffuse. V30 is close to the original DGP and does not resolve fine expression detail.',
    ],
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def main():
    target = FOLDER / 'visual_review.json'
    assert not target.exists(), 'Preserve the prior review'
    prep = read(FOLDER / 'preparation.json'); plan = read(FOLDER / 'plan.json')
    assert prep['complete'] and prep['exact_cells_checked'] == 250
    assert sha(FOLDER / 'plan.json') == prep['plan_sha256']
    assert plan['regions'] == REGIONS and not plan['image_processing_or_resize']
    for name, digest in prep['sheet_sha256'].items(): assert sha(FOLDER / name) == digest
    for name, digest in prep['source_bindings_sha256'].items(): assert sha(ROOT / name) == digest
    observed = []
    lookup = {r['id']: r for r in prep['rows']}
    for sheet in prep['sheets']:
        ref = sheet['reference']; notes = OBSERVATIONS[ref]
        assert len(notes) == len(sheet['cases']) == 5
        for profile, cid, note in zip(PROFILES, sheet['cases'], notes):
            row = lookup[cid]; assert row['profile'] == profile and row['reference'] == ref
            observed.append({'id': cid, 'reference': ref, 'source': row['source'], 'profile': profile,
                             'sheet': sheet['file'], 'regions_inspected': REGIONS,
                             'exposed_by50': row['exposed_by50'], 'observations': note,
                             'V30_useful_whole_face_gain_established': False,
                             'identity_accuracy_inferred': False})
    assert [r['id'] for r in observed] == plan['case_ids'] and len(observed) == 50
    audit_path = ROOT / 'outputs/cctv_dgp_broader_mean_v30_independent_audit_r1.json'
    if audit_path.exists():
        audit = read(audit_path)
        assert audit['complete'] and audit['VM_failure_retained'] and not audit['necessary_capacity_pass']
    report = {'complete': True, 'date': '2026-10-07', 'reviewer': 'Primary Codex agent',
              'method': 'All10 sheets actually viewed at original resolution; all50 rows and five comparison columns inspected before recording',
              'cases_reviewed': 50, 'comparison_cells_reviewed': 250, 'sheets_reviewed': 10,
              'plan_sha256': sha(FOLDER / 'plan.json'), 'preparation_sha256': sha(FOLDER / 'preparation.json'),
              'recorder_source_sha256': sha(Path(__file__)), 'sheet_sha256': prep['sheet_sha256'],
              'R1_audit_sha256': sha(audit_path) if audit_path.exists() else None,
              'R1_audit_pending_at_recording': not audit_path.exists(), 'rows': observed,
              'scope': 'Photographic paired synthetic TRAIN diagnostic only; unexposed TRAIN is not held-out evidence',
              'summary': 'Broad visible appearance is generally retained, but the stopped V30 outputs remain soft and close to the unchanged DGP. Small contrast or boundary changes do not establish useful whole-face clarity. V29 update50 often has stronger coarse boundaries, without qualifying either candidate on development data.',
              'all_visible_features_remain_important': True, 'early_one_percent_failure_retained': True,
              'V30_resume_permitted': False, 'native_or_reserved_used': False,
              'independent_final_review': False, 'app_promotion': False, 'goal_complete': False}
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: report[k] for k in ['complete', 'cases_reviewed', 'comparison_cells_reviewed', 'R1_audit_pending_at_recording', 'app_promotion', 'goal_complete']}))


if __name__ == '__main__':
    main()
