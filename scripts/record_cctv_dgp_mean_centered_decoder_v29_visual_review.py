"""Primary-assistant observations from all ten actual V29 TRAIN sheets."""
from pathlib import Path
import json

from prepare_cctv_dgp_mean_centered_decoder_v29_visual_review import sha, read, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_visual_review'
NOTES = {
    'tr_ffhq_00084': [
        'Clear glasses, eyes, nostrils and open lips remain visible; cheeks and skin stay smoother than the input and paired target.',
        'Blur has clearer eye, nostril, lip and jaw boundaries than unchanged DGP; clear-glasses frames remain unreadable and mouth/expression differ from the target.',
        'Lowlight gains readable coarse eyes, nostrils and lip contour; glasses and exact eye/gaze appearance remain unresolved.',
        'Motion gains eye, nose and lip structure and a clearer face outline; readable glasses and exact gaze are still absent.',
        'Compound gains coarse eyes, nose and lips, with uneven dark eye contrast; glasses and finer expression remain unresolved.',
    ],
    'tr_ffhq_00323': [
        'Clear eyes, nose and closed mouth remain readable, with stronger boundaries and smoother wrinkles than the target.',
        'Blur gains distinct coarse eyes, nose and closed lips; dark eye detail and skin texture remain estimated and diffuse.',
        'Lowlight gains eyes, nostrils, mouth and cheek contour; the broad expression remains, without exact fine eye structure.',
        'Motion gains clearer eye, nose and lip boundaries; the broad face arrangement survives, while wrinkles and gaze stay soft.',
        'Compound has more readable eyes, nose, lips and outline; dark eyes and blurred fine expression remain unlike the paired target.',
    ],
    'tr_ffhq_00178': [
        'Clear large eyes, smile, hair and face outline retain their broad appearance, with some baseline skin smoothing.',
        'Blur gains coarse eye, nose, smile and outline structure; dark eyes and uncertain tooth boundaries still differ from the target.',
        'Lowlight gains eyes, nose and smile contrast; eyelids, gaze and teeth remain soft estimates.',
        'Motion gains readable eye/nose/smile boundaries; finer eyes and tooth arrangement are still unresolved.',
        'Compound gains coarse dark eyes and lips; the smile becomes flatter and finer expression remains uncertain.',
    ],
    'tr_ffhq_00616': [
        'Clear eyes, nose, smile, cap, hair, earrings and visible hand remain present, with smoother skin and tooth edges.',
        'Blur gains eye, nose and smile boundaries and a clearer outline; cap and broad hair remain, while teeth are diffuse.',
        'Lowlight gains coarse eyes, nose and smiling lips; earrings and fine eye/tooth detail remain difficult to read.',
        'Motion gains readable eyes and smile; cap, hair, earrings and visible hand remain broadly present.',
        'Compound gains coarse eyes, nose and smile, but finer teeth and exact expression stay unresolved.',
    ],
    'tr_ffhq_01210': [
        'Clear mildly turned eyes, glasses, nose, teeth and outline remain visible; skin and fine glasses detail stay smoothed.',
        'Blur gains eyes, nose and smiling lips; glasses remain faint or unreadable and lips/tooth boundaries smear.',
        'Lowlight gains nose, eye and mouth boundaries; exact glasses, gaze and smiling tooth detail stay uncertain.',
        'Motion gains eyes, nostrils, mouth and face contour; glasses are only faintly suggested and tooth detail is soft.',
        'Compound gains coarse eyes and mouth, with faint glasses and uneven lip/cheek detail; expression remains an estimate.',
    ],
    'tr_asian_00048': [
        'Clear eyes, nostrils, lips, facial hair and outline remain present; boundary contrast changes while the broad appearance survives.',
        'Blur gains nostrils, lip contour and cheek outline; eyes/gaze and facial hair remain diffuse and differ from the target.',
        'Lowlight gains nose, mouth and cheek structure; finer eyes and facial-hair pattern remain uncertain.',
        'Motion gains eye, nostril and lip boundaries; exact gaze and facial hair stay soft.',
        'Compound gains a more readable nose and lips, while dark eyes, facial hair and fine expression remain diffuse.',
    ],
    'tr_asian_00133': [
        'Clear eyes, nose and open mouth remain readable; dark eyelid and lip boundaries become slightly stronger.',
        'Blur gains narrow dark eyes, nostrils and lip contour; the open-mouth shape and fine expression still differ from the target.',
        'Lowlight gains coarse eyes, nose and mouth; exact lip shape, gaze and tooth detail remain unresolved.',
        'Motion gains eyes, nose and open-mouth boundaries; finer eye shape and mouth detail remain estimated.',
        'Compound gains coarse eyes and lips; nose and fine expression remain diffuse with uneven contrast.',
    ],
    'tr_asian_00176': [
        'Clear eyes, nose, closed lips, cheeks, clothing and outline remain broadly present; eye boundaries are somewhat darker.',
        'Blur gains nose, lip and cheek boundaries; uneven dark eye shapes and uncertain gaze differ from the paired target.',
        'Lowlight gains eyes, nose and closed lips; fine eyelids and cheek/mouth structure stay soft.',
        'Motion gains readable eye, nose and mouth arrangement and a clearer outline; exact eyes and fine expression stay unresolved.',
        'Compound gains coarse eyes and mouth, with uneven eye contrast and diffuse nose and outline details.',
    ],
    'tr_asian_00180': [
        'Clear mildly turned eyes, nose and open mouth retain the broad appearance; smoothing remains in fine skin and mouth detail.',
        'Blur gains coarse eyes, nostrils and open lips; precise gaze and mouth shape still differ from the target.',
        'Lowlight gains readable eyes, nose and mouth; lips and fine expression remain soft estimates.',
        'Motion gains eye/nose/open-mouth boundaries and face contour; finer gaze and expression stay uncertain.',
        'Compound gains dark coarse eyes and mouth; nose and detailed expression remain diffuse.',
    ],
    'tr_asian_00196': [
        'Clear smile, eyes, eyebrows, outline and visible hand remain present; stronger local boundaries retain the broad expression.',
        'Blur gains eye, nose, smile and outline structure; gaze and tooth detail remain diffuse, with the visible hand retained.',
        'Lowlight gains a more readable smile, nose and eyes; fine eyelids and mouth detail remain unresolved.',
        'Motion gains readable eyes, nose and smile; the broad hand/face arrangement survives with uncertain exact gaze.',
        'Compound gains coarse eyes, nose and smiling lips, with uneven eye contrast and diffuse finer expression.',
    ],
}


def main():
    prep, plan = read(OUT / 'preparation.json'), read(OUT / 'plan.json')
    assert prep['complete'] and prep['exact_source_cells_verified'] == 200
    assert sha(OUT / 'plan.json') == prep['plan_sha256']
    audit = read(ROOT / 'outputs/cctv_dgp_mean_centered_decoder_v29_independent_audit.json')
    assert audit['complete'] and audit['necessary_capacity_pass']
    assert set(NOTES) == {s['reference'] for s in prep['sheets']}
    rows = []
    for sheet in prep['sheets']:
        assert sha(OUT / sheet['file']) == prep['sheet_sha256'][sheet['file']]
        assert len(NOTES[sheet['reference']]) == len(sheet['cases']) == 5
        for cid, note in zip(sheet['cases'], NOTES[sheet['reference']]):
            case = next(r for r in prep['rows'] if r['id'] == cid)
            rows.append({'id': cid, 'reference': case['reference'], 'source': case['source'],
                         'profile': case['profile'], 'sheet': sheet['file'],
                         'regions_reviewed': plan['regions'], 'note': note,
                         'checkpoint_qualified_for_app': False})
    assert [r['id'] for r in rows] == plan['case_ids'] and len(rows) == 50
    write(OUT / 'visual_review.json', {
        'complete': True, 'date': '2026-10-06', 'reviewer': 'Codex primary assistant',
        'method': 'Viewed every actual1072x1516 sheet with original image detail; all200 unchanged256x256 cells compared across input, original DGP, V29 final800 and paired TRAIN target.',
        'cases_reviewed': 50, 'exact_source_cells': 200, 'rows': rows,
        'preparation_sha256': sha(OUT / 'preparation.json'),
        'sheet_sha256': prep['sheet_sha256'], 'runner_sha256': sha(Path(__file__)),
        'necessary_capacity_pass': True,
        'conclusion': 'Coarse eye/nose/mouth arrangement and face outline are more readable across degraded TRAIN examples; the original quantitative preservation gates pass. Strong degradation still leaves glasses, gaze, teeth and expression uncertain or altered. This supports separately frozen development inference, not app qualification or recovery of exact identity.',
        'training_visual_coverage_complete': True,
        'broader_development_review_justified': True,
        'source_labels_are_not_ethnicity': True, 'native_or_reserved_used': False,
        'independent_final_review': False, 'app_promotion': False, 'goal_complete': False,
    })
    print(json.dumps({'complete': True, 'cases_reviewed': 50, 'next': 'Frozen development inference'}))


if __name__ == '__main__':
    main()
