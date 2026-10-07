"""Record all ten sheets actually viewed by the primary assistant at original detail."""
import json
from pathlib import Path

from prepare_cctv_dgp_active_original_decoder_v28_visual_review import sha, read, write

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_active_original_decoder_v28_visual_review'
NOTES = {
    'tr_ffhq_00084': [
        'Clear glasses, eyes, nostrils and open mouth remain present, with slightly stronger boundaries and baseline smoothing.',
        'Blur has more distinct eye, nostril and lip boundaries, but clear-glasses frames remain unresolved and eye/nose appearance differs from the target.',
        'Lowlight gains coarse eyes, nose and mouth boundaries; glasses remain unresolved, and finer visible structure remains soft.',
        'Motion has stronger eye, nose and lip contrast; glasses and exact eye/gaze detail remain unresolved.',
        'Compound has clearer coarse eyes and lips, with uneven dark eye contrast and absent readable glasses; overall finer appearance remains unresolved.',
    ],
    'tr_ffhq_00323': [
        'Clear eyes and closed mouth are slightly more defined; wrinkles and visible appearance remain softened.',
        'Blur gains coarse eye, nose, lip and cheek boundaries, with very dark eyes and unresolved fine wrinkles.',
        'Lowlight gains eye and nose boundaries and a more readable mouth/outline, while remaining diffuse.',
        'Motion gains eye and lip contrast and keeps the coarse expression; fine eye/skin structure remains unresolved.',
        'Compound gains coarse eyes, nose, mouth and outline; dark eyes and softened expression still differ from the paired training target.',
    ],
    'tr_ffhq_00178': [
        'Clear eyes, smile and hair retain their broad appearance, with modest boundary contrast changes.',
        'Blur gains eye, nose and smile boundaries, but eyes are very dark and tooth/gaze detail remains unresolved.',
        'Lowlight gains coarse eye, nose and mouth structure; finer appearance remains soft and smile detail differs from the target.',
        'Motion gains eye/nose/mouth contrast and a readable smile, without establishing exact eyes or teeth.',
        'Compound gains coarse eyes and mouth, with strongly darkened eyes, altered smile detail and unresolved fine features.',
    ],
    'tr_ffhq_00616': [
        'Clear eyes and smile are modestly more defined; cap, earrings, hair and the visible hand remain present.',
        'Blur gains eye, nose and smile structure and preserves the coarse face outline; fine eyes and teeth remain soft.',
        'Lowlight gains eye/nose/mouth contrast; expression is more readable, with unresolved fine eye and tooth boundaries.',
        'Motion gains stronger eye and smile boundaries; hair, cap, earrings and visible hand remain present.',
        'Compound gains coarse eyes, nose and smile, while expression and tooth detail remain diffuse.',
    ],
    'tr_ffhq_01210': [
        'Clear mildly turned eyes, glasses, nose and smile remain present with slight contrast changes.',
        'Blur gains coarse eyes, nose and smile, but glasses and lip/tooth boundaries remain smeared.',
        'Lowlight gains nose and smile definition; readable clear-glasses and exact eye detail remain unresolved.',
        'Motion gains nose, mouth and outline definition; clear-glasses and exact eye/gaze detail remain soft.',
        'Compound gains coarse eyes and mouth; glasses remain unreadable and lips have smeared/doubled-looking boundaries.',
    ],
    'tr_asian_00048': [
        'Clear eyes, nostrils, lips and facial hair show contrast/softness changes; small changes do not waive measured clear-group failures.',
        'Blur gains coarse eyes, nostrils, lips and cheek outline, while eyes/gaze and facial-hair detail differ from the paired target.',
        'Lowlight gains nose and mouth boundaries, retaining diffuse eye and facial-hair detail.',
        'Motion gains eye, nostril and lip boundaries, with unresolved fine hair and exact eye/gaze appearance.',
        'Compound gains a more readable nose/mouth and coarse outline; eyes and facial-hair structure remain diffuse.',
    ],
    'tr_asian_00133': [
        'Clear eyes, nose and open mouth remain readable; stronger dark eye/mouth boundaries alter fine appearance slightly.',
        'Blur gains coarse eye, nostril and lip boundaries, but eye shape is very dark/narrow and mouth detail remains unresolved.',
        'Lowlight gains eye/nose/mouth contrast; exact gaze, lips and visible fine structure remain unresolved.',
        'Motion gains eye and lip contrast, with altered narrow eye appearance and softer mouth detail than the target.',
        'Compound gains coarse eyes and lips with uneven nose/mouth detail; fine expression remains diffuse.',
    ],
    'tr_asian_00176': [
        'Clear eyes and mouth boundaries are slightly stronger; hair, cheeks, chin and clothing remain present.',
        'Blur gains dark eye, nose and mouth boundaries; eye/gaze appearance and finer cheek/mouth structure differ from the target.',
        'Lowlight gains coarse eyes, nose and mouth; finer visible structure stays soft.',
        'Motion gains stronger eye and lip contrast, with dark eye/gaze changes and a softened outline.',
        'Compound has more readable coarse eyes and mouth, with uneven eye contrast and diffuse nose/outline detail.',
    ],
    'tr_asian_00180': [
        'Clear mildly turned eyes, nose and open mouth remain readable, with modest contrast/boundary changes.',
        'Blur gains eye, nose and open-mouth boundaries; precise gaze and mouth shape remain different/soft.',
        'Lowlight gains coarse eye and mouth structure; fine nose and expression detail remain unresolved.',
        'Motion gains eye/nose/mouth contrast and a readable outline; exact eye/gaze detail stays unresolved.',
        'Compound gains dark coarse eyes and mouth, while nose and finer expression remain diffuse.',
    ],
    'tr_asian_00196': [
        'Clear smile, eyes and visible hand remain present; stronger local contrast changes fine appearance.',
        'Blur gains eye/nose/mouth and outline contrast, but gaze and smile detail differ from the target; the visible hand remains.',
        'Lowlight gains a more readable coarse smile and nose; eyes and mouth detail remain diffuse.',
        'Motion gains eye/nose/mouth contrast and keeps the broad smile/hand, with unresolved exact gaze and finer expression.',
        'Compound has somewhat clearer coarse eyes and smile, with uneven dark eye contrast and diffuse fine detail.',
    ],
}


def main():
    prep = read(OUT / 'preparation.json');plan = read(OUT / 'plan.json')
    assert prep['complete'] and prep['exact_source_cells_verified'] == 200
    assert sha(OUT / 'plan.json') == prep['plan_sha256']
    assert set(NOTES) == {row['reference'] for row in prep['sheets']}
    rows = []
    for sheet in prep['sheets']:
        assert sha(OUT / sheet['file']) == prep['sheet_sha256'][sheet['file']]
        for cid, note in zip(sheet['cases'], NOTES[sheet['reference']]):
            case = next(row for row in prep['rows'] if row['id'] == cid)
            rows.append({'id': cid, 'reference': case['reference'], 'source': case['source'],
                         'profile': case['profile'], 'sheet': sheet['file'], 'regions_reviewed': plan['regions'],
                         'note': note, 'checkpoint_qualified_for_app': False})
    assert len(rows) == 50 and [row['id'] for row in rows] == plan['case_ids']
    review = {'complete': True, 'date': '2026-10-06', 'reviewer': 'Codex primary assistant',
              'method': 'Viewed all ten actual1072x1516 sheets using original image detail, comparing each unchanged256x256 input, original DGP, V28 final800 and paired photographic TRAIN target.',
              'cases_reviewed': 50, 'exact_source_cells': 200, 'rows': rows,
              'preparation_sha256': sha(OUT / 'preparation.json'), 'sheet_sha256': prep['sheet_sha256'],
              'runner_sha256': sha(Path(__file__)),
              'conclusion': 'Several degraded TRAIN faces have visibly stronger coarse eye/nose/mouth boundaries and more readable overall facial arrangement. Finer eyes, glasses, gaze and expression remain unresolved or altered in strong degradation. All original capacity/preservation/brightness failures remain; these ten training references do not establish generalization or native CCTV usefulness.',
              'source_labels_are_not_ethnicity': True, 'native_or_reserved_used': False,
              'independent_final_review': False, 'app_promotion': False, 'goal_complete': False}
    write(OUT / 'visual_review.json', review)
    print(json.dumps({'complete': True, 'cases_reviewed': 50, 'fixed_gate_failures_retained': True}))


if __name__ == '__main__':
    main()
