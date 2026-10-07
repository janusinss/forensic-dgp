"""Record observations made after viewing all ten exact-pixel TRAIN sheets."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_profile_batches_v31_failure_review_v1'
PROFILES = ['clear', 'blur_lr24', 'lowlight_lr32', 'motion_lr48', 'compound_lr24']
# Every row below was inspected in the original-detail view_image sheets.
# Order in each row: eyes, nose, mouth, outline, visible appearance.
OBSERVATIONS = {
    'tr_ffhq_00084': [
        ['Eye placement and clear-glasses shape remain visible.', 'Nostrils and nose direction remain readable.', 'Open lips and tooth band remain visible.', 'Jaw and cheeks retain their broad shape.', 'Light hair and earrings remain; skin and hair texture are softened.'],
        ['Eyes and glasses merge into a diffuse horizontal area.', 'Nose is a broad light shape with weak nostril definition.', 'Lip opening is soft; individual teeth are unresolved.', 'Cheek and jaw silhouette survives without fine contours.', 'Block edges are smoother; glasses and hair detail remain unresolved.'],
        ['Dark eye area lacks readable frame and eyelid edges.', 'Nose direction is broad and nostrils are indistinct.', 'Dark mouth opening survives with weak lip boundaries.', 'Rounded cheeks remain visible but soft.', 'Low-light tone and hair silhouette remain; no clear added detail.'],
        ['Eye locations survive but clear frames are poorly delineated.', 'Nostrils are more readable than in the blur row but remain soft.', 'Open lips remain, without fine tooth separation.', 'Jaw and cheek placement remain broad.', 'Smoothing reduces blocks; hair, glasses and earrings lack target detail.'],
        ['Eye and glasses information is highly diffuse.', 'Nose anatomy is not reliably delineated.', 'A dark mouth patch has little usable lip detail.', 'Only broad face and hair silhouette remains.', 'Severe dark compound input needs a clearer crop for useful structure.'],
    ],
    'tr_ffhq_00323': [
        ['Eye positions and gaze broadly follow the input.', 'Nose tip and nostrils remain readable.', 'Closed slight smile remains.', 'Broad jaw and ears remain aligned.', 'Short grey hair and skin folds are smoothed relative to input.'],
        ['Eyes become dark soft patches.', 'Nose shading survives with diffuse edges.', 'Thin smile is visible but lip detail is weak.', 'Jaw silhouette survives.', 'Block edges reduce; forehead and short hair remain feature-poor.'],
        ['Dark eye positions remain without clear iris edges.', 'Nose shape is coarse and low contrast.', 'Slight smile survives but lacks sharp corners.', 'Cheeks and jaw remain soft.', 'Dim tone and short-hair silhouette remain; clarity gain is small.'],
        ['Eye locations remain; detailed gaze is unresolved.', 'Nose tip and nostrils remain broad.', 'Smile is readable but softened.', 'Angular jaw remains visible.', 'Less blockiness does not recover fine hair or skin structure.'],
        ['Eyes are indistinct dark patches.', 'Nose is mostly a central shadow.', 'Mouth is barely delineated.', 'Broad head shape remains without useful interior detail.', 'Dark compound case needs a clearer crop.'],
    ],
    'tr_ffhq_00178': [
        ['Wide eyes remain; eyelash detail is softened.', 'Small nose and nostrils remain visible.', 'Smile and tooth band remain.', 'Rounded cheeks and chin retain their shape.', 'Light bob hairstyle survives; fine hair and skin are smoothed.'],
        ['Eyes become dark pools without fine boundaries.', 'Nose has weak small-scale definition.', 'Smile opening survives but tooth detail is lost.', 'Rounded face silhouette remains.', 'Block reduction gives a smoother image without useful extra detail.'],
        ['Dark eyes remain but eyelid detail is weak.', 'Nose is diffuse and low contrast.', 'Smile remains; teeth are poorly separated.', 'Cheek outline stays broad.', 'Dim face and hair remain soft despite smoother blocks.'],
        ['Eye shapes are more readable than in the blur row but soft.', 'Nose placement remains without crisp nostrils.', 'Smile and pale tooth band survive without individual separation.', 'Chin and cheek arrangement remain.', 'Hair strands and facial detail remain below the paired target.'],
        ['Eyes are indistinct dark spots.', 'Nose cannot be reliably delineated.', 'Smile is a diffuse mouth patch.', 'Only broad cheek and chin shape survives.', 'Dark compound case lacks enough detail for a useful clear reconstruction.'],
    ],
    'tr_ffhq_00616': [
        ['Eye placement and broad gaze remain visible.', 'Nose shape is preserved with softened edges.', 'Wide smile and teeth remain readable.', 'Chin and cheeks retain their broad placement.', 'Graduation cap, dark hair and earrings remain; fine texture is soft.'],
        ['Eye and eyelid boundaries remain faint.', 'Nose is diffuse.', 'Smile survives as a soft tooth band.', 'Face silhouette remains under the cap.', 'Blocks reduce but cap, hair and jewellery detail stays soft.'],
        ['Eyes are low-contrast soft marks.', 'Nose has little defined anatomy.', 'Dim smile survives without clear tooth detail.', 'Cheeks and chin remain broad.', 'Low-light cap and hair silhouette remain without added clear structure.'],
        ['Eye positions are readable, with soft lids.', 'Nose remains broad without fine edge clarity.', 'Smile is readable; tooth boundaries are soft.', 'Cheek and chin placement remains.', 'Cap, dark hair and earrings remain without convincing incremental clarity.'],
        ['Eye detail is weak and diffuse.', 'Nose is not usefully defined.', 'Mouth is a faint soft opening.', 'A broad face silhouette survives.', 'Compound case remains too indistinct for useful facial detail.'],
    ],
    'tr_ffhq_01210': [
        ['Clear glasses and eye placement remain.', 'Nose direction and nostrils remain readable.', 'Open smile and tooth band remain.', 'Cheeks and chin broadly match input.', 'Long dark hair remains; hair, skin and clear lenses are softened.'],
        ['Glasses and eyes merge into diffuse brow-level shading.', 'Nose edges are weak.', 'Mouth is open but teeth and lip boundaries are unresolved.', 'Head outline remains broad.', 'Less blockiness does not restore useful clear-glasses detail.'],
        ['Glasses and eyelids remain poorly delineated.', 'Nose shading survives without crisp contours.', 'Dim open smile survives with weak lips.', 'Cheek and chin placement remains soft.', 'Low-light tone and dark hair remain without substantial clarity.'],
        ['A broad horizontal eye/glasses band remains.', 'Nose is visible but soft.', 'Smile and tooth band survive without fine boundaries.', 'Jaw and cheeks remain broad.', 'Block smoothing does not recover clear frames or hair strands.'],
        ['Eyes and glasses are barely readable.', 'Nose is diffuse.', 'Mouth is a faint low-contrast patch.', 'Broad face and hair silhouette remains.', 'Dark compound case needs a clearer crop.'],
    ],
    'tr_asian_00048': [
        ['Eye placement and broad gaze remain.', 'Nose tip and direction remain.', 'Closed mouth and moustache remain readable.', 'Cheek and jaw proportions broadly follow input.', 'Short hair and facial hair survive; fine texture is smoothed.'],
        ['Eyes become diffuse narrow marks.', 'Nose is a broad light/dark shape.', 'Mouth line and moustache remain soft.', 'Broad jaw survives.', 'Smoothing removes block edges without distinct facial-hair detail.'],
        ['Dark eyes remain without fine corners.', 'Nose shading is broad.', 'Mouth line and facial-hair patch remain.', 'Cheek and jaw outline remains soft.', 'Low-light tone remains; fine structure is unresolved.'],
        ['Eye placement survives with soft edges.', 'Nose tip and nostrils remain broad.', 'Mouth line and facial hair remain readable but soft.', 'Jaw placement broadly survives.', 'Texture is smoother without convincing extra facial detail.'],
        ['Eye region is hazy and lacks clear lids.', 'Nose anatomy is highly diffuse.', 'Mouth and moustache merge into a dark area.', 'Only broad cheek and jaw shape remains.', 'Severe compound case lacks useful fine structure.'],
    ],
    'tr_asian_00133': [
        ['Narrow dark eyes and their placement remain.', 'Nose and nostrils remain readable.', 'Open smile and lower lip remain.', 'Rounded cheek and chin shape remains.', 'Short fringe remains; cheek texture is smoother.'],
        ['Eyes remain dark narrow marks without fine edges.', 'Small nose is diffuse.', 'Open mouth survives but internal detail is soft.', 'Rounded outline remains.', 'Blocks reduce; hair fringe and skin remain soft.'],
        ['Dark eye placement survives without fine eyelids.', 'Nose is low contrast.', 'Open mouth and lower lip remain diffuse.', 'Broad cheeks and chin remain.', 'Dim colour remains without useful incremental detail.'],
        ['Eye locations remain but fine gaze is unresolved.', 'Nose and nostril locations survive softly.', 'Open mouth remains; internal detail is indistinct.', 'Rounded cheeks remain aligned.', 'Fringe and skin are smoother without a whole-face clarity gain.'],
        ['Eyes are dark narrow blurred marks.', 'Nose is poorly defined.', 'Mouth becomes a dark horizontal patch.', 'Broad cheek silhouette survives.', 'Green/magenta degraded tone and missing detail make this insufficient.'],
    ],
    'tr_asian_00176': [
        ['Wide eyes and broad gaze remain.', 'Small nose remains readable.', 'Pursed mouth remains.', 'Rounded cheeks and chin remain.', 'Cap edge and coloured collar survive with softer face texture.'],
        ['Eyes become dark dots with weak surrounding edges.', 'Nose is a diffuse central shape.', 'Mouth line remains soft.', 'Round face silhouette remains.', 'Blocks reduce without recovering fine eyelid or collar detail.'],
        ['Eye locations remain dark and soft.', 'Nose definition is low contrast.', 'Pursed mouth survives without clear lip edges.', 'Rounded jaw and cheeks remain.', 'Dim cap and collar remain; facial detail is weak.'],
        ['Eye locations remain without fine lid detail.', 'Nose tip remains broad.', 'Pursed lip line is visible but soft.', 'Round cheek and chin geometry survives.', 'Smoothing does not establish distinct added whole-face detail.'],
        ['Eye detail is ambiguous and diffuse.', 'Nose cannot be reliably delineated.', 'Mouth is a weak dark mark.', 'Only broad rounded silhouette remains.', 'Dark pink/purple compound distortion needs a clearer crop.'],
    ],
    'tr_asian_00180': [
        ['Broad sideways/upward gaze and eye placement remain.', 'Nose rim and nostrils remain visible.', 'Open mouth and lower lip remain.', 'Rounded cheek and chin placement remains.', 'Dark hair edge and red garment remain; skin is smoother.'],
        ['Eyes become dark spots; detailed gaze is lost.', 'Nose is a broad bright shape.', 'Mouth opening survives but lip boundaries are diffuse.', 'Rounded face silhouette survives.', 'Block smoothing does not restore gaze or fine lip structure.'],
        ['Dark eye locations remain without fine lids.', 'Nose has weak low-contrast anatomy.', 'Dark mouth opening remains.', 'Broad jaw and cheeks remain.', 'Dim tone and garment survive with little extra useful detail.'],
        ['Eye locations survive but precise gaze is soft.', 'Nose rim remains broad.', 'Lip opening is readable without fine corners.', 'Rounded cheek and chin outline remains.', 'Smoother face remains substantially below target detail.'],
        ['Eyes are indistinct dark dots.', 'Nose structure is poorly defined.', 'Mouth is a dark diffuse opening.', 'Only broad head outline survives.', 'Dark compound input needs a clearer crop for useful structure.'],
    ],
    'tr_asian_00196': [
        ['Narrow eyes and brows remain readable.', 'Nose shading and direction remain.', 'Smile and pale tooth line remain.', 'Jaw and cheek shape remains.', 'Hand beside the face remains visible; skin texture is softened.'],
        ['Eyes and brows become diffuse marks.', 'Nose is a weak broad shadow.', 'Smile line survives without tooth detail.', 'Cheek and chin silhouette remains.', 'Adjacent hand survives; smoother blocks do not restore fine anatomy.'],
        ['Dark eye and brow placement survives softly.', 'Nose shadow remains without clear fine edges.', 'Smile is a diffuse line.', 'Broad jaw and cheeks remain.', 'Dim face and adjacent hand remain without substantial clarity.'],
        ['Narrow eye and brow locations survive with soft edges.', 'Nose shading is broad.', 'Smile is readable without sharp tooth/lip boundaries.', 'Cheek and chin outline remains.', 'Hand placement remains; smoothing is not a convincing new structure gain.'],
        ['Eye and gaze details are uncertain.', 'Nose is highly diffuse.', 'Mouth is barely readable.', 'Broad cheek shape remains.', 'Dark face and hand silhouette remain; finer facial information is insufficient.'],
    ],
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def main():
    prepared, plan = read(OUT / 'preparation.json'), read(OUT / 'plan.json')
    checked = read(OUT / 'independent_preparation_audit.json')
    assert checked['complete'] and checked['preparation_sha256'] == sha(OUT / 'preparation.json')
    assert not prepared['early_stop']['pass'] and not checked['preservation_regressions_at50']
    rows, sheets = [], []
    for sheet in prepared['sheets']:
        reference = sheet['reference']
        assert len(OBSERVATIONS[reference]) == len(sheet['cases']) == 5
        assert sha(OUT / sheet['file']) == prepared['sheet_sha256'][sheet['file']]
        sheets.append({'file': sheet['file'], 'sha256': sha(OUT / sheet['file']),
                       'requested_detail': 'original', 'actually_viewed': True, 'dimensions': [1336, 1516],
                       'method': 'tools.view_image; all five 256px columns and all five rows inspected'})
        for profile, cid, notes in zip(PROFILES, sheet['cases'], OBSERVATIONS[reference], strict=True):
            assert cid.endswith('_' + profile) and len(notes) == len(plan['regions']) == 5
            rows.append({'id': cid, 'reference': reference, 'profile': profile,
                         'regions_inspected': plan['regions'],
                         'observations': dict(zip(plan['regions'], notes, strict=True)),
                         'V31_vs_V30': 'Little visible incremental change; no convincing whole-face clarity gain established.',
                         'V31_vs_retained': 'Broad visible structure remains, with smoothing; fine degraded detail is still weak.',
                         'paired_target_is_TRAIN_only': True,
                         'clearer_crop_needed_for_useful_fine_structure': profile == 'compound_lr24'})
    assert [r['id'] for r in rows] == plan['case_ids'] and len(rows) == 50 and len(sheets) == 10
    receipt = {'complete': True, 'recorded_utc': datetime.now(timezone.utc).isoformat(),
               'recorder_source_sha256': sha(Path(__file__)),
               'independent_return_audit_sha256': sha(ROOT / 'outputs/cctv_dgp_profile_batches_v31_independent_audit.json'),
               'preparation_sha256': sha(OUT / 'preparation.json'),
               'independent_preparation_audit_sha256': sha(OUT / 'independent_preparation_audit.json'),
               'cases_reviewed': 50, 'comparison_cells_reviewed': 250, 'rows': rows,
               'actually_viewed_sheets': sheets, 'useful_whole_face_gain_established': False,
               'early_numerical_failure_waived': False,
               'scope': 'Agent visual review of fixed paired photographic TRAIN previews; not human independent final review',
               'native_or_reserved_used': False, 'independent_final_review': False,
               'identity_or_ethnicity_claim': False, 'Zamboanga_performance_claim': False,
               'neural_or_gradient_calls': 0, 'optimizer_updates': 0, 'app_promotion': False, 'goal_complete': False}
    with (OUT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({'complete': True, 'cases_reviewed': 50, 'comparison_cells_reviewed': 250,
                      'useful_whole_face_gain_established': False, 'independent_final_review': False}))


if __name__ == '__main__':
    main()
