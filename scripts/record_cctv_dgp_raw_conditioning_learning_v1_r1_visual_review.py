"""Bind the twenty directly viewed, original-resolution comparison sheets."""
from datetime import datetime, timezone
from PIL import Image
from import_cctv_dgp_raw_conditioning_learning_v1_r1_return import ROOT, ANALYSIS, sha, read, write


NOTES = {
    'b036': 'Clear glasses, eyes, lips and hair are softer than the target. All three degraded trials remain diffuse around the glasses and mouth; no convincing advantage over initial.',
    'b051': 'Clear hair and skin texture are softened. Blur and compound rows lose eye and lip definition; motion outputs remain smooth with no convincing trial improvement.',
    'b076': 'Clear eyelash, lip and hair detail stays softened. Low-light and compound eye/nose/mouth structure remains diffuse across initial and every trial.',
    'b085': 'Clear tooth boundaries and brows are softened. Degraded teeth become a diffuse band; all trial columns retain the same broad smile and blurred eyes.',
    'b092': 'Clear teeth, eyelids and hair highlights are less defined than the target. Degraded eye and tooth details stay blurred in every trial.',
    'b146': 'Clear eyelash and hair edges are softened. Blur, low-light and compound rows remain broad smooth facial shapes; the trials give no convincing detail gain.',
    'b169': 'Clear eyebrows and tooth boundaries are softened. Degraded smile, eyes and nose are diffuse; the three candidate columns remain visually close to initial.',
    'b357': 'Clear eye-liner and lip edges are softened. Degraded eyes and lips remain broad blobs; preserved visible hair is not a completion test.',
    'b384': 'Clear eyebrow, stubble and hair detail is softened. Degraded cheek, nose and mouth edges remain smooth; none of the trials visibly restores target detail.',
    'b386': 'Clear glasses, hair and tooth edges are softened. Degraded glasses/eyes become poorly defined and mouths remain blurred across all trials.',
    'b391': 'The target has limited fine detail, but eyebrow, beard and lip edges still exceed the smooth clear output. All degraded trials remain diffuse.',
    'b392': 'Clear lip, tooth and hair detail is smoothed. Blur and compound rows leave poorly defined eyes and an open-mouth region; trials show no convincing advantage.',
    'b393': 'The source/target is already soft. Clear lip and eye boundaries are further smoothed, and all degraded candidate columns remain indistinct.',
    'b394': 'Clear nostril, lip and eye edges are softened. Blur, low-light and compound rows remain smooth; no convincing trial improvement in the visible mouth.',
    'b395': 'Clear brows and tooth boundaries are softened. Degraded features stay diffuse. The visible hand remains context; no hidden-region completion was performed.',
    'b480': 'Clear teeth and eyebrow detail are softened. Degraded eyes and smile stay broad and blurry; all three trials remain close to initial.',
    'b615': 'Portrait padding remains visible. Clear teeth and hair detail are softened; degraded eye/mouth and visible-hand context stay blurred. No completion is tested.',
    'b651': 'Clear visible hair and skin texture are softened. Degraded visible eye, nose and lip contours stay diffuse. Hair is retained rather than reconstructed away.',
    'b701': 'Portrait padding and head covering are retained. Clear skin detail is softened, with degraded eyes/lips diffuse in every candidate; no covered-region estimate is tested.',
    'b745': 'The target is already soft. Clear tooth and hair separation is further smoothed; degraded eyes and smile remain blurry, with no convincing trial clarity gain.',
}


def main():
    audited = read(ANALYSIS / 'failure_forensic_audit.json')
    assert audited['complete'] and not audited['cross_run_derivative_parity_pass']
    planned = audited['all_planned_visual_sheets']
    assert len(planned) == 20 and {Image.open(ROOT/r['file']).size for r in planned} == {(1536,1400)}
    assert {r['file'].split('/')[-1].removesuffix('.png') for r in planned} == set(NOTES)
    rows = []
    for row in planned:
        path = ROOT / row['file']; assert sha(path) == row['sha256'] and row['cells'] == 30
        rows.append(dict(file=row['file'],sha256=row['sha256'],cells=30,all30_cells_directly_viewed=True,
                         requested_image_detail='original',observations=NOTES[path.stem]))
    write(ANALYSIS / 'visual_review.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),
        recorder_sha256=sha(__file__),failure_forensic_audit_sha256=sha(ANALYSIS/'failure_forensic_audit.json'),
        reviewer='Primary agent directly viewed every original-resolution sheet through view_image',
        all20_sheets_and600_cells_directly_viewed=True, sheets=rows,
        rows=100, columns=['input','paired TRAIN target','initial','r00_trial00','r00_trial01','r00_trial02'],
        profiles=['clear','blur_lr24','lowlight_lr32','motion_lr48','compound_lr24'],
        conclusion='No convincing whole-face clarity gain over the permanent original DGP; degraded eye, mouth and hair definition remains soft.',
        scientific_preservation_rejections_not_overridden=True,original_cross_run_parity_check_still_failed=True,
        paired_photographic_TRAIN_only=True,native_CCTV_used=False,independent_final_review=False,
        automatic_or_assisted_completion_qualification=False,model_qualified=False,goal_complete=False))
    print(dict(complete=True,sheets=20,cells=600,useful_new_restoration_established=False))


if __name__=='__main__':main()
