"""Record actual original-size review of all twenty V36 pages."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_finite_clearance_probe_v36_analysis_v1'
NOTES = {
    'exposed_01': 'Clear gaze and microphone remain, with softer eye and lip edges. Severe degraded eyes and mouth remain broad; all four trials look close to the original DGP.',
    'exposed_02': 'Clear spectacles and mouth opening remain, with smoothing. Degraded frames, nostrils and teeth merge or lose definition; the clearance trials do not visibly restore those details.',
    'exposed_03': 'Clear smile and face outline remain with soft skin. Blur and compound inputs retain weak eye and tooth definition; motion retains a coarse expression, similarly across trials.',
    'exposed_04': 'Clear eyes and face placement remain recognizable, with softened lips. Severe blur and compound profiles lose eyelid and nostril structure; trial differences remain subtle.',
    'exposed_05': 'Clear gaze and mouth placement remain. Degraded eyes and lip edges remain broad, while motion keeps more recognizable facial structure; no clear new detail separates the four trials.',
    'exposed_06': 'Clear glasses, smile and hair remain with softer fine detail. Degraded glasses and eyes merge and teeth become a broad light band; trial appearances remain close to the original.',
    'exposed_07': 'Clear smile and hair placement remain. Motion retains the expression, while blur and compound profiles leave poorly resolved eyes and tooth edges; all trial scales remain similarly soft.',
    'exposed_08': 'Clear eyes, nose and mouth remain with smoothing. Degraded eyes become small dark regions and lip boundaries weaken; the trials do not visibly recover the missing definition.',
    'exposed_09': 'Clear gaze and hair remain recognizable, with softer eyelids and lips. Severe degraded eyes and mouth remain broad; motion preserves more coarse structure, without an obvious trial gain.',
    'exposed_10': 'The clear mildly turned face remains recognizable. Degraded nose and lips are indistinct, especially in compound input; all scales retain the same coarse appearance.',
    'unexposed_01': 'Clear spectacle frames, mouth and hair remain with smoothing. Degraded lenses, gaze, nostrils and teeth merge; the trial columns do not resolve the weak features.',
    'unexposed_02': 'Clear eye positions, nose and face outline remain with softer contours. Degraded brow and lip detail remain broad; motion retains more outline but trial gains are not convincingly visible.',
    'unexposed_03': 'Clear facial hair, gaze and lip placement remain. Severe profiles weaken eye, nostril and facial-hair separation; motion retains coarse features, similarly at every scale.',
    'unexposed_04': 'Clear eyes and smile remain with softened tooth and skin detail. Blur and compound profiles leave eye and tooth definition weak; all trial appearances remain close to the original.',
    'unexposed_05': 'Clear open mouth and gaze remain recognizable. Degraded mouth opening survives coarsely, while teeth, eyelids and nostrils remain poorly defined; trial scales look similar.',
    'unexposed_06': 'Clear smile, gaze and hair remain with smoothing. Motion retains coarse teeth and expression; severe blur and compound profiles remain too soft to demonstrate useful new structural detail.',
    'unexposed_07': 'Clear face outline and eye placement remain with softer lips. Severe profiles reduce eyes to dark regions and weaken cheek and mouth separation; no convincing additional structure appears in a trial.',
    'unexposed_08': 'Clear mildly turned gaze and open mouth remain. Severe profiles retain coarse eyes and mouth regions but lack eyelid, nostril and tooth definition; trials remain visually similar.',
    'unexposed_09': 'Clear spectacles and mouth remain with softened contours. Degraded frames merge with the eyes and tooth detail weakens; motion preserves more coarse structure, without a clear trial repair.',
    'unexposed_10': 'Clear face and adjacent hand remain with smoothing. Severe degraded features remain indistinct; the hand is visible input, not a covering-completion or hidden-identity test.'}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    analysis = json.loads((OUT / 'analysis.json').read_text())
    assert analysis['complete'] and analysis['jointly_eligible_subset_variants'] == []
    pages = []
    for page in analysis['visual_pages']:
        path = ROOT / page['path']
        assert sha(path) == page['sha256'] and path.stem in NOTES
        pages.append({'path': page['path'], 'sha256': page['sha256'], 'case_ids': page['case_ids'],
            'actually_viewed': True, 'image_detail': 'original', 'native_cell_size': [256, 256],
            'observation': NOTES[path.stem]})
    assert len(pages) == 20 and len({c for page in pages for c in page['case_ids']}) == 100
    result = {'complete': True, 'analysis_sha256': sha(OUT / 'analysis.json'),
        'observer': 'Implementing assistant actual image inspection; not independent final visual review',
        'pages': pages, 'actually_viewed_pages': 20, 'actually_viewed_cases': 100,
        'model_output_cells_reviewed': 500, 'input_target_output_cells_reviewed': 700,
        'all_scales_visible_structure_qualified': False, 'all_measured_preservation_failures_retained': True,
        'softness_alone_not_failure': True, 'clear_glasses_and_visible_hair_not_removal_targets': True,
        'paired_photographic_TRAIN_only': True, 'native_or_reserved_final_used': False,
        'input_sufficiency_threshold_newly_qualified': False, 'hidden_identity_recovery_claim': False,
        'covering_family_quality_newly_qualified': False, 'independent_final_review': False,
        'app_adoption': False, 'goal_complete': False, 'script_sha256': sha(Path(__file__))}
    with (OUT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'cases_viewed': 100, 'model_outputs_viewed': 500, 'app_adoption': False}))


if __name__ == '__main__':
    main()
