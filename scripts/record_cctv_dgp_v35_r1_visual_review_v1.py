"""Record actual full-size inspection of all20 pages,100 cases and500 model outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_group_guard_probe_v35_r1_analysis_v1_r1'
NOTES = {
    'exposed_01': 'Clear eye and lip edges are softer than the target; microphone and hair remain. Degraded gaze and lip definition remain broad; trials show no convincing new structure.',
    'exposed_02': 'Clear glasses remain recognizable, with softened lens and mouth boundaries. Severe profiles leave glasses, nostrils and the open mouth indistinct across all trials.',
    'exposed_03': 'The clear smile and face outline remain, with skin and lip smoothing. Blur and compound cases lose eye and tooth definition; motion retains a coarse smile, without convincing trial improvement.',
    'exposed_04': 'Clear eye outlines and mouth are softened. Degraded eyelids, nostrils and lips remain indistinct; the four trial appearances remain close to the original DGP.',
    'exposed_05': 'Clear eye and mouth placement remain recognizable. Degraded eyelids and lip edges remain broad; motion keeps more structure but trials do not visibly resolve the weak features.',
    'exposed_06': 'Clear spectacles and smile remain with softened fine detail. Degraded glasses merge with eye regions and tooth edges disappear; all trial scales remain similarly soft.',
    'exposed_07': 'Clear smile and hair placement remain with softened skin detail. Blur and compound cases have poorly resolved eyes and mouth; motion retains a coarse expression without a clear trial gain.',
    'exposed_08': 'Clear eyes, nose and mouth remain, with smoothing. Degraded eyes become small dark regions and mouth boundaries weaken; trials remain close to the original.',
    'exposed_09': 'Clear face and hair remain recognizable with softer eyelids and lip boundaries. Degraded eyes and mouth remain broad; no trial demonstrates a convincing structural repair.',
    'exposed_10': 'The clear mildly turned face remains, with softer eye and lip detail. Degraded nostrils and mouth remain indistinct; the coarse motion outline survives but trials remain visually similar.',
    'unexposed_01': 'Clear spectacle frames remain with softened eyes, nostrils and teeth. Degraded frames and gaze largely merge; trial differences do not resolve the weak facial structure.',
    'unexposed_02': 'Clear eye and mouth positions remain with softened skin contours. Degraded brow, nose and lip detail remain broad; motion keeps more outline but there is no convincing new trial detail.',
    'unexposed_03': 'Clear facial hair remains, with smoother eye and nose edges. Severe profiles weaken facial-hair and lip separation; coarse motion features survive, similarly in all trials.',
    'unexposed_04': 'Clear eye and smile placement remain, with softened fine edges. Blur and compound profiles leave tooth and eyelid structure weak; trials show no clear improvement over the original.',
    'unexposed_05': 'Clear open mouth and eyes remain with smoothing. Degraded mouth opening survives coarsely while teeth, nostrils and eyelids remain unresolved; trial scales look similar.',
    'unexposed_06': 'Clear eyes and smile remain with softened skin detail. Motion retains coarse eyes and teeth, while severe blur and compound inputs remain poorly resolved across trials.',
    'unexposed_07': 'Clear face outline remains with softened nose and lip edges. Degraded eyes become small dark regions and cheek and mouth boundaries blur; trials remain close to the original.',
    'unexposed_08': 'Clear gaze and open mouth remain recognizable with softer contours. Severe inputs retain only coarse eye and mouth regions; all four trial scales remain similarly soft.',
    'unexposed_09': 'Clear spectacles and open mouth remain, with softened fine detail. Degraded glasses merge with the eye region and tooth contours weaken; motion keeps more structure without a convincing trial repair.',
    'unexposed_10': 'Clear face and adjacent hand remain with softened eyes and mouth. Degraded features remain indistinct; the hand is not used as evidence of completion or hidden identity.'}


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
        'all_scales_visible_structure_qualified': False, 'preservation_failures_retained': True,
        'paired_photographic_TRAIN_only': True, 'native_or_reserved_final_used': False,
        'input_sufficiency_threshold_newly_qualified': False, 'hidden_identity_recovery_claim': False,
        'covering_family_quality_newly_qualified': False, 'independent_final_review': False,
        'app_adoption': False, 'goal_complete': False, 'script_sha256': sha(Path(__file__))}
    with (OUT / 'visual_review.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'cases_viewed': 100, 'model_outputs_viewed': 500, 'app_adoption': False}))


if __name__ == '__main__':
    main()
