"""Record actual inspection of all twenty V38 pages after viewing them."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_delivered_margin_probe_v38_analysis_v1'
NOTES = {
    'exposed_01': 'Clear gaze, open lips and microphone remain with smoothing. Blur and compound cases leave broad dark eyes and poorly defined lips. The four trial columns remain close to the original DGP; smaller passing steps do not demonstrate a visible clarity gain.',
    'exposed_02': 'Clear spectacles and open mouth remain, with softer eyelids and tooth edges. Degraded spectacle frames, eyes and teeth merge; motion retains a coarse mouth opening. No trial resolves the missing fine structure convincingly.',
    'exposed_03': 'Clear smile, hair and face contour remain. Motion preserves more coarse expression than severe blur or compound input. The latter retain broad eyes and a weak mouth boundary in all trial columns.',
    'exposed_04': 'Clear eye placement and lip shape remain with softened skin. Severe degraded eyelids, nostrils and lips remain poorly resolved; the motion row is coarsely recognizable. Trial differences are subtle at the native cell size.',
    'exposed_05': 'Clear gaze, lip placement and adjacent visible hand remain. Degraded eye regions and mouth edges remain broad; motion keeps more of the expression. Neither passing smaller step shows convincing new facial definition.',
    'exposed_06': 'Clear spectacles, smile and non-obstructing hair remain. Blur and low-light frames merge with eyes and teeth become a broad light region. Motion retains more coarse structure, while the four trials remain similarly soft.',
    'exposed_07': 'Clear smile and hair placement remain. Motion retains recognizable coarse teeth and expression. Severe blur and compound rows retain weak eye and tooth boundaries at every trial scale.',
    'exposed_08': 'Clear eye positions, nose and closed-mouth placement remain with smoothing. Severe profiles leave small dark eyes and a weak lip edge. The trial columns do not visibly restore eyelid or nostril definition.',
    'exposed_09': 'Clear gaze, face outline and hair remain with softer contours. Motion retains a coarse nose and mouth. Blur, low-light and compound rows remain too indistinct to demonstrate a useful new structural gain from the small steps.',
    'exposed_10': 'The clear mildly turned face and closed-lip shape remain recognizable. Degraded nose and mouth boundaries weaken; the motion mouth is coarser than the target. All trial columns retain nearly the same appearance.',
    'unexposed_01': 'Clear spectacle shape, gaze and mouth opening remain with smoothing. Degraded lenses, nostrils and lip boundaries merge. Motion keeps the broad expression but the trials do not convincingly repair the lost detail.',
    'unexposed_02': 'Clear eye positions, nose and face contour remain with softened lip and skin detail. Severe blur and compound rows leave broad eyes and a faint mouth; motion retains more contour. Differences between trial scales are subtle.',
    'unexposed_03': 'Clear facial hair, eye placement and mouth remain recognizable. Severe profiles weaken eyelid, nostril and facial-hair separation; motion preserves more coarse features. The passing small steps remain close to the original DGP.',
    'unexposed_04': 'Clear smile, eyes and hair placement remain with softer tooth edges. Motion retains the open smile coarsely. Blur and compound rows still lack eye and tooth definition in every trial.',
    'unexposed_05': 'Clear eye and open-mouth placement remain. Degraded mouth opening survives as a broad region, while eyelids, nostrils and tooth detail remain weak. Trial appearances are nearly indistinguishable at 256 pixels.',
    'unexposed_06': 'Clear smile, gaze and non-obstructing hair remain with smoothing. Motion retains coarse teeth and expression; blur and compound rows remain too soft to show useful new detail from any tested step.',
    'unexposed_07': 'Clear face outline and eye placement remain. Severe profiles reduce eyes to dark regions and weaken cheek/nose/mouth separation. The trial columns remain similarly broad and do not supply clear new structure.',
    'unexposed_08': 'Clear mildly turned gaze and open mouth remain. Motion retains the expression coarsely. Severe profiles lack eyelid, nostril and mouth-boundary definition; smaller passing steps have only subtle visible differences.',
    'unexposed_09': 'Clear spectacles, teeth and mouth placement remain with softer edges. Degraded frames merge into eyes and teeth lose separation. Motion retains more structure but the trials do not visibly repair the weak regions.',
    'unexposed_10': 'Clear face and adjacent hand remain with smoothing. Degraded eyes, nose and lips stay indistinct, especially in compound input. The visible hand is retained input context, not a covering-completion or hidden-identity test.'
}


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    analysis = json.loads((OUT / 'analysis.json').read_text())
    assert analysis['complete']
    pages = []
    for page in analysis['visual_pages']:
        path = ROOT / page['path']
        assert sha(path) == page['sha256'] and path.stem in NOTES
        pages.append({'path': page['path'], 'sha256': page['sha256'], 'case_ids': page['case_ids'],
            'actually_viewed': True, 'image_detail': 'original', 'native_cell_size': [256, 256],
            'observation': NOTES[path.stem]})
    assert len(pages) == len(NOTES) == 20 and len({c for page in pages for c in page['case_ids']}) == 100
    result = {'complete': True, 'analysis_sha256': sha(OUT / 'analysis.json'),
        'observer': 'Implementing assistant actual development image inspection; not independent final review',
        'pages': pages, 'actually_viewed_pages': 20, 'actually_viewed_cases': 100,
        'model_output_cells_reviewed': 500, 'input_target_output_cells_reviewed': 700,
        'subset_preservation_passing_variants': analysis['jointly_eligible_subset_variants'],
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
