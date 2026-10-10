"""Record the actual original-detail review of ten optimized TRAIN sheets."""
from datetime import datetime, timezone
from pathlib import Path
from cctv_dgp_spatial_fit_v40_contract import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_v40_learning_signal_v1_analysis'


def main():
    ready = read(OUT/'analysis.json'); checked = read(OUT/'independent_analysis_audit.json')
    assert checked['complete'] and checked['analysis_sha256'] == sha(OUT/'analysis.json')
    notes = [
        'Microphone and visible hair remain broadly present; severe blur and compound eyes/nose/mouth stay unresolved. No convincing stopped50 gain.',
        'Clear glasses remain in the clear control; degraded eye/frame and mouth detail remain soft. Stopped50 is visually close to original DGP.',
        'Clear smile and outline remain broad; blur/compound eye, nose and tooth boundaries stay diffuse at both states.',
        'Clear eyes and mouth remain broad with smoothing; severe degradation retains diffuse facial boundaries. No clear stopped50 definition gain.',
        'Clear hair and visible hand remain; eye/nose/lip boundaries remain blurred across degradation. No convincing incremental gain.',
        'Clear glasses, curls and broad smile remain; degraded eye/frame and teeth detail remain unresolved. Stopped50 is close to baseline.',
        'Clear smile, hair and outline remain broad; degraded features remain diffuse. No convincing incremental structural gain.',
        'Clear outline and eyes remain broad; degraded eyes/nose/lip boundaries remain soft and compound is indistinct at both states.',
        'Clear eye/nose/lip outline remains broad with smoothing; severe blur and compound detail stay unresolved. No convincing stopped50 gain.',
        'Clear mild turn, hair and lips remain broad; degraded central features remain diffuse. No convincing stopped50 gain.',
    ]
    pages = []
    for page, note in zip(ready['pages'], notes):
        assert sha(OUT/page['path']) == page['sha256']
        pages.append({'path': page['path'], 'sha256': page['sha256'], 'reference': page['reference'], 'source': page['source'],
                      'ids': page['ids'], 'all_five_profiles_actually_viewed': True, 'observation': note})
    write(OUT/'visual_review.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
        'reviewer': 'Primary agent; all ten view_image calls at original256-cell detail',
        'analysis_sha256': sha(OUT/'analysis.json'), 'source_sha256': sha(Path(__file__)), 'pages': pages,
        'all50_optimized_TRAIN_cases_and200_cells_actually_viewed': True,
        'all_visible_features_considered': ['eyes', 'nose', 'mouth', 'face_outline', 'visible_appearance'],
        'convincing_incremental_structure_gain': False, 'training_capacity_pass': False,
        'input_only_criteria_or_case_labels_changed': False, 'no_identity_or_ethnicity_inference': True,
        'native_or_reserved_used': False, 'new_neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0,
        'app_promotion': False, 'goal_complete': False})
    print({'complete': True, 'optimized_TRAIN_cases_reviewed': 50, 'app_promotion': False})


if __name__ == '__main__': main()
