"""Record implementing-assistant observations after actually viewing all20 sheets."""
from datetime import datetime, timezone
from pathlib import Path
from cctv_dgp_spatial_fit_v40_contract import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_pcgrad_fit_v41_analysis'


def main():
    analysis = read(OUT/'analysis.json'); checked = read(OUT/'independent_analysis_audit.json')
    assert checked['complete'] and checked['exact_unscaled_sheet_cells'] == 500
    assert checked['analysis_sha256'] == sha(OUT/'analysis.json')
    notes = {
        'tr_ffhq_00084': 'Clear glasses, hair, eye pair and open lips remain in the baseline arrangement. Degraded lens/eyelid, nose and lip edges stay diffuse; V41 offers no convincing added definition over V40 or initial.',
        'tr_ffhq_00323': 'Eye pair, nose, closed-mouth smile and head outline stay near initial. Fine eyelid and lip detail remains weak in the four degradation profiles.',
        'tr_ffhq_00178': 'Open eyes, smiling mouth and hair outline remain near initial. Degraded eyes, teeth and lip boundaries remain soft; the broad smoothing already exists in the retained DGP.',
        'tr_ffhq_00616': 'Eye pair, smile, headwear and hair outline remain near initial. Fine eye/teeth definition is still weak in the severe profiles; V41 does not add convincing clarity.',
        'tr_ffhq_01210': 'Clear-glasses frames, open-mouth shape and hair remain near the baseline. Degraded frame, eyelid, nose and tooth boundaries are still indistinct.',
        'tr_asian_00048': 'Eye pair, nose, facial hair, closed lips and head outline remain near initial. The severe central features remain diffuse; no convincing incremental structure is visible.',
        'tr_asian_00133': 'Eye slits, open smile and cheek outline remain near initial. Degraded lip/teeth, eyelid and nasal edges remain soft across V40 and V41.',
        'tr_asian_00176': 'Eye pair, closed mouth and rounded cheek outline remain near initial. Degraded boundaries around eyes, nose and lips are still broad and soft.',
        'tr_asian_00180': 'Eye pair, open mouth and cheek outline remain in the baseline arrangement. Fine eyelid, nostril and lip boundaries are not convincingly clarified by V41.',
        'tr_asian_00196': 'Eye pair, smile, hair and the visible hand beside the cheek remain near initial. Degraded eyelids, nose, lip and cheek/hand edges remain soft.',
        'tr_ffhq_37935': 'Eye pair, open lips, hair and visible cheek microphone remain near initial. Blur/lowlight/compound features remain diffuse; V41 supplies no convincing new eye/nose/mouth definition.',
        'tr_ffhq_55330': 'Clear glasses, eye pair, open mouth and face outline remain near initial. Degraded frames, eyelids and mouth boundaries stay weak; no convincing incremental improvement is visible.',
        'tr_asian_09639': 'Mildly turned face, smile and hair remain in the baseline arrangement, including the surrounding gray canvas. Eye/nose/teeth contours remain soft in degraded profiles.',
        'tr_ffhq_63174': 'Wide eye pair, nose, lips and head outline remain near initial. Severe profiles retain diffuse eyes/nose/mouth boundaries without convincing V41 definition.',
        'tr_asian_02292': 'Eye pair, broad smile, hair and the visible hand at the upper edge remain near the baseline. Degraded eyelid, nose, lip and cheek edges remain soft.',
        'tr_ffhq_53271': 'Clear glasses, smile, visible hair and face outline remain near initial. Degraded frame/eye and tooth/lip boundaries stay soft in V40 and V41.',
        'tr_asian_08999': 'Eye pair, broad smile, hair and cheek outline remain near initial. Degraded eye/nose/mouth boundaries remain diffuse; color smoothing is already present in the retained DGP.',
        'tr_asian_03298': 'Eye pair, nose, closed-mouth smile and cheek outline remain near initial. Fine degraded feature definition remains weak in all four degradation profiles.',
        'tr_ffhq_55007': 'Eye pair, closed lips, ordinary hair and face outline remain near initial. Degraded eye and lip detail remains soft without convincing added V41 structure.',
        'tr_asian_09839': 'Mild face turn, eye pair, nose profile, closed lips and ordinary hair remain near initial. Degraded central features stay diffuse; V41 has no convincing added boundary definition.'}
    profiles = {
        'clear': 'Clear-control broad appearance remains near the baseline; existing DGP softening remains visible.',
        'blur_lr24': 'Baseline reduces block appearance, while fine visible feature boundaries remain soft.',
        'lowlight_lr32': 'Noise and block appearance are smoother in the baseline; V41 does not establish useful added definition.',
        'motion_lr48': 'Some broad facial layout remains visible; new eye/nose/mouth/outline definition is not convincing.',
        'compound_lr24': 'Severe dark/blurred structure stays weak; this output does not change input-only insufficiency labels.'}
    rows = []
    for page in analysis['pages']:
        assert sha(OUT/page['path']) == page['sha256'] and page['reference'] in notes
        for cid in page['ids']:
            profile = next(key for key in profiles if cid.endswith('_'+key))
            rows.append({'id': cid, 'cohort': page['cohort'], 'source': page['source'], 'profile': profile,
                         'all_visible_features_considered': ['eyes', 'nose', 'mouth', 'face_outline', 'visible_appearance'],
                         'observations': notes[page['reference']]+' '+profiles[profile],
                         'decision': 'Retain the stopped failure; no app adoption or gate override'})
    assert len(rows) == len({r['id'] for r in rows}) == 100 and len(analysis['pages']) == 20
    write(OUT/'visual_review.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
          'source_sha256': sha(Path(__file__)), 'analysis_sha256': sha(OUT/'analysis.json'),
          'independent_analysis_audit_sha256': sha(OUT/'independent_analysis_audit.json'),
          'reviewer': 'Implementing assistant development inspection; not independent final review',
          'all100_TRAIN_cases_and500_cells_actually_viewed': True,
          'all20_sheets_actually_viewed_at_original256_cell_detail': True,
          'all_visible_features_considered': ['eyes', 'nose', 'mouth', 'face_outline', 'visible_appearance'],
          'pages': analysis['pages'], 'rows': rows, 'reference_notes': notes,
          'convincing_added_structure_over_retained_DGP': False, 'quality_qualification': False,
          'input_only_labels_or_gates_changed': False, 'cohorts_are_TRAIN_not_independent_evaluation': True,
          'source_names_not_ethnicity': True, 'native_or_reserved_used': False,
          'new_neural_calls': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0,
          'app_promotion': False, 'independent_final_review': False, 'goal_complete': False})
    print({'complete': True, 'actually_reviewed_cases': 100, 'cells': 500, 'app_promotion': False}, flush=True)


if __name__ == '__main__': main()
