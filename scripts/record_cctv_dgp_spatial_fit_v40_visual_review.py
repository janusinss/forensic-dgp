"""Record actual inspection of all50 prospective TRAIN previews; no model calls."""
from pathlib import Path
from datetime import datetime, timezone
from cctv_dgp_spatial_fit_v40_contract import read, write, sha
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/cctv_dgp_spatial_fit_v40_failure_review'


def main():
    plan, ready, checked = [read(OUT/name) for name in ['plan.json', 'preparation.json', 'independent_preparation_audit.json']]
    assert checked['complete'] and checked['exact_page_cells'] == 250 and checked['preparation_sha256'] == sha(OUT/'preparation.json')
    notes = {
        'tr_ffhq_00084': 'Clear glasses, lips and head outline remain near update0; degraded frames/eyes and lip edges remain soft.',
        'tr_ffhq_00323': 'Eye pair, nose and smile remain near update0; degraded feature edges remain weak without useful new definition.',
        'tr_ffhq_00178': 'Eye pair, broad smile and hair outline stay in the baseline arrangement; degraded eyes/teeth and lip contour remain soft.',
        'tr_ffhq_00616': 'Eyes, smile, headwear and face outline remain near update0; degraded eyebrows/teeth and mouth edges remain soft.',
        'tr_ffhq_01210': 'Clear-glasses and open-mouth geometry remain near update0; severe frames, gaze and mouth remain indistinct.',
        'tr_asian_00048': 'Eye pair, nose, facial hair and expression stay near update0; degraded eyes/mouth are not convincingly clarified.',
        'tr_asian_00133': 'Eye and mouth arrangement remains near update0, preserving a broad smile; degraded eyelids/lips and outline remain soft.',
        'tr_asian_00176': 'Broad eye and closed-mouth arrangement stays near update0; degraded eyes/nose and cheek outline remain soft.',
        'tr_asian_00180': 'Eye pair, open mouth and cheek outline remain near update0; candidate does not visibly clarify eye/nose/lip boundaries.',
        'tr_asian_00196': 'Eye pair, smile and hand beside cheek stay near update0; eye/nose/smile and cheek/hand edges remain soft.'}
    profiles = {
        'clear': 'Baseline softening remains, without an apparent conspicuous new pose/expression change.',
        'blur_lr24': 'Input blocks are smoother in the baseline; stopped50 supplies no convincing added feature definition.',
        'lowlight_lr32': 'The dark/noisy input is smoother in the baseline; the candidate remains soft without convincing added definition.',
        'motion_lr48': 'Baseline already reduces block/motion appearance; no convincing additional feature gain is visible.',
        'compound_lr24': 'Severe dark/blurred features remain weak; the failure does not justify output-based input relabeling.'}
    refs = {cid: page['reference'] for page in ready['pages'] for cid in page['ids']}
    rows = [{'id': row['id'], 'source': row['source'], 'profile': row['profile'], 'optimized_by50': row['optimized_by50'],
             'regions': plan['regions'], 'all5_regions_actually_reviewed': True,
             'observed_comparison': notes[refs[row['id']]]+' '+profiles[row['profile']],
             'decision': 'Retain stopped candidate and failed capacity requirement; no app promotion',
             'independent_final_review': False} for row in ready['rows']]
    assert [row['id'] for row in rows] == plan['case_ids'] and len(rows) == 50
    write(OUT/'visual_review.json', {'complete': True, 'UTC': datetime.now(timezone.utc).isoformat(),
          'reviewer': 'Implementing assistant development inspection; not independent final review',
          'plan_sha256': sha(OUT/'plan.json'), 'preparation_sha256': sha(OUT/'preparation.json'),
          'independent_preparation_audit_sha256': sha(OUT/'independent_preparation_audit.json'), 'source_sha256': sha(Path(__file__)),
          'all50_previews_and250_comparison_cells_actually_viewed': True, 'all10_sheets_actually_viewed_at_original256_cell_detail': True,
          'pages': ready['pages'], 'rows': rows, 'finding': 'Broad appearance remains near the retained DGP; tone/texture changes do not establish useful added structure.',
          'invalid_assumption': 'Nonzero improvement gradients and a new spatial decoder necessarily yield useful structural learning under the declared recipe',
          'not_a_proven_unique_cause': True, 'no_gate_override_or_threshold_change': True, 'fixed50_directly_optimized_cases': 0,
          'all_are_TRAIN_not_held_out_evaluation': True, 'source_names_not_ethnicity': True, 'no_new_native_or_reserved_used': True,
          'quality_qualification': False, 'app_promotion': False, 'independent_final_review': False,
          'new_model_forwards': 0, 'local_gradient_calls': 0, 'local_optimizer_updates': 0, 'VM_calls': 0, 'goal_complete': False})
    print({'complete': True, 'actually_reviewed': 50, 'sheets': 10, 'app_promotion': False}, flush=True)


if __name__ == '__main__': main()
