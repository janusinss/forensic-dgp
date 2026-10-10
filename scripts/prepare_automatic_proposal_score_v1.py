"""Freeze existing inputs, masks and production-source bindings before inference."""
from datetime import datetime, timezone
from automatic_proposal_score_v1_common import ROOT, OUT, BUDGETS, sha, read, write, bindings


def main():
    assert not OUT.exists()
    parent_path = ROOT/'outputs/completion_feature_fusion_off_v1/protocol.json'
    assert sha(parent_path) == '53c038ea900c1b8d0c602679435cf794b6770708196183ec6a1773437becf95b'
    parent = read(parent_path); cases = []
    previous = read(ROOT/'outputs/dgp_app_covering_review_v3/results.json')
    assert previous['complete'] and read(ROOT/'outputs/dgp_app_covering_review_v3/saved_output_audit.json')['complete']
    sources = {}
    for item in parent['cases']:
        case = {k: item[k] for k in ['id', 'base_id', 'family', 'condition', 'input', 'automatic',
            'reviewed', 'input_review', 'input_only_note', 'exposure', 'rejected']}
        if item.get('masks'): case['masks'] = item['masks']
        cases.append(case)
        names = [case['input'], case['automatic'], case['reviewed']]+list(case.get('masks', {}).values())
        for name in names:
            digest = sha(ROOT/name); assert digest == parent['sources_sha256'][name], name
            sources[name] = digest
    for name in ['dgp_face_workflow_v3.py', 'face_workflow.py', 'completion.py', 'completion_inference.py',
        'face_workflow_web.py', 'static/face_workflow.js', 'templates/face_workflow.html',
        'outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth',
        'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth', 'checkpoints/dgp_zamboanga_final.pth',
        'outputs/completion_feature_fusion_off_v1/protocol.json', 'outputs/dgp_app_covering_review_v3/plan.json',
        'outputs/dgp_app_covering_review_v3/results.json', 'outputs/dgp_app_covering_review_v3/saved_output_audit.json',
        'CCTV_DGP_AUTOMATIC_PROPOSAL_SCORE_V1_DESIGN.md', 'scripts/automatic_proposal_score_v1_common.py',
        'scripts/prepare_automatic_proposal_score_v1.py', 'scripts/run_automatic_proposal_score_v1.py',
        'scripts/audit_automatic_proposal_score_v1.py', 'scripts/supervise_automatic_proposal_score_v1.py']:
        sources[name] = sha(ROOT/name)
    assert sources['outputs/downloaded_completion/outputs/completion_pilot/epoch_2.pth'] == 'c2d4cd180226413af63bcfc2a3e17461cfa95f87aff9a8998b893fc3afc42e93'
    protocol = {'format': 'automatic-proposal-score-trace-v1', 'UTC': datetime.now(timezone.utc).isoformat(),
        'cases': cases, 'sources_sha256': sources, 'budgets': BUDGETS, 'threshold': .5, 'margin_pixels_256': 3,
        'completion_forwards': 0, 'DGP_forwards': 0, 'optimizer_updates': 0, 'gradient_queries': 0,
        'native_or_final_used': False, 'app_adoption': False, 'goal_complete': False,
        'historical_masks_are_expert_truth': False, 'hidden_ground_truth': None,
        'threshold_or_margin_search': False, 'automatic_generation_quality_qualified': False,
        'input_exclusions_preserved': True, 'derivative_or_training_calls_permitted': False,
        'replay_ids': ['00_cloth_mask_native', '00_cloth_mask_degraded']}
    bindings(protocol); OUT.mkdir(); write(OUT/'protocol.json', protocol)
    print({'prepared': True, 'protocol_sha256': sha(OUT/'protocol.json'), 'detector_forwards': 0})


if __name__ == '__main__': main()
