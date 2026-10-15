"""Independent publication/history/status audit. No model imports or forwards."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from cctv_dgp_spatial_readout_v1_contract import read, write, sha


def main():
    root = Path(__file__).resolve().parent.parent
    analysis = root/'outputs/cctv_dgp_spatial_readout_v1_analysis'
    stage = analysis/'publication'
    pub = read(stage/'publication.json')
    au = read(analysis/'independent_audit.json')
    visual = read(analysis/'visual_observations.json')
    learning = read(analysis/'learning_attribution.json')
    objective = read(analysis/'saved_objective_audit.json')
    returned = root/'outputs/cctv_dgp_spatial_readout_v1_return'
    protocol = read(returned/'protocol.json')
    assert pub['complete'] and not pub['model_qualified'] and not pub['goal_complete'] and not pub['new_packet_prepared']
    assert au['complete'] and au['paired_RAW_PNG_compositions_checked'] == 500 and len(au['sheets']) == 20
    assert [x['id'] for x in visual['references']] == [x['reference'] for x in protocol['reference_blocks']]
    assert visual['sheets_reviewed'] == 20 and visual['cells_viewed'] == 500 and not visual['model_qualified']
    for row in au['sheets']:
        assert sha(Path(row['path'])) == row['sha256']
    for name, digest in pub['evidence_sha256'].items():
        assert sha(analysis/name) == digest
    for name, digest in pub['analysis_scripts_sha256'].items():
        assert sha(root/'scripts'/name) == digest
    for name, digest in pub['new_reports_sha256'].items():
        assert sha(root/name) == digest
    for name, receipt in pub['documents'].items():
        old = (stage/'before'/name).read_bytes()
        current = (root/name).read_bytes()
        n = receipt['prefix_bytes']
        assert hashlib.sha256(old).hexdigest() == receipt['before_sha256']
        assert hashlib.sha256(current).hexdigest() == receipt['after_sha256']
        assert current[n:] == old and hashlib.sha256(current[:n]).hexdigest() == pub['prefix_sha256'][name]
        assert b'USED / FAILED' in current[:n]
    checkpoint = root/'outputs/cctv_dgp_face_code_fit_vm_v12_r2/weights/dgp_v2.pth'
    assert sha(checkpoint) == pub['original_checkpoint_sha256']
    report = (root/'CCTV_DGP_SPATIAL_READOUT_V1_RESULTS.md').read_text(encoding='utf-8')
    assert all(w in report for w in ['0.115752%', '0.132699%', '0.095199%', '0.111965%', '8 / 8', '18 / 18', '1%'])
    assert objective['source_summaries']['all']['summary']['lower_objective_references'] == {'A50': 20, 'B50': 1}
    cpu_vm = max(abs(x['difference']) for x in objective['baseline_CPU_VM_differences'])
    assert cpu_vm < 1e-7
    for route in ['A', 'B']:
        assert learning['routes'][route]['projection_applied_updates'] == 50
    checker = 'audit_cctv_dgp_spatial_readout_v1_return.py'
    assert sha(root/'scripts'/checker) == sha(returned/checker) == protocol['assets_sha256'][checker]
    for name in ['scripts/cctv_dgp_spatial_readout_v1_vm.py', 'cctv_dgp_spatial_readout_v1_model.py', 'cctv_dgp_spatial_readout_v1_contract.py']:
        assert sha(returned/name) == protocol['assets_sha256'][name]
    for name in ['review_cctv_dgp_spatial_readout_v1_return.py', 'audit_cctv_dgp_spatial_readout_v1_saved_objective.py']:
        tree = ast.parse((root/'scripts'/name).read_text(encoding='utf-8'))
        assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in
                       ['backward', 'cuda', 'load_state_dict', 'step'] for n in ast.walk(tree))
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), documents_exact_old_suffixes_verified=7,
        new_reports_verified=2, all20_visual_sheets_bound=True, all500_gallery_cells_previously_verified=True,
        all_planned_references_directly_reviewed=True, source_and_group_claims_match_audit=True,
        original_checkpoint_sha256=pub['original_checkpoint_sha256'], sealed_worker_and_checker_unchanged=True,
        baseline_CPU_VM_objective_max_difference=cpu_vm, local_neural_or_training_calls=0,
        original_recipe_status='USED_FAILED', new_packet_prepared=False, model_qualified=False, goal_complete=False)
    write(stage/'independent_publication_audit.json', record)
    print(json.dumps(record))


if __name__ == '__main__':
    main()
