"""Prepare a distinct exact checker with only replay layout and receipt path corrected."""
import ast
from pathlib import Path
from completion_feature_fusion_off_v1_common import ROOT, OUT, sha, read, write, verify_bindings


def main():
    p = read(OUT/'protocol.json'); verify_bindings(p)
    diagnostic = read(OUT/'replay_layout_diagnostic/results.json')
    assert diagnostic['complete'] and diagnostic['actual_network_forwards'] == 2
    assert diagnostic['root_cause_confirmed'] == 'Replay changed memory layout from actual channels-last to C-contiguous'
    assert diagnostic['rows'][0]['stages']['internal512']['exact'] and not diagnostic['rows'][0]['stages']['logits']['exact']
    assert all(v['exact'] for v in diagnostic['rows'][1]['stages'].values())
    original_path = ROOT/'scripts/audit_completion_feature_fusion_off_v1.py'
    original = original_path.read_text(encoding='utf-8')
    bound = p['sources_sha256'][original_path.relative_to(ROOT).as_posix()]
    assert sha(original_path) == bound == read(OUT/'replay_layout_diagnostic/contract.json')['original_checker_sha256']
    old = "torch.from_numpy(replay_source['neural_input512']), w=0, adain=False"
    new = "torch.from_numpy(replay_source['neural_input512']).to(memory_format=torch.channels_last), w=0, adain=False"
    old_receipt = "OUT/'independent_saved_output_audit.json'"
    new_receipt = "OUT/'independent_saved_output_audit_r1.json'"
    assert original.count(old) == original.count(old_receipt) == 1
    revised = original.replace(old, new).replace(old_receipt, new_receipt)
    assert revised.replace(new, old).replace(new_receipt, old_receipt) == original
    assert ast.dump(ast.parse(revised.replace(new, old).replace(new_receipt, old_receipt))) == ast.dump(ast.parse(original))
    target = ROOT/'scripts/audit_completion_feature_fusion_off_v1_r1.py'
    with target.open('x', encoding='utf-8', newline='\n') as stream: stream.write(revised)
    write(OUT/'audit_r1_contract.json', {
        'original_protocol_sha256': sha(OUT/'protocol.json'), 'original_checker_sha256': bound,
        'original_failure_sha256': sha(OUT/'replay_layout_diagnostic/original_checker_failure.json'),
        'diagnostic_contract_sha256': sha(OUT/'replay_layout_diagnostic/contract.json'),
        'diagnostic_results_sha256': sha(OUT/'replay_layout_diagnostic/results.json'),
        'R1_checker_sha256': sha(target), 'preparer_sha256': sha(Path(__file__)),
        'original_audit_retained': True, 'changes': ['Convert fresh replay input to actual adapter channels-last layout', 'Write a distinct R1 receipt'],
        'inverse_source_and_AST_exact': True, 'all_exact_comparisons_unchanged': True,
        'model_output_or_quality_gates_changed': False, 'outputs_regenerated': 0, 'new_tolerance': False,
        'gradient_calls': 0, 'optimizer_updates': 0, 'app_changes': False})
    print({'complete': True, 'checker_sha256': sha(target), 'changes': 2}, flush=True)


if __name__ == '__main__': main()
