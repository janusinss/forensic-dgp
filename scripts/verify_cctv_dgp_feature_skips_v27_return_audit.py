"""Independent full-source migration readback for prospective V27 return audits."""
import ast
import json
from pathlib import Path
import time

from verify_cctv_dgp_feature_skips_v27 import ROOT, NEW, OUT, PIN, sha, tree, same, function, target_name

OLD_PIN = 'f9ccf86ee8e87ca87d43e849df7c0deae8e511e0db1a175b86ab279ec6964994'


def backwards(text):
    for new, old in [
        ('import_cctv_dgp_feature_skips_v27', 'import_cctv_dgp_batchmatched_identity_v26'),
        ('cctv_dgp_feature_skips_v27_return_rules', 'cctv_dgp_batchmatched_identity_v26_return_rules'),
        ('audit_cctv_dgp_feature_skips_v27_execution', 'audit_cctv_dgp_batchmatched_identity_v26_execution'),
        ('cctv_dgp_feature_skips_vm_v27', 'cctv_dgp_batchmatched_identity_vm_v26'),
        ('cctv_dgp_feature_skips_v27_return', 'cctv_dgp_batchmatched_identity_v26_return'),
        ('cctv_dgp_feature_skips_v27_independent_audit.json', 'cctv_dgp_batchmatched_identity_v26_independent_audit.json'),
        ('cctv-dgp-feature-skips-v27', 'cctv-dgp-batchmatched-identity-v26'),
        ('scripts/cctv_dgp_feature_skips_v27_vm.py', 'scripts/cctv_dgp_batchmatched_identity_v26_vm.py'),
        ('scripts/run_v27.sh', 'scripts/run_v26.sh'),
        ('cctv_dgp_feature_skips_v27_preflight.py', 'cctv_dgp_batchmatched_identity_v26_preflight.py'),
        ('scripts/install_v27.py', 'scripts/install_v26.py'),
        ('cctv_dgp_feature_skips_v27.py', 'cctv_dgp_spatial_features_v25.py'),
        ('cctv_dgp_feature_skips_v27_prepare.py', 'cctv_dgp_spatial_features_v25_prepare.py'),
        ('dgp-direct-feature-skips-capacity-v27', 'dgp-spatial-batchmatched-identity-capacity-v26'),
        ('all36_', 'all26_'), ('V27', 'V26'), (PIN, OLD_PIN)]:
        text = text.replace(new, old)
    return text


def main():
    started = time.monotonic()
    path = OUT / 'return_audit_preparation.json'
    record = json.loads(path.read_text(encoding='utf-8'))
    assert record['complete'] and record['protocol_sha256'] == PIN and sha(NEW / 'protocol.json') == PIN
    for name, digest in {**record['basis_sha256'], **record['generated_source_sha256']}.items():
        assert sha(ROOT / name) == digest, name
    # Importer retains every original member/path/hash/resource/nonfinite JSON guard.
    imported = backwards((ROOT / 'scripts/import_cctv_dgp_feature_skips_v27.py').read_text(encoding='utf-8'))
    same(tree(imported), tree((ROOT / 'scripts/import_cctv_dgp_batchmatched_identity_v26.py').read_text(encoding='utf-8')),
        'Original full safe importer differs beyond declared names/pin')
    audited = backwards((ROOT / 'scripts/audit_cctv_dgp_feature_skips_v27.py').read_text(encoding='utf-8'))
    audited = audited.replace("len(p['assets_sha256']) == 246", "len(p['assets_sha256']) == 240")
    audited = audited.replace("receipt['trainable_parameters'] == 55524", "receipt['trainable_parameters'] == 53781")
    audited = audited.replace("'<pinned-local-V26-class-only>'", "'<pinned-local-V25-class-only>'")
    current = tree(audited); initial = function(current, 'verify_initial_state')
    index = next(i for i, statement in enumerate(initial.body) if target_name(statement) == 'original_path')
    additions = initial.body[index:index + 6]
    assert [target_name(statement) for statement in additions] == ['original_path', 'p', None, 'original', None, None]
    assert all(isinstance(additions[i], ast.Expr) and isinstance(additions[i].value, ast.Call) for i in [2, 4, 5])
    assert 'closed_V26_initial_head_sha256' in ast.unparse(additions[2])
    assert 'len(original) == 28' in ast.unparse(additions[4]) and 'torch.equal(saved[name], value)' in ast.unparse(additions[4])
    assert "torch.count_nonzero(value) == 0" in ast.unparse(additions[5])
    del initial.body[index:index + 6]
    checks = [node for node in ast.walk(initial) if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp)]
    assert len(checks) == 1 and isinstance(checks[0].test.op, ast.Or)
    assert ast.unparse(checks[0].test.values[1]) == "name.startswith('feature_skips.')"
    checks[0].test = checks[0].test.values[0]
    same(current, tree((ROOT / 'scripts/audit_cctv_dgp_batchmatched_identity_v26.py').read_text(encoding='utf-8')),
        'Original full saved-output/group/capacity/state/feature/CPU replay audit changed beyond declared new head and initial proof')
    executed = backwards((ROOT / 'scripts/audit_cctv_dgp_feature_skips_v27_execution.py').read_text(encoding='utf-8'))
    current = tree(executed); check = function(current, 'check_execution')
    updates = next(node for node in check.body if isinstance(node, ast.If) and ast.unparse(node.test) == "execution['updates'] > 0")
    index = next(i for i, statement in enumerate(updates.body) if target_name(statement) == 'skip_values')
    assert ast.unparse(updates.body[index].value) == "gradient.get('per_feature_skip_weight_gradient_sum_squares')"
    guard = ast.unparse(updates.body[index + 1])
    for text in ['math.isfinite(value)', 'value > 0', "per_tensor['feature_skips.' + index + '.weight']"]:
        assert text in guard
    del updates.body[index:index + 2]
    returned = check.body[-1].value
    key = next(i for i, node in enumerate(returned.keys) if node.value == 'five_direct_feature_skip_preoptimizer_gradients_verified')
    del returned.keys[key]; del returned.values[key]
    same(current, tree((ROOT / 'scripts/audit_cctv_dgp_batchmatched_identity_v26_execution.py').read_text(encoding='utf-8')),
        'Original full execution/gradient/timing/neural count audit differs beyond new positive skip-gradient guard')
    ruled = backwards((ROOT / 'scripts/cctv_dgp_feature_skips_v27_return_rules.py').read_text(encoding='utf-8'))
    ruled = ruled.replace("len(p['assets_sha256']) == 246 and len(p['inherited_assets']) == 240", "len(p['assets_sha256']) == 240 and len(p['inherited_assets']) == 235")
    current = tree(ruled); original = tree((ROOT / 'scripts/cctv_dgp_batchmatched_identity_v26_return_rules.py').read_text(encoding='utf-8'))
    proof = function(current, 'check_identity_preflights')
    preflight = next(node for node in proof.body if isinstance(node, ast.If) and ast.unparse(node.test) == 'preflights')
    extra = preflight.body.pop()
    for token in ["shared_V26_initial_tensors_exact", '28', 'new_feature_skip_tensors_exact_zero', '10', '55524']:
        assert token in ast.unparse(extra)
    # Installation switches a source-bound copy receipt to zero-copy hardlinks.
    current.body = [node for node in current.body if not (isinstance(node, ast.FunctionDef) and node.name == 'check_installation')]
    original.body = [node for node in original.body if not (isinstance(node, ast.FunctionDef) and node.name == 'check_installation')]
    same(current, original, 'Original full case-role/finite/exact-zero identity rules changed beyond declarations')
    result = {'complete': True, 'date': '2026-10-06', 'checker_sha256': sha(Path(__file__)),
        'preparation_sha256': sha(path), 'protocol_sha256': PIN,
        'four_original_full_source_bodies_compared': True,
        'safe_importer_unchanged_except_names_and_pin': True,
        'all_original_CPU_replay_group_capacity_state_feature_cohort_and_scientific_guards_retained': True,
        'original_execution_timing_and_neural_counts_retained': True,
        'all36_initial_exact_zero_identity_and_five_positive_skip_gradients_required': True,
        'original_initial_tensor_bytes_and_zero_new_shortcuts_required': True,
        'declared_installation_receipt_change_to_zero_copy_hardlinks': True,
        'installation_receipt_and_tamper_tests_required': True,
        'neural_or_gradient_calls': 0, 'optimizer_updates': 0, 'VM_actions': False,
        'app_promotion': False, 'actual_V27_return_pending': True, 'goal_complete': False,
        'seconds': time.monotonic() - started}
    destination = OUT / 'independent_return_audit_source_check.json'
    with destination.open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
