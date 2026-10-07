"""Prospective V27 safe import and CPU replay, retaining original V26 guards."""
import ast
from pathlib import Path
import time

from prepare_cctv_dgp_feature_skips_v27 import ROOT, NEW, OLD, OUT, read, sha, write, once

PIN = '22f76701e317b38d945838a6767239c5636cff534b61ddfbf248cc42ba08a82e'
OLD_PIN = 'f9ccf86ee8e87ca87d43e849df7c0deae8e511e0db1a175b86ab279ec6964994'


def source_names(text):
    for old, new in [
        ('scripts/cctv_dgp_batchmatched_identity_v26_vm.py', 'scripts/cctv_dgp_feature_skips_v27_vm.py'),
        ('scripts/run_v26.sh', 'scripts/run_v27.sh'),
        ('cctv_dgp_batchmatched_identity_v26_preflight.py', 'cctv_dgp_feature_skips_v27_preflight.py'),
        ('scripts/install_v26.py', 'scripts/install_v27.py'),
        ('cctv_dgp_spatial_features_v25.py', 'cctv_dgp_feature_skips_v27.py'),
        ('cctv_dgp_spatial_features_v25_prepare.py', 'cctv_dgp_feature_skips_v27_prepare.py')]:
        text = text.replace(old, new)
    return text


def rules(text):
    text = source_names(text).replace(OLD_PIN, PIN)
    text = text.replace("outputs/cctv_dgp_batchmatched_identity_vm_v26'", "outputs/cctv_dgp_feature_skips_vm_v27'")
    text = text.replace("p['format'] == 'dgp-spatial-batchmatched-identity-capacity-v26'", "p['format'] == 'dgp-direct-feature-skips-capacity-v27'")
    text = once(text, "len(p['assets_sha256']) == 240 and len(p['inherited_assets']) == 235", "len(p['assets_sha256']) == 246 and len(p['inherited_assets']) == 240")
    module = ast.parse(text); node = next(n for n in module.body if isinstance(n, ast.FunctionDef) and n.name == 'check_installation')
    lines = text.splitlines(keepends=True)
    replacement = '''def check_installation(returned, p, bundle=BUNDLE):
    receipt = read(returned / 'installation_receipt.json')
    policy = {'complete': True, 'protocol_sha256': PIN,
        'inherited_assets_verified_and_hardlinked': 240, 'assets_verified': 246,
        'inherited_data_bytes_copied': 0, 'old_V26_protocol_sha256': p['closed_V26_protocol_sha256'],
        'old_V26_failure_preserved': True, 'old_V26_initial_head_preserved': True,
        'cap_seconds': 60, 'model_or_gradient_calls': 0, 'optimizer_updates': 0,
        'original_files_changed': False, 'copy_fallback_permitted': False, 'quality_acceptance_not_implied': True}
    require(set(receipt) == set(policy) | {'seconds', 'inherited_logical_bytes'}, 'Installation receipt schema differs')
    fixed_fields(receipt, policy, 'Installation policy differs')
    finite(receipt['seconds'], 'Installation timing differs', positive=True, cap=60)
    expected = sum((bundle / name).stat().st_size for name in p['inherited_assets'])
    require(type(receipt['inherited_logical_bytes']) is int and receipt['inherited_logical_bytes'] == expected,
        'Hardlinked logical byte count differs')
    original = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return'
    require(sha(original / 'outputs/early_structure_stop.json') == p['closed_V26_early_failure_sha256'] and
        read(original / 'outputs/early_structure_stop.json')['pass'] is False, 'Original V26 failure changed')
    require(sha(original / 'outputs/update0/head.pth') == p['closed_V26_initial_head_sha256'], 'Original initial head changed')
    return {'complete': True, 'inherited_assets': 240, 'reconstructed_assets': 246,
        'hardlinked_logical_bytes': expected, 'inherited_data_bytes_copied': 0,
        'seconds': receipt['seconds'], 'old_failure_and_initial_head_preserved': True,
        'model_or_gradient_calls': 0,
        'limit': 'Exact receipt/source/asset linkage, not a new VM filesystem observation'}
'''
    text = ''.join(lines[:node.lineno - 1]) + replacement + ''.join(lines[node.end_lineno:])
    text = text.replace('all26_', 'all36_')
    anchor = "        finite(receipt.get('seconds'), 'Main preflight timing differs', positive=True, cap=300)"
    text = once(text, anchor, anchor + '''
        require(receipt.get('shared_V26_initial_tensors_exact') == 28 and
            receipt.get('new_feature_skip_tensors_exact_zero') == 10 and receipt.get('trainable_parameters') == 55524,
            'Shared original initialization or exact new zero shortcuts differ')
''')
    return text.replace('Frozen V26', 'Frozen V27').replace('local V26 protocol', 'local V27 protocol').replace('V26 preflight neural counts', 'V27 preflight neural counts')


def importer(text):
    text = text.replace(OLD_PIN, PIN)
    text = text.replace('cctv_dgp_batchmatched_identity_v26_return_rules', 'cctv_dgp_feature_skips_v27_return_rules')
    text = text.replace('cctv_dgp_batchmatched_identity_v26_return', 'cctv_dgp_feature_skips_v27_return')
    text = text.replace('cctv-dgp-batchmatched-identity-v26', 'cctv-dgp-feature-skips-v27')
    return text.replace('V26', 'V27')


def auditor(text):
    text = text.replace('from import_cctv_dgp_batchmatched_identity_v26 import', 'from import_cctv_dgp_feature_skips_v27 import')
    text = text.replace('cctv_dgp_batchmatched_identity_v26_return_rules', 'cctv_dgp_feature_skips_v27_return_rules')
    text = text.replace('audit_cctv_dgp_batchmatched_identity_v26_execution', 'audit_cctv_dgp_feature_skips_v27_execution')
    text = text.replace("outputs/cctv_dgp_batchmatched_identity_vm_v26'", "outputs/cctv_dgp_feature_skips_vm_v27'")
    text = text.replace('outputs/cctv_dgp_batchmatched_identity_v26_return', 'outputs/cctv_dgp_feature_skips_v27_return')
    text = text.replace('outputs/cctv_dgp_batchmatched_identity_v26_independent_audit.json', 'outputs/cctv_dgp_feature_skips_v27_independent_audit.json')
    text = text.replace("p['format'] == 'dgp-spatial-batchmatched-identity-capacity-v26' and len(p['assets_sha256']) == 240",
        "p['format'] == 'dgp-direct-feature-skips-capacity-v27' and len(p['assets_sha256']) == 246")
    text = text.replace("bundle / 'cctv_dgp_spatial_features_v25.py'", "bundle / 'cctv_dgp_feature_skips_v27.py'")
    text = text.replace("'<pinned-local-V25-class-only>'", "'<pinned-local-V27-class-only>'")
    text = once(text, "if name in ('kernel', 'reflect_indices', 'direct.weight', 'direct.bias', 'tail.weight', 'tail.bias'):",
        "if name in ('kernel', 'reflect_indices', 'direct.weight', 'direct.bias', 'tail.weight', 'tail.bias') or name.startswith('feature_skips.'):")
    anchor = "    maximum = 0.\n    for name, value in saved.items():"
    addition = '''    original_path = ROOT / 'outputs/cctv_dgp_batchmatched_identity_v26_return/outputs/update0/head.pth'
    p = read(BUNDLE / 'protocol.json')
    require(sha(original_path) == p['closed_V26_initial_head_sha256'], 'Original V26 initial head differs')
    original = torch.load(original_path, map_location='cpu', weights_only=True)
    require(len(original) == 28 and all(torch.equal(saved[name], value) for name, value in original.items()),
        'All original V26 initialization bytes must remain exact on VM')
    require(all(torch.count_nonzero(value) == 0 for name, value in saved.items() if name.startswith('feature_skips.')),
        'New feature shortcuts must initially be exactly zero')
    maximum = 0.
    for name, value in saved.items():'''
    text = once(text, anchor, addition)
    text = once(text, "receipt['trainable_parameters'] == 53781", "receipt['trainable_parameters'] == 55524")
    return text.replace('Independent V26', 'Independent V27').replace('local V26 protocol', 'local V27 protocol')


def execution(text):
    text = text.replace('from import_cctv_dgp_batchmatched_identity_v26 import', 'from import_cctv_dgp_feature_skips_v27 import')
    anchor = "        require(gradient.get('neural_forward_counts') == {'detail_head': 111, 'DGP': 50, 'DGP_CPU': 4, 'fixed_recognizer': 171},\n                'One-batch neural counts differ')"
    text = once(text, anchor, anchor + '''
        skip_values = gradient.get('per_feature_skip_weight_gradient_sum_squares')
        require(isinstance(skip_values, dict) and set(skip_values) == {'0','1','2','3','4'} and
            all(type(value) in (int,float) and math.isfinite(value) and value > 0 and
                value == per_tensor['feature_skips.' + index + '.weight'] for index,value in skip_values.items()),
            'All five new DGP-feature shortcuts require finite nonzero pre-optimizer gradients')
''')
    text = once(text, "            'direct_gradient_sum_squares': gradient['direct_gradient_sum_squares'] if gradient else None,",
        "            'direct_gradient_sum_squares': gradient['direct_gradient_sum_squares'] if gradient else None,\n            'five_direct_feature_skip_preoptimizer_gradients_verified': gradient is not None and execution['updates'] > 0,")
    return text.replace('Independent V26', 'Independent V27')


def main():
    started = time.monotonic()
    assert sha(NEW / 'protocol.json') == PIN
    assert not (OUT / 'return_audit_preparation.json').exists()
    jobs = [
        ('scripts/cctv_dgp_batchmatched_identity_v26_return_rules.py', 'scripts/cctv_dgp_feature_skips_v27_return_rules.py', rules),
        ('scripts/import_cctv_dgp_batchmatched_identity_v26.py', 'scripts/import_cctv_dgp_feature_skips_v27.py', importer),
        ('scripts/audit_cctv_dgp_batchmatched_identity_v26.py', 'scripts/audit_cctv_dgp_feature_skips_v27.py', auditor),
        ('scripts/audit_cctv_dgp_batchmatched_identity_v26_execution.py', 'scripts/audit_cctv_dgp_feature_skips_v27_execution.py', execution),
    ]
    results = {}; basis = {}
    for source, destination, transform in jobs:
        assert not (ROOT / destination).exists()
        text = transform((ROOT / source).read_text(encoding='utf-8'))
        ast.parse(text, feature_version=(3, 10))
        basis[source] = sha(ROOT / source)
        results[destination] = text
    for destination, text in results.items():
        with (ROOT / destination).open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
    write(OUT / 'return_audit_preparation.json', {'complete': True, 'date': '2026-10-06',
        'protocol_sha256': PIN, 'basis_sha256': basis,
        'generated_source_sha256': {name: sha(ROOT / name) for name in results},
        'scope': 'Prospective safe import and independent local CPU replay; actual V27 return pending',
        'archive_member_cap': 1600, 'local_CPU_seconds_cap': 600,
        'all_original_import_feature_cohort_replay_quality_timing_guards_retained': True,
        'original_shared_initial_tensors_and_zero_new_readouts_required': True,
        '36_gradient_groups_and_five_skip_preoptimizer_gradients_required': True,
        'source_body_migration_requires_independent_verification': True,
        'local_neural_calls': 0, 'local_gradient_calls': 0, 'optimizer_updates': 0,
        'VM_actions': False, 'app_promotion': False, 'goal_complete': False,
        'seconds': time.monotonic() - started})
    print('V27 prospective importer/full CPU audit prepared; original numerical and scientific bounds retained.')


if __name__ == '__main__':
    main()
