"""Frozen V27 archive roles and independent installation/identity receipt checks."""
import math
from pathlib import Path
import re

from import_cctv_dgp_spatial_features_v25 import read, require, sha

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_feature_skips_vm_v27'
PIN = '22f76701e317b38d945838a6767239c5636cff534b61ddfbf248cc42ba08a82e'
REQUIRED_SOURCES = (
    'protocol.json', 'schedule.json', 'scripts/cctv_dgp_feature_skips_v27_vm.py',
    'scripts/run_v27.sh', 'cctv_dgp_batchmatched_identity_v26.py',
    'cctv_dgp_feature_skips_v27_preflight.py', 'scripts/install_v27.py',
    'cctv_dgp_feature_skips_v27.py', 'cctv_dgp_feature_skips_v27_prepare.py',
    'cctv_dgp_degraded_objective_v24.py',
)
REQUIRED_RECEIPTS = ('installation_receipt.json', 'supervisor_receipt.json', 'trainer_exit_code.txt', 'trainer.log')
OUTPUT_RECEIPTS = {
    'results.json', 'failure.json', 'execution_receipt.json', 'cohort_loss_setup.json',
    'frozen_DGP_features.json', 'one_batch_gradient_preflight.json',
    'feature_path_gradient_update2.json', 'timing_update20.json', 'early_structure_stop.json', 'stopped_head.pth',
}


def protocol(bundle=BUNDLE):
    require(sha(bundle / 'protocol.json') == PIN, 'Frozen local V27 protocol differs')
    p = read(bundle / 'protocol.json')
    require(p['format'] == 'dgp-direct-feature-skips-capacity-v27' and
            len(p['assets_sha256']) == 246 and len(p['inherited_assets']) == 240 and
            len(p['cases']) == 50 and len({case['id'] for case in p['cases']}) == 50,
            'Frozen V27 data/source roles differ')
    return p


def source_hashes(bundle=BUNDLE):
    p = protocol(bundle)
    result = {}
    for name in REQUIRED_SOURCES:
        digest = sha(bundle / name)
        expected = PIN if name == 'protocol.json' else p['assets_sha256'][name]
        require(digest == expected, 'Frozen local source differs: ' + name)
        result[name] = digest
    return result


def allowed_return_name(name, p):
    """Exact source/receipt roles; binary files may belong only to frozen TRAIN cases."""
    if name in set(REQUIRED_SOURCES) | set(REQUIRED_RECEIPTS) | {'export_manifest.json'}:
        return True
    if re.fullmatch(r'(?:preflight|feature_preflight|batchmatched_identity_preflight)_[0-9]+\.json', name):
        return True
    pieces = name.split('/')
    if len(pieces) == 2 and pieces[0] == 'outputs' and pieces[1] in OUTPUT_RECEIPTS:
        return True
    ids = {case['id'] for case in p['cases']}
    if len(pieces) == 3 and pieces[:2] == ['outputs', 'frozen_DGP_features']:
        return any(pieces[2] == case + '_fpn' + str(scale) + '.npy' for case in ids for scale in range(5))
    if len(pieces) != 3 or pieces[0] != 'outputs' or pieces[1] not in {'update0', 'update50', 'update400', 'update800'}:
        return False
    if pieces[2] in {'head.pth', 'metrics.json'}:
        return True
    suffixes = ['.npy', '.png', '_embedding.npy']
    if pieces[1] == 'update0':
        suffixes.append('_target_embedding.npy')
    return any(pieces[2] == case + suffix for case in ids for suffix in suffixes)


def finite(value, label, *, positive=False, cap=None):
    require(type(value) in (int, float) and math.isfinite(value) and
            (value > 0 if positive else value >= 0) and (cap is None or value <= cap), label)


def fixed_fields(receipt, expected, label):
    for key, value in expected.items():
        require(key in receipt and type(receipt[key]) is type(value) and receipt[key] == value,
                label + ': ' + key)


def check_installation(returned, p, bundle=BUNDLE):
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


def check_identity_preflights(returned, p):
    paths = sorted(returned.glob('batchmatched_identity_preflight_*.json'))
    preflights = [read(path) for path in sorted(returned.glob('preflight_*.json'))]
    require(len(paths) <= 1 and len(preflights) <= 1, 'One supervised neural preflight only')
    has_training = (returned / 'outputs').exists() and any((returned / 'outputs').iterdir())
    if not paths:
        require(not preflights and not has_training, 'Training or completed preflight lacks exact-zero identity proof')
        return {'proof_present': False, 'completed_main_preflight': False, 'VM_gradient_queries_checked': 0,
                'limit': 'Preflight failed before completed identity proof; no optimizer or quality acceptance'}
    proof = read(paths[0])
    policy = {
        'complete': True, 'cases': 50, 'batches': 10, 'gradient_calls': 20,
        'head_forwards': 10, 'recognizer_forwards': 20, 'optimizer_constructed': False,
        'optimizer_updates': 0, 'baseline_and_prediction_identical_cases': 50,
        'head_grad_buffers_empty': True, 'frozen_features_and_recognizer_have_no_gradients': True,
        'same_original_penalty_weight': 5, 'identity_margin': 0, 'quality_gate_relaxed': False,
    }
    numeric = {'legacy_component_value', 'legacy_component_gradient_norm',
               'batchmatched_component_value', 'batchmatched_component_gradient_norm', 'seconds'}
    require(set(proof) == set(policy) | numeric | {'head_state_before_after', 'recognizer_state_before_after', 'rows'},
            'Identity proof schema differs')
    fixed_fields(proof, policy, 'Identity proof policy differs')
    for key in ['head_state_before_after', 'recognizer_state_before_after']:
        require(isinstance(proof[key], str) and re.fullmatch(r'[0-9a-f]{64}', proof[key]), 'Identity proof state differs')
    finite(proof['seconds'], 'Identity proof timing differs', positive=True, cap=180)
    finite(proof['legacy_component_value'], 'Legacy discrepancy must have positive value', positive=True)
    finite(proof['legacy_component_gradient_norm'], 'Legacy discrepancy must have positive gradient', positive=True)
    for key in ['batchmatched_component_value', 'batchmatched_component_gradient_norm']:
        finite(proof[key], 'Nonfinite matched identity proof')
        require(proof[key] == 0, 'Matched identity penalty/gradient must be exactly zero')
    rows = proof['rows']
    require(isinstance(rows, list) and len(rows) == 10, 'Identity proof needs ten batches')
    row_policy = {'exact_cached_baseline_cases': 5, 'exact_reference_prediction_cosines': True,
                  'all36_matched_gradient_tensors_exactly_zero': True}
    values = []
    norms = []
    for index, row in enumerate(rows):
        require(isinstance(row, dict) and set(row) == set(row_policy) | {
            'ids', 'legacy_component_value', 'legacy_component_gradient_norm',
            'batchmatched_component_value', 'batchmatched_component_gradient_norm'}, 'Identity batch schema differs')
        fixed_fields(row, row_policy, 'Identity batch policy differs')
        require(row['ids'] == [case['id'] for case in p['cases'][index * 5:(index + 1) * 5]],
                'Identity batch case order differs')
        finite(row['legacy_component_value'], 'Nonfinite legacy batch value')
        finite(row['legacy_component_gradient_norm'], 'Nonfinite legacy batch gradient')
        values.append(row['legacy_component_value'])
        norms.append(row['legacy_component_gradient_norm'])
        for key in ['batchmatched_component_value', 'batchmatched_component_gradient_norm']:
            finite(row[key], 'Nonfinite matched identity batch')
            require(row[key] == 0, 'Every matched batch penalty/gradient must be exactly zero')
    require(abs(sum(values) - proof['legacy_component_value']) <= 1e-12, 'Legacy batch scalar arithmetic differs')
    require(proof['legacy_component_gradient_norm'] <= sum(norms) + 1e-12, 'Legacy summed gradient norm exceeds triangle bound')
    if preflights:
        receipt = preflights[0]
        require(receipt.get('complete') is True and receipt.get('protocol_sha256') == PIN and
                receipt.get('batchmatched_identity_proof') == proof and
                receipt.get('initial_head_state') == proof['head_state_before_after'] and
                receipt.get('recognizer_state') == proof['recognizer_state_before_after'],
                'Main preflight and identity proof linkage differs')
        require(receipt.get('neural_forward_counts') == p['prospective_return_audit']['expected_preflight_forward_counts'],
                'V27 preflight neural counts differ')
        finite(receipt.get('seconds'), 'Main preflight timing differs', positive=True, cap=300)
        require(receipt.get('shared_V26_initial_tensors_exact') == 28 and
            receipt.get('new_feature_skip_tensors_exact_zero') == 10 and receipt.get('trainable_parameters') == 55524,
            'Shared original initialization or exact new zero shortcuts differ')

    else:
        require(not has_training, 'Training lacks completed main preflight')
    return {'proof_present': True, 'completed_main_preflight': bool(preflights),
            'cases': 50, 'batches': 10, 'VM_gradient_queries_checked': 20,
            'legacy_component_value': proof['legacy_component_value'],
            'legacy_component_gradient_norm': proof['legacy_component_gradient_norm'],
            'batchmatched_component_value': 0, 'batchmatched_component_gradient_norm': 0,
            'all36_exact_zero_gradient_assertions_source_bound': True,
            'head_state': proof['head_state_before_after'], 'recognizer_state': proof['recognizer_state_before_after'],
            'seconds': proof['seconds'], 'optimizer_updates': 0,
            'limit': 'Source-bound L4 assertions, receipts and batch arithmetic; local audit does not rerun derivatives'}
