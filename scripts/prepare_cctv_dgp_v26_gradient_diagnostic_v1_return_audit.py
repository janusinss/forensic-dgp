"""Freeze prospective V26 matrix return checks; no return or L4 claim."""
import ast
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = time.monotonic()
    pin = 'cbbb9898dfdc6bc7b3409a21b26254b7286131d032e757047da9712c66b551f1'
    folder = ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_preparation'
    import sys
    sys.path.insert(0, str(ROOT / 'scripts'))
    from verify_cctv_dgp_v26_gradient_diagnostic_v1 import sha, read
    assert sha(ROOT / 'outputs/cctv_dgp_v26_gradient_diagnostic_v1_vm/protocol.json') == pin
    bindings = {}
    common = [('v25_gradient_diagnostic_v1', 'v26_gradient_diagnostic_v1'),
        ('cctv-dgp-v25-gradient', 'cctv-dgp-v26-gradient'), ('closed_V25', 'closed_V26'),
        ('cctv_dgp_spatial_features_vm_v25', 'cctv_dgp_batchmatched_identity_vm_v26'),
        ('cctv_dgp_spatial_features_v25_return', 'cctv_dgp_batchmatched_identity_v26_return'),
        ('cctv_dgp_spatial_features_v25_loss_audit_v1', 'cctv_dgp_batchmatched_identity_v26_loss_audit_v1'),
        ('V25', 'V26')]
    for name in ['import_cctv_dgp_v25_gradient_diagnostic_v1.py', 'audit_cctv_dgp_v25_gradient_diagnostic_v1_return.py']:
        original = ROOT / 'scripts' / name
        text = original.read_text(encoding='utf-8')
        for before, after in common:
            text = text.replace(before, after)
        if name.startswith('import_'):
            oldpin = "PIN = '884412f5e3dab571e630cdadb4f962046c8da309ebe6b62eac4d5af0e29d281a'"
            assert oldpin in text
            text = text.replace(oldpin, "PIN = '" + pin + "'")
        else:
            assert "'recognizer_forwards':120" in text
            text = text.replace("'recognizer_forwards':120", "'recognizer_forwards':70")
            check = "    # At zero tails the old comparison reports numerical identity cost on the same image.\n    require(rows[0]['component_values'][6]>0 and rows[0]['component_norms'][6]>0 and exact==50,'Demonstrated baseline identity-loss discrepancy')"
            assert check in text
            text = text.replace(check, "    # Corrected reference must stay exactly zero in every initial batch and array.\n    require(rows[0]['component_values'][6]==0 and rows[0]['component_norms'][6]==0 and exact==50,'Initial corrected identity proof differs')\n    require(np.count_nonzero(arrays[0][6])==0 and all(b['cohort_weighted_component_values'][6]==0 and b['cohort_weighted_component_gradient_norms'][6]==0 for b in r['snapshots'][0]['batches']),'Initial corrected identity batch/gradient must be exactly zero')")
            text = text.replace("'identity_baseline_numerical_discrepancy_confirmed':True", "'corrected_initial_identity_exact_zero_verified':True")
        target = ROOT / 'scripts' / name.replace('v25_gradient_diagnostic_v1', 'v26_gradient_diagnostic_v1')
        assert not target.exists(), target
        ast.parse(text, feature_version=(3, 10))
        with target.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(text)
        bindings[original.relative_to(ROOT).as_posix()] = sha(original)
        bindings[target.relative_to(ROOT).as_posix()] = sha(target)
    receipt = {'complete': True, 'date': '2026-10-06', 'protocol_sha256': pin,
        'builder_sha256': sha(Path(__file__)), 'source_bindings_sha256': bindings,
        'original_archive_caps_and_matrix_arithmetic_retained': True,
        'exact_zero_initial_corrected_identity_required': True,
        'actual_L4_return_present': False, 'neural_calls': 0, 'local_gradient_calls': 0,
        'optimizer_updates': 0, 'VM_actions': False, 'app_promotion': False,
        'goal_complete': False, 'seconds': time.monotonic() - started}
    with (folder / 'prospective_return_audit_preparation.json').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in receipt.items() if k != 'source_bindings_sha256'}, indent=2))


if __name__ == '__main__':
    main()
