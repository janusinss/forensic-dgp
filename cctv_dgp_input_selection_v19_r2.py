"""Separate inference-only normalization correction; original V19 remains frozen."""
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
HISTORICAL = HERE / 'original_v19/cctv_dgp_input_selection_v19.py'
if not HISTORICAL.is_file():
    # Working source/tests before packaging; the VM bundle requires its own
    # byte-identical original_v19 subtree, never a randomly located dependency.
    HISTORICAL = HERE / 'outputs/cctv_dgp_input_selection_vm_v19/cctv_dgp_input_selection_v19.py'
spec = importlib.util.spec_from_file_location('_immutable_v19_contract_for_r2', HISTORICAL)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)

from dgp_input_normalization_v19_r2 import POLICY as NORMALIZATION_POLICY

ORIGINAL_PIN = '2d707ebba97524867c4ce7610666e3abeb29638eb9054a8025983b6476e628d2'
DIAGNOSTIC_RESULTS = '8a8ac06d6c5948e23c7196c03106a76a1e4902a316952fc47c86fa1a41fc4671'
DIAGNOSTIC_ARCHIVE = '9d910721986df024a3ba346766c6385487603bb5174ebc771456c3d8159d7f1c'
PLAN = 'input_selection_protocol_v19_r2.json'
FORMAT = 'dgp-input-selection-canonical-baseline-v19-r2'
ARMS = original.ARMS
DESIGN = {**original.DESIGN, 'normalization_policy': NORMALIZATION_POLICY,
          'additional_canonical_DGP_forwards': 520, 'internal_spatial_base_raw_exports': 520,
          'original_scientific_gates_changed': False}
sha, read, write, safe, require = original.sha, original.read, original.write, original.safe, original.require
aggregate, validate_automatic_alias = original.aggregate, original.validate_automatic_alias


def expected_counts():
    return {**original.expected_counts(), 'dgp': 1090}


def validate_result_scope(result, pin):
    original.validate_result_scope(result, pin)
    require(result['normalization_policy'] == NORMALIZATION_POLICY
            and result['original_scientific_gates_changed'] is False
            and result['internal_spatial_base_raw_exports'] == 520,
            'R2 declared normalization/raw scope differs')


def verify(root, parent, r2, mixed, baseline, pin):
    root = Path(root)
    require(sha(root / PLAN) == pin == (root / 'protocol.sha256').read_text().strip(), 'V19 R2 protocol differs')
    p = read(root / PLAN)
    require(p['format'] == FORMAT and p['design'] == DESIGN
            and p['original_protocol_sha256'] == ORIGINAL_PIN, 'Separate R2 finite design differs')
    for name, digest in p['assets_sha256'].items():
        require(sha(safe(root, name)) == digest, 'R2 asset differs:' + name)
    require(sha(HISTORICAL) == p['assets_sha256']['original_v19/cctv_dgp_input_selection_v19.py'],
            'Imported historical contract differs')
    historical = root / 'original_v19'
    require(sha(historical / original.PLAN) == ORIGINAL_PIN, 'Original V19 protocol changed')
    hp, v, old = original.verify(historical, parent, r2, mixed, baseline, ORIGINAL_PIN)
    for key in ['references', 'cases', 'training_parity_cases', 'training_parity_references',
                'preview_reference_ids', 'data_assets_sha256', 'terminal_state_hash',
                'baseline_results_sha256', 'training', 'validation_used', 'native_used',
                'native_reserved_used', 'teacher_used', 'production_promoted']:
        require(p[key] == hp[key], 'Original cohort/weights/scope differs:' + key)
    # Every historical byte is included and remains pinned, rather than changing
    # old modules or manufacturing a receipt that says the failed run succeeded.
    expected = {name: digest for name, digest in hp['assets_sha256'].items()}
    expected.update({original.PLAN: ORIGINAL_PIN, 'protocol.sha256': sha(historical / 'protocol.sha256')})
    actual = {name.removeprefix('original_v19/'): digest for name, digest in p['assets_sha256'].items()
              if name.startswith('original_v19/')}
    require(actual == expected, 'Original V19 asset set differs')
    diag = read(root / 'diagnostic/results.json')
    audit = read(root / 'diagnostic/local_independent_audit.json')
    receipt = read(root / 'lineage/diagnostic_export.json')
    require(sha(root / 'diagnostic/results.json') == DIAGNOSTIC_RESULTS
            and diag['complete'] and diag['normalization_hypothesis_confirmed'] is True
            and audit['complete'] and audit['full_diagnostic_audit']
            and audit['results_sha256'] == DIAGNOSTIC_RESULTS
            and audit['archive_sha256'] == DIAGNOSTIC_ARCHIVE
            and audit['normalization_hypothesis_confirmed']
            and receipt['complete'] and receipt['diagnostic_succeeded']
            and receipt['archive_sha256'] == DIAGNOSTIC_ARCHIVE
            and receipt['results_sha256'] == DIAGNOSTIC_RESULTS,
            'Confirmed independently audited normalization diagnostic required')
    require(diag['DGP_state_before'] == diag['DGP_state_after']
            == read(root / 'lineage/failed_execution.json')['states_before']['dgp']
            and diag['protected_original_fingerprints_before'] == diag['protected_original_fingerprints_after'],
            'Original diagnostic DGP/file invariant differs')
    for name, digest in diag['artifacts_sha256'].items():
        require(sha(safe(root / 'diagnostic', name)) == digest, 'Diagnostic raw evidence differs')
    require(p['training'] is False and p['validation_used'] is True and all(p[key] is False for key in
            ['native_used', 'native_reserved_used', 'teacher_used', 'production_promoted']), 'R2 scope differs')
    sys.path.insert(0, str(historical))
    sys.path.insert(0, str(root))
    return p, v, old
