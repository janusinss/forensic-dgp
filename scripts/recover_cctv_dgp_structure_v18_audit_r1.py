"""One external numeric-only audit correction; preserves the frozen V18 return."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_structure_vm_v18'
RETURN = ROOT / 'outputs/cctv_dgp_structure_return_v18'
RECOVERY = ROOT / 'outputs/cctv_dgp_structure_v18_audit_recovery_r1'
PROTOCOL = 'e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb'

OLD = """    expected_weights = q.normalized_group_weights(means)
    v.require(weighting['raw_MSE_means'] == means and weighting['weights'] == expected_weights, 'Training-only weighting differs')"""
NEW = """    from cctv_dgp_structure_audit_numeric_v18_r1 import verify_training_weighting
    reference_errors = {key: [] for key in means}
    for cid, case in cases.items():
        error = cached[cid]['dgp_base'].transpose(1, 2, 0) - targets[case['reference_id']].astype(np.float32) / 255
        reference_errors[case['source'] + '/' + case['profile']].append(
            float(np.square(error[masks[case['reference_id']]]).mean(dtype=np.float64)))
    reference_means = {key: float(np.mean(values, dtype=np.float64)) for key, values in reference_errors.items()}
    expected_weights, numeric_check = verify_training_weighting(weighting, means, reference_means, q.normalized_group_weights)
    v.write(receipt.parent / 'training_accumulation_check.json', numeric_check)"""


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def corrected_source(source):
    if source.count(OLD) != 1:
        raise ValueError('Require exact single frozen accumulation assertion')
    value = source.replace(OLD, NEW)
    ast.parse(value, feature_version=(3, 10))
    return value


def recover():
    started = time.monotonic()
    protocol = json.loads((BUNDLE / 'structure_protocol_v18.json').read_text())
    original = BUNDLE / 'scripts/audit_cctv_dgp_structure_v18.py'
    failure = RETURN / 'local_audit.log'
    results = RETURN / 'outputs/structure_v18/results.json'
    terminal = json.loads((ROOT / 'outputs/supervisor_completion_v18.json').read_text())
    if RECOVERY.exists():
        raise ValueError('Preserve existing correction; do not repeat/overwrite')
    if sha(BUNDLE / 'structure_protocol_v18.json') != PROTOCOL or terminal['protocol_sha256'] != PROTOCOL:
        raise ValueError('Require unchanged frozen V18 protocol')
    if sha(original) != protocol['assets_sha256']['scripts/audit_cctv_dgp_structure_v18.py']:
        raise ValueError('Frozen auditor changed')
    if not failure.is_file() or 'ValueError: Training-only weighting differs' not in failure.read_text():
        raise ValueError('Preserved original accumulation failure required')
    if sha(results) != terminal['results_sha256'] or not terminal['complete']:
        raise ValueError('Require completed unchanged transferred result')
    before = {'protocol': sha(BUNDLE / 'structure_protocol_v18.json'), 'original_auditor': sha(original),
              'results': sha(results), 'original_audit_failure': sha(failure)}
    RECOVERY.mkdir()
    corrected = RECOVERY / original.name
    with corrected.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(corrected_source(original.read_text(encoding='utf-8')))
    numeric = ROOT / 'cctv_dgp_structure_audit_numeric_v18_r1.py'
    shutil.copyfile(numeric, RECOVERY / numeric.name)
    receipt = {'complete': True, 'protocol_sha256': PROTOCOL, 'results_sha256': before['results'],
        'correction': 'Only exact training-error accumulation assertion becomes float64-reference/float32-roundoff check; recorded coefficients still exact',
        'before': before, 'corrected_auditor_sha256': sha(corrected), 'numeric_helper_sha256': sha(numeric),
        'recovery_script_sha256': sha(Path(__file__)), 'audit_cap_seconds': 300,
        'external_process_timeout_seconds': 330, 'training_restarted': False, 'quality_gates_changed': False}
    (RECOVERY / 'correction_provenance.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    args = [sys.executable, '-B', '-X', 'utf8', '-u', str(corrected), '--root', str(BUNDLE),
        '--parent', str(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'),
        '--r2', str(ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'),
        '--mixed', str(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'),
        '--baseline', str(ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'),
        '--expected-sha', PROTOCOL, '--results', str(results.parent),
        '--receipt', str(RECOVERY / 'local_full_audit.json'), '--replay-head']
    with (RECOVERY / 'audit.log').open('x', encoding='utf-8') as log:
        result = subprocess.run(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=330)
    print((RECOVERY / 'audit.log').read_text()[-2600:], flush=True)
    after = {'protocol': sha(BUNDLE / 'structure_protocol_v18.json'), 'original_auditor': sha(original),
             'results': sha(results), 'original_audit_failure': sha(failure)}
    if before != after:
        raise ValueError('Original protocol/source/results/failure changed during independent audit')
    if result.returncode:
        raise ValueError('Corrected audit failed; retain evidence and do not retrain')
    full = json.loads((RECOVERY / 'local_full_audit.json').read_text())
    if not full['complete'] or full['head_forwards'] != 250 or full['backward_calls'] or full['optimizer_updates']:
        raise ValueError('Completed frozen inference-only audit required')
    receipt.update({'originals_unchanged': True, 'audit_receipt_sha256': sha(RECOVERY / 'local_full_audit.json'),
                    'seconds': time.monotonic() - started})
    with (RECOVERY / 'recovery_completion.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(json.dumps({'complete': True, 'originals_unchanged': True, 'head_forwards': 250,
                      'backward_calls': 0, 'optimizer_updates': 0, 'seconds': receipt['seconds']}), flush=True)


if __name__ == '__main__':
    recover()
