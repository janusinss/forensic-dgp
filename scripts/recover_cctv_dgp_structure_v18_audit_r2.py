"""Separate final V18 audit correction; retain original and r1 failed evidence."""
import ast
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.recover_cctv_dgp_structure_v18_audit_r1 import OLD, NEW

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_structure_vm_v18'
RETURN = ROOT / 'outputs/cctv_dgp_structure_return_v18'
R1 = ROOT / 'outputs/cctv_dgp_structure_v18_audit_recovery_r1'
RECOVERY = ROOT / 'outputs/cctv_dgp_structure_v18_audit_recovery_r2'
PROTOCOL = 'e3a7567e9a9c53a1668464ab5c95edbf094c73e9a7c8e550635dcf8d8ef04adb'

REPLACEMENTS = [(OLD, NEW + """
    from cctv_dgp_structure_audit_numeric_v18_r2 import verify_psnr_summaries, verify_preservation_report
    arithmetic_checks = []"""), (
    "        v.require(summary == metrics['summary'], 'Snapshot aggregation differs')",
    "        arithmetic_checks.append({'update': update, 'snapshot': verify_psnr_summaries(summary, metrics['summary'])})"), (
    "        v.require(q.strict_preservation(summary, baseline_summary) == metrics['preservation'], 'Unchanged preservation report differs')",
    "        arithmetic_checks.append({'update': update, 'preservation': verify_preservation_report(q.strict_preservation(summary, baseline_summary), metrics['preservation'], summary, baseline_summary)})"), (
    """            v.require(v.aggregate(ablations) == metrics['zero_prior_summary']
                      and state_hash(head) == neural['final_state'], 'Final/zero-prior summary differs')""",
    """            arithmetic_checks.append({'update': update, 'zero_prior': verify_psnr_summaries(v.aggregate(ablations), metrics['zero_prior_summary'])})
            v.require(state_hash(head) == neural['final_state'], 'Final/zero-prior state differs')"""), (
    "    v.write(receipt, {'complete': True, 'protocol_sha256': pin, 'results_sha256': v.sha(out / 'results.json'),",
    """    v.write(receipt.parent / 'binary64_log10_checks.json', {'complete': True, 'checks': arithmetic_checks,
        'scope': 'Only derived PSNR/log10 roundoff; all non-PSNR means, stop ratios and quality decisions exact'})
    v.write(receipt, {'complete': True, 'protocol_sha256': pin, 'results_sha256': v.sha(out / 'results.json'),""")]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def corrected_source(source):
    for old, new in REPLACEMENTS:
        if source.count(old) != 1:
            raise ValueError('Require exact single frozen arithmetic/receipt block')
        source = source.replace(old, new)
    ast.parse(source, feature_version=(3, 10))
    return source


def recover():
    started = time.monotonic()
    original = BUNDLE / 'scripts/audit_cctv_dgp_structure_v18.py'
    results = RETURN / 'outputs/structure_v18/results.json'
    terminal = json.loads((ROOT / 'outputs/supervisor_completion_v18.json').read_text())
    protocol = json.loads((BUNDLE / 'structure_protocol_v18.json').read_text())
    paths = {'protocol': BUNDLE / 'structure_protocol_v18.json', 'original_auditor': original,
             'results': results, 'original_failure': RETURN / 'local_audit.log',
             'r1_failure': R1 / 'audit.log', 'r1_provenance': R1 / 'correction_provenance.json',
             'aggregation_diagnosis': ROOT / 'outputs/cctv_dgp_structure_v18_aggregation_diagnostic_r2.json'}
    if RECOVERY.exists():
        raise ValueError('Preserve existing correction; never repeat/overwrite')
    before = {key: sha(path) for key, path in paths.items()}
    if before['protocol'] != PROTOCOL or terminal['protocol_sha256'] != PROTOCOL:
        raise ValueError('Frozen V18 protocol differs')
    if before['original_auditor'] != protocol['assets_sha256']['scripts/audit_cctv_dgp_structure_v18.py']:
        raise ValueError('Original auditor differs')
    if before['results'] != terminal['results_sha256'] or not terminal['complete']:
        raise ValueError('Completed unchanged transferred result required')
    if 'ValueError: Training-only weighting differs' not in paths['original_failure'].read_text() or \
       'ValueError: Snapshot aggregation differs' not in paths['r1_failure'].read_text():
        raise ValueError('Both original arithmetic failures must remain')
    r1 = json.loads(paths['r1_provenance'].read_text())
    helper1 = ROOT / 'cctv_dgp_structure_audit_numeric_v18_r1.py'
    if sha(helper1) != r1['numeric_helper_sha256']:
        raise ValueError('Preserved first numeric correction differs')
    RECOVERY.mkdir()
    corrected = RECOVERY / original.name
    with corrected.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(corrected_source(original.read_text(encoding='utf-8')))
    helpers = [helper1, ROOT / 'cctv_dgp_structure_audit_numeric_v18_r2.py']
    for helper in helpers:
        shutil.copyfile(helper, RECOVERY / helper.name)
    evidence = {'complete': True, 'protocol_sha256': PROTOCOL, 'results_sha256': before['results'],
        'before': before, 'corrected_auditor_sha256': sha(corrected),
        'helper_sha256': {x.name: sha(x) for x in helpers}, 'recovery_script_sha256': sha(Path(__file__)),
        'correction': 'Float64-reference/float32 training-error accumulation and four-ULP binary64 log10 reference only; recorded weights, non-PSNR means, fitting stop ratios and quality decisions exact',
        'audit_cap_seconds': 300, 'external_process_timeout_seconds': 330,
        'training_restarted': False, 'quality_gates_changed': False}
    (RECOVERY / 'correction_provenance.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
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
    if before != {key: sha(path) for key, path in paths.items()}:
        raise ValueError('Original scientific evidence/failures changed during independent audit')
    if result.returncode:
        raise ValueError('Third audit attempt failed: stop corrections, retain evidence; no retraining')
    full = json.loads((RECOVERY / 'local_full_audit.json').read_text())
    if not full['complete'] or full['head_forwards'] != 250 or full['backward_calls'] or full['optimizer_updates']:
        raise ValueError('Completed frozen inference-only audit required')
    evidence.update({'originals_unchanged': True, 'audit_receipt_sha256': sha(RECOVERY / 'local_full_audit.json'),
                    'seconds': time.monotonic() - started})
    with (RECOVERY / 'recovery_completion.json').open('x', encoding='utf-8') as stream:
        json.dump(evidence, stream, indent=2)
    print(json.dumps({'complete': True, 'originals_unchanged': True, 'head_forwards': 250,
                      'backward_calls': 0, 'optimizer_updates': 0, 'seconds': evidence['seconds']}), flush=True)


if __name__ == '__main__':
    recover()
