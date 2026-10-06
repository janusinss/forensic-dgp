"""Replay the full R2 arithmetic audit after its stale sample-count check failed.

This is a local audit-only recovery. The frozen pilot, its failed audit, trained
checkpoints and export stay unchanged. No training, inference or selection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'outputs/cctv_dgp_broader_codes_vm_v16_r2'
PLAN = 'broader_codes_protocol_v16_r2.json'
PIN = '4f64cf2204458f8fadcab1824fc4bc293c65803e123fdebb304f7b5bd3a502a0'
AUDITOR = 'scripts/audit_cctv_dgp_broader_codes_v16.py'
AUDITOR_SHA = '0ec24b386ccfc210f4efe126d48d1e3e4064c38574bf158f139f310b7d6472fd'
REPLACEMENTS = (
    (b"('cache_timing.json', 'references', 20, 900)",
     b"('cache_timing.json', 'references', d['cache_timing_at_reference'], d['cache_cap_seconds'])"),
    (b"('fit_timing.json', 'update', 25, 1200)",
     b"('fit_timing.json', 'update', d['timing_update'], d['fit_cap_seconds'])"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def corrected_source(original):
    """Permit precisely the two protocol-bound count/cap tuple replacements."""
    require(hashlib.sha256(original).hexdigest() == AUDITOR_SHA,
            'Recovery is only for the unchanged frozen R2 auditor')
    corrected = original
    for before, after in REPLACEMENTS:
        require(corrected.count(before) == 1 and after not in corrected,
                'Stale audit tuple is absent or ambiguous')
        corrected = corrected.replace(before, after, 1)
    restored = corrected
    for before, after in REPLACEMENTS:
        restored = restored.replace(after, before, 1)
    require(restored == original, 'Recovery altered more than the timing tuples')
    compile(corrected, 'audit_corrected.py', 'exec')
    return corrected


def recover(return_root, recovery_to):
    return_root, recovery_to = return_root.resolve(), recovery_to.resolve()
    require(return_root.is_relative_to(ROOT / 'outputs') and return_root.is_dir(),
            'Require a locally imported return under outputs')
    require(recovery_to.is_relative_to(ROOT / 'outputs') and not recovery_to.exists()
            and not recovery_to.is_relative_to(return_root)
            and not recovery_to.is_relative_to(BUNDLE),
            'Require a fresh, separate recovery directory under outputs')
    require(sha(BUNDLE / PLAN) == sha(return_root / PLAN) == PIN,
            'Frozen/returned R2 protocol differs')
    protocol = read(BUNDLE / PLAN)
    require(protocol['assets_sha256'][AUDITOR] == AUDITOR_SHA,
            'Frozen auditor is not bound to the R2 protocol')
    for name, expected in protocol['assets_sha256'].items():
        require(sha(BUNDLE / name) == sha(return_root / name) == expected,
                'Frozen/returned source differs: ' + name)

    prefix = read(return_root / 'local_failure_import.json')
    require(prefix['complete'] is True and prefix['success'] is False
            and prefix['protocol_sha256'] == PIN
            and prefix['completed_trace_records_checked'] == 3128
            and prefix['exposures_checked'] == 31280,
            'First import and preserve the failure export with all completed updates')
    failure = read(return_root / 'supervisor_failure.json')
    audit_log = return_root / 'audit.log'
    require(failure['complete'] is False and failure['protocol_sha256'] == PIN
            and failure['resume_permitted'] is False
            and failure['error_type'] == 'CalledProcessError'
            and 'audit_cctv_dgp_broader_codes_v16.py' in failure['error']
            and audit_log.read_text(encoding='utf-8').rstrip().endswith(
                'ValueError: Timing stop receipt differs'),
            'Recovery is restricted to the reported post-training timing audit failure')
    out = return_root / 'outputs/broader_codes_v16_r2'
    results = read(out / 'results.json')
    require(results['complete'] is True and results['protocol_sha256'] == PIN
            and results['optimizer_updates'] == 3128
            and results['backward_calls'] == 3129
            and results['training_exposures'] == 31280,
            'Full trained results are required; this cannot recover interrupted training')
    require(protocol['design']['cache_timing_at_reference'] == 30
            and read(out / 'cache_timing.json')['references'] == 30
            and protocol['design']['audit_cap_seconds'] == 240,
            'Expected R2 sample count or audit cap differs')

    original = (BUNDLE / AUDITOR).read_bytes()
    corrected = corrected_source(original)
    recovery_to.mkdir()
    executable = recovery_to / 'audit_corrected.py'
    with executable.open('xb') as stream:
        stream.write(corrected)
    evidence = {'format': 'v16-r2-audit-only-recovery-1', 'protocol_sha256': PIN,
        'frozen_auditor_sha256': AUDITOR_SHA, 'corrected_auditor_sha256': sha(executable),
        'recovery_script_sha256': sha(Path(__file__)),
        'results_sha256': sha(out / 'results.json'), 'original_audit_log_sha256': sha(audit_log),
        'supervisor_failure_sha256': sha(return_root / 'supervisor_failure.json'),
        'prefix_import_sha256': sha(return_root / 'local_failure_import.json'),
        'return_root': str(return_root), 'stale_cache_sample_count': 20,
        'protocol_cache_sample_count': 30, 'audit_cap_seconds': 240,
        'correction': 'Only timing tuple counts/caps now read the unchanged protocol design.',
        'original_failure_preserved': True, 'recipe_changed': False,
        'neural_forwards': 0, 'backward_calls': 0, 'optimizer_updates': 0,
        'checkpoint_selected': False, 'production_promoted': False}
    write_new(recovery_to / 'recovery_request.json', evidence)
    receipt = recovery_to / 'local_full_audit.json'
    command = [sys.executable, '-X', 'utf8', '-u', str(executable), '--root', str(BUNDLE),
        '--parent', str(ROOT / 'outputs/cctv_dgp_face_code_fit_vm_v12_r2'),
        '--mixed', str(ROOT / 'outputs/cctv_dgp_mixed_vm_v9_r2'),
        '--baseline', str(ROOT / 'outputs/cctv_dgp_generalization_return_v15/outputs/generalization_v15'),
        '--expected-sha', PIN, '--results', str(out), '--receipt', str(receipt)]
    start = time.monotonic()
    try:
        with (recovery_to / 'audit.log').open('x', encoding='utf-8') as log:
            child = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   check=False, timeout=240)
        require(child.returncode == 0, 'Full corrected audit failed; inspect the new audit.log')
        audited = read(receipt)
        require(audited['complete'] is True and audited['protocol_sha256'] == PIN
                and audited['results_sha256'] == evidence['results_sha256']
                and audited['auditor_sha256'] == evidence['corrected_auditor_sha256']
                and all(audited[k] == 0 for k in ['neural_forwards', 'backward_calls', 'optimizer_updates']),
                'Corrected audit receipt binding differs')
        require(sha(BUNDLE / AUDITOR) == sha(return_root / AUDITOR) == AUDITOR_SHA
                and sha(return_root / 'audit.log') == evidence['original_audit_log_sha256']
                and sha(out / 'results.json') == evidence['results_sha256'],
                'Original source/failure/results changed during recovery')
        completion = {**evidence, 'complete': True, 'full_audit_passed': True,
            'full_audit_receipt_sha256': sha(receipt), 'seconds': time.monotonic() - start,
            'limitation': 'Serialized provenance/pixel/logit/embedding audit only. Quality guards, visual development review and independent final review still determine usefulness.'}
        write_new(recovery_to / 'recovery_completion.json', completion)
        print({'complete': True, 'full_audit_passed': True,
               'receipt': str(receipt), 'seconds': completion['seconds']}, flush=True)
    except BaseException as error:
        write_new(recovery_to / 'recovery_failure.json', {**evidence, 'complete': False,
            'error_type': type(error).__name__, 'error': str(error),
            'seconds': time.monotonic() - start, 'retry_or_training_authorized': False})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--return-root', type=Path, required=True)
    parser.add_argument('--recovery-to', type=Path, required=True)
    args = parser.parse_args()
    recover(args.return_root, args.recovery_to)
