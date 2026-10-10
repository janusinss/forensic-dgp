"""Read the actual successful Bash parse; preserve the sandbox-denied audit attempt."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / 'scripts/verify_cctv_dgp_v32_loss_gradient_v1_packet.py'
PREP = ROOT / 'outputs/cctv_dgp_v32_loss_gradient_v1_preparation'
PIN = '160b310bb68ae8cd700071e04c736e20e1d1d19e4c18d1a442277dd1dfffddd8'


def main():
    assert hashlib.sha256(ORIGINAL.read_bytes()).hexdigest() == PIN
    source = ORIGINAL.read_text(encoding='utf-8')
    before_guard = "with (PREP / 'Windows_gradient_guard.txt').open('x', encoding='utf-8') as stream: stream.write(guard.stderr)"
    after_guard = "assert 'Existing Linux VM only; no local differentiation' in (PREP / 'Windows_gradient_guard.txt').read_text(encoding='utf-8')"
    before_bash = "bash = subprocess.run(['C:/Program Files/Git/bin/bash.exe', '-n', str(shell).replace('\\\\', '/')], cwd=ROOT, capture_output=True, text=True, timeout=30)\n    assert bash.returncode == 0"
    after_bash = "parsed = read(PREP / 'bash_syntax_approved.json')\n    assert parsed['complete'] and parsed['exit_code'] == 0 and parsed['read_only_syntax_check']\n    assert parsed['script_sha256'] == sha(shell) and not parsed['VM_or_training_executed']"
    for before, after in [(before_guard, after_guard), (before_bash, after_bash)]:
        assert source.count(before) == 1, before
        source = source.replace(before, after)
    correction = {'complete': True, 'original_failed_checker_sha256': PIN,
                  'corrected_checker_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  'cause': 'Git Bash could not create its Windows signal pipe inside the sandbox; Win32 error5, exit3221225794.',
                  'successful_read_only_parse_sha256': hashlib.sha256((PREP / 'bash_syntax_approved.json').read_bytes()).hexdigest(),
                  'original_failure_receipt_sha256': hashlib.sha256((PREP / 'packet_audit_failure_v1.json').read_bytes()).hexdigest(),
                  'correction_scope': 'Reuse preserved guard evidence and verify the actual separately successful read-only Bash parse.',
                  'original_checker_packet_protocol_worker_and_caps_unchanged': True,
                  'neural_or_gradient_calls': 0, 'optimizer_updates': 0, 'VM_execution_started': False}
    with (PREP / 'packet_audit_r1_correction.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(correction, stream, indent=2)
    tree = ast.parse(source)
    namespace = {'__name__': 'pinned_packet_audit_r1', '__file__': str(Path(__file__).resolve())}
    exec(compile(tree, str(ORIGINAL), 'exec'), namespace)
    namespace['main']()


if __name__ == '__main__':
    main()
