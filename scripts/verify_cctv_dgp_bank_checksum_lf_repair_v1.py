"""Check the repaired checksum using GNU tools without installation or training."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_bank_checksum_lf_repair_v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not (OUT / 'verification.json').exists()
    p = json.loads((OUT / 'preparation.json').read_text())
    assert p['complete']
    archive = ROOT / 'outputs/cctv-dgp-bank-comparison-v1-execution.tar.gz'
    checksum = Path(str(archive) + '.sha256')
    expected = (p['archive_sha256'] + '  ' + archive.name + '\n').encode('ascii')
    assert checksum.read_bytes() == expected and sha(checksum) == p['corrected_checksum_sha256']
    legacy = (OUT / 'before' / checksum.name).read_bytes()
    assert legacy == expected[:-1] + b'\r\n'
    tr = Path('C:/Program Files/Git/usr/bin/tr.exe')
    check = Path('C:/Program Files/Git/usr/bin/sha256sum.exe')
    bash = Path('C:/Program Files/Git/bin/bash.exe')
    assert all(path.is_file() for path in [tr, check, bash])
    normalized = subprocess.run([str(tr), '-d', r'\r'], input=legacy, capture_output=True, check=True, timeout=10).stdout
    assert normalized == expected
    commands = {}
    for label, value in [('repaired_LF', expected), ('uploaded_CRLF_after_tr', normalized),
                         ('wrong_hash_refused', b'0'*64 + expected[64:])]:
        result = subprocess.run([str(check), '-c', '-'], input=value, cwd=archive.parent,
                                capture_output=True, timeout=60)
        success = label != 'wrong_hash_refused'
        assert (result.returncode == 0) == success
        if success:
            assert archive.name.encode() + b': OK' in result.stdout
        commands[label] = dict(exit_code=result.returncode, stdout=result.stdout.decode(errors='replace'),
                               stderr=result.stderr.decode(errors='replace'))
    guide = (ROOT / 'CCTV_DGP_BANK_COMPARISON_V1_VM.md').read_text()
    fragment = guide.split('3. Install the verified new packet', 1)[1].split('```bash\n', 1)[1].split('```', 1)[0]
    result = subprocess.run([str(bash), '--noprofile', '--norc', '-n', '-c', fragment],
                            capture_output=True, check=True, timeout=10)
    assert "tr -d '\\r'" in fragment and 'sha256sum -c - &&' in fragment
    for file in ['prepare_cctv_dgp_bank_comparison_v1.py', 'audit_cctv_dgp_bank_comparison_transfer_v1.py']:
        ast.parse((ROOT / 'scripts' / file).read_text(), filename=file, feature_version=(3,10))
    assert (ROOT / 'PROJECT_HANDOFF.md').read_bytes().endswith((OUT / 'before/PROJECT_HANDOFF.md').read_bytes())
    assert sha(ROOT / 'outputs/cctv_dgp_bank_comparison_v1_vm/protocol.json') == p['protocol_sha256']
    assert sha(archive) == p['archive_sha256'] and archive.stat().st_size == p['archive_bytes']
    after = {name: sha(ROOT / name) for name in p['previous_snapshots']}
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(Path(__file__)),
        preparation_sha256=sha(OUT / 'preparation.json'), archive_sha256=p['archive_sha256'], protocol_sha256=p['protocol_sha256'],
        checksum_bytes=len(expected), GNU_tr_legacy_CRLF_normalization_verified=True, GNU_checksum_outcomes=commands,
        replacement_step3_Bash_syntax_checked=True, Python310_syntax_checked=True,
        previous_handoff_preserved_as_exact_suffix=True, after_sha256=after,
        original_audit_retained=True, archive_rebuilt=False, model_gradient_or_training_calls=0,
        VM_connection_or_training_launch_performed=False, manual_step3_pending=True)
    with (OUT / 'verification.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    print(dict(complete=True, GNU_checksum_checks_passed=True, checksum_bytes=len(expected), archive_unchanged=True,
               VM_commands_executed=False), flush=True)


if __name__ == '__main__':
    main()
