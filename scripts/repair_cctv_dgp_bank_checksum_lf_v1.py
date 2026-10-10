"""Preserve checksum-format failure evidence; no archive rebuild or VM execution."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_bank_checksum_lf_repair_v1'
ARCHIVE = ROOT / 'outputs/cctv-dgp-bank-comparison-v1-execution.tar.gz'
PIN = 'ece56acfdf440fc1ced9f5d9d53ba39db8c3a4da48fa0b57bee5346d265297a0'
PROTOCOL = '0e4f9645abc99f6d980f3b20240b196f65f3e130472795faa2f964df1a5a3b3e'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert not OUT.exists()
    checksum = Path(str(ARCHIVE) + '.sha256')
    old = checksum.read_bytes()
    expected = (PIN + '  ' + ARCHIVE.name + '\n').encode('ascii')
    assert old == expected[:-1] + b'\r\n'
    assert ARCHIVE.stat().st_size == 809058558 and sha(ARCHIVE) == PIN
    protocol = ROOT / 'outputs/cctv_dgp_bank_comparison_v1_vm/protocol.json'
    assert sha(protocol) == PROTOCOL
    OUT.mkdir(); backup = OUT / 'before'; backup.mkdir()
    paths = [checksum, ROOT / 'CCTV_DGP_BANK_COMPARISON_V1_VM.md', ROOT / 'PROJECT_HANDOFF.md',
             ROOT / 'scripts/prepare_cctv_dgp_bank_comparison_v1.py',
             ROOT / 'scripts/audit_cctv_dgp_bank_comparison_transfer_v1.py']
    snapshots = {}
    for path in paths:
        destination = backup / path.name
        shutil.copyfile(path, destination)
        assert sha(destination) == sha(path)
        snapshots[path.relative_to(ROOT).as_posix()] = sha(path)
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        previous_checksum_bytes=len(old), previous_checksum_sha256=sha(backup / checksum.name),
        corrected_checksum_bytes=len(expected), corrected_checksum_sha256=hashlib.sha256(expected).hexdigest(),
        previous_snapshots=snapshots, archive_bytes=ARCHIVE.stat().st_size, archive_sha256=PIN,
        protocol_sha256=PROTOCOL, archive_rebuilt=False, packet_or_training_recipe_modified=False,
        local_model_gradient_or_training_calls=0, VM_connection_or_launch_performed=False,
        remote_installation_status='User reported checksum failure before extraction; corrected VM command awaits manual execution')
    with (OUT / 'preparation.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False) + '\n')
    checksum.write_bytes(expected)
    assert checksum.read_bytes() == expected
    print(dict(complete=True, corrected_checksum_bytes=len(expected), LF_only=True,
               archive_sha256=PIN, archive_rebuilt=False, VM_commands_executed=False), flush=True)


if __name__ == '__main__':
    main()
