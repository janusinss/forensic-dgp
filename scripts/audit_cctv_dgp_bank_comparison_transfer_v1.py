"""Independent archive/member hash and adverse-contract prerequisite checks."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def audit(root, archive, receipt):
    assert not receipt.exists()
    stamp = time.monotonic(); root = root.resolve()
    sys.path.insert(0, str(root))
    from cctv_dgp_bank_comparison_v1_contract import verify, validate, NAME, schedule
    pin = sha(root/'protocol.json'); p = verify(root, pin)
    expected = {NAME+'/'+n: digest for n, digest in p['assets_sha256'].items()}
    expected[NAME+'/protocol.json'] = pin
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            assert member.isfile() and not member.issym() and not member.islnk()
            assert member.name in expected and member.name not in seen
            h = hashlib.sha256(); stream = tar.extractfile(member)
            for block in iter(lambda: stream.read(4 * 1024**2), b''):
                h.update(block)
            assert h.hexdigest() == expected[member.name], member.name
            seen.add(member.name)
    assert seen == set(expected)
    checksum = sha(archive)
    checksum_bytes = Path(str(archive)+'.sha256').read_bytes()
    assert checksum_bytes == (checksum+'  '+archive.name+'\n').encode('ascii'), 'Transfer checksum must have the exact filename and ASCII LF bytes; reject CRLF/BOM'
    # These changes must be rejected, rather than weakening a gate/scope/root.
    adverse = []
    for label in ['lower_gate', 'final_native_role', 'repeat_exposure', 'remove_clear_anchor', 'replace_baseline']:
        q = copy.deepcopy(p)
        if label == 'lower_gate':
            q['scientific_thresholds']['early_structure_gain'] = .005
        elif label == 'final_native_role':
            q['native_development'][0]['role'] = 'final'
        elif label == 'repeat_exposure':
            q['schedule'][1] = q['schedule'][0][:]
        elif label == 'remove_clear_anchor':
            q['active_clear_blur_anchor_coefficients'][0] = 0
        else:
            q['original_checkpoint_sha256'] = '0'*64
        try:
            validate(q)
        except AssertionError:
            adverse.append(label)
        else:
            raise AssertionError('Adverse protocol accepted: '+label)
    result = subprocess.run([sys.executable, '-B', str(root/'scripts/cctv_dgp_bank_comparison_v1_vm.py'),
        '--root', str(root), '--protocol-sha', pin, '--verify-transfer'], capture_output=True, text=True, check=True, timeout=120)
    result = json.loads(result.stdout)
    assert result['complete'] and result['optimizer_updates'] == result['neural_or_gradient_calls'] == 0
    schedule_rows = schedule(p['cases'])
    assert len(schedule_rows) == 50 and len({i for row in schedule_rows for i in row}) == 250
    record = {'complete': True, 'checker_sha256': sha(__file__), 'protocol_sha256': pin,
        'archive_sha256': checksum, 'archive_bytes': archive.stat().st_size,
        'archive_members_checked': len(seen), 'packet_assets': len(p['assets_sha256']),
        'adverse_contract_changes_refused': adverse, 'actual_transfer_CLI': result,
        'local_gradient_queries': 0, 'local_optimizer_updates': 0, 'local_neural_calls': 0,
        'CUDA_preflight_pending': True, 'manual_training_pending': True,
        'model_qualified': False, 'goal_complete': False, 'seconds': time.monotonic()-stamp}
    with receipt.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(record, indent=2, allow_nan=False)+'\n')
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True); parser.add_argument('--receipt', type=Path, required=True)
    a = parser.parse_args(); audit(a.root, a.archive, a.receipt)
