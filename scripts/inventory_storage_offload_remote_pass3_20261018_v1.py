"""Read only the three Drive objects after successful streamed SHA verification."""
import argparse
from datetime import datetime, timezone
import json
import subprocess
import time
from pathlib import Path
import storage_offload_pass3_20261018_v1 as transfer
from apply_storage_offload_pass3_20261018_v1 import remote_rows_check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-receipt', required=True)
    parser.add_argument('--verify-sha256', required=True)
    args = parser.parse_args()
    planpath = transfer.OUT / 'plan_v1.json'
    expected = 'ec5b59e1b537c61f0a485ea1a911bf86af551db6bf1dd55d78a051d96d054e2e'
    transfer.require(transfer.sha(planpath) == expected, 'Exact reviewed plan required')
    plan = transfer.read(planpath)
    path = transfer.ROOT / transfer.relative(args.verify_receipt)
    transfer.require(path.is_relative_to(transfer.OUT) and transfer.sha(path) == args.verify_sha256,
                     'Exact completed verify receipt required')
    verify = transfer.read(path)
    transfer.require(verify.get('complete') is True and verify.get('streamed_downloaded_SHA256_verified') is True
                     and verify.get('mode') == 'verify' and verify.get('child_terminal') is True
                     and verify.get('plan_sha256') == expected, 'Full remote byte-stream verification first')
    connection = transfer.connection()
    command = [str(transfer.EXE), '--config', str(transfer.CONFIG), 'lsjson', transfer.DESTINATION,
               '--recursive', '--files-only', '--hash', '--retries', '1', '--low-level-retries', '3',
               '--timeout', '30s', '--contimeout', '20s', '--max-depth', '2']
    transfer.save(transfer.OUT / 'remote_inventory_intent_v1.json', {
        'command': command, 'plan_sha256': expected, 'verify_receipt': transfer.pin(path),
        'connection': connection, 'readonly_network': True, 'max_seconds': 120,
        'UTC': datetime.now(timezone.utc).isoformat()})
    started = time.monotonic()
    result = subprocess.run(command, capture_output=True, timeout=120,
                            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    transfer.require(len(result.stdout) + len(result.stderr) <= 1024**2, 'Bounded three-object response required')
    with (transfer.OUT / 'remote_inventory_stderr_v1.txt').open('xb') as stream:
        stream.write(result.stderr)
    transfer.require(result.returncode == 0, 'Read-only remote inventory failed; retain originals')
    rows = json.loads(result.stdout)
    remote_rows_check(plan, rows)
    pin = transfer.save(transfer.OUT / 'remote_objects_after_verify_v1.json', rows)
    receipt = {'complete': True, 'plan_sha256': expected, 'verify_receipt': transfer.pin(path),
        'remote_inventory': pin, 'objects': 3, 'bytes': plan['upload_bytes'],
        'destination': transfer.DESTINATION, 'UTC': datetime.now(timezone.utc).isoformat(),
        'seconds': time.monotonic() - started, 'files_removed': 0, 'VM_calls': 0}
    transfer.save(transfer.OUT / 'remote_inventory_receipt_v1.json', receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
