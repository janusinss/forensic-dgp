"""Verify the command-only import correction on the existing VM; no training."""
import base64
import hashlib
import json
from pathlib import Path
import shlex

import storage_gcloud_20261006_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v31_launch_import_v1'
BUNDLE = ROOT / 'outputs/cctv_dgp_profile_batches_vm_v31'
PIN = 'ae91d25066bcb532a8a88bdc48fe401c4b18d73e8d348b47b5d012d5df8e485e'
KEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'

REMOTE = r'''
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

assert os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis'
assert str(Path.home()) == '/home/janusdominic0'
root = Path.home() / 'forensic-dgp/cctv_dgp_profile_batches_vm_v31'
assert not (root / 'outputs').exists() and not (root / 'trainer.log').exists(), 'Do not touch an active or prior run'
for name, digest in packet_pins.items():
    path = (root / name).resolve()
    assert path.is_relative_to(root) and path.is_file() and not path.is_symlink()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
env = dict(os.environ)
env['PYTHONPATH'] = str(root) + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
python = Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/python'
cmd = [str(python), '-B', '-u', 'scripts/cctv_dgp_profile_batches_v31_vm.py',
       '--root', '.', '--protocol-sha', protocol_pin, '--verify-transfer']
start = time.monotonic()
result = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, timeout=240)
assert result.returncode == 0, result.stdout + result.stderr
receipt = json.loads(result.stdout)
assert receipt['complete'] and receipt['training_cases'] == 3905 and receipt['training_references'] == 781
assert receipt['optimizer_updates'] == receipt['neural_or_gradient_calls'] == 0
assert not result.stderr
assert not (root / 'outputs').exists() and not (root / 'trainer.log').exists()
print(json.dumps({'complete': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
                  'read_only_transfer_preflight': True, 'root_added_as_first_PYTHONPATH_entry': True,
                  'packet_files_verified': len(packet_pins), 'protocol_sha256': protocol_pin,
                  'transfer_receipt': receipt, 'seconds': time.monotonic() - start,
                  'VM_files_changed': 0, 'training_started_by_agent': False,
                  'outputs_created': False, 'trainer_log_created': False}, indent=2))
'''


def main():
    assert hashlib.sha256((BUNDLE / 'protocol.json').read_bytes()).hexdigest() == PIN
    p = json.loads((BUNDLE / 'protocol.json').read_text())
    pins = {'protocol.json': PIN, **p['assets_sha256']}
    for name, digest in pins.items():
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == digest
    OUT.mkdir(exist_ok=False)
    transport.OUT = OUT
    remote = 'packet_pins=' + repr(pins) + '\nprotocol_pin=' + repr(PIN) + '\n' + REMOTE
    bootstrap = 'import base64;exec(base64.b64decode(' + repr(base64.b64encode(remote.encode()).decode()) + '))'
    transport.run(['compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + transport.BASE +
                  ['--ssh-flag=-batch', '--ssh-flag=-hostkey', '--ssh-flag=' + KEY,
                   '--command=python3 -B -c ' + shlex.quote(bootstrap)], 'corrected_transfer_check', 270, True)


if __name__ == '__main__': main()
