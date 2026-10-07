"""Read-only maintenance snapshot; never execute a pilot or change the VM."""
import base64
from pathlib import Path
import shlex

import storage_gcloud_20261006_v1 as transport

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cctv_dgp_v31_launch_status_v1'
KEY = 'SHA256:pGOZAAcM7szdr2S1Ty+8UORXEPPAhxMDw7jDfzZ4I0E'

REMOTE = r'''
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

assert os.uname().nodename.split('.')[0] == 'forensic-dgp-thesis'
assert str(Path.home()) == '/home/janusdominic0'
root = Path.home() / 'forensic-dgp/cctv_dgp_profile_batches_vm_v31'
def command(args):
    r = subprocess.run(args, capture_output=True, text=True, timeout=10)
    return {'exit_code': r.returncode, 'stdout': r.stdout[-16000:], 'stderr': r.stderr[-2000:]}
def tail(path, lines=18):
    if not path.is_file(): return {'exists': False}
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size - 24000))
        data = stream.read().decode('utf-8', errors='replace')
    return {'exists': True, 'bytes': path.stat().st_size, 'last_lines': data.splitlines()[-lines:]}
procs = []
for path in Path('/proc').iterdir():
    if not path.name.isdigit(): continue
    try:
        args = (path / 'cmdline').read_bytes().decode(errors='replace').split('\0')
        if any(arg.endswith('cctv_dgp_profile_batches_v31_vm.py') or arg.endswith('scripts/run_v31.sh') for arg in args):
            procs.append({'pid': int(path.name), 'args': [a for a in args if a],
                          'status': (path / 'status').read_text().splitlines()[:3]})
    except (FileNotFoundError, PermissionError, ProcessLookupError): pass
protocol = root / 'protocol.json'
disk = shutil.disk_usage(Path.home() / 'forensic-dgp')
receipt = {'complete': True, 'snapshot_utc': datetime.now(timezone.utc).isoformat(),
           'read_only': True, 'VM_writes': 0, 'training_started_by_agent': False,
           'root_exists': root.is_dir(),
           'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest() if protocol.is_file() else None,
           'V31_processes': procs,
           'tmux_panes': command(['tmux', 'list-panes', '-t', 'dgp_profile_batches_v31', '-F', '#{session_name} #{pane_id} #{pane_pid} #{pane_current_command} #{pane_in_mode}']),
           'tmux_last_lines': command(['tmux', 'capture-pane', '-p', '-t', 'dgp_profile_batches_v31', '-S', '-16']),
           'trainer_log': tail(root / 'trainer.log'),
           'outputs_present': sorted(p.name for p in (root / 'outputs').iterdir())[:25] if (root / 'outputs').is_dir() else [],
           'GPU': command(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv,noheader']),
           'free_bytes': disk.free}
export = Path.home() / 'cctv-dgp-profile-batches-v31-export.json'
if export.is_file() and export.stat().st_size < 16000:
    receipt['export_receipt'] = json.loads(export.read_text())
print(json.dumps(receipt, indent=2))
'''


def main():
    OUT.mkdir(exist_ok=False)
    transport.OUT = OUT
    source = 'import base64;exec(base64.b64decode(' + repr(base64.b64encode(REMOTE.encode()).decode()) + '))'
    transport.run(['compute', 'ssh', 'janusdominic0@forensic-dgp-thesis'] + transport.BASE +
                  ['--ssh-flag=-batch', '--ssh-flag=-hostkey', '--ssh-flag=' + KEY,
                   '--command=python3 -B -c ' + shlex.quote(source)], 'launch_status', 60, True)


if __name__ == '__main__': main()
