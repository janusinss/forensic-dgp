"""User-run one-time bootstrap for the existing L4; creates detached tmux itself."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shlex
import shutil
import subprocess
import sys
import tarfile
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): digest.update(block)
    return digest.hexdigest()


def preflight_code(root, parent, r2, mixed, baseline, pin):
    return ('import sys;from pathlib import Path;sys.path.insert(0,' + repr(str(parent)) + ');'
        'sys.path.insert(0,' + repr(str(root)) + ');from cctv_dgp_targets_v6 import require_vm;'
        'from cctv_dgp_input_selection_v19_r2 import verify;require_vm(Path(' + repr(str(root)) + '));'
        'verify(' + ','.join('Path(' + repr(str(path)) + ')' for path in [root, parent, r2, mixed, baseline])
        + ',' + repr(pin) + ');import torch,torchvision;'
        'assert (Path(' + repr(str(root.parent / 'cctv_dgp_vm_bundle/cuda_runtime_before.txt'))
        + ').read_text().splitlines()==[torch.__version__,torchvision.__version__]),"Existing VM runtime changed; no reinstall";'
        'print("V19 source/data/parity-lineage/CUDA availability/runtime preflight passed")')


def live_tmux_tasks(output):
    allowed = {'bash', 'sh', 'zsh', 'fish', 'tmux', 'tail', 'less', 'watch', 'htop', 'top'}
    tasks = []
    for line in output.splitlines():
        fields = line.split('\t')
        if len(fields) != 3:
            raise ValueError('Unrecognized tmux process-state output; do not assume idle')
        session, command, dead = fields
        if dead != '1' and command not in allowed: tasks.append({'session': session, 'command': command})
    return tasks


def launch(pin, archive_sha):
    if sys.platform != 'linux' or os.uname().nodename.split('.')[0] != 'forensic-dgp-thesis':
        raise RuntimeError('Launch only on the existing forensic-dgp-thesis Linux VM')
    for value in [pin, archive_sha]:
        if len(value) != 64 or any(x not in '0123456789abcdef' for x in value): raise ValueError('Require lowercase SHA256')
    repo = (Path.home() / 'forensic-dgp').resolve()
    root = repo / 'cctv_dgp_input_selection_vm_v19_r2'
    parent = repo / 'cctv_dgp_face_code_fit_vm_v12_r2'
    r2 = repo / 'cctv_dgp_broader_codes_vm_v16_r2'
    mixed = repo / 'cctv_dgp_mixed_vm_v9_r2'
    baseline = repo / 'cctv_dgp_generalization_vm_v15/outputs/generalization_v15'
    if root.exists() or not root.resolve().is_relative_to(repo): raise RuntimeError('Preserve existing/partial V19; no repeat/resume')
    if shutil.disk_usage(repo).free < 4 * 1024**3: raise RuntimeError('Need4GiB free disk')
    python = repo / 'cctv_dgp_vm_bundle/.venv/bin/python'
    if not python.is_file(): raise RuntimeError('Existing VM runtime missing; no reinstall')

    def idle():
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True, timeout=20)
        panes = subprocess.run(['tmux', 'list-panes', '-a', '-F', '#{session_name}\t#{pane_current_command}\t#{pane_dead}'],
                               capture_output=True, text=True, timeout=20)
        if panes.returncode not in [0, 1]: raise RuntimeError('Cannot verify tmux process state')
        tasks = live_tmux_tasks(panes.stdout) if panes.returncode == 0 else []
        if apps.stdout.strip() or tasks:
            raise RuntimeError('Competing GPU/task; preserve it and do not launch: ' + repr(tasks))
        exists = subprocess.run(['tmux', 'has-session', '-t', 'dgp_input_selection_v19_r2'], capture_output=True, timeout=20)
        if exists.returncode != 1: raise RuntimeError('Preserve existing V19 session; no duplicate launch')

    idle()
    archive = Path.home() / 'cctv-dgp-input-selection-v19-r2-execution.tar.gz'
    if sha(archive) != archive_sha or Path(str(archive) + '.sha256').read_text().split() != [archive_sha, archive.name]:
        raise ValueError('Transfer hash/sidecar differs')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        names = set()
        if sum(member.size for member in members) > 128 * 1024**2: raise ValueError('Inference archive exceeds128MiB')
        for member in members:
            name, path = member.name, PurePosixPath(member.name)
            if (not member.isfile() or not name or path.is_absolute() or '..' in path.parts or ':' in name
                or '\\' in name or str(path) != name or name in names or not (root / name).resolve().is_relative_to(root)):
                raise ValueError('Unsafe/duplicate transfer member')
            names.add(name)
        root.mkdir()
        stream.extractall(root, members=members)
    if sha(root / 'input_selection_protocol_v19_r2.json') != pin: raise ValueError('Frozen protocol differs')
    plan = json.loads((root / 'input_selection_protocol_v19_r2.json').read_text())
    if sha(Path(__file__)) != plan['assets_sha256']['scripts/launch_cctv_dgp_input_selection_v19_r2.py']:
        raise ValueError('Bootstrap executable differs')
    command = [str(python), '-B', '-u', '-c', preflight_code(root, parent, r2, mixed, baseline, pin)]
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        receipt = {'complete': result.returncode == 0, 'returncode': result.returncode, 'command': command,
            'stdout': result.stdout, 'stderr': result.stderr, 'seconds': time.monotonic() - started,
            'timeout_seconds': 120, 'inference_started': False}
    except subprocess.TimeoutExpired as error:
        decode = lambda value: value.decode('utf-8', errors='replace') if isinstance(value, bytes) else (value or '')
        receipt = {'complete': False, 'timed_out': True, 'command': command, 'stdout': decode(error.stdout),
                   'stderr': decode(error.stderr), 'timeout_seconds': 120, 'inference_started': False}
    with (root / 'bootstrap_preflight.json').open('x', encoding='utf-8') as stream:
        json.dump(receipt, stream, indent=2)
    print(receipt['stdout'] + receipt['stderr'], end='', flush=True)
    if not receipt['complete']: raise RuntimeError('Preflight failed; preserve root and printed evidence')
    idle()
    environment = {'PYTHONDONTWRITEBYTECODE': '1'}
    receipt = {'complete': True, 'protocol_sha256': pin, 'archive_sha256': archive_sha,
        'launcher_sha256': sha(Path(__file__)), 'session': 'dgp_input_selection_v19_r2', 'root': str(root),
        'started_unix': time.time(), 'automatic_resume': False, 'production_promoted': False,
        'source_import_environment': environment, 'preflight_sha256': sha(root / 'bootstrap_preflight.json')}
    with (root / 'supervisor_launch.json').open('x', encoding='utf-8') as stream: json.dump(receipt, stream, indent=2)
    args = [str(python), '-B', '-u', str(root / 'scripts/supervise_cctv_dgp_input_selection_v19_r2.py'),
        '--root', str(root), '--parent', str(parent), '--r2', str(r2), '--mixed', str(mixed),
        '--baseline', str(baseline), '--expected-sha', pin]
    shell = 'set -C; exec ' + shlex.join(['env', 'PYTHONDONTWRITEBYTECODE=1'] + args)
    shell += ' > ' + shlex.quote(str(root / 'supervisor.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'dgp_input_selection_v19_r2', '-c', str(root), shell], check=True, timeout=20)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha', required=True)
    parser.add_argument('--archive-sha', required=True)
    a = parser.parse_args()
    launch(a.protocol_sha, a.archive_sha)
