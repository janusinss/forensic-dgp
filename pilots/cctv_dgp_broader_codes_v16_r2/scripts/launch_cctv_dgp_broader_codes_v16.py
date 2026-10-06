"""One-time manual V16 r2 bootstrap; preserve failed V16 and use its existing L4 assets."""
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
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def preflight_code(root, parent, mixed, baseline, pin):
    return ('import sys;from pathlib import Path;'
            'sys.path.insert(0,' + repr(str(parent)) + ');'
            'sys.path.insert(0,' + repr(str(root)) + ');'
            'from cctv_dgp_targets_v6 import require_vm;from cctv_dgp_broader_codes_v16 import verify;'
            'require_vm(' + repr(str(root)) + ');'
            'verify(' + ','.join('Path(' + repr(str(path)) + ')' for path in [root, parent, mixed, baseline]) + ',' + repr(pin) + ');'
            'print("V16 r2 full source/data/CUDA availability preflight passed")')


def launch(pin, archive_sha):
    if sys.platform != 'linux' or os.uname().nodename.split('.')[0] != 'forensic-dgp-thesis':
        raise RuntimeError('Launch only on the existing forensic-dgp-thesis Linux VM')
    repo = (Path.home() / 'forensic-dgp').resolve()
    root = repo / 'cctv_dgp_broader_codes_vm_v16_r2'
    parent = repo / 'cctv_dgp_face_code_fit_vm_v12_r2'
    mixed = repo / 'cctv_dgp_mixed_vm_v9_r2'
    baseline = repo / 'cctv_dgp_generalization_vm_v15/outputs/generalization_v15'
    if root.exists() or not root.resolve().is_relative_to(repo):
        raise RuntimeError('Preserve existing/partial V16 r2; no repeat/resume')
    if shutil.disk_usage(repo).free < 12 * 1024**3:
        raise RuntimeError('Need12GiB free disk')

    def idle():
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              capture_output=True, text=True, check=True, timeout=20)
        sessions = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True, timeout=20)
        if apps.stdout.strip() or sessions.returncode != 1 or sessions.stdout.strip():
            raise RuntimeError('Competing GPU/tmux task; preserve it and do not launch')

    idle()
    archive = Path.home() / 'cctv-dgp-broader-codes-v16-r2-execution.tar.gz'
    if sha(archive) != archive_sha or Path(str(archive) + '.sha256').read_text().split() != [archive_sha, archive.name]:
        raise ValueError('Transfer checksum differs')
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers(); names = set()
        if sum(member.size for member in members) > 256 * 1024**2:
            raise ValueError('Thin execution archive exceeds256MiB')
        for member in members:
            name = member.name; path = PurePosixPath(name)
            if (not member.isfile() or not name or path.is_absolute() or '..' in path.parts or '\\' in name or
                    ':' in name or str(path) != name or name in names or not (root / name).resolve().is_relative_to(root)):
                raise ValueError('Unsafe/duplicate execution member')
            names.add(name)
        root.mkdir(); stream.extractall(root, members=members)
    if sha(root / 'broader_codes_protocol_v16_r2.json') != pin:
        raise ValueError('Protocol differs')
    plan = json.loads((root / 'broader_codes_protocol_v16_r2.json').read_text())
    if sha(Path(__file__)) != plan['assets_sha256']['scripts/launch_cctv_dgp_broader_codes_v16.py']:
        raise ValueError('Bootstrap executable differs')
    python = repo / 'cctv_dgp_vm_bundle/.venv/bin/python'
    if not python.is_file():
        raise RuntimeError('Existing VM runtime missing; no reinstall')
    code = preflight_code(root, parent, mixed, baseline, pin)
    command = [str(python), '-B', '-X', 'pycache_prefix=' + str(root / 'unused-bytecode'), '-u', '-c', code]
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else (value or '')
        receipt = {'complete': False, 'timed_out': True, 'timeout_seconds': 120, 'command': command,
                   'stdout': decode(error.stdout), 'stderr': decode(error.stderr), 'training_started': False}
        with (root / 'bootstrap_preflight.json').open('x') as stream:
            json.dump(receipt, stream, indent=2)
        print(receipt['stdout'] + receipt['stderr'], flush=True)
        raise RuntimeError('Bootstrap preflight exceeded120s; preserve partial root') from error
    receipt = {'complete': result.returncode == 0, 'command': command, 'returncode': result.returncode,
               'stdout': result.stdout, 'stderr': result.stderr, 'seconds': time.monotonic() - started,
               'timeout_seconds': 120, 'training_started': False}
    with (root / 'bootstrap_preflight.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
    print(result.stdout, end='', flush=True)
    print(result.stderr, end='', file=sys.stderr, flush=True)
    if result.returncode:
        raise RuntimeError('Bootstrap preflight failed; preserve root and provide the printed error')
    idle()
    environment = {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPYCACHEPREFIX': str(root / 'unused-bytecode')}
    launch_receipt = {'complete': True, 'protocol_sha256': pin, 'archive_sha256': archive_sha,
                      'launcher_sha256': sha(Path(__file__)), 'session': 'dgp_broader_codes_v16_r2',
                      'root': str(root), 'started_unix': time.time(), 'automatic_resume': False,
                      'production_promoted': False, 'source_import_environment': environment,
                      'preflight_sha256': sha(root / 'bootstrap_preflight.json')}
    with (root / 'supervisor_launch.json').open('x') as stream:
        json.dump(launch_receipt, stream, indent=2)
    arguments = [str(python), '-u', str(root / 'scripts/supervise_cctv_dgp_broader_codes_v16.py'),
                 '--root', str(root), '--parent', str(parent), '--mixed', str(mixed),
                 '--baseline', str(baseline), '--expected-sha', pin]
    shell_command = 'set -C; exec ' + shlex.join(['env'] + [key + '=' + value for key, value in environment.items()] + arguments)
    shell_command += ' > ' + shlex.quote(str(root / 'supervisor.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'dgp_broader_codes_v16_r2', '-c', str(root), shell_command],
                   check=True, timeout=20)
    print(json.dumps(launch_receipt), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha', required=True)
    parser.add_argument('--archive-sha', required=True)
    args = parser.parse_args()
    launch(args.protocol_sha, args.archive_sha)
