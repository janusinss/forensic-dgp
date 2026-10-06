"""One-time browser-SSH launch, using existing L4 paths and verified assets only."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def launch(pin, archive_sha):
    if sys.platform != 'linux' or os.uname().nodename.split('.')[0] != 'forensic-dgp-thesis':
        raise RuntimeError('Launch only on the existing forensic-dgp-thesis VM')
    repo = (Path.home() / 'forensic-dgp').resolve(); root = repo / 'cctv_dgp_broader_codes_vm_v16'
    parent = repo / 'cctv_dgp_face_code_fit_vm_v12_r2'; mixed = repo / 'cctv_dgp_mixed_vm_v9_r2'
    baseline = repo / 'cctv_dgp_generalization_vm_v15/outputs/generalization_v15'
    if root.exists() or not root.resolve().is_relative_to(repo): raise RuntimeError('Preserve existing/partial V16')
    if shutil.disk_usage(repo).free < 12 * 1024**3: raise RuntimeError('Need12GiB free disk')

    def idle():
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], capture_output=True, text=True, check=True)
        sessions = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True)
        if apps.stdout.strip() or sessions.returncode != 1 or sessions.stdout.strip():
            raise RuntimeError('Competing GPU/tmux task; do not launch')

    idle(); archive = Path.home() / 'cctv-dgp-broader-codes-v16-execution.tar.gz'
    if sha(archive) != archive_sha or Path(str(archive) + '.sha256').read_text().split() != [archive_sha, archive.name]:
        raise ValueError('Transfer checksum differs')
    # Bootstrap solely from checksum-verified plain files; no archive code is executed.
    from pathlib import PurePosixPath
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers(); names = set()
        if sum(m.size for m in members) > 256 * 1024**2: raise ValueError('Thin execution archive exceeds256MiB')
        for member in members:
            name = member.name; path = PurePosixPath(name)
            if (not member.isfile() or not name or path.is_absolute() or '..' in path.parts or '\\' in name or
                    ':' in name or str(path) != name or name in names or not (root / name).resolve().is_relative_to(root)):
                raise ValueError('Unsafe/duplicate execution member')
            names.add(name)
        root.mkdir(); stream.extractall(root, members=members)
    if sha(root / 'broader_codes_protocol_v16.json') != pin: raise ValueError('Protocol differs')
    plan = json.loads((root / 'broader_codes_protocol_v16.json').read_text())
    if sha(Path(__file__)) != plan['assets_sha256']['scripts/launch_cctv_dgp_broader_codes_v16.py']:
        raise ValueError('Launcher differs')
    python = repo / 'cctv_dgp_vm_bundle/.venv/bin/python'
    if not python.is_file(): raise RuntimeError('Existing VM runtime missing; no reinstall')
    code = 'import sys;sys.path.insert(0,' + repr(str(parent)) + ');sys.path.insert(0,' + repr(str(root)) + ');from cctv_dgp_targets_v6 import require_vm;from cctv_dgp_broader_codes_v16 import verify;require_vm(' + repr(str(root)) + ');verify(' + ','.join(repr(str(x)) for x in [root, parent, mixed, baseline, pin]) + ');print("V16 full data/source/CUDA preflight passed")'
    result = subprocess.run([str(python), '-c', code], capture_output=True, text=True, check=True, timeout=120)
    print(result.stdout, flush=True); idle()
    v = {'complete': True, 'protocol_sha256': pin, 'archive_sha256': archive_sha,
        'launcher_sha256': sha(Path(__file__)), 'session': 'dgp_broader_codes_v16', 'root': str(root),
        'started_unix': time.time(), 'automatic_resume': False, 'production_promoted': False}
    with (root / 'supervisor_launch.json').open('x') as stream: json.dump(v, stream, indent=2)
    command = 'exec ' + shlex.join([str(python), '-u', str(root / 'scripts/supervise_cctv_dgp_broader_codes_v16.py'),
        '--root', str(root), '--parent', str(parent), '--mixed', str(mixed), '--baseline', str(baseline),
        '--expected-sha', pin]) + ' > ' + shlex.quote(str(root / 'supervisor.log')) + ' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', 'dgp_broader_codes_v16', '-c', str(root), command], check=True)
    print(json.dumps(v), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-sha', required=True); parser.add_argument('--archive-sha', required=True)
    a = parser.parse_args(); launch(a.protocol_sha, a.archive_sha)
