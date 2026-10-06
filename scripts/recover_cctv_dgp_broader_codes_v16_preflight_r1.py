"""Recover only the pinned V16 string/Path bootstrap failure; no recipe changes.

Default is verification only. --launch is an explicit manual VM action and opens
the unchanged finite supervisor in detached tmux. Never extract, edit or resume
an existing training run. Evidence is retained outside the frozen bundle and
embedded in supervisor_launch.json when launching, so it returns with results.
"""
import argparse
from datetime import datetime, timezone
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
import uuid

PROTOCOL_SHA = '4331c28c97a659846eab76ee0dee9150811a00da3c6139d8d24b7905173a6681'
ARCHIVE_SHA = 'a40f68ccdeb7e7cf2e1420bef8c984a7faad6120461073da4362bcf2fb8ab65e'
ORIGINAL_LAUNCHER_SHA = '09b1a40613295842e45c30fe70880e5b8415ee66013e636ed8c11cf53b17197f'
PATH_TYPE_ERROR = "TypeError: unsupported operand type(s) for /: 'str' and 'str'"
SESSION = 'dgp_broader_codes_v16'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def idle():
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          capture_output=True, text=True, check=True, timeout=20)
    sessions = subprocess.run(['tmux', 'list-sessions'], capture_output=True, text=True, timeout=20)
    if apps.stdout.strip() or sessions.returncode != 1 or sessions.stdout.strip():
        raise RuntimeError('Competing GPU/tmux task; preserve it and do not launch')


def verify_extracted(root, archive):
    """Compare every installed byte with the original archive; reject run state."""
    if not root.is_dir() or root.is_symlink():
        raise RuntimeError('Require the original extracted V16 directory; no extraction or replacement')
    if sha(archive) != ARCHIVE_SHA:
        raise ValueError('Original execution archive differs')
    if Path(str(archive) + '.sha256').read_text().split() != [ARCHIVE_SHA, archive.name]:
        raise ValueError('Original archive sidecar differs')
    expected = set()
    with tarfile.open(archive, 'r:gz') as stream:
        members = stream.getmembers()
        if sum(member.size for member in members) > 256 * 1024**2:
            raise ValueError('Execution archive exceeds256MiB')
        for member in members:
            name = member.name
            path = PurePosixPath(name)
            target = root / name
            if (not member.isfile() or not name or path.is_absolute() or '..' in path.parts
                    or '\\' in name or ':' in name or str(path) != name or name in expected
                    or not target.resolve().is_relative_to(root.resolve()) or target.is_symlink()):
                raise ValueError('Unsafe or duplicate execution member')
            expected.add(name)
            data = stream.extractfile(member).read()
            if not target.is_file() or sha(target) != hashlib.sha256(data).hexdigest():
                raise ValueError('Preserve changed/missing installed asset: ' + name)
    # The original failed imports can leave Python bytecode, but no run evidence.
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ValueError('Preserve unexpected symlink: ' + str(path.relative_to(root)))
        if not path.is_file():
            continue
        name = path.relative_to(root).as_posix()
        bytecode = '__pycache__' in path.relative_to(root).parts and path.suffix == '.pyc'
        if name not in expected and not bytecode:
            raise RuntimeError('Preserve existing training/unknown evidence: ' + name)
    if (root / 'outputs').exists():
        raise RuntimeError('Preserve existing outputs, even an empty directory; no recovery/resume')
    if sha(root / 'broader_codes_protocol_v16.json') != PROTOCOL_SHA:
        raise ValueError('Original protocol differs')
    plan = json.loads((root / 'broader_codes_protocol_v16.json').read_text())
    if plan['assets_sha256']['scripts/launch_cctv_dgp_broader_codes_v16.py'] != ORIGINAL_LAUNCHER_SHA:
        raise ValueError('Original bootstrap binding differs')
    return plan


def preflight_code(root, parent, mixed, baseline, *, corrected):
    paths = [root, parent, mixed, baseline]
    arguments = ','.join(('Path(' + repr(str(path)) + ')' if corrected else repr(str(path)))
                         for path in paths)
    return ('import sys;from pathlib import Path;'
            'sys.path.insert(0,' + repr(str(parent)) + ');'
            'sys.path.insert(0,' + repr(str(root)) + ');'
            'from cctv_dgp_targets_v6 import require_vm;'
            'from cctv_dgp_broader_codes_v16 import verify;'
            'require_vm(' + repr(str(root)) + ');'
            'verify(' + arguments + ',' + repr(PROTOCOL_SHA) + ');'
            'print("V16 source/data/CUDA availability preflight passed")')


def probe(python, code, evidence, name, *, show_output=True):
    # Read source instead of bytecode left by the original failed imports.
    command = [str(python), '-B', '-X', 'pycache_prefix=' + str(evidence / 'unused-bytecode'), '-u', '-c', code]
    started = time.monotonic()
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        receipt = {'command': command, 'returncode': result.returncode, 'stdout': result.stdout,
                   'stderr': result.stderr, 'seconds': time.monotonic() - started,
                   'timeout_seconds': 120, 'training_started': False}
    except subprocess.TimeoutExpired as error:
        def decode(value):
            return value.decode('utf-8', errors='replace') if isinstance(value, bytes) else (value or '')
        receipt = {'command': command, 'returncode': None, 'stdout': decode(error.stdout),
                   'stderr': decode(error.stderr), 'seconds': time.monotonic() - started,
                   'timeout_seconds': 120, 'timed_out': True, 'training_started': False}
        write(evidence / (name + '.json'), receipt)
        print(receipt['stdout'] + receipt['stderr'], flush=True)
        raise TimeoutError('Preflight exceeded120s; evidence preserved at ' + str(evidence)) from error
    write(evidence / (name + '.json'), receipt)
    if show_output:
        print(receipt['stdout'], end='', flush=True)
        print(receipt['stderr'], end='', file=sys.stderr, flush=True)
    return receipt


def recover(expected_sha, *, launch=False):
    if sys.platform != 'linux' or os.uname().nodename.split('.')[0] != 'forensic-dgp-thesis':
        raise RuntimeError('Recovery permitted only on the existing forensic-dgp-thesis Linux VM')
    actual_sha = sha(Path(__file__))
    if actual_sha != expected_sha:
        raise ValueError('Recovery file transfer hash differs')
    repo = (Path.home() / 'forensic-dgp').resolve()
    root = repo / 'cctv_dgp_broader_codes_vm_v16'
    parent = repo / 'cctv_dgp_face_code_fit_vm_v12_r2'
    mixed = repo / 'cctv_dgp_mixed_vm_v9_r2'
    baseline = repo / 'cctv_dgp_generalization_vm_v15/outputs/generalization_v15'
    python = repo / 'cctv_dgp_vm_bundle/.venv/bin/python'
    if not python.is_file():
        raise RuntimeError('Existing VM runtime missing; no reinstall')
    if shutil.disk_usage(repo).free < 12 * 1024**3:
        raise RuntimeError('Need12GiB free disk')
    idle()
    verify_extracted(root, Path.home() / 'cctv-dgp-broader-codes-v16-execution.tar.gz')
    if sha(Path.home() / 'launch_cctv_dgp_broader_codes_v16.py') != ORIGINAL_LAUNCHER_SHA:
        raise ValueError('Preserve changed original uploaded bootstrap')
    evidence_parent = repo / 'cctv_dgp_broader_codes_v16_preflight_recovery'
    evidence_parent.mkdir(exist_ok=True)
    if evidence_parent.is_symlink() or not evidence_parent.resolve().is_relative_to(repo):
        raise ValueError('Recovery evidence path escapes repository')
    attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    evidence = evidence_parent / attempt
    evidence.mkdir()
    print('Preflight evidence: ' + str(evidence), flush=True)
    original = probe(python, preflight_code(root, parent, mixed, baseline, corrected=False),
                     evidence, 'original_failure', show_output=False)
    if original['returncode'] == 0 or PATH_TYPE_ERROR not in original['stderr']:
        print(original['stdout'] + original['stderr'], flush=True)
        raise RuntimeError('Failure is not the known V16 string/Path bug; stop and provide the printed error')
    print('Confirmed original string/Path preflight error; full traceback retained. Checking corrected call.', flush=True)
    corrected = probe(python, preflight_code(root, parent, mixed, baseline, corrected=True),
                      evidence, 'corrected_preflight')
    if corrected['returncode'] != 0:
        raise RuntimeError('Corrected preflight failed; stop and provide the printed error')
    provenance = {'format': 'v16-bootstrap-path-recovery-r1', 'launcher_sha256': actual_sha,
                  'launcher_source': Path(__file__).read_text(encoding='utf-8'),
                  'original_failure': original, 'corrected_preflight': corrected,
                  'evidence_directory': str(evidence), 'training_recipe_changed': False,
                  'frozen_bundle_changed': False, 'training_resume': False}
    write(evidence / 'verified.json', {'complete': True, 'protocol_sha256': PROTOCOL_SHA,
                                    'recovery': provenance, 'training_started': False})
    if not launch:
        print('VERIFIED ONLY; no supervisor/training launch. Use --launch for the manual one-time launch.', flush=True)
        return
    idle()
    # Recheck all installed bytes and refuse any run state immediately before launch.
    verify_extracted(root, Path.home() / 'cctv-dgp-broader-codes-v16-execution.tar.gz')
    receipt = {'complete': True, 'protocol_sha256': PROTOCOL_SHA, 'archive_sha256': ARCHIVE_SHA,
               'launcher_sha256': actual_sha, 'frozen_bootstrap_sha256': ORIGINAL_LAUNCHER_SHA,
               'session': SESSION, 'root': str(root), 'started_unix': time.time(),
               'automatic_resume': False, 'production_promoted': False, 'preflight_recovery': provenance}
    environment = {'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONPYCACHEPREFIX': str(evidence / 'unused-bytecode')}
    receipt['source_import_environment'] = environment
    write(root / 'supervisor_launch.json', receipt)
    arguments = [str(python), '-u', str(root / 'scripts/supervise_cctv_dgp_broader_codes_v16.py'),
                 '--root', str(root), '--parent', str(parent), '--mixed', str(mixed),
                 '--baseline', str(baseline), '--expected-sha', PROTOCOL_SHA]
    command = 'set -C; exec ' + shlex.join(['env'] + [key + '=' + value for key, value in environment.items()] + arguments)
    command += ' > ' + shlex.quote(str(root / 'supervisor.log')) + ' 2>&1'
    try:
        subprocess.run(['tmux', 'new-session', '-d', '-s', SESSION, '-c', str(root), command],
                       check=True, timeout=20)
    except BaseException as error:
        write(evidence / 'tmux_launch_failure.json', {'complete': False, 'error': str(error),
                                                    'resume_permitted': False})
        raise
    write(evidence / 'launch.json', {'complete': True, 'session': SESSION,
                                   'supervisor_launch_sha256': sha(root / 'supervisor_launch.json')})
    print(json.dumps({'tmux_launched': True, 'session': SESSION, 'root': str(root),
                      'protocol_sha256': PROTOCOL_SHA, 'recipe_changed': False}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected-sha', required=True, help='SHA256 of this separate recovery file')
    parser.add_argument('--launch', action='store_true', help='Explicit manual launch after verified preflight')
    args = parser.parse_args()
    recover(args.expected_sha, launch=args.launch)
