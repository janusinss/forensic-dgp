"""Bounded static packet/data verification; importing this module is inert."""
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys

STAGE_NAME = 'cctv_codeformer_u96_spatial_feasibility_v1_vm'
OUTPUT_RELATIVE = 'outputs/u96_spatial_feasibility_v1'
FORMAT = 'complete-U96-spatial-feasibility-manual-L4-v1'
MAX_FILE = 2 * 1024**3
PACKAGE = 'research/codeformer_u96_spatial_feasibility_v1'
PARENT_PROTOCOL_FIXED = (197079, 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e')
MODEL_FIXED = {
    'r2': (376609465, 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'),
    'u96': (212827, '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'),
}
PROTOTYPE_FIXED = {
    'model': (34041, '5035a46ab37c7852e0f3bf6f6f3213158d63be98df25e67be16ad7d3d91edd5e'),
    'objective': (5368, '03caf0f22764b2c87f32ed678058d5c512f7cc3326da270a64e4c723276f8078'),
    'direction': (15832, '06eb83ebda4c64d799f988de02e23d00e6e5b2181cc93b6f4ecb8d62cf7479e2'),
    'contract': (9978, '1f33e474b9013b7a9fbee00c81369178d66641ea62949e457c318fcd666f9f1d'),
}
LIMITS = {
    'full_forward_attempts': 218, 'full_forward_returns': 218,
    'common_initializer_component_forwards': 16, 'gradient_queries': 32,
    'initial_monitor_guard_forwards': 56, 'independent_parity_forwards': 2,
    'proposal_forwards': 144, 'proposal_applications': 2,
    'optimizer_constructions': 0, 'optimizer_steps': 0,
    'automatic_retry_resume': 0, 'worker_seconds': 1200, 'external_seconds': 1260,
    'termination_grace_seconds': 20, 'per_phase_seconds': 180,
    'export_seconds': 300, 'export_external_seconds': 330,
    'initial_free_disk_bytes': 8 * 1024**3, 'continuing_free_disk_bytes': 4 * 1024**3,
    'maximum_RSS_bytes': 10 * 1024**3, 'maximum_CUDA_reserved_bytes': 18 * 1024**3,
    'maximum_output_bytes': 2 * 1024**3, 'maximum_archive_bytes': 2 * 1024**3,
    'complete_model_state_witnesses': 3,
}
RUNTIME_FILES = frozenset({
    'install.json', 'install_failure.json', 'run.log', 'export.log',
    'supervisor.start.json', 'supervisor.worker.json', 'supervisor.export.json',
    'supervisor.worker.started.json', 'supervisor.export.started.json',
    'supervisor.json', 'supervisor_failure.json', 'export.json', 'export_failure.json',
})


def need(ok, message):
    if not ok:
        raise ValueError(message)


def directory(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Ordinary unlinked directory required: ' + str(item))
    return path.resolve(strict=True)


def regular(path):
    path = Path(path).absolute()
    directory(path.parent)
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
         and not getattr(info, 'st_file_attributes', 0) & 1024
         and info.st_nlink == 1 and 0 <= info.st_size <= MAX_FILE,
         'Bounded single-link ordinary file required: ' + str(path))
    return path.resolve(strict=True)


def sha256(path):
    digest = hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def pin(root, path):
    root = directory(root)
    path = regular(path)
    need(path.is_relative_to(root), 'Pinned file escapes stage')
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    need((before.st_ino, before.st_mtime_ns, before.st_size)
         == (after.st_ino, after.st_mtime_ns, after.st_size), 'File changed during hashing')
    return {'path': path.relative_to(root).as_posix(), 'bytes': after.st_size, 'sha256': digest}


def relative(value):
    need(type(value) is str and bool(value) and '\\' not in value and ':' not in value,
         'Canonical relative POSIX path required')
    path = PurePosixPath(value)
    need(value != '.' and not path.is_absolute() and path.as_posix() == value
         and all(part not in ('', '.', '..') for part in path.parts), 'Path escape refused')
    return path


def strict_json(raw):
    need(type(raw) is bytes, 'JSON bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON constant')
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=invalid)
    def finite(item):
        if type(item) is float:
            need(math.isfinite(item), 'Finite JSON numbers required')
        elif type(item) is dict:
            for child in item.values():
                finite(child)
        elif type(item) is list:
            for child in item:
                finite(child)
    finite(value)
    return value


def _binding(root, files, value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'Exact artifact pin required')
    relative(value['path'])
    need(value['path'] in files and files[value['path']] == {
        'bytes': value['bytes'], 'sha256': value['sha256']}, 'Artifact missing from sealed file table')
    need(pin(root, root / value['path']) == value, 'Bound artifact differs: ' + value['path'])


def verify_static(root, expected_sha, *, missing_weights_for_install=False):
    """Verify bytes and retained admission only; construct no numerical/model object."""
    root = directory(root)
    need(type(missing_weights_for_install) is bool, 'Explicit install mode required')
    raw = regular(root / 'protocol.json').read_bytes()
    need(type(expected_sha) is str and re.fullmatch('[0-9a-f]{64}', expected_sha)
         and hashlib.sha256(raw).hexdigest() == expected_sha, 'Exact supplied protocol SHA required')
    need(b'\r' not in raw and raw.endswith(b'\n'), 'New sealed protocol must use LF and final newline')
    p = strict_json(raw)
    need(type(p) is dict and p['format'] == FORMAT and p['stage_name'] == STAGE_NAME,
         'Known distinct manual feasibility packet required')
    need(p['limits'] == LIMITS and all(type(x) is int for x in p['limits'].values()),
         'Exact finite counter/time/resource limits required')
    need(p['manual_launch_only'] is True and p['automatic_retry_resume'] is False,
         'Manual fresh-stage execution only')
    files = p['files']
    need(type(files) is dict and 0 < len(files) <= 1000 and 'protocol.json' not in files,
         'Bounded nonrecursive file table required')
    need(set(p['model']) == set(MODEL_FIXED) and set(p['prototype']) == set(PROTOTYPE_FIXED),
         'Exact complete initializer and four reviewed prototypes required')
    omitted = {b['path'] for b in p['model'].values()} if missing_weights_for_install else set()
    allowed_dirs = {'outputs'}
    for name, value in files.items():
        relative(name)
        need(type(value) is dict and set(value) == {'bytes', 'sha256'}
             and type(value['bytes']) is int and 0 <= value['bytes'] <= 512 * 1024**2
             and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
             'Exact bounded file content binding required')
        allowed_dirs.update(x.as_posix() for x in PurePosixPath(name).parents if x.as_posix() != '.')
        if name in omitted and not (root / name).exists():
            continue
        need(pin(root, root / name) == {'path': name, **value}, 'Static file differs: ' + name)
        if name.startswith((PACKAGE + '/', 'scripts/')) and name.endswith(('.py', '.sh')):
            need(b'\r' not in (root / name).read_bytes(), 'New executable source requires LF: ' + name)
    for item in root.rglob('*'):
        name = item.relative_to(root).as_posix()
        if item.is_dir():
            directory(item)
            need(name in allowed_dirs or name == OUTPUT_RELATIVE or name.startswith(OUTPUT_RELATIVE + '/'),
                 'Unexpected directory; no unreviewed caches: ' + name)
        else:
            regular(item)
            need(name in files or name == 'protocol.json' or name in RUNTIME_FILES
                 or name.startswith(OUTPUT_RELATIVE + '/'), 'Unexpected file namespace: ' + name)
    for key, fixed in MODEL_FIXED.items():
        value = p['model'][key]
        need(set(value) == {'path', 'bytes', 'sha256'} and value['path'] in files
             and files[value['path']] == {'bytes': value['bytes'], 'sha256': value['sha256']}
             and (value['bytes'], value['sha256']) == fixed and value['path'].startswith('weights/'),
             'Exact retained R2 plus U96 initializer required')
    for key, fixed in PROTOTYPE_FIXED.items():
        value = p['prototype'][key]
        need(value['path'] == PACKAGE + '/' + key + '.py'
             and (value['bytes'], value['sha256']) == fixed, 'Exact reviewed core prototype required')
        _binding(root, files, value)
    parent_pin = p['parent_protocol']
    need(parent_pin['path'] == 'evidence/spatial_parent_protocol.json'
         and (parent_pin['bytes'], parent_pin['sha256']) == PARENT_PROTOCOL_FIXED,
         'Exact immutable readout parent required')
    _binding(root, files, parent_pin)
    parent_bytes = (root / parent_pin['path']).read_bytes()
    parent = strict_json(parent_bytes)
    for name in ('model', 'data', 'cases', 'guards'):
        need(p[name] == parent[name], 'Retained parent ' + name + ' must remain exact')
    need(p['quality_policy'] == parent['quality_policy'], 'Inherited quality requirements cannot change')
    # Import only the two already hash-checked inert helpers. No Torch/model import.
    contract = importlib.import_module(__package__ + '.contract')
    objective = importlib.import_module(__package__ + '.objective')
    for module, key in ((contract, 'contract'), (objective, 'objective')):
        active = regular(module.__file__)
        need((active.stat().st_size, sha256(active)) == PROTOTYPE_FIXED[key],
             'Active inert helper bytes differ from reviewed prototype')
    selected = contract.select_parent_cases(parent_bytes)
    need(p['feasibility'] == contract.prospective_contract() and p['limits'] == contract.LIMITS,
         'Exact finite source-reviewed feasibility contract required')
    need(len(contract.operation_plan(selected)) == 218, 'Exact218-operation schedule required')
    need(len(p['cases']) == 96 and len(p['guards']) == 24, 'Retain all96 paired cases and24 guards')
    def bindings(item):
        if type(item) is dict:
            if set(item) == {'path', 'bytes', 'sha256'}:
                _binding(root, files, item)
            else:
                for child in item.values():
                    bindings(child)
        elif type(item) is list:
            for child in item:
                bindings(child)
    bindings(p['data']); bindings(p['cases']); bindings(p['guards'])
    for name in ('readout_reuse', 'inherited_DEV_evidence'):
        if name in p:
            _binding(root, files, p[name])
    need(p.get('model_qualified', False) is False and p.get('training_success_not_implied', True) is True,
         'Transfer verification cannot imply model qualification')
    return p


verify = verify_static


def require_vm(root):
    root = directory(root)
    need(Path(__file__).resolve() == root / PACKAGE / 'packet.py', 'Use the exact staged packet helper')
    need(platform.system() == 'Linux' and sys.version_info >= (3, 10), 'Manual Linux Python3.10+ required')
    need(root == directory(Path.home() / 'forensic-dgp' / STAGE_NAME)
         and directory(Path.cwd()) == root, 'Exact home/stage/cwd required')
    need(directory(sys.prefix) == directory(Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv'),
         'Retained VM virtual environment required')
    need(bool(os.environ.get('TMUX', '').strip()), 'Launch manually inside tmux')
    free = shutil.disk_usage(root).free
    need(free >= LIMITS['initial_free_disk_bytes'], 'Initial8GiB free required')
    need(not (root / OUTPUT_RELATIVE / 'started.json').exists(), 'Started diagnostic immutable; no retry/resume')
    return {'system': platform.system(), 'python': platform.python_version(), 'root': str(root),
            'tmux': True, 'initial_free_disk_bytes': free, 'manual_execution_only': True}
