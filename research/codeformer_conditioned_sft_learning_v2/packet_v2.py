"""Static Conditioned SFT V2 packet verification; zero framework/model imports on load."""
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

STAGE_NAME = 'cctv_codeformer_conditioned_sft_learning_v2_vm'
OUTPUT_RELATIVE = 'outputs/conditioned_sft_learning_v2'
PACKAGE = 'research/codeformer_conditioned_sft_learning_v2'
FORMAT = 'conditioned-sft-v2-top-down-dilated-pyramid-freq-head-manual-L4-v1'
MAX_FILE = 2 * 1024**3
MAX_STATIC_FILE = 512 * 1024**2

MODEL_FIXED = {
    'r2': (376609465, 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'),
    'u96': (212827, '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'),
}
HEAD_FIXED = (685818, '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17')

RUNTIME_FILES = frozenset({
    'install.json', 'install_failure.json', 'run.log', 'export.log',
    'supervisor.start.json', 'supervisor.worker.json', 'supervisor.export.json',
    'supervisor.worker.started.json', 'supervisor.export.started.json',
    'supervisor.json', 'supervisor_failure.json', 'export.json', 'export_failure.json',
})


def need(condition, message):
    if not condition:
        raise ValueError(message)


def directory(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not (getattr(info, 'st_file_attributes', 0) & 1024),
             'Ordinary unlinked directory required: ' + str(item))
    return path.resolve(strict=True)


def regular(path):
    path = Path(path).absolute()
    directory(path.parent)
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode)
         and not (getattr(info, 'st_file_attributes', 0) & 1024)
         and info.st_nlink == 1 and 0 <= info.st_size <= MAX_FILE,
         'Bounded single-link ordinary file required: ' + str(path))
    return path.resolve(strict=True)


def sha256(path):
    path = regular(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) ==
         (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size),
         'File changed during hashing')
    return digest.hexdigest()


def pin(root, path):
    root, path = directory(root), regular(path)
    need(path.is_relative_to(root), 'Pinned file escapes exact stage')
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) ==
         (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size),
         'File changed during pinning')
    return {'path': path.relative_to(root).as_posix(), 'bytes': after.st_size, 'sha256': digest}


def relative(value):
    need(type(value) is str and bool(value) and '\\' not in value and ':' not in value
         and '\x00' not in value and all(ord(c) >= 32 for c in value),
         'Canonical relative POSIX path required')
    parsed = PurePosixPath(value)
    need(value != '.' and not parsed.is_absolute() and parsed.as_posix() == value
         and all(part not in ('', '.', '..') for part in parsed.parts),
         'Path escape or alias refused')
    return parsed


def strict_json(raw):
    need(type(raw) is bytes and len(raw) <= 32 * 1024**2, 'Bounded exact JSON bytes required')

    def pairs(items):
        out = {}
        for key, value in items:
            need(key not in out, 'Duplicate JSON key refused')
            out[key] = value
        return out

    def reject(value):
        raise ValueError('Nonfinite JSON token refused: ' + str(value))

    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs, parse_constant=reject)

    def finite(item, depth=0):
        need(depth <= 64, 'Bounded JSON nesting required')
        if type(item) is float:
            need(math.isfinite(item), 'Finite JSON numbers required')
        elif type(item) is dict:
            for child in item.values():
                finite(child, depth + 1)
        elif type(item) is list:
            for child in item:
                finite(child, depth + 1)

    finite(value)
    return value


def _binding(root, files, value, *, omitted=()):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'Exact artifact pin required')
    relative(value['path'])
    need(value['path'] in files and files[value['path']] == {'bytes': value['bytes'], 'sha256': value['sha256']},
         'Artifact absent from sealed file table: ' + str(value.get('path')))
    if value['path'] in omitted and not (root / value['path']).exists():
        return
    need(pin(root, root / value['path']) == value, 'Bound artifact differs: ' + value['path'])


def _module(root, files, name):
    relative_path = name.replace('.', '/') + '.py'
    need(relative_path in files, 'Imported helper must have a sealed file binding: ' + relative_path)
    _binding(root, files, {'path': relative_path, **files[relative_path]})
    module = importlib.import_module(name)
    need(Path(module.__file__).resolve() == root / relative_path, 'Loaded source origin differs: ' + name)
    return module


def verify_static(root, expected_sha, *, missing_weights_for_install=False):
    """Authenticate protocol.json and every static stage file before any neural import."""
    root = directory(root)
    need(type(missing_weights_for_install) is bool, 'Explicit install mode required')
    raw = regular(root / 'protocol.json').read_bytes()
    need(type(expected_sha) is str and re.fullmatch('[0-9a-f]{64}', expected_sha) is not None
         and hashlib.sha256(raw).hexdigest() == expected_sha,
         'Exact supplied protocol SHA256 required')
    need(b'\r' not in raw and raw.endswith(b'\n'), 'Sealed protocol requires LF and final newline')
    protocol = strict_json(raw)
    need(type(protocol) is dict and protocol.get('format') == FORMAT
         and protocol.get('stage_name') == STAGE_NAME
         and protocol.get('output') == OUTPUT_RELATIVE,
         'Exact Conditioned SFT V2 packet format required')
    need(protocol.get('manual_launch_only') is True
         and protocol.get('automatic_retry_resume') is False
         and protocol.get('model_qualified') is False
         and protocol.get('training_success_not_implied') is True,
         'Fresh manual finite stage required')

    files = protocol.get('files')
    need(type(files) is dict and 0 < len(files) <= 2200 and 'protocol.json' not in files,
         'Bounded nonrecursive static file table required')
    model = protocol.get('model')
    need(type(model) is dict and set(model) == set(MODEL_FIXED), 'Exact retained R2/U96 model pins required')
    omitted = {value['path'] for value in model.values()} if missing_weights_for_install else set()

    allowed_dirs = {'outputs'}
    for name, expected in files.items():
        parsed = relative(name)
        need('__pycache__' not in parsed.parts and not name.endswith(('.pyc', '.pyo')),
             'Bytecode and unreviewed caches are forbidden')
        need(type(expected) is dict and set(expected) == {'bytes', 'sha256'}
             and type(expected['bytes']) is int and 0 <= expected['bytes'] <= MAX_STATIC_FILE
             and type(expected['sha256']) is str and re.fullmatch('[0-9a-f]{64}', expected['sha256']) is not None,
             'Exact bounded static content pin required: ' + name)
        allowed_dirs.update(p.as_posix() for p in parsed.parents if p.as_posix() != '.')
        _binding(root, files, {'path': name, **expected}, omitted=omitted)
        if name.startswith((PACKAGE + '/', 'scripts/')) and name.endswith(('.py', '.sh')):
            need(b'\r' not in (root / name).read_bytes(), 'Executable source requires LF: ' + name)

    for item in root.rglob('*'):
        name = item.relative_to(root).as_posix()
        if item.is_dir():
            directory(item)
            need(name in allowed_dirs or name == OUTPUT_RELATIVE or name.startswith(OUTPUT_RELATIVE + '/'),
                 'Unexpected directory: ' + name)
        else:
            regular(item)
            need(name in files or name == 'protocol.json' or name in RUNTIME_FILES
                 or name.startswith(OUTPUT_RELATIVE + '/'),
                 'Unexpected file: ' + name)

    for key, fixed in MODEL_FIXED.items():
        value = model[key]
        expected_rel = 'weights/' + ('codeformer_r2.pth' if key == 'r2' else 'codeformer_u96_adapter.pth')
        need(value.get('path') == expected_rel and (value.get('bytes'), value.get('sha256')) == fixed,
             'Exact retained initializer bytes required: ' + key)
        _binding(root, files, value, omitted=omitted)

    head = protocol['u64_head']
    need(head.get('path') == 'weights/terminal_u0064_heads.npz'
         and (head.get('bytes'), head.get('sha256')) == HEAD_FIXED,
         'Exact trained-u64 heads required')
    _binding(root, files, head)

    contract = _module(root, files, __package__ + '.contract_v2')
    _module(root, files, __package__ + '.conditioning_v2')
    _module(root, files, __package__ + '.objective_v2')
    _module(root, files, __package__ + '.architecture_v2')

    for field, fixed in contract.EVIDENCE_PINS.items():
        binding = protocol['evidence'][field]
        need((binding.get('path'), binding.get('bytes'), binding.get('sha256')) == fixed,
             'Exact retained evidence required: ' + field)
        _binding(root, files, binding)

    need(protocol['comparison'] == contract.prospective_contract_v2()
         and protocol['limits'] == contract.LIMITS,
         'Exact Conditioned SFT V2 prospective contract required')

    cohorts = contract.select_v2_cohorts(
        (root / protocol['evidence']['v2_manifest']['path']).read_bytes(),
        (root / protocol['evidence']['v1_protocol']['path']).read_bytes(),
    )
    need(protocol['selection'] == cohorts, 'Protocol selection must match rederived V2 cohorts')

    def check_bindings(node):
        if type(node) is dict:
            if set(node) == {'path', 'bytes', 'sha256'}:
                _binding(root, files, node)
            else:
                for child in node.values():
                    check_bindings(child)
        elif type(node) is list:
            for child in node:
                check_bindings(child)

    check_bindings(cohorts)
    check_bindings(protocol['controls'])
    check_bindings(protocol['u64_parity'])
    return protocol


verify = verify_static


def require_vm(root):
    root = directory(root)
    need(Path(__file__).resolve() == root / PACKAGE / 'packet_v2.py', 'Use exact staged packet_v2 source')
    need(platform.system() == 'Linux' and sys.version_info >= (3, 10), 'Manual Linux Python3.10+ required')
    need(root == directory(Path.home() / 'forensic-dgp' / STAGE_NAME) and directory(Path.cwd()) == root,
         'Exact manual conditioned-sft-v2 stage/cwd required')
    need(directory(sys.prefix) == directory(Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv'),
         'Retained VM virtual environment required')
    need(bool(os.environ.get('TMUX', '').strip()), 'Manual tmux session required')
    free = shutil.disk_usage(root).free
    need(free >= 24 * 1024**3, 'At least 24 GiB free disk required before worker launch')
    need(not (root / OUTPUT_RELATIVE).exists(), 'Complete or partial prior run immutable; no retry/resume')
    return {
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'stage_root': str(root),
        'venv': str(Path(sys.prefix).resolve()),
        'free_disk_bytes': free,
    }
