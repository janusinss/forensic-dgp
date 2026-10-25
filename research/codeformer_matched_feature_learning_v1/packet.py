"""Static matched-packet verification; no framework/model/worker on import."""
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

STAGE_NAME = 'cctv_codeformer_matched_feature_learning_v1_vm'
OUTPUT_RELATIVE = 'outputs/matched_feature_learning_v1'
PACKAGE = 'research/codeformer_matched_feature_learning_v1'
FORMAT = 'matched-heads-versus-heads-U96-adapters-manual-L4-v1'
MAX_FILE = 2 * 1024**3
MAX_STATIC_FILE = 512 * 1024**2
MODEL_FIXED = {
    'r2': (376609465, 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'),
    'u96': (212827, '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'),
}
HEAD_FIXED = (685818, '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17')
EVIDENCE_FIXED = {
    'admission': ('evidence/matched_data_admission_v1.json', 9653,
                  '888b191bf6639760e324f2a7d77885fd6b9d13e439a6bbf64251372513fc89b0'),
    'candidate_data': ('evidence/matched_candidate_data_v1.json', 323833,
                       '23c402f82c9dea0b52f32a828446b5f961c52d249a7c9881e260459e0147e9ce'),
    'parent_protocol': ('evidence/u64_parent_protocol.json', 197079,
                        'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'),
    'historical_pixel_audit': ('evidence/u64_historical_pixel_audit.json', 2677742,
                              '8b73cbb901452faff8b3679bcf9d077d4d8c98a90e461c18af260565fd04c18d'),
    'historical_archive_index': ('evidence/u64_historical_archive_index.json', 1103555,
                                '0f376b6bff438914fb9e1a9f1c00ae0ea236ee06aaa00496e333a60cd9013356'),
    'retained_continuation': ('evidence/retained_continuation_protocol.json', 279879,
                             '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'),
}
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
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
             not getattr(info, 'st_file_attributes', 0) & 1024,
             'Ordinary unlinked directory required: ' + str(item))
    return path.resolve(strict=True)


def regular(path):
    path = Path(path).absolute()
    directory(path.parent)
    info = path.lstat()
    need(stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
         not getattr(info, 'st_file_attributes', 0) & 1024 and
         info.st_nlink == 1 and 0 <= info.st_size <= MAX_FILE,
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
         (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size), 'File changed during hashing')
    return digest.hexdigest()


def pin(root, path):
    root, path = directory(root), regular(path)
    need(path.is_relative_to(root), 'Pinned file escapes exact stage')
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_mtime_ns, before.st_size) ==
         (after.st_dev, after.st_ino, after.st_mtime_ns, after.st_size), 'File changed during pinning')
    return {'path': path.relative_to(root).as_posix(), 'bytes': after.st_size, 'sha256': digest}


def relative(value):
    need(type(value) is str and bool(value) and '\\' not in value and ':' not in value and
         '\x00' not in value and all(ord(c) >= 32 for c in value), 'Canonical relative POSIX path required')
    parsed = PurePosixPath(value)
    need(value != '.' and not parsed.is_absolute() and parsed.as_posix() == value and
         all(part not in ('', '.', '..') for part in parsed.parts), 'Path escape or alias refused')
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
        raise ValueError('Nonfinite JSON token refused: ' + value)
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
    need(value['path'] in files and files[value['path']] ==
         {'bytes': value['bytes'], 'sha256': value['sha256']}, 'Artifact absent from sealed file table')
    if value['path'] in omitted and not (root / value['path']).exists():
        return
    need(pin(root, root / value['path']) == value, 'Bound artifact differs: ' + value['path'])


def _module(root, files, name):
    relative_path = name.replace('.', '/') + '.py'
    need(relative_path in files, 'Imported helper must have a sealed file binding')
    _binding(root, files, {'path': relative_path, **files[relative_path]})
    module = importlib.import_module(name)
    need(Path(module.__file__).resolve() == root / relative_path, 'Loaded source origin differs: ' + name)
    return module


def verify_static(root, expected_sha, *, missing_weights_for_install=False):
    """Hash/authenticate the entire static scope before importing pure helpers.

    This verifies transfers and admission, never pixel decode or model quality.
    The two exact retained R2/U96 artifacts alone may be absent at installation.
    """
    root = directory(root)
    need(type(missing_weights_for_install) is bool, 'Explicit install mode required')
    raw = regular(root / 'protocol.json').read_bytes()
    need(type(expected_sha) is str and re.fullmatch('[0-9a-f]{64}', expected_sha) is not None and
         hashlib.sha256(raw).hexdigest() == expected_sha, 'Exact supplied protocol SHA256 required')
    need(b'\r' not in raw and raw.endswith(b'\n'), 'Sealed protocol requires LF/final newline')
    protocol = strict_json(raw)
    need(type(protocol) is dict and protocol.get('format') == FORMAT and
         protocol.get('stage_name') == STAGE_NAME and protocol.get('output') == OUTPUT_RELATIVE,
         'Exact new matched-comparison packet required')
    need(protocol.get('manual_launch_only') is True and protocol.get('automatic_retry_resume') is False and
         protocol.get('model_qualified') is False and protocol.get('training_success_not_implied') is True,
         'Fresh manual finite pilot; transfer is not model qualification')
    files = protocol.get('files')
    need(type(files) is dict and 0 < len(files) <= 1600 and 'protocol.json' not in files,
         'Bounded nonrecursive static file table required')
    model = protocol.get('model')
    need(type(model) is dict and set(model) == set(MODEL_FIXED), 'Exact retained R2/U96 model pins required')
    omitted = {value['path'] for value in model.values()} if missing_weights_for_install else set()
    allowed_dirs = {'outputs'}
    for name, expected in files.items():
        parsed = relative(name)
        need('__pycache__' not in parsed.parts and not name.endswith(('.pyc', '.pyo')),
             'Bytecode and unreviewed import caches are forbidden')
        need(type(expected) is dict and set(expected) == {'bytes', 'sha256'} and
             type(expected['bytes']) is int and 0 <= expected['bytes'] <= MAX_STATIC_FILE and
             type(expected['sha256']) is str and re.fullmatch('[0-9a-f]{64}', expected['sha256']) is not None,
             'Exact bounded static content pin required')
        allowed_dirs.update(p.as_posix() for p in parsed.parents if p.as_posix() != '.')
        _binding(root, files, {'path': name, **expected}, omitted=omitted)
        if name.startswith((PACKAGE + '/', 'scripts/')) and name.endswith(('.py', '.sh')):
            need(b'\r' not in (root / name).read_bytes(), 'New executable source requires LF')
    # Reject unreviewed caches and namespace objects, including __pycache__.
    for item in root.rglob('*'):
        name = item.relative_to(root).as_posix()
        if item.is_dir():
            directory(item)
            need(name in allowed_dirs or name == OUTPUT_RELATIVE or name.startswith(OUTPUT_RELATIVE + '/'),
                 'Unexpected directory: ' + name)
        else:
            regular(item)
            need(name in files or name == 'protocol.json' or name in RUNTIME_FILES or
                 name.startswith(OUTPUT_RELATIVE + '/'), 'Unexpected file: ' + name)
    for key, fixed in MODEL_FIXED.items():
        value = model[key]
        need(value.get('path') == 'weights/' + ('codeformer_r2.pth' if key == 'r2' else 'codeformer_u96_adapter.pth') and
             (value.get('bytes'), value.get('sha256')) == fixed, 'Exact retained initializer bytes required')
        _binding(root, files, value, omitted=omitted)
    head = protocol['u64_head']
    need(head.get('path') == 'weights/terminal_u0064_heads.npz' and
         (head.get('bytes'), head.get('sha256')) == HEAD_FIXED, 'Original trained-u64 heads required')
    _binding(root, files, head)
    for field, fixed in EVIDENCE_FIXED.items():
        binding = protocol[field]
        need((binding.get('path'), binding.get('bytes'), binding.get('sha256')) == fixed,
             'Exact retained evidence required: ' + field)
        _binding(root, files, binding)
    contract = _module(root, files, __package__ + '.contract')
    _module(root, files, __package__ + '.objective')
    need(protocol['comparison'] == contract.prospective_contract() and protocol['limits'] == contract.LIMITS and
         all(type(value) is int for value in protocol['limits'].values()), 'Exact finite matched contract required')
    selection = contract.select_cases((root / protocol['admission']['path']).read_bytes(),
                                     (root / protocol['candidate_data']['path']).read_bytes(),
                                     (root / protocol['parent_protocol']['path']).read_bytes())
    need(protocol['selection'] == selection, 'Packet selection differs from rederived admitted bytes')
    need(protocol['operation_plan'] == contract.operation_plan(selection), 'Exact finite bank-qualified schedule required')
    for bank in ('new_optimization', 'new_monitors'):
        for row in selection[bank]:
            _binding(root, files, row['input']); _binding(root, files, row['target'])
    def bindings(value):
        if type(value) is dict:
            if set(value) == {'path', 'bytes', 'sha256'}:
                _binding(root, files, value)
            else:
                for child in value.values():
                    bindings(child)
        elif type(value) is list:
            for child in value:
                bindings(child)
    bindings(selection['old_quality'])
    old = selection['old_quality']
    controls, parity = protocol['controls'], protocol['u64_parity']
    retained = strict_json((root / protocol['retained_continuation']['path']).read_bytes())
    need(controls == retained['controls'] and parity == retained['u64_parity'] and
         protocol['quality_policy'] == retained['quality_policy'],
         'Cached controls, witnesses and original quality policy must stay exact')
    need(type(controls) is list and [r.get('id') for r in controls] == [r['id'] for r in old],
         'Exact ordered56 frozen U96 controls required')
    need(type(parity) is list and [r.get('id') for r in parity] ==
         [r['id'] for r in selection['old_monitors'][:4]], 'Exact four historical u64 parity witnesses required')
    for bank in (controls, parity):
        for row in bank:
            need(type(row) is dict and set(row) == {'id', 'raw', 'png', 'record'}, 'Exact cached-control schema required')
            for field in ('raw', 'png', 'record'):
                _binding(root, files, row[field])
    for name in ('visual_plan', 'source_review', 'source_tests'):
        _binding(root, files, protocol[name])
    return protocol


verify = verify_static


def require_vm(root):
    root = directory(root)
    need(Path(__file__).resolve() == root / PACKAGE / 'packet.py', 'Use exact staged packet source')
    need(platform.system() == 'Linux' and sys.version_info >= (3, 10), 'Manual Linux Python3.10+ required')
    need(root == directory(Path.home() / 'forensic-dgp' / STAGE_NAME) and directory(Path.cwd()) == root,
         'Exact manual matched stage/cwd required')
    need(directory(sys.prefix) == directory(Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv'),
         'Retained VM virtual environment required')
    need(bool(os.environ.get('TMUX', '').strip()), 'Manual tmux required')
    from .contract import LIMITS
    free = shutil.disk_usage(root).free
    need(free >= LIMITS['initial_free_disk_bytes'], 'Initial32GiB free required')
    need(not (root / OUTPUT_RELATIVE).exists(), 'Complete or partial prior run immutable; no retry/resume')
    return {'system': platform.system(), 'python': platform.python_version(), 'root': str(root),
            'tmux': True, 'initial_free_disk_bytes': free, 'manual_execution_only': True}
