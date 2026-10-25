"""Exact static/data/VM guards; verification never imports numerical libraries."""
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import stat
import sys

STAGE_NAME = 'cctv_codeformer_u96_postfusion_smoothanchors_v1_vm'
OUTPUT_RELATIVE = 'outputs/u96_postfusion_smoothanchors_v1'
FORMAT = 'complete-U96-postfusion-image-smoothanchors-manual-L4-v1'
MAX_FILE = 2 * 1024**3
LIMITS = {
    'full_forward_attempts': 712, 'full_forward_returns': 712,
    'reference_forwards': 120, 'candidate_snapshot_forwards': 336,
    'candidate_training_forwards': 256, 'backward_attempts': 256,
    'backward_returns': 256, 'optimizer_constructions': 1,
    'controlled_hook_exception_probes': 1,
    'optimizer_update_attempts': 64, 'optimizer_updates': 64,
    'internal_seconds': 2700, 'external_seconds': 2760, 'per_phase_seconds': 180,
    'initial_free_disk_bytes': 8 * 1024**3, 'continuing_disk_reserve': 4 * 1024**3,
    'maximum_RSS_bytes': 10 * 1024**3, 'maximum_CUDA_reserved_bytes': 18 * 1024**3,
    'maximum_output_bytes': 1536 * 1024**2,
}
MODEL_FIXED = {
    'r2': (376609465, 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'),
    'u96': (212827, '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'),
}
PROTOTYPE_FIXED = {
    'source': (23240, 'ca2b466161932068c01980bb41fcc08e12ee8f0120f17c8fdf47898f35934aa8'),
    'design': (15781, 'cf4ce39beefa2bef41149de6399f4f8b86cac3bdddf5ab6818651631ef0567be'),
    'review': (4966, '5da647e31ea3c6f6f635ece331943291aed94e043adfe502438541d9b3471ab4'),
}
LEGACY_PROTOCOL_FIXED = (463820, 'c355679d2475a65e19338a6c4fa7e8682b0c154442070e06b92a073037ea50ed')
SLOT_ROLES = ['local_gaussian_1p2', 'global_gaussian_1p2', 'clear', 'lowlight_linear_0p35']
LEARNING_FIXED = {
    'format': 'u96-postfusion-image-smoothanchors-B64-v1', 'max_updates': 64,
    'anchor_criterion': {'name': 'SmoothL1', 'beta': .01, 'reduction': 'mean',
                         'scope': ['clear_RGB', 'clear_adjacent_residual', 'dark_block16_luminance']},
    'microbatches_per_update': 4, 'slot_roles': SLOT_ROLES,
    'sampler_offsets': [0, 4, 8, 12], 'snapshot_updates': [0, 1, 4, 16, 32, 64],
    'warmup_updates': 4, 'gradient_clip_max_norm': 1.0,
    'optimizer': {'type': 'Adam', 'lr': 5e-5, 'betas': [.9, .99], 'eps': 1e-8,
                  'weight_decay': 0., 'amsgrad': False, 'foreach': False, 'fused': False,
                  'capturable': False, 'differentiable': False, 'maximize': False},
    'loss': {'image': 1., 'adjacent': .25, 'local_inside': .5, 'local_outside': .5,
             'clear_anchor': 2., 'dark_illumination_anchor': 2.},
    'local_half_open_roi': [64, 64, 192, 192], 'luminance_weights': [.2126, .7152, .0722],
    'illumination_block_size': 16,
    'initial_heads_canonical_sha256': '2e7826c07d31298eecf07d7f4d53286aa86247c00157dea9005242f23902726c',
}


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
         and info.st_nlink == 1 and info.st_size <= MAX_FILE,
         'Bounded single-link ordinary file required: ' + str(path))
    return path.resolve(strict=True)


def sha256(path):
    result = hashlib.sha256()
    with regular(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def pin(root, path):
    root = directory(root)
    path = regular(path)
    need(path.is_relative_to(root), 'Pinned file must stay inside stage')
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    need((before.st_ino, before.st_mtime_ns, before.st_size)
         == (after.st_ino, after.st_mtime_ns, after.st_size), 'File changed during hashing')
    return {'path': path.relative_to(root).as_posix(), 'bytes': after.st_size, 'sha256': digest}


def relative(value):
    need(type(value) is str and value and '\\' not in value and ':' not in value,
         'Canonical relative POSIX path required')
    path = PurePosixPath(value)
    need(value != '.' and not path.is_absolute() and path.as_posix() == value
         and all(part not in ('', '.', '..') for part in path.parts), 'Path escape refused')
    return path


def strict_json(data):
    def pairs(items):
        answer = {}
        for key, value in items:
            need(key not in answer, 'Duplicate JSON key')
            answer[key] = value
        return answer
    value = json.loads(data.decode('utf-8'), object_pairs_hook=pairs,
                       parse_constant=lambda _: need(False, 'Nonfinite JSON constant'))
    def finite(item):
        if type(item) is float:
            need(math.isfinite(item), 'Finite JSON value required')
        elif type(item) is dict:
            for child in item.values():
                finite(child)
        elif type(item) is list:
            for child in item:
                finite(child)
    finite(value)
    return value


def verify(root, expected_sha, *, missing_weights_for_install=False):
    root = directory(root)
    need(type(expected_sha) is str and len(expected_sha) == 64
         and sha256(root / 'protocol.json') == expected_sha, 'Exact supplied protocol SHA required')
    p = strict_json((root / 'protocol.json').read_bytes())
    need(p['format'] == FORMAT and p['stage_name'] == STAGE_NAME, 'Known manual pilot required')
    need(p['limits'] == LIMITS and all(type(v) is int for v in p['limits'].values()),
         'Exact finite counter/time/resource caps required')
    need(p['manual_launch_only'] is True and p['automatic_retry_resume'] is False,
         'Manual fresh-stage execution only')
    learning = dict(p['learning'])
    optimize = learning.pop('optimization_reference_order')
    monitor = learning.pop('monitor_reference_order')
    need(learning == LEARNING_FIXED, 'Reviewed learning contract differs')
    need(len(optimize) == len(set(optimize)) == 16 and len(monitor) == len(set(monitor)) == 8
         and not set(optimize) & set(monitor), '16/8 disjoint photograph roles required')
    files = p['files']
    need('protocol.json' not in files, 'No recursive protocol self-pin')
    omitted = {item['path'] for item in p['model'].values()} if missing_weights_for_install else set()
    allowed_dirs = {'outputs'}
    for name, binding in files.items():
        relative(name)
        need(set(binding) == {'bytes', 'sha256'} and type(binding['bytes']) is int
             and 0 <= binding['bytes'] <= 512 * 1024**2
             and type(binding['sha256']) is str and len(binding['sha256']) == 64
             and all(char in '0123456789abcdef' for char in binding['sha256']), 'Static byte binding required')
        allowed_dirs.update(parent.as_posix() for parent in PurePosixPath(name).parents if parent.as_posix() != '.')
        if name in omitted and not (root / name).exists():
            continue
        actual = pin(root, root / name)
        need(actual == {'path': name, **binding}, 'Static file differs: ' + name)
    for item in root.rglob('*'):
        name = item.relative_to(root).as_posix()
        if item.is_dir():
            directory(item)
            need(name in allowed_dirs or name == OUTPUT_RELATIVE
                 or name.startswith(OUTPUT_RELATIVE + '/'), 'Unexpected directory: ' + name)
        else:
            regular(item)
            need(name in files or name == 'protocol.json' or name.startswith(OUTPUT_RELATIVE + '/')
                 or name in ('run.log', 'supervisor.json', 'export.json', 'install.json'),
                 'Unexpected file namespace: ' + name)
    for section, fixed in (('model', MODEL_FIXED), ('prototype', PROTOTYPE_FIXED)):
        need(set(p[section]) == set(fixed), 'Exact ' + section + ' metadata required')
        for key, binding in p[section].items():
            need(set(binding) == {'path', 'bytes', 'sha256'} and binding['path'] in files
                 and files[binding['path']] == {k: binding[k] for k in ('bytes', 'sha256')}
                 and (binding['bytes'], binding['sha256']) == fixed[key], 'Exact reviewed artifact required')
    cases = p['cases']
    need(len(cases) == len({row['id'] for row in cases}) == 96, 'Exactly96 unique cases required')
    for cohort, refs in (('optimize', optimize), ('monitor', monitor)):
        selected = [row for row in cases if row['cohort'] == cohort]
        need(len(selected) == len(refs) * 4, 'Exact role table size required')
        need({(row['reference_id'], row['role']) for row in selected}
             == {(ref, role) for ref in refs for role in SLOT_ROLES}, 'Fixed photo/role table differs')
    for row in cases:
        need(row['split'] == 'TRAIN' and type(row['optimization_permitted']) is bool
             and row['optimization_permitted'] == (row['cohort'] == 'optimize'), 'Explicit TRAIN optimization admission required')
        for key in ('input', 'target'):
            binding = row[key]
            need(binding == pin(root, root / binding['path']) and binding['path'] in files, 'Exact input/target join required')
        if row['role'] == 'clear':
            need(row['input'] == row['target'], 'Clear input must be exact clean target')
    need(len(p['guards']) == len({row['id'] for row in p['guards']}) == 24, 'Retain exactly24 unique historical guards')
    need({role: sum(row['role'] == role for row in p['guards']) for role in ('clear', 'dark', 'filter')}
         == {'clear': 8, 'dark': 8, 'filter': 8}, 'Retained guard role counts differ')
    need(not {row['reference_id'] for row in p['guards']} & set(optimize + monitor), 'Guard photographs cannot enter new cohort')
    for row in p['guards']:
        need(row['split'] == 'TRAIN', 'Guard membership changed')
        if row['role'] == 'dark':
            need(row['target'] is None and row['behavior_teacher']['kind'] == 'r2', 'Legacy dark target/teacher changed')
        if row['role'] == 'clear':
            need(row['behavior_teacher']['kind'] == 'original', 'Legacy clear teacher changed')
    for binding in p['data'].values():
        need(binding == pin(root, root / binding['path']) and binding['path'] in files, 'Exact evidence join required')
    legacy_binding = p['data']['legacy_protocol']
    need((legacy_binding['bytes'], legacy_binding['sha256']) == LEGACY_PROTOCOL_FIXED,
         'Exact historical preservation protocol required; no rebased guard limits')
    legacy = strict_json((root / legacy_binding['path']).read_bytes())
    need(p['guards'] == legacy['guards'], 'Preserve the original24 guard rows, paths, teachers and initial measurements exactly')
    def guard_bindings(value):
        if type(value) is dict:
            if set(value) == {'path', 'bytes', 'sha256'}:
                need(value['path'] in files and value == pin(root, root / value['path']),
                     'Historical guard asset changed')
            else:
                for child in value.values():
                    guard_bindings(child)
        elif type(value) is list:
            for child in value:
                guard_bindings(child)
    guard_bindings(p['guards'])
    admission = strict_json((root / p['data']['admission']['path']).read_bytes())
    need(admission['complete'] is True and admission['optimization_permitted'] is True,
         'Retained original input-only optimization admission required')
    need(set(admission['admitted_optimization_ids']) == {row['id'] for row in cases if row['cohort'] == 'optimize'}
         and set(admission['admitted_monitor_ids']) == {row['id'] for row in cases if row['cohort'] == 'monitor'},
         'Explicit admitted case sets differ')
    reuse = strict_json((root / p['data']['conditional_reuse']['path']).read_bytes())
    need(reuse['complete'] is True and reuse['optimization_permitted'] is True,
         'Explicit conditional reuse admission required for distinct anchor-criterion trial')
    need(reuse['optimization_reference_order'] == optimize and reuse['monitor_reference_order'] == monitor,
         'Conditional reuse must preserve original16/8 photograph ordering')
    need(set(reuse['admitted_optimization_ids']) == set(admission['admitted_optimization_ids'])
         and set(reuse['admitted_monitor_ids']) == set(admission['admitted_monitor_ids']),
         'No silent data replacement in objective comparison')
    return p


def require_vm(root, p):
    root = directory(root)
    need(platform.system() == 'Linux' and sys.version_info >= (3, 10), 'Manual Linux Python3.10+ required')
    need(root == directory(Path.home() / 'forensic-dgp' / STAGE_NAME)
         and directory(Path.cwd()) == root, 'Exact home/stage and working directory required')
    need(directory(sys.prefix) == directory(Path.home() / 'forensic-dgp/cctv_dgp_vm_bundle/.venv'),
         'Retained VM virtual environment required')
    need(bool(os.environ.get('TMUX')), 'Launch manually inside tmux')
    free = shutil.disk_usage(root).free
    need(free >= LIMITS['initial_free_disk_bytes'], 'Initial8GiB free required')
    need(not (root / OUTPUT_RELATIVE / 'started.json').exists(), 'Started runs immutable; no automatic retry/resume')
    return {'system': platform.system(), 'python': platform.python_version(),
            'root': str(root), 'tmux': True, 'initial_free_disk_bytes': free,
            'actual_training_permitted_by_this_manual_launch': True}
