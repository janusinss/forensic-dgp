"""Pure-CPU contract for the zero-update bounded-fusion feasibility measure."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

PACKAGE_DIR = 'cctv_codeformer_bounded_fusion_feasibility_v1_vm'
OUTPUTS = 'outputs/bounded_fusion_feasibility_v1'
PREFIX = 'cctv-codeformer-bounded-fusion-feasibility-v1'
DESIGN_SHA256 = 'aeb5fe8871e44e31a9af49df0e883b6153a0d987f5e39b1f9b927e91fec9f4d4'
DESIGN_PATH = 'instructions/CCTV_CODEFORMER_BOUNDED_FUSION_FEASIBILITY_V1_DESIGN.md'
ASSETS = {
    'original': {'parent_dir': 'cctv_codeformer_fp32_comparison_v1_vm', 'path': 'weights/codeformer.pth',
        'sha256': '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7',
        'bytes': 376637898, 'parent_protocol_sha256': '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'},
    'stopped': {'parent_dir': 'cctv_codeformer_fp32_comparison_v1_vm',
        'path': 'outputs/fp32_comparison_v1/stopped_checkpoint.pt',
        'sha256': '133c5a1e15e069bdd2abda4ebf818d3f4ff59b966677762753e894ff42e2583d',
        'bytes': 855229944, 'parent_protocol_sha256': '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'},
}
CALL_CAPS = {'codeformer_forwards': 170, 'fusion_block_evaluations': 960,
    'hq_teacher_forwards': 0, 'behavior_teacher_forwards': 0, 'lpips_forwards': 0,
    'gradient_queries': 0, 'backward_calls': 0, 'optimizer_attempts': 0, 'optimizer_updates': 0}
LIMITS = {'initialization_seconds': 300, 'measurement_seconds': 600, 'worker_seconds': 900,
    'export_seconds': 300, 'initial_free_bytes': 8 * 1024**3, 'reserve_free_bytes': 4 * 1024**3,
    'result_bytes': 5 * 1024**3 // 4, 'archive_bytes': 1449551462,
    'metadata_bytes': 32 * 1024**2, 'regular_files': 2500, 'temporary_bytes': 128 * 1024**2,
    'max_reserved_GPU_bytes': 22 * 1024**3, 'PNG_file_bytes': 198656,
    'array_file_bytes': 3 * 128 * 64 * 64 * 4 + 4096}
# A retained canary block can be(256,64,64), larger than one native512 RGB.
LIMITS['array_file_bytes'] = 256 * 64 * 64 * 4 + 4096
GATE = {'q10': 0.21007671428419458, 'q90': 0.5546973290483728,
    'beta': .10, 'epsilon': 1e-6, 'gate_floor': .25, 'gate_ceiling': 1.0, 'central_support': [32, 224]}
IMMUTABLE_SOURCES = {
    'research/codeformer_adaptation_v2/model.py': '8cb64d7a2c3fac46fe38805ac54c00287a050b0473608a87dfe650b5d9e508e3',
    'third_party/codeformer/codeformer_arch.py': 'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb'}
SELECTED_PAIRS = ((2, 3), (7, 3), (10, 3), (11, 1), (15, 1), (16, 1), (17, 3), (19, 1),
                  (27, 1), (30, 1), (32, 1), (33, 3), (37, 1), (45, 1), (47, 3), (48, 1))
SELECTED_ORDER = tuple((update, slot) for update, first in SELECTED_PAIRS for slot in (first, first + 1))
CANARY_IDS = ['train_u0011_slot1', 'train_u0032_slot2']
HEX = re.compile(r'^[a-f0-9]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9\u00b9\u00b2\u00b3]|lpt[1-9\u00b9\u00b2\u00b3])(?:\.|$)', re.I)
REQUIRED_SOURCES = ('research/codeformer_bounded_fusion_feasibility_v1/contract.py',
    'research/codeformer_bounded_fusion_feasibility_v1/worker.py',
    'research/codeformer_bounded_fusion_feasibility_v1/neural.py',
    'scripts/supervise_codeformer_bounded_fusion_feasibility_v1.py')
SPEC = {'format': 'codeformer-bounded-fusion-feasibility-finite-manual-v1',
    'outputs': OUTPUTS, 'export_prefix': PREFIX, 'assets': ASSETS, 'call_caps': CALL_CAPS,
    'limits': LIMITS, 'gate': GATE, 'seed': 20261015, 'optimizer_updates': 0, 'no_automatic_training': True}


def require(value, message):
    if not value:
        raise ValueError(message)


def exact(value, expected):
    """JSON numbers/bools must retain their declared types as well as values."""
    if type(value) is not type(expected):
        return False
    if isinstance(expected, dict):
        return value.keys() == expected.keys() and all(exact(value[k], item) for k, item in expected.items())
    if isinstance(expected, list):
        return len(value) == len(expected) and all(exact(a, b) for a, b in zip(value, expected))
    return value == expected


def canonical(name):
    require(isinstance(name, str) and name, 'Missing relative artifact path')
    p = PurePosixPath(name)
    require(p.parts and not p.is_absolute() and p.as_posix() == name and
            all(x not in ('', '.', '..') for x in p.parts), 'Noncanonical artifact path')
    require(not any(c in name for c in '\\:"<>|?*') and all(not DEVICE.match(x) and
            not x.endswith((' ', '.')) and len(x.encode('utf-16-le')) <= 510 and
            all(ord(c) >= 32 for c in x) for x in p.parts), 'Nonportable artifact path')
    return p


def no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            require(not item.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 1024,
                    'Symlink/reparse path forbidden')
    return path


def bounded(root, name, *, must_exist=True):
    path = no_links(Path(root).joinpath(*canonical(name).parts))
    require(path.is_relative_to(no_links(root)), 'Artifact escapes root')
    if must_exist:
        require(stat.S_ISREG(path.lstat().st_mode), 'Expected regular artifact: ' + name)
    return path


def sha256(path):
    path = no_links(path); before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'Hash input must be regular')
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            h.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Artifact changed during hash')
    return h.hexdigest()


def read_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field'); result[key] = value
        return result
    payload = no_links(path).read_bytes()
    require(len(payload) <= 8 * 1024**2, 'Metadata file exceeds8MiB')
    return json.loads(payload.decode('utf-8-sig'), object_pairs_hook=pairs,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write_new_json(path, value):
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path = no_links(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(payload); output.flush(); os.fsync(output.fileno())


def validate_protocol(p):
    require(all(exact(p.get(key), value) for key, value in SPEC.items()),
            'Fixed finite measurement/gate/training/seed controls differ')
    runtime = p.get('runtime_expected', {})
    require(isinstance(runtime, dict) and set(runtime) == {'torch', 'numpy', 'CUDA', 'GPU', 'capability'} and
            all(isinstance(runtime[name], str) and runtime[name] for name in ('torch', 'numpy', 'CUDA')) and
            runtime['GPU'] == 'NVIDIA L4' and runtime['capability'] == [8, 9],
            'Exact retained L4 runtime core required')
    files = p.get('files_sha256'); exports = p.get('export_files')
    require(isinstance(files, dict) and files and isinstance(exports, list) and
            len(exports) == len(set(exports)) and set(exports) <= set(files), 'Explicit source/input maps required')
    folded = set()
    for name, digest in files.items():
        canonical(name); require(HEX.fullmatch(digest) and name.casefold() not in folded and name != 'protocol.json',
                                 'Invalid/colliding bound-file namespace')
        folded.add(name.casefold())
    require(all(name in files for name in REQUIRED_SOURCES) and files.get(DESIGN_PATH) == DESIGN_SHA256,
            'Executing sources/design not pinned')
    require(all(files.get(name) == digest for name, digest in IMMUTABLE_SOURCES.items()),
            'Immutable CodeFormer model/architecture source differs')
    for name in exports:
        require(not name.startswith('data/') and Path(name).suffix.lower() not in {'.npy', '.png', '.pth', '.pt'},
                'Do not duplicate immutable input arrays/models in result export')
    cases, calibration, canaries = p.get('cases'), p.get('calibration_inputs'), p.get('canary_ids')
    require(isinstance(cases, list) and len(cases) == 32 and len({x['id'] for x in cases}) == 32 and
            isinstance(calibration, list) and len(calibration) == 200 and len({x['id'] for x in calibration}) == 200,
            'Exact32-case/200-input calibration census required')
    require([(x['update'], x['slot']) for x in calibration] == [(u, s) for u in range(1, 51) for s in range(1, 5)],
            'Calibration chronological TRAIN prefix differs')
    require(tuple((x['update'], x['slot']) for x in cases) == SELECTED_ORDER,
            'Prospective sixteen-pair photometry-overlap cohort differs')
    require(canaries == CANARY_IDS, 'Two fixed overlap-stress canaries required')
    by_id = {x['id']: x for x in calibration}
    for row in cases + calibration:
        require(type(row['slot']) is int and row['slot'] in (1, 2, 3, 4) and
                type(row['update']) is int and 1 <= row['update'] <= 50 and
                row['id'] == 'train_u%04d_slot%d' % (row['update'], row['slot']), 'TRAIN-only case identity required')
        require(row['input_path'] == 'data/inputs/' + row['id'] + '.png', 'Exact retained input path required')
        require(files.get(row['input_path']) == row['input_sha256'], 'Case input not protocol-bound')
    for row in cases:
        require(row['reference_id'].startswith('tr_ffhq_') and
                all(row[k] == by_id[row['id']][k] for k in ('update', 'slot', 'input_path', 'input_sha256')),
                'Selected cohort must retain exact calibration input bindings')
        for path_key, sha_key in [('source_path', 'source_sha256'), ('target512_path', 'target512_sha256'),
                                 ('target256_path', 'target256_sha256')]:
            require(files.get(row[path_key]) == row[sha_key], 'Case source/target not protocol-bound')
    require(len({row['reference_id'] for row in cases}) == 16 and
            all(cases[i]['reference_id'] == cases[i + 1]['reference_id'] for i in range(0, 32, 2)),
            'Sixteen matched photographic reference pairs required')
    return p


def resolve_assets(root, p):
    result = {}
    for name, asset in p['assets'].items():
        parent = no_links(Path(root).parent / asset['parent_dir'])
        require(parent.parent == no_links(root).parent and parent.is_dir(), 'Exact sibling asset root required')
        require(sha256(bounded(parent, 'protocol.json')) == asset['parent_protocol_sha256'], 'Parent protocol differs')
        file = bounded(parent, asset['path'])
        require(file.stat().st_size == asset['bytes'] and sha256(file) == asset['sha256'], 'Parent model asset differs')
        result[name] = file
    return result


def verify_transfer(root, protocol_sha, *, verify_parent=True):
    root = no_links(root)
    require(HEX.fullmatch(protocol_sha) and sha256(bounded(root, 'protocol.json')) == protocol_sha,
            'Exact sealed protocol SHA required')
    p = validate_protocol(read_json(root / 'protocol.json'))
    for name, digest in p['files_sha256'].items():
        require(sha256(bounded(root, name)) == digest, 'Static source/input differs: ' + name)
    actual = set()
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in list(dirs):
            path = Path(folder) / name; no_links(path)
            if path == root / 'outputs':
                dirs.remove(name)
        for name in names:
            file = Path(folder) / name; actual.add(file.relative_to(root).as_posix())
    require(actual == set(p['files_sha256']) | {'protocol.json'}, 'Unbound importable/static packet namespace')
    if verify_parent:
        resolve_assets(root, p)
    return p
