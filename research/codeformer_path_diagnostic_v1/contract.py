"""Pure CPU transfer contract for one finite zero-update measurement.

No neural imports, network, VM access, model construction or optimizer.
The exact stopped parent checkpoint stays an external read-only dependency.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat


PARENT_PROTOCOL_SHA256 = '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'
PARENT_DIR = 'cctv_codeformer_adaptation_v2_vm'
PACKAGE_DIR = 'cctv_codeformer_path_diagnostic_v1_vm'
POSITIONS = ((1, 1), (1, 2), (1, 3), (1, 4), (24, 1), (24, 2), (24, 3), (24, 4))
REFERENCE_IDS = ('tr_ffhq_16572', 'tr_ffhq_16572', 'tr_ffhq_20546', 'tr_ffhq_20546',
                 'tr_ffhq_15401', 'tr_ffhq_15401', 'tr_ffhq_61144', 'tr_ffhq_61144')
ASSETS = {
    'initializer': {'path': 'weights/codeformer.pth', 'sha256':
        '1009e537e0c2a07d4cabce6355f53cb66767cd4b4297ec7a4a64ca4b8a5684b7', 'bytes': 376637898},
    'stopped_checkpoint': {'path': 'outputs/adaptation_v2/stopped_checkpoint.pt', 'sha256':
        '3863236cd5b2ef01432e7498836d3c37096903d84a293f76b4a65bcd1fa7037e', 'bytes': 855229880},
    'teacher': {'path': 'weights/vqgan_code1024.pth', 'sha256':
        '4d1c6741b3cffcbdc2cd1a12b2c3c2442282e042d5de66909cb643d4fa31b20f', 'bytes': 255078528},
    'perceptual_features': {'path': 'perceptual/vgg16_features.pth', 'sha256':
        'a5b412426cfb06d2196179871dff81a905f97688cd46f2f16df1b63d5b251279', 'bytes': 58867573},
    'perceptual_calibration': {'path': 'perceptual/lpips_vgg_v01.pth', 'sha256':
        'a78928a0af1e5f0fcb1f3b9e8f8c3a2a5a3de244d830ad5c1feddc79b8432868', 'bytes': 7289},
    'perceptual_manifest': {'path': 'perceptual/manifest.json', 'sha256':
        '9bf1b2812f5f85d34ee826d66f9118a10a5b0a4c7680a3c5400aae16045b4f39', 'bytes': 1995},
}
STOPPED_CHECKPOINT = ASSETS['stopped_checkpoint']
REQUIRED_SOURCES = (
    'research/codeformer_path_diagnostic_v1/__init__.py',
    'research/codeformer_path_diagnostic_v1/contract.py',
    'research/codeformer_path_diagnostic_v1/neural.py',
    'research/codeformer_path_diagnostic_v1/worker.py',
    'scripts/prepare_codeformer_path_diagnostic_v1.py',
    'scripts/supervise_codeformer_path_diagnostic_v1.py',
)
# Independent of frozen parent controls: these are the prospective diagnostic's
# limits. It constructs no optimizer and cannot continue a training trajectory.
SPEC = {
    'format': 'codeformer-path-diagnostic-finite-manual-v1', 'seed': 20261015,
    'parent_dir': PARENT_DIR, 'parent_protocol_sha256': PARENT_PROTOCOL_SHA256,
    'assets': ASSETS,
    'outputs': 'outputs/path_diagnostic_v1',
    'export_prefix': 'cctv-codeformer-path-diagnostic-v1',
    'call_caps': {'codeformer_forwards': 104, 'hq_teacher_forwards': 4,
             'lpips_forwards': 10, 'gradient_queries': 64, 'optimizer_updates': 0},
    'limits': {'worker_seconds': 900, 'export_seconds': 300,
        'initial_free_bytes': 6 * 1024**3, 'reserve_free_bytes': 4 * 1024**3,
        'result_bytes': 256 * 1024**2, 'archive_bytes': 272 * 1024**2,
        'max_reserved_GPU_bytes': 22 * 1024**3},
}
HEX = re.compile(r'^[0-9a-f]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)', re.I)


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def write_new_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, allow_nan=False, ensure_ascii=False) + '\n').encode('utf-8')
    with path.open('xb') as target:
        target.write(payload)


def canonical(relative):
    require(isinstance(relative, str) and relative, 'Missing relative artifact path')
    value = PurePosixPath(relative)
    require(not value.is_absolute() and value.as_posix() == relative and '..' not in value.parts
        and ':' not in relative and '\\' not in relative, 'Noncanonical or escaped artifact path')
    require(all(part not in ('', '.') and not part.endswith(('.', ' ')) and
                not DEVICE.match(part) and all(ord(char) >= 32 for char in part) for part in value.parts),
            'Nonportable artifact name')
    return value


def bounded(root, relative, *, must_exist=True):
    relative = canonical(relative)
    require(Path(root).is_dir(), 'Missing artifact root directory')
    root = Path(root).resolve(strict=True)
    result = root.joinpath(*relative.parts)
    require(result.resolve(strict=must_exist).is_relative_to(root), 'Artifact path escape')
    for item in (result, *result.parents):
        if item == root.parent:
            break
        if not item.exists():
            continue
        info = item.lstat()
        require(not item.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400,
                'Symlink/reparse artifact forbidden')
    if must_exist:
        require(stat.S_ISREG(result.lstat().st_mode), 'Missing regular artifact')
    return result


def parent_root(root, protocol):
    root = Path(root).resolve(strict=True)
    require(protocol.get('parent_dir') == PARENT_DIR, 'Unexpected parent directory')
    parent = root.parent / PARENT_DIR
    require(parent.is_dir() and not parent.is_symlink(), 'Missing regular sibling parent directory')
    require(parent.resolve(strict=True).parent == root.parent and parent.resolve() != root,
            'Parent root must remain a distinct direct sibling')
    info = parent.lstat()
    require(not getattr(info, 'st_file_attributes', 0) & 0x400, 'Parent reparse point forbidden')
    return parent


def case_id(update, slot):
    return f'train_u{update:04d}_slot{slot}'


def validate_protocol(protocol):
    require(isinstance(protocol, dict), 'Protocol must be an object')
    for name, expected in SPEC.items():
        require(protocol.get(name) == expected, 'Diagnostic control differs: ' + name)
    files, parent_files = protocol.get('files_sha256'), protocol.get('parent_files_sha256')
    for label, mapping in [('packet', files), ('parent', parent_files)]:
        require(isinstance(mapping, dict) and bool(mapping), 'Missing exact ' + label + ' file bindings')
        folded = set()
        for relative, digest in mapping.items():
            canonical(relative)
            require(relative.casefold() not in folded and isinstance(digest, str) and HEX.fullmatch(digest),
                    'Malformed/colliding ' + label + ' binding')
            folded.add(relative.casefold())
        require('protocol.json' not in mapping, 'Protocol cannot bind itself')
    require(all(parent_files.get(asset['path']) == asset['sha256'] for asset in ASSETS.values()),
            'External model/teacher/checkpoint assets must be exact parent bindings')
    require(set(REQUIRED_SOURCES) <= set(files), 'Incomplete diagnostic executable source bindings')
    cases = protocol.get('cases')
    require(isinstance(cases, list) and [(r.get('update'), r.get('slot')) for r in cases] == list(POSITIONS),
            'Only eight fixed chronological TRAIN cases are permitted')
    require([r.get('reference_id') for r in cases] == list(REFERENCE_IDS),
            'The already audited fixed TRAIN photograph choices differ')
    ids, sources = set(), set()
    for row in cases:
        update, slot = row['update'], row['slot']
        require(type(update) is int and type(slot) is int and
                row['id'] == case_id(update, slot) and row['id'] not in ids and
                re.fullmatch(r'tr_ffhq_[0-9]{5}', row['reference_id']) and
                type(row['image_eligible']) is bool and row['image_eligible'] == (update == 24 or slot >= 3),
                'TRAIN case identity/eligibility differs')
        ids.add(row['id']); sources.add(row['reference_id'])
        for path_key, hash_key in [('input_path', 'input_sha256'), ('clean512_path', 'clean512_sha256'),
                                   ('pair_record_path', 'pair_record_sha256')]:
            require(files.get(row[path_key]) == row[hash_key], 'Unbound case payload: ' + path_key)
        behavior = row.get('behavior')
        if slot == 3:
            require(isinstance(behavior, dict), 'Normal-B case needs its exact saved behavior anchor')
            for path_key, hash_key in [('delivered_path', 'delivered_sha256'), ('logits_path', 'logits_sha256')]:
                require(files.get(behavior[path_key]) == behavior[hash_key], 'Unbound behavior anchor')
        else:
            require(behavior is None, 'Only normal-B may contain behavior anchors')
    require(len(sources) == 4, 'Fixed diagnostic requires exactly four TRAIN photographs')
    for left, right in ((0, 1), (2, 3), (4, 5), (6, 7)):
        a, b = cases[left], cases[right]
        require(a['reference_id'] == b['reference_id'] and a['clean512_path'] == b['clean512_path'] and
                a['clean512_sha256'] == b['clean512_sha256'], 'Matched pair source/target differs')
    design = protocol.get('instruction')
    require(isinstance(design, dict) and files.get(design.get('path')) == design.get('sha256')
            and HEX.fullmatch(design.get('sha256', '')), 'Reviewed diagnostic instruction must be bound')
    exports = protocol.get('export_files')
    require(isinstance(exports, list) and len(exports) == len(set(exports)) and
            all(name in files and not name.startswith(('data/', 'weights/')) for name in exports),
            'Export only explicit minimal source/notices/provenance companions')
    require(protocol.get('optimizer_updates') == 0 and protocol.get('no_automatic_training') is True,
            'Diagnostic must forbid optimizer updates and automatic training')
    return protocol


def verify_transfer(root, expected_sha, verify_parent=True):
    require(isinstance(expected_sha, str) and HEX.fullmatch(expected_sha), 'Expected exact protocol SHA256')
    root = Path(root).resolve(strict=True)
    protocol_path = bounded(root, 'protocol.json')
    require(sha256(protocol_path) == expected_sha, 'Protocol SHA256 differs')
    protocol = validate_protocol(read_json(protocol_path))
    expected = set(protocol['files_sha256']) | {'protocol.json'}
    actual, folded = set(), set()
    for folder, dirs, files in __import__('os').walk(root, followlinks=False):
        # Run outputs/logs are allowed after launch, never accepted as assets.
        dirs[:] = [d for d in dirs if (Path(folder) / d).relative_to(root).as_posix() != 'outputs']
        for directory in dirs:
            path = Path(folder) / directory
            require(not path.is_symlink() and not getattr(path.lstat(), 'st_file_attributes', 0) & 0x400,
                    'Packet directory link forbidden')
        for filename in files:
            relative = (Path(folder) / filename).relative_to(root).as_posix()
            bounded(root, relative)
            require(relative.casefold() not in folded, 'Case-colliding packet namespace')
            folded.add(relative.casefold()); actual.add(relative)
    require(actual == expected, 'Exact diagnostic packet namespace differs')
    for name, expected in protocol['files_sha256'].items():
        require(sha256(bounded(root, name)) == expected, 'Packet file fingerprint differs: ' + name)
    # CPU-only exact PNG->float32 and clean-array checks establish the actual
    # tensors consumed by this measurement, without regenerating a camera path.
    for row in protocol['cases']:
        pair = read_json(bounded(root, row['pair_record_path']))
        require(pair.get('role') == 'train' and pair.get('reference_id') == row['reference_id'] and
                pair.get('update') == row['update'] and pair.get('slot') == row['slot'] and
                pair['parameters']['image_eligible'] == row['image_eligible'] and
                pair.get('saved_input_sha256') == row['input_sha256'] and
                pair.get('member') == ('normal' if row['slot'] % 2 else 'dark') and
                pair.get('pair_role') == ('A' if row['slot'] <= 2 else 'B'),
                'Saved case is not the declared exact TRAIN slot')
        verify_case_pixels(root, row, pair)
    if verify_parent:
        parent = parent_root(root, protocol)
        require(sha256(bounded(parent, 'protocol.json')) == PARENT_PROTOCOL_SHA256,
                'Parent protocol SHA256 differs')
        for name, expected in protocol['parent_files_sha256'].items():
            path = bounded(parent, name)
            require(sha256(path) == expected, 'External parent dependency differs: ' + name)
        for asset in ASSETS.values():
            require(bounded(parent, asset['path']).stat().st_size == asset['bytes'],
                    'External parent asset size differs')
    return protocol


def verify_case_pixels(root, row, pair):
    """Read-only CPU restoration of frozen PNG/array inputs; no model call."""
    import numpy as np
    from PIL import Image
    with Image.open(bounded(root, row['input_path'])) as image:
        require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (256, 256),
                'Exact TRAIN observation must remain RGB256 PNG')
        pixels = np.asarray(image).copy()
    floats = np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))
    require(hashlib.sha256(pixels.tobytes()).hexdigest() == pair['input256_rgb8_sha256'] and
            hashlib.sha256(floats.tobytes()).hexdigest() == pair['input256_float32_sha256'],
            'Saved PNG does not restore the exact V2 TRAIN float32 tensor')
    with bounded(root, row['clean512_path']).open('rb') as source:
        clean = np.load(source, allow_pickle=False)
        require(source.read(1) == b'', 'Unexpected trailing clean-target bytes')
    require(clean.dtype == np.float32 and clean.shape == (512, 512, 3) and
            clean.flags.c_contiguous and np.isfinite(clean).all() and
            clean.min() >= 0 and clean.max() <= 1, 'Clean target dtype/geometry/range differs')
    require(hashlib.sha256(clean.tobytes()).hexdigest() == pair['clean512_float32_sha256'],
            'Clean512 tensor differs from the exact saved TRAIN target')
    if row['behavior'] is not None:
        for label, shape in [('delivered', (1, 3, 256, 256)), ('logits', (1, 256, 1024))]:
            with bounded(root, row['behavior'][label + '_path']).open('rb') as source:
                value = np.load(source, allow_pickle=False)
                require(source.read(1) == b'', 'Unexpected trailing behavior-anchor bytes')
            require(value.dtype == np.float32 and value.shape == shape and value.flags.c_contiguous
                    and np.isfinite(value).all(), 'Behavior anchor dtype/geometry/nonfinite differs')
            if label == 'delivered':
                require(value.min() >= 0 and value.max() <= 1, 'Behavior delivered range differs')
    return {'input_PNG_float32_exact': True, 'clean512_float32_exact': True,
            'camera_regeneration': False, 'neural_calls': 0}
