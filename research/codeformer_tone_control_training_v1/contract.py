"""Standard-library-only contract for the finite manual tone-controller pilot."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat

FORMAT = 'codeformer-tone-control-training-finite-manual-v1'
VM_NAME = 'cctv_codeformer_tone_control_training_v1_vm'
PARENT_NAME = 'cctv_codeformer_adain_gradient_isolation_v1_vm'
MODULE = 'research.codeformer_tone_control_training_v1.worker'
OUTPUTS = 'outputs/tone_control_training_v1'
PREFIX = 'cctv-codeformer-tone-control-training-v1'
LOCK_NAME = 'tone_control_training_v1.lock'
DESIGN_SHA256 = '7c78e745739980d36e66a8ac7eb64dd59281ff1af21e973dab60cc67cad8dbfe'
PARENT_PROTOCOL_SHA = '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
PARENT_PROTOCOL_CANONICAL_SHA = '3b0644acae19f82d429483c4a64fddc6f5e61a329add05a80d2c3f727624e30e'
DEPENDENCIES_SHA256 = '0cf7922ff74d9a01139d943b9f27b3b15a1d6b0e0d3e5bbc6d668c77fcaff8b9'
RETAINED_OUTPUTS_SHA256 = '915bb9084ce5752f8a4cd4df0eb4d6a6b44c0f12b09b4703d25efa2764e2ec74'
HISTORICAL_EXPORT_MANIFEST_SHA256 = '0091134a5653825fb3a06d63e3907d1e57521f607e67da5d39ea856cdceff805'
HISTORICAL_IMPORT_RECEIPT_SHA256 = '4be3058a4ee9782cc314510884c303749b86ae5b8540a2200fbfa429dc6ad70a'
HEX = re.compile('[0-9a-f]{64}')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            if budget is not None:
                budget()
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()


def equal(a, b):
    return json.dumps(a, sort_keys=True, separators=(',', ':'), allow_nan=False) == \
           json.dumps(b, sort_keys=True, separators=(',', ':'), allow_nan=False)


def no_links(path):
    path = Path(path).absolute()
    for part in [path] + list(path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and
                    not (getattr(info, 'st_file_attributes', 0) & 1024),
                    'Linked/reparse path forbidden: ' + str(part))
    return path


def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value and
            '\x00' not in value, 'Portable relative path required')
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and relative.as_posix() == value and
            all(part not in ('', '.', '..') and not part.endswith((' ', '.')) and
                not re.match(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', part)
                for part in relative.parts), 'Unsafe relative path')
    return relative


def safe_path(base, relative, must_exist=True):
    base = no_links(base).resolve(strict=True)
    path = no_links(base.joinpath(*safe_relative(relative).parts))
    require(path.resolve(strict=must_exist).is_relative_to(base), 'Path escapes declared root')
    if must_exist:
        require(stat.S_ISREG(path.lstat().st_mode), 'Exact regular file required')
    return path


def write_new_json(path, value):
    path = no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def manual_root(root):
    require(platform.system() == 'Linux', 'Manual NVIDIA L4 VM only; local neural calls forbidden')
    root = no_links(root).resolve(strict=True)
    expected = no_links(Path.home() / 'forensic-dgp' / VM_NAME).resolve(strict=False)
    require(root == expected, 'Unexpected manual VM root')
    require(bool(os.environ.get('TMUX')), 'Manual tmux session required before framework import')
    return root


def resolve_parent(root):
    root = no_links(root).resolve(strict=True)
    parent = no_links(root.parent / PARENT_NAME).resolve(strict=True)
    require(parent.is_dir() and parent.parent == root.parent and parent != root,
            'Exact read-only AdaIN sibling required')
    return parent


_design_file = Path(__file__).with_name('design.json')
require(sha256(_design_file) == DESIGN_SHA256, 'Prospective design bytes changed')
DESIGN_BYTES = _design_file.stat().st_size
DESIGN = read_json(_design_file)
LIMITS = DESIGN['resource_caps']
TRAINING = DESIGN['training']
EVALUATION = DESIGN['evaluation']
CENSUS = DESIGN['call_census_maximum']
OPERATION_KEYS = ('CodeFormer_TRAIN_original', 'CodeFormer_TRAIN_full_coarse',
    'CodeFormer_original_DEV_bank', 'CodeFormer_zero_head_DEV_bank', 'CodeFormer_head50_DEV_bank',
    'CodeFormer_parity', 'controller_TRAIN', 'controller_DEV_zero_and50',
    'controller_math_prerequisites', 'controller_final_known_stress', 'head_autograd_queries',
    'head_backward_calls', 'head_optimizer_step_attempts', 'TRAIN_input_writes')
COUNTS = {name: CENSUS[name] for name in OPERATION_KEYS}
ASSETS = {'initializer': {'path': DESIGN['parents']['reference']['checkpoint'],
    'sha256': DESIGN['parents']['reference']['checkpoint_sha256'],
    'bytes': DESIGN['parents']['reference']['checkpoint_bytes']},
    'base': {'path': DESIGN['parents']['restorer']['checkpoint'],
    'sha256': DESIGN['parents']['restorer']['checkpoint_sha256'],
    'bytes': DESIGN['parents']['restorer']['checkpoint_bytes']}}
PARENT_CONTROLS = {
    'validation_saved': {'manifest': 'validation_saved_v1.json',
        'sha256': 'ff76007be1f55e70e12e5d5c3aad00c53e4564326e6d10eeb1e12a38b9178d84',
        'source_protocol_sha256': 'e463222441c0930a01a85ee0711a1ea31b0d80ed3487a4a5459c71e65f517d3e',
        'case_count': 318},
    'training_parity': {'path': 'training_parity_v1.json',
        'sha256': '743c073b92dd1bcfe37210a854277fb31b5e6dbe562d1229fa05903b41c5c6c6',
        'cases': 200, 'updates': 50, 'behavior_anchors': 50,
        'source_protocol_sha256': '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'}}
CHECKPOINT_PATHS = [OUTPUTS + '/' + slot + '/checkpoint.pt'
                    for slot in DESIGN['state_and_return']['checkpoint_slots']]
REQUIRED_SOURCE_FILES = {'research/__init__.py',
    'scripts/supervise_codeformer_tone_control_training_v1_vm.py',
    *('research/codeformer_tone_control_training_v1/' + name for name in
      ('__init__.py', 'contract.py', 'data.py', 'budget.py', 'model.py', 'worker.py', 'transport.py', 'design.json'))}


def verify_binding_map(mapping, label):
    require(isinstance(mapping, dict) and mapping, 'Nonempty binding map required: ' + label)
    lowered = set()
    for name, row in mapping.items():
        safe_relative(name)
        require(name.casefold() not in lowered, 'Case-colliding binding: ' + label)
        lowered.add(name.casefold())
        require(isinstance(row, dict) and set(row) == {'sha256', 'bytes'} and
                isinstance(row['sha256'], str) and HEX.fullmatch(row['sha256']) and
                type(row['bytes']) is int and row['bytes'] >= 0, 'Invalid binding: ' + label)
    return mapping


def parent_parity_paths():
    return {state: {case: {'raw': 'outputs/adain_gradient_isolation_v1/snapshot_' + update + '/' +
                          case + '/raw.npy', 'png': 'outputs/adain_gradient_isolation_v1/snapshot_' +
                          update + '/' + case + '/native.png'}
        for case in DESIGN['prerequisites']['frozen_parity_cases']}
        for state, update in [('original', '0000'), ('base', '0050')]}


def verify_protocol(p):
    require(isinstance(p, dict) and p.get('format') == FORMAT, 'Exact tone-controller format required')
    require(p.get('parent_dir') == PARENT_NAME and p.get('parent_protocol_sha256') == PARENT_PROTOCOL_SHA and
            p.get('outputs') == OUTPUTS and p.get('export_prefix') == PREFIX and p.get('module') == MODULE,
            'Exact new/parent namespace required')
    require(equal(p.get('design'), DESIGN) and p.get('design_sha256') == DESIGN_SHA256,
            'Frozen prospective design differs')
    require(equal(p.get('call_caps'), COUNTS) and equal(p.get('limits'), LIMITS) and
            sum(COUNTS.values()) == CENSUS['operation_reservations_including_writes'] == 3090,
            'Source-derived operation/resource caps differ')
    require(equal(p.get('assets'), ASSETS), 'Exact original and historical model assets required')
    for key, value in PARENT_CONTROLS.items():
        require(equal(p.get(key), value), 'Historical data control differs: ' + key)
    require(p.get('data_manifest') == 'references.json' and p.get('comparison_cases') == 'comparison_cases.json',
            'Canonical data manifests required')
    old = p.get('historical_parent_protocol')
    require(isinstance(old, dict) and canonical_sha256(old) == PARENT_PROTOCOL_CANONICAL_SHA,
            'Exact historical parent protocol required as provenance, not a new optimizer recipe')
    deps = verify_binding_map(p.get('dependencies'), 'read-only parent')
    require(len(deps) == 2011 and canonical_sha256(deps) == DEPENDENCIES_SHA256,
            'Exact original2011 parent dependency manifest required')
    require(deps.get('protocol.json', {}).get('sha256') == PARENT_PROTOCOL_SHA and
            all(deps.get(name, {}).get('sha256') == digest for name, digest in old['files_sha256'].items()),
            'Entire immutable historical parent namespace required')
    retained = verify_binding_map(p.get('retained_output_dependencies'), 'retained parent outputs')
    require(len(retained) == 1624 and canonical_sha256(retained) == RETAINED_OUTPUTS_SHA256 and
            not set(retained) & set(deps), 'Exact additional1624 historical RAW/PNG files required')
    require(equal(p.get('parent_parity_outputs'), parent_parity_paths()) and
            all(name in retained for cases in parent_parity_paths().values()
                for item in cases.values() for name in item.values()), 'Exact eight parity output paths required')
    for row in ASSETS.values():
        require(deps.get(row['path']) == {key: row[key] for key in ('sha256', 'bytes')},
                'Named model asset differs from dependency')
    for row in PARENT_CONTROLS.values():
        name = row.get('path', row.get('manifest'))
        require(deps.get(name, {}).get('sha256') == row['sha256'], 'Unbound historical data control')
    files, sizes = p.get('files_sha256'), p.get('files_bytes')
    require(isinstance(files, dict) and isinstance(sizes, dict) and set(files) == set(sizes) and
            REQUIRED_SOURCE_FILES <= set(files) and 'protocol.json' not in files and len(files) <= 48,
            'Exact executing-source manifest required')
    verify_binding_map({name: {'sha256': digest, 'bytes': sizes[name]} for name, digest in files.items()},
                       'new source')
    require(files['research/codeformer_tone_control_training_v1/design.json'] == DESIGN_SHA256 and
            sizes['research/codeformer_tone_control_training_v1/design.json'] == DESIGN_BYTES and
            sum(sizes.values()) <= 16 * 1024 ** 2 and
            sizes['research/__init__.py'] == 0 and
            files['research/__init__.py'] == hashlib.sha256(b'').hexdigest(),
            'Bound design and intentional empty research namespace required')
    document = p.get('design_document', {})
    require(document.get('path') == 'instructions/CCTV_CODEFORMER_TONE_CONTROL_TRAINING_V1_DESIGN.md' and
            files.get(document.get('path')) == document.get('sha256'), 'Bound prospective design document required')
    require(p.get('status') == 'PREPARED_UNRUN' and p.get('manual_tmux_required') is True and
            p.get('new_training_recipe') is True and p.get('actual_candidate_checkpoints_required') is True,
            'Manual unrun genuine checkpoint pilot required')
    for name in ('automatic_training', 'automatic_resume', 'model_qualified', 'automatic_app_promotion',
                 'quality_gates_changed', 'FINAL_used', 'old_optimizer_loaded', 'cached_gradient_reuse'):
        require(p.get(name) is False, 'Forbidden launch, qualification or reuse flag: ' + name)
    return p
