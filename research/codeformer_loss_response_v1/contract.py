"""Standard-library-only prospective diagnostic contract. Import executes no model."""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import stat

FORMAT = 'codeformer-loss-response-v1'
CASES_FORMAT = 'codeformer-loss-response-cases-v1'
VM_NAME = 'cctv_codeformer_loss_response_v1_vm'
PARENT_NAME = 'cctv_codeformer_adain_gradient_isolation_v1_vm'
PARENT_PROTOCOL_SHA = '4e2cc86857628617edbb313b6ef2911602a63ad4e1f9862afe07c57c0ba770c5'
SOURCE_UPDATES = [1, 6, 15, 24]
COMPONENTS = ['token_ce', 'latent_mse', 'pixel_l1_rgb', 'lpips_vgg',
              'coarse_rgb_l1', 'behavior_rgb_l1', 'behavior_token_kl']
WEIGHTS = dict(zip(COMPONENTS, [.5, 1., 1., 1., 4., 2., .1]))
BRANCHES = ['baseline', 'full'] + ['drop:' + name for name in COMPONENTS] + ['momentum_only']
COUNTS = {
    'candidate_source_forwards': 32, 'candidate_probe_forwards': 320,
    'restoration_repeat_forwards': 32, 'behavior_teacher_forwards': 4,
    'initializer_parity_candidate_forwards': 2, 'initializer_parity_benchmark_forwards': 2,
    'perceptual_source_forwards': 24, 'perceptual_prerequisite_forwards': 1,
    'HQ_teacher_forwards': 8, 'component_gradient_queries': 160,
    'full_gradient_queries': 32, 'perceptual_prerequisite_gradient_queries': 1,
    'optimizer_step_attempts': 18, 'disposable_optimizer_steps': 18,
    'committed_training_updates': 0, 'backward_calls': 0,
}
LIMITS = {
    'worker_seconds': 1800, 'initialization_seconds': 300,
    'source_seconds': 600, 'probe_seconds': 900, 'export_seconds': 600,
    'preflight_seconds': 300, 'reserve_free_bytes': 4 * 1024**3,
    'required_free_bytes': 14 * 1024**3,
    'result_payload_bytes': 4 * 1024**3, 'archive_bytes': 9 * 1024**3 // 2,
    'metadata_bytes': 128 * 1024**2, 'regular_file_cap': 12000,
    'CUDA_reserved_bytes': 22 * 1024**3, 'RSS_bytes': 8 * 1024**3,
    'array_file_bytes': 256 * 1024**2, 'PNG_file_bytes': 3 * 512 * 512 + 4096,
    'log_bytes': 2 * 1024**2,
}
OPTIMIZER = {'name': 'Adam', 'betas': [.9, .99], 'eps': 1e-8, 'weight_decay': 0.,
             'fixed_group_rates': {'encoder_token': 1e-5, 'fusion': 2e-5},
             'gradient_accumulation_denominator': 16,
             'rate_policy': 'Recorded parent update50 rates, identical at both measurement states; no scheduler or continuation',
             'branch_restore_required': True, 'chained_steps': False,
             'None_is_not_zero': True}
NUMERICAL = {'fullsum_absolute_tolerance': 2e-5,
             'fullsum_relative_tolerance': 2e-4,
             'fixed_state_repeat_maximum_error': 0.0,
             'initializer_parity_maximum_error': 1e-6,
             'quality_thresholds_changed': False}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def sha256(path, budget=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for payload in iter(lambda: stream.read(1024**2), b''):
            if budget is not None:
                budget()
            digest.update(payload)
    return digest.hexdigest()

def read_json(path):
    def unique(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))

def no_links(path):
    path = Path(path).absolute()
    for part in [path] + list(path.parents):
        if part.exists() or part.is_symlink():
            info = part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not (getattr(info, 'st_file_attributes', 0) & 1024),
                    'Linked/reparse path forbidden: ' + str(part))
    return path

def safe_relative(value):
    require(isinstance(value, str) and value and '\\' not in value and ':' not in value and '\x00' not in value,
            'Portable relative path required')
    relative = PurePosixPath(value)
    require(not relative.is_absolute() and relative.as_posix() == value and
            all(part not in ('', '.', '..') and not part.endswith((' ', '.')) and
                not re.match(r'(?i)^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', part)
                for part in relative.parts), 'Unsafe relative path')
    return relative

def safe_path(base, relative, must_exist=True):
    base = no_links(base).resolve(strict=True)
    parts = safe_relative(relative).parts
    path = no_links(base.joinpath(*parts))
    require(path.resolve(strict=must_exist).is_relative_to(base), 'Path escapes declared root')
    if must_exist:
        info = path.lstat()
        require(stat.S_ISREG(info.st_mode), 'Exact regular file required')
    return path

def write_new_json(path, value):
    path = no_links(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())

def manual_root(root):
    require(platform.system() == 'Linux', 'Manual NVIDIA L4 VM execution only; local neural calls forbidden')
    root = no_links(root).resolve(strict=True)
    expected = no_links(Path.home() / 'forensic-dgp' / VM_NAME).resolve(strict=False)
    require(root == expected, 'Unexpected manual VM root')
    return root

def resolve_parent(root):
    root = no_links(root).resolve(strict=True)
    parent = no_links(root.parent / PARENT_NAME).resolve(strict=True)
    require(parent.is_dir() and parent.parent == root.parent and parent != root, 'Exact read-only parent required')
    return parent

def verify_protocol(p):
    require(p.get('format') == FORMAT and p.get('parent_directory') == PARENT_NAME and
            p.get('parent_protocol_sha256') == PARENT_PROTOCOL_SHA, 'Protocol/parent identity differs')
    require(p.get('seed') == 20261015 and p.get('outputs') == 'outputs/loss_response_v1',
            'Fixed seed/output namespace differs')
    require(p.get('source_updates') == SOURCE_UPDATES and p.get('cases_file') == 'cases.json' and
            p.get('snapshots') == ['original0', 'stopped50'] and p.get('branches') == BRANCHES,
            'Fixed diagnostic census differs')
    require(p.get('loss_weights') == WEIGHTS and p.get('optimizer') == OPTIMIZER and
            p.get('numerical') == NUMERICAL and p.get('limits') == LIMITS and
            p.get('call_caps') == COUNTS, 'Frozen mathematical/resource controls differ')
    require(p.get('total_neural_forwards') == 425 and p.get('total_gradient_queries') == 193,
            'Exact finite neural/gradient census differs')
    require(p.get('automatic_app_promotion') is False and p.get('model_qualified') is False and
            p.get('new_training_recipe') is False and p.get('export_candidate_checkpoint') is False,
            'Diagnostic cannot authorize model promotion or chained training')
    require(p.get('manual_tmux_required') is True and p.get('all_branches_disposable') is True and
            p.get('quality_gates_changed') is False and p.get('status') == 'PREPARED_UNRUN',
            'Manual unrun disposable diagnostic status required')
    own, sizes = p.get('files_sha256'), p.get('files_bytes')
    require(isinstance(own, dict) and isinstance(sizes, dict) and set(own) == set(sizes) and
            'protocol.json' not in own and 'cases.json' in own, 'Exact own-file manifest required')
    for name, digest in own.items():
        safe_relative(name)
        require(re.fullmatch('[0-9a-f]{64}', digest) is not None and type(sizes[name]) is int and sizes[name] >= 0,
                'Invalid own-file fingerprint')
    deps = p.get('dependencies')
    require(isinstance(deps, dict) and deps and 'protocol.json' in deps, 'Read-only parent bindings required')
    for name, record in deps.items():
        safe_relative(name)
        require(set(record) == {'sha256', 'bytes'} and re.fullmatch('[0-9a-f]{64}', record['sha256']) and
                type(record['bytes']) is int and record['bytes'] >= 0, 'Invalid dependency fingerprint')
    require(deps['protocol.json']['sha256'] == PARENT_PROTOCOL_SHA, 'Parent protocol fingerprint differs')
    assets = p.get('assets')
    require(isinstance(assets, dict) and set(assets) == {'initializer', 'teacher', 'perceptual_features',
            'perceptual_calibration', 'original_checkpoint', 'stopped_checkpoint'}, 'Exact six assets required')
    for record in assets.values():
        require(record.get('path') in deps and {k: record[k] for k in ('sha256', 'bytes')} == deps[record['path']],
                'Asset must be an exact parent dependency')
    return p

def verify_cases(document):
    require(document.get('format') == CASES_FORMAT and document.get('source_updates') == SOURCE_UPDATES,
            'Case role/selection differs')
    rows = document.get('cases')
    require(isinstance(rows, list) and len(rows) == 16, 'Exactly16 TRAIN source/probe observations required')
    expected = [(update, slot) for update in SOURCE_UPDATES for slot in (1,2,3,4)]
    require([(r['update'], r['slot']) for r in rows] == expected, 'Canonical case order differs')
    for row in rows:
        require(row['id'] == 'u%04d_s%d' % (row['update'], row['slot']) and
                row['record']['role'] == 'train' and row['reference_id'].startswith('tr_ffhq_') and
                row['reference_id'] == row['record']['reference_id'] and
                row['update'] == row['record']['update'] and row['slot'] == row['record']['slot'] and
                row['profile'] == row['record']['profile'] and
                row['source_path'] == row['record']['source_path'] and
                row['parameters'] == row['record']['parameters'] and
                row['parameters']['behavior_anchor'] == (row['slot'] == 3), 'TRAIN role/case metadata differs')
        prefix = 'outputs/adain_gradient_isolation_v1/training_inputs/update_%04d/slot_%d/' % (row['update'],row['slot'])
        require(row['input_path'] == prefix + 'input.png' and
                row['pair_record_path'] == prefix + 'pair_record.json' and
                row['source_path'] == 'data/sources/' + row['reference_id'] + '.png',
                'Canonical source/input path join differs')
        hq = 'outputs/adain_gradient_isolation_v1/HQ_cache/' + row['reference_id'] + '/'
        require(row['hq_indices_path'] == hq + 'indices.npy' and row['hq_features_path'] == hq + 'features.npy' and
                row['hq_record_path'] == hq + 'record.json', 'Canonical HQ cache join differs')
        behavior_keys = {'behavior_delivered_path','behavior_logits_path','behavior_record_path'}
        require(behavior_keys.issubset(row) if row['slot'] == 3 else not behavior_keys.intersection(row),
                'Behavior target support differs')
        if row['slot'] == 3:
            anchor = 'outputs/adain_gradient_isolation_v1/behavior_anchors/update_%04d/' % row['update']
            require(row['behavior_delivered_path'] == anchor + 'delivered256.npy' and
                    row['behavior_logits_path'] == anchor + 'logits.npy' and
                    row['behavior_record_path'] == anchor + 'record.json', 'Canonical behavior path join differs')
        for key, value in row.items():
            if key.endswith('_path'):
                safe_relative(value)
    require(sum(bool(r['parameters']['image_eligible']) for r in rows) == 12 and
            len({r['reference_id'] for r in rows}) == 8 and
            len({r['id'] for r in rows}) == 16, 'Eligibility/reference census differs')
    require([r['parameters']['variant'] for r in rows if r['slot'] == 1] ==
            ['severe24','severe28','severe32','severe40'], 'Severe-source coverage differs')
    require([r['parameters']['variant'] for r in rows if r['slot'] == 3] ==
            ['clear256','clear128','light256','light64'], 'Protected-control coverage differs')
    return document
