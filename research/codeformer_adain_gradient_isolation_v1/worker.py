"""Finite, human-launched AdaIN image-gradient isolation comparison; import/static checks never train.

--run is Linux/L4/home guarded. No resume CLI, app switch or qualification flag
exists. A completed export means only that evidence was packaged successfully.
"""
import argparse
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import gzip
import json
import math
import os
from pathlib import Path, PurePosixPath
import platform
import random
import re
import shutil
import signal
import socket
import stat
import subprocess
import sys
import tarfile
import time
import traceback
import uuid


DEFAULT_PROTOCOL = {
    'format': 'codeformer-adain-gradient-isolation-finite-manual-v1',
    'seed': 20261015,
    'parent_dir': 'cctv_codeformer_fp32_comparison_v1_vm',
    'parent_protocol_sha256': '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1',
    'data_manifest': 'references.json',
    'comparison_cases': 'comparison_cases.json',
    'outputs': 'outputs/adain_gradient_isolation_v1',
    'export_prefix': 'cctv-codeformer-adain-gradient-isolation-v1',
    'gradient_route': {'policy': 'candidate-AdaIN-style-detach-only-v1',
        'witness_update': 3, 'witness_slots': [1, 2], 'forwards': 4,
        'gradient_queries': 8, 'before_Adam_construction': True,
        'forward_equality': 'bit-exact', 'CPU_CUDA_RNG_restored': True},
    'design': {'path': 'instructions/CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_DESIGN.md',
               'sha256': 'a406f77459695ef194676e0cde78499017b71b07b85bb5eb8d91917045939b93',
               'readback_path': 'provenance/adain_gradient_isolation_v1/INDEPENDENT_DESIGN_READBACK_V1.json'},
    'training': {
        'max_updates': 50, 'schedule_horizon_updates': 600, 'data_horizon_updates': 600,
        'microbatch': 1, 'accumulate': 4,
        'snapshots': [0, 50], 'benchmark_updates': 20,
        'optimizer': 'Adam', 'betas': [0.9, 0.99], 'eps': 1e-8,
        'weight_decay': 0.0, 'encoder_token_lr': 1e-5, 'fusion_lr': 2e-5,
        'warmup_updates': 50, 'cosine_end_lr': 2e-6,
        'autocast': 'disabled-FP32', 'grad_scaler': False, 'gradient_clipping': None,
        'GAN': False, 'EMA': False, 'resume': False,
        'max_gradient_queries': 9, 'max_backward_calls': 200,
        'loss_weights': {'token_ce': 0.5, 'latent_mse': 1.0,
                         'pixel_l1_rgb': 1.0, 'lpips_vgg': 1.0,
                         'coarse_rgb_l1': 4.0, 'behavior_rgb_l1': 2.0,
                         'behavior_token_kl': 0.1},
        'pair_policy': 'paired-linear-light-keyed-fields-v2',
        'severe_edges': [24, 28, 32, 40],
        'control_variants': ['clear256', 'clear128', 'clear64', 'light256', 'light128', 'light64'],
        'cycle': 'u=update-1; Aedge=edges[u%4]; Bvariant=variants[(u//4)%6]',
        'behavior_slot': 3, 'distillation_temperature': 2.0,
        'call_caps': {'optimizer_step_attempts': 50, 'backward_calls': 200,
                      'gradient_queries': 9, 'route_witness_forwards': 4, 'route_witness_perceptual_forwards': 4, 'perceptual_prerequisite_forwards': 1,
                      'candidate_training_forwards': 200, 'perceptual_training_forwards': 148,
                      'behavior_teacher_forwards': 50, 'teacher_forwards': 391,
                      'snapshot_forwards': 812, 'baseline_forwards': 2,
                      'initializer_parity_candidate_forwards': 2, 'training_input_writes': 200},
    },
    'limits': {
        'initialization_seconds': 600, 'witness_seconds': 180, 'benchmark_seconds': 600,
        'projected_training_seconds': 900, 'training_seconds': 900,
        'snapshot_seconds': 600, 'worker_seconds': 3000,
        'export_seconds': 600, 'initial_free_bytes': 16 * 1024**3,
        'reserve_free_bytes': 4 * 1024**3,
        'result_payload_bytes': 3758096384, 'archive_bytes': 3865470566,
        'checkpoint_temporary_bytes': 1024**3, 'PNG_file_bytes': 198656,
        'metadata_bytes': 128 * 1024**2, 'training_history_line_bytes': 65536,
        'CUDA_reserved_bytes': 22 * 1024**3,
    },
    'evaluation': {
        'validation_references': 53,
        'profiles': ['clear', 'light', 'blur', 'jpeg', 'lowlight', 'compound'],
        'comparison_cases': 88, 'native_unpaired': 48, 'old_paired': 40,
        'clear_views': ['raw', 'native_png'],
        'mean_clear_mae_delta_stop': 0.01,
        'worst_per_case_clear_mae_delta_stop': 0.03,
        'normal_blur_cases': 53, 'mean_blur_photometric_delta_stop': 0.01,
        'worst_per_case_blur_photometric_delta_stop': 0.03,
        'blur_photometric_metrics': ['luminance_bias', 'coarse_rgb_mae'],
        'quality_guard_is_historical_one_percent_gate': False,
        'baseline': 'same_VM_update0_FP32', 'fidelity_weight': 1.0,
        'adain': True, 'alignment': 'none', 'display_processing': 'none',
        'delivered_pixels': 256, 'internal_pixels': 512,
        'validation_target_policy': 'retained direct RGB1024-to-Pillow-LANCZOS256',
    },
}

ASSET_LABELS = {'initializer', 'teacher', 'perceptual_features', 'perceptual_calibration'}
HEX = re.compile(r'^[0-9a-f]{64}$')
SOURCE_PROTOCOL_V1 = 'e463222441c0930a01a85ee0711a1ea31b0d80ed3487a4a5459c71e65f517d3e'
SOURCE_PROTOCOL_V2 = '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'
SOURCE_PROTOCOL_FP32 = '8acff86e31c3d135994fdac6ed781374e036150ce1ae2110db82ca0dc5a3aff1'
SOURCE_DATA_DESIGN_SHA256 = 'a7105b577863e22c0bc35ce12d310234a510eec41c8115ebeec2e9b46bb511b9'
PACKAGE_DIR = 'cctv_codeformer_adain_gradient_isolation_v1_vm'
REGULAR_OUTPUT_FILE_CAP = 7000


class PilotStop(RuntimeError):
    """Prospective timing/resource/quality stop, never a useful-model claim."""


class CappedWriter:
    """Refuse a write before an exact temporary/archive byte ceiling is crossed."""
    def __init__(self, stream, cap):
        self.stream, self.cap = stream, cap

    def write(self, data):
        if self.stream.tell() + len(data) > self.cap:
            raise PilotStop('Declared bounded file-write byte cap reached')
        return self.stream.write(data)

    def __getattr__(self, name):
        return getattr(self.stream, name)


def output_accounting(out):
    final, temporary, metadata, regular_files = 0, 0, 0, 0
    if Path(out).exists():
        for path in Path(out).rglob('*'):
            require(not path.is_symlink(), 'Output symlink is not permitted')
            if path.is_file():
                regular_files += 1
                if '.tmp-' in path.name:
                    temporary += path.stat().st_size
                else:
                    final += path.stat().st_size
                if path.suffix.lower() in {'.json', '.jsonl', '.log', '.txt', '.sha256'}:
                    metadata += path.stat().st_size
    return {'final_bytes': final, 'temporary_bytes': temporary, 'metadata_bytes': metadata,
            'regular_files': regular_files}


def checkpoint_byte_cap(has_optimizer_state):
    """Match the storage proof, including a partially initialized first Adam step."""
    require(type(has_optimizer_state) is bool, 'Optimizer-state presence must be an explicit boolean')
    return (832 if has_optimizer_state else 384) * 1024**2


def reserve_call(counts, name, cap):
    """Reserve before a call so a cap never permits one extra invocation."""
    require(type(cap) is int and cap >= 0 and type(counts.get(name)) is int,
            'Invalid finite call counter')
    require(0 <= counts[name] < cap, 'Declared finite call cap reached: ' + name)
    counts[name] += 1


def require_payload_room(out, limits, additional_bytes):
    require(type(additional_bytes) is int and additional_bytes >= 0, 'Invalid payload reservation')
    accounting = output_accounting(out)
    require(accounting['final_bytes'] + additional_bytes <= limits['result_payload_bytes'],
            'Prospective write exceeds the declared 3.5 GiB result payload cap')
    return accounting


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    def fields(rows):
        result = {}
        for key, value in rows:
            require(key not in result, 'Duplicate JSON field: ' + key)
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding='utf-8-sig'), object_pairs_hook=fields,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def bounded(root, relative, *, must_exist=True):
    require(isinstance(relative, str) and relative, 'Empty relative path')
    rel = PurePosixPath(relative)
    require(not rel.is_absolute() and rel.as_posix() == relative and
            '..' not in rel.parts and ':' not in relative and '\\' not in relative,
            'Noncanonical or escaped packet path: ' + relative)
    root = Path(root).resolve(strict=True)
    result = root.joinpath(*rel.parts)
    require(result.resolve(strict=must_exist).is_relative_to(root), 'Packet path escape')
    for ancestor in [result, *result.parents]:
        if ancestor == root.parent:
            break
        if not ancestor.exists():
            continue
        info = ancestor.lstat()
        require(not ancestor.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 0x400,
                'Symlink/reparse point is not an execution asset')
    if must_exist:
        require(stat.S_ISREG(result.lstat().st_mode), 'Expected a regular file')
    return result


def atomic_json(path, value, *, replace=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(replace or not path.exists(), 'Retain existing evidence: ' + str(path))
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temp.open('x', encoding='utf-8', newline='\n') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()  # Only this newly created exact temporary file.


def validate_protocol(protocol):
    """The comparison cannot loosen guards or extend its fifty-step treatment."""
    require(isinstance(protocol, dict), 'Protocol must be an object')
    for key, expected in DEFAULT_PROTOCOL.items():
        require(protocol.get(key) == expected, 'FP32 comparison protocol control differs: ' + key)
    assets = protocol.get('assets')
    require(isinstance(assets, dict) and ASSET_LABELS == set(assets), 'Missing or undeclared model assets')
    files = protocol.get('files_sha256')
    require(isinstance(files, dict) and files, 'Missing exact source/data binding list')
    for name, digest in files.items():
        require(isinstance(name, str) and isinstance(digest, str) and HEX.fullmatch(digest),
                'Malformed bound-file SHA256')
        require(name != 'protocol.json', 'Protocol cannot hash itself')
    for label, asset in assets.items():
        require(isinstance(asset, dict) and {'path', 'sha256', 'bytes'} <= set(asset),
                'Asset metadata missing: ' + label)
        require(HEX.fullmatch(asset['sha256']) is not None and
                type(asset['bytes']) is int and asset['bytes'] > 0, 'Invalid asset fingerprint')
    for name in [protocol['data_manifest'], protocol['comparison_cases']]:
        require(name in files, 'Manifest must be protocol-bound: ' + name)
    export_files = protocol.get('export_files')
    require(isinstance(export_files, list) and export_files and len(set(export_files)) == len(export_files),
            'Explicit export source/manifest/license list required')
    asset_paths = {row['path'] for row in assets.values()}
    for name in export_files:
        require(name in files and name not in asset_paths and not name.startswith('data/'),
                'Do not duplicate immutable model/dataset assets in result exports')
        require(Path(name).suffix.lower() in {'.py', '.md', '.txt', '.json', '.csv', '.yaml', '.yml', '.sh', '.sha256'} or
                Path(name).name.upper() in {'LICENSE', 'COPYING', 'NOTICE', 'AUTHORS'},
                'Only source/manifest/license companions may be exported outside outputs')
    require(protocol['data_manifest'] in export_files and protocol['comparison_cases'] in export_files,
            'Export must retain exact reference and comparison ledgers')
    require(files.get('third_party/codeformer/codeformer_arch.py') ==
            'ee9758522173b72fc1f292df825b9a32179aebfd96142ececf9aa68f6e12e9eb' and
            'research/codeformer_adain_gradient_isolation_v1/gradient_route.py' in files,
            'Exact architecture and isolated gradient helper must be bound')
    require('research/codeformer_adain_gradient_isolation_v1/worker.py' in files,
            'Executing worker source is not bound')
    require(files.get(protocol['design']['path']) == protocol['design']['sha256'],
            'Reviewed V2 design bytes must be bound')
    require(protocol['design']['readback_path'] in files,
            'Independent design readback must be bound')
    parity = protocol.get('training_parity')
    require(isinstance(parity, dict) and parity.get('path') == 'training_parity_v1.json' and
            parity.get('cases') == 200 and parity.get('updates') == 50 and parity.get('behavior_anchors') == 50 and
            parity.get('source_protocol_sha256') == SOURCE_PROTOCOL_V2 and
            HEX.fullmatch(parity.get('sha256', '')) is not None and
            files.get(parity['path']) == parity['sha256'] and parity['path'] in export_files,
            'Exact complete first-fifty training parity manifest must be bound and exported')
    require(files.get('research/codeformer_adaptation_v2/data.py') ==
            '3fcc25c68c7df44aa9f3f5d9dcaa8dea837fea684a597082170750d31f4dc815' and
            files.get('research/codeformer_adaptation_v2/model.py') ==
            '8cb64d7a2c3fac46fe38805ac54c00287a050b0473608a87dfe650b5d9e508e3',
            'Historical data and model computations must remain byte-identical')
    for field, relative in [('source_readiness', 'source_readiness.json'),
                            ('resource_ledger', 'provenance/resource_ledger.json')]:
        row = protocol.get(field)
        require(isinstance(row, dict) and row.get('path') == relative and
                HEX.fullmatch(row.get('sha256', '')) is not None and files.get(relative) == row['sha256'] and
                relative in export_files, 'Required approved/source resource record is not bound: ' + field)
    hq = protocol.get('hq_cache_parity')
    require(isinstance(hq, dict) and hq.get('path') == 'hq_cache_parity_v1.json' and
            hq.get('cases') == 391 and hq.get('source_protocol_sha256') == SOURCE_PROTOCOL_FP32 and
            HEX.fullmatch(hq.get('sha256', '')) is not None and
            files.get(hq['path']) == hq['sha256'] and hq['path'] in export_files,
            'Exact retained FP32 HQ target parity manifest must be bound/exported')
    saved = protocol.get('validation_saved')
    require(isinstance(saved, dict) and saved.get('case_count') == 318 and
            saved.get('source_protocol_sha256') == SOURCE_PROTOCOL_V1 and
            saved.get('manifest') == 'validation_saved_v1.json' and
            HEX.fullmatch(saved.get('sha256', '')) is not None and
            files.get(saved['manifest']) == saved['sha256'],
            'Exact portable V1 validation manifest binding is required')
    require(saved['manifest'] in export_files, 'Export must retain saved validation provenance')
    prior = protocol.get('prior_baseline')
    require(isinstance(prior, dict) and prior.get('metrics') == 'provenance/v1/snapshot_0000_metrics.json' and
            prior.get('source_protocol_sha256') == SOURCE_PROTOCOL_V1 and
            files.get(prior['metrics']) == prior.get('sha256') and
            HEX.fullmatch(prior.get('sha256', '')) is not None and prior['metrics'] in export_files,
            'Exact prior V1 baseline metrics provenance is required')
    return protocol


def learning_rate(update, base, training=None):
    training = DEFAULT_PROTOCOL['training'] if training is None else training
    require(type(update) is int and 1 <= update <= training['max_updates'], 'Schedule position outside pilot')
    warmup = training['warmup_updates']
    if update <= warmup:
        return base * update / warmup
    position = (update - warmup) / (training['schedule_horizon_updates'] - warmup)
    return training['cosine_end_lr'] + (base - training['cosine_end_lr']) * (1 + math.cos(math.pi * position)) / 2


def expected_call_caps(training=None):
    """Derive calls from the first-fifty schedule, without importing image/Torch code."""
    training = DEFAULT_PROTOCOL['training'] if training is None else training
    maximum = training['max_updates']
    eligible = sum(2 + (2 if ((update - 1) % 4) >= 2 else 0) for update in range(1, maximum + 1))
    return {'optimizer_step_attempts': maximum, 'backward_calls': maximum * 4,
        'gradient_queries': 9, 'route_witness_forwards': 4, 'route_witness_perceptual_forwards': 4, 'perceptual_prerequisite_forwards': 1,
        'candidate_training_forwards': maximum * 4, 'perceptual_training_forwards': eligible,
        'behavior_teacher_forwards': maximum, 'teacher_forwards': 391,
        'snapshot_forwards': len(training['snapshots']) * 406, 'baseline_forwards': 2,
        'initializer_parity_candidate_forwards': 2, 'training_input_writes': maximum * 4}


RUNTIME_KEYS = ('python', 'torch', 'numpy', 'Pillow', 'OpenCV', 'CUDA', 'device', 'capability')


def algorithm_metadata(torch, *, environment=None):
    """Read actual flags only; preserve the retained algorithm-setting policy."""
    environment = os.environ if environment is None else environment
    return {'deterministic_algorithms_enabled': bool(torch.are_deterministic_algorithms_enabled()),
            'cudnn_deterministic': bool(torch.backends.cudnn.deterministic),
            'cudnn_benchmark': bool(torch.backends.cudnn.benchmark),
            'cuda_matmul_allow_tf32': bool(torch.backends.cuda.matmul.allow_tf32),
            'cudnn_allow_tf32': bool(torch.backends.cudnn.allow_tf32),
            'CUBLAS_WORKSPACE_CONFIG': environment.get('CUBLAS_WORKSPACE_CONFIG'),
            'metadata_only_no_additional_algorithm_override': True,
            'unrecorded_parent_flags_not_assumed_equal': True}


def verify_runtime_parity(actual, expected):
    """Pin computational versions/device; disk free and package inventory are measurements."""
    require(all(name in actual and name in expected and actual[name] == expected[name] for name in RUNTIME_KEYS),
            'Runtime versions/device differ from the retained V2 first-fifty experiment')
    return {'complete': True, 'matched_keys': list(RUNTIME_KEYS),
        'current_values': {name: actual[name] for name in RUNTIME_KEYS},
        'disk_free_and_package_inventory_are_recorded_not_equality_controls': True}


def verify_source_readiness(protocol, readback):
    require(readback.get('complete') is True and readback.get('approved_for_protocol_freeze') is True
            and readback.get('training_authorized') is False, 'Independent source readiness is not approved')
    bindings = readback.get('source_sha256')
    required = {'research/codeformer_adain_gradient_isolation_v1/worker.py',
        'research/codeformer_adain_gradient_isolation_v1/gradient_route.py',
        'scripts/prepare_codeformer_adain_gradient_isolation_v1.py',
        'scripts/supervise_codeformer_adain_gradient_isolation_v1.py',
        'scripts/seal_codeformer_adain_gradient_isolation_v1.py'}
    require(isinstance(bindings, dict) and required <= set(bindings) and all(
        protocol['files_sha256'].get(name) == digest for name, digest in bindings.items()),
        'Approved source hash set differs from the executing packet')
    require(all(name in bindings for name in protocol['files_sha256'] if name.startswith(
        ('research/codeformer_adain_gradient_isolation_v1/', 'tests/test_codeformer_adain_gradient_isolation'))),
        'New implementation/audit/test sources are missing from independent readiness')
    return bindings


def verify_packet_namespace(root, protocol, *, staged_dependencies=False):
    """Reject unbound importable files; only named operational receipts are extra."""
    root = Path(root).resolve(strict=True)
    dynamic = {'dependency_install_receipt.json', 'dependency_install_journal.jsonl',
               'preflight.log', 'trainer.log', 'export.log', 'supervisor_receipt.json', 'adain_gradient_isolation_v1.lock'}
    expected = set(protocol['files_sha256']) | {'protocol.json'}
    if staged_dependencies:
        expected -= set(protocol['dependency_files_sha256'])
    actual, folded = set(), set()
    for folder, dirs, files in os.walk(root, followlinks=False):
        for directory in list(dirs):
            path = Path(folder) / directory
            require(not path.is_symlink() and not getattr(path.lstat(), 'st_file_attributes', 0) & 0x400,
                    'Packet directory link/reparse point forbidden')
            if path == root / 'outputs':
                dirs.remove(directory)
        for filename in files:
            path = Path(folder) / filename
            relative = path.relative_to(root).as_posix()
            bounded(root, relative)
            require(relative.casefold() not in folded, 'Case-colliding packet files')
            folded.add(relative.casefold())
            if relative not in dynamic:
                actual.add(relative)
    require(actual == expected, 'Exact installed/staged source namespace differs')
    return len(actual)


def parity_manifest(root, protocol, checked, *, resolver=None):
    """Validate fixed role/order/file bindings; numerical pixels are checked before forward."""
    resolve = (lambda name: bounded(root, name)) if resolver is None else resolver
    supplied = protocol['training_parity']
    value = read_json(resolve(supplied['path']))
    require(value.get('format') == 'codeformer-v2-first50-training-parity-v1' and
            value.get('source_protocol_sha256') == SOURCE_PROTOCOL_V2,
            'Training parity provenance differs')
    cases = value.get('cases')
    require(isinstance(cases, list) and len(cases) == 200 and
            [(row.get('update'), row.get('slot')) for row in cases] ==
            [(update, slot) for update in range(1, 51) for slot in range(1, 5)],
            'Exact chronological two-hundred observation ledger required')
    for row in cases:
        require(re.fullmatch(r'tr_ffhq_[0-9]{5}', row.get('reference_id', '')) is not None,
                'Only approved TRAIN reference IDs may occur in parity')
        for path_key, digest_key in [('input_path', 'input_sha256'), ('pair_record_path', 'pair_record_sha256'),
                                     ('clean_source_path', 'source_sha256')]:
            require(checked.get(row.get(path_key)) == row.get(digest_key), 'Parity file not source-bound: ' + path_key)
        for key in ('input256_rgb8_sha256', 'input256_float32_sha256',
                    'clean512_rgb8_sha256', 'clean512_float32_sha256'):
            require(HEX.fullmatch(row.get(key, '')) is not None, 'Missing exact parity tensor digest')
        expected = row['slot'] >= 3 or (row['update'] - 1) % 4 >= 2
        require(type(row.get('image_eligible')) is bool and row['image_eligible'] == expected,
                'Parity image eligibility differs')
        record = read_json(resolve(row['pair_record_path']))
        require(record['role'] == 'train' and record['reference_id'] == row['reference_id'] and
                record['update'] == row['update'] and record['slot'] == row['slot'] and
                record['source_sha256'] == row['source_sha256'] and record['profile'] == row['profile'] and
                all(record[key] == row[key] for key in ('input256_rgb8_sha256', 'input256_float32_sha256',
                    'clean512_rgb8_sha256', 'clean512_float32_sha256')),
                'Parity exposure record differs')
    require(len({row['reference_id'] for row in cases}) == 100, 'First-fifty TRAIN reference count differs')
    updates = value.get('updates')
    require(isinstance(updates, list) and [row.get('update') for row in updates] == list(range(1, 51)),
            'Parity trajectory must have fifty ordered positions')
    for row in updates:
        expected = {name: learning_rate(row['update'], protocol['training'][base], protocol['training'])
                    for name, base in [('encoder_token', 'encoder_token_lr'), ('fusion', 'fusion_lr')]}
        require(row.get('group_lrs') == expected, 'First-fifty learning rates changed')
        selection = row.get('TRAIN_selection')
        require(isinstance(selection, dict) and selection.get('update') == row['update'] and
                selection.get('id_a') == cases[(row['update'] - 1) * 4]['reference_id'] and
                selection.get('id_b') == cases[(row['update'] - 1) * 4 + 2]['reference_id'],
                'Parity sampler trajectory differs')
    anchors = value.get('behavior_anchors')
    require(isinstance(anchors, list) and [row.get('update') for row in anchors] == list(range(1, 51)),
            'Exactly fifty chronological normal-B anchors required')
    for row in anchors:
        case = cases[(row['update'] - 1) * 4 + 2]
        require(row.get('slot') == 3 and row.get('reference_id') == case['reference_id'] and
                row.get('input256_float32_sha256') == case['input256_float32_sha256'] and
                row.get('teacher_FP32_eval_frozen') is True and HEX.fullmatch(row.get('teacher_state_sha256', '')),
                'Behavior anchor identity/precision differs')
        for path_key, hash_key in [('delivered_path', 'delivered_sha256'), ('logits_path', 'logits_sha256'),
                                   ('pair_record_path', 'pair_record_sha256')]:
            require(checked.get(row.get(path_key)) == row.get(hash_key), 'Unbound retained behavior target')
    expected_runtime = value.get('runtime_expected')
    require(isinstance(expected_runtime, dict) and all(name in expected_runtime for name in
            ('torch', 'numpy', 'Pillow', 'OpenCV', 'CUDA', 'device')), 'Exact V2 runtime version policy required')
    require(expected_call_caps(protocol['training']) == protocol['training']['call_caps'],
            'Declared calls disagree with first-fifty schedule')
    return value


def array_sha(value, np):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def verify_pair_parity(root, expected, pair, np, Image):
    """All checks happen before a training forward; no pixel or role substitution."""
    record = pair.record
    require(record['role'] == 'train' and all(record[name] == expected[name] for name in
            ('update', 'slot', 'reference_id', 'source_sha256', 'profile', 'input256_rgb8_sha256',
             'input256_float32_sha256', 'clean512_rgb8_sha256', 'clean512_float32_sha256')),
            'Generated exposure/clean-target provenance differs from V2')
    path = bounded(root, expected['input_path'])
    require(sha256(path) == expected['input_sha256'], 'Retained V2 TRAIN PNG changed')
    with Image.open(path) as image:
        require(image.mode == 'RGB' and image.size == (256, 256), 'Retained TRAIN input geometry changed')
        pixels = np.asarray(image).copy()
    reconstructed = np.ascontiguousarray(pixels.astype(np.float32) / np.float32(255))
    generated_pixels = np.rint(pair.input256 * 255).astype(np.uint8)
    clean_pixels = np.rint(pair.clean512 * 255).astype(np.uint8)
    require(pair.input256.dtype == np.float32 and pair.input256.shape == (256, 256, 3) and
            pair.clean512.dtype == np.float32 and pair.clean512.shape == (512, 512, 3) and
            np.isfinite(pair.input256).all() and np.isfinite(pair.clean512).all(), 'Invalid generated pair tensors')
    require(np.array_equal(reconstructed, pair.input256) and np.array_equal(pixels, generated_pixels) and
            array_sha(pixels, np) == expected['input256_rgb8_sha256'] and
            array_sha(pair.input256, np) == expected['input256_float32_sha256'] and
            array_sha(pair.clean512, np) == expected['clean512_float32_sha256'] and
            array_sha(clean_pixels, np) == expected['clean512_rgb8_sha256'],
            'V2 exact TRAIN input or clean512 tensor parity failed before forward')
    require(bool(pair.parameters['image_eligible']) == expected['image_eligible'], 'Loss eligibility changed')
    return {'complete': True, 'update': record['update'], 'slot': record['slot'],
        'retained_input_sha256': expected['input_sha256'], 'exact_input_float32': True,
        'exact_clean512_float32': True, 'before_candidate_training_forward': True}


def load_retained_anchor_arrays(root, expected, np):
    retained = {}
    for name, shape in [('delivered', (1, 3, 256, 256)), ('logits', (1, 256, 1024))]:
        path = bounded(root, expected[name + '_path'])
        require(sha256(path) == expected[name + '_sha256'], 'Original behavior anchor bytes changed')
        with path.open('rb') as stream:
            saved = np.load(stream, allow_pickle=False)
            require(not stream.read(1), 'Trailing bytes in behavior target')
        require(saved.shape == shape and saved.dtype == np.float32 and np.isfinite(saved).all(),
                'Retained original behavior target shape/dtype/finiteness differs')
        if name == 'delivered':
            require(saved.min() >= 0 and saved.max() <= 1, 'Original delivered target out of range')
        retained[name] = saved
    return retained


def verify_anchor_parity(root, expected, delivered, logits, np, *, retained=None):
    """Describe new online numerics; optimization always uses exact old arrays."""
    retained = load_retained_anchor_arrays(root, expected, np) if retained is None else retained
    result = {'complete': True, 'update': expected['update'], 'input256_float32_sha256':
              expected['input256_float32_sha256'], 'comparisons': {},
              'behavior_loss_targets': 'exact retained V2 arrays, not recomputed online arrays',
              'online_numerical_differences_are_reported_not_a_new_quality_gate': True}
    for name, actual in [('delivered', delivered), ('logits', logits)]:
        saved = retained[name]
        require(actual.shape == saved.shape and actual.dtype == np.float32 and np.isfinite(actual).all(),
                'Online FP32 anchor shape/dtype/finiteness differs')
        difference = np.abs(actual.astype(np.float64) - saved.astype(np.float64))
        result['comparisons'][name] = {'exact_array_equal': bool(np.array_equal(actual, saved)),
            'maximum_absolute': float(difference.max()), 'mean_absolute': float(difference.mean()),
            'online_tensor_sha256': array_sha(actual, np), 'used_target_tensor_sha256': array_sha(saved, np),
            'retained_file_sha256': expected[name + '_sha256']}
    return result


def initial_path_gate(image_l1, token_kl, dtype_record):
    require(all(type(value) in (int, float) and math.isfinite(value) for value in (image_l1, token_kl)),
            'Nonfinite initial path check')
    require(dtype_record['logits_dtype'] == 'torch.float32' and
            dtype_record['probabilities_dtype'] == 'torch.float32' and
            dtype_record['native_output_dtype'] == 'torch.float32' and dtype_record['autocast_enabled'] is False,
            'Candidate path is not pure FP32')
    require(0 <= image_l1 <= 1e-6 and abs(token_kl) <= 1e-6,
            'Initial FP32 image/KL path check failed before first Adam update')
    return {'pass': True, 'image_anchor_l1': image_l1, 'token_KL': token_kl, 'bound': 1e-6,
            'optimizer_updates_before_check': 0}


@contextmanager
def capture_candidate_path(net, torch, *, witness_arrays=False):
    """Observe the already-budgeted forward; never performs a neural call."""
    captured = {}
    original = net.quantize.get_codebook_feat
    functional = torch.nn.functional
    original_softmax = functional.softmax
    def lookup(indices, shape):
        require('selected_tokens' not in captured, 'Unexpected repeated candidate dictionary lookup')
        captured['selected_tokens'] = indices.detach().cpu().clone()
        return original(indices, shape)
    def logits_hook(module, arguments, output):
        captured['logits_dtype'] = str(output.dtype)
    def output_hook(module, arguments, output):
        require(isinstance(output, tuple) and len(output) == 3, 'Unexpected CodeFormer witness return')
        captured['witness_logits'] = output[1].detach().float().cpu().numpy().copy()
        captured['witness_latent'] = output[2].detach().float().cpu().numpy().copy()
    def softmax(*args, **kwargs):
        output = original_softmax(*args, **kwargs)
        if 'logits_dtype' in captured and 'probabilities_dtype' not in captured:
            captured['probabilities_dtype'] = str(output.dtype)
        return output
    handle = net.idx_pred_layer.register_forward_hook(logits_hook)
    output_handle = net.register_forward_hook(output_hook) if witness_arrays else None
    try:
        net.quantize.get_codebook_feat = lookup
        functional.softmax = softmax
        yield captured
    finally:
        net.quantize.get_codebook_feat = original
        functional.softmax = original_softmax
        handle.remove()
        if output_handle is not None:
            output_handle.remove()


def clear_quality_gate(candidate, baseline, evaluation=None):
    """Mean and worst per-case deltas, separately for RAW and delivered PNG."""
    evaluation = DEFAULT_PROTOCOL['evaluation'] if evaluation is None else evaluation
    require(set(candidate) == set(baseline) and bool(candidate), 'Clear control namespace differs')
    report = {'pass': True, 'views': {}, 'qualification_not_implied': True}
    for view in evaluation['clear_views']:
        deltas = {}
        for identifier in sorted(candidate):
            a, b = candidate[identifier][view], baseline[identifier][view]
            require(math.isfinite(a) and math.isfinite(b), 'Nonfinite clear metric')
            deltas[identifier] = a - b
        mean = sum(deltas.values()) / len(deltas)
        worst = max(deltas.values())
        passed = mean <= evaluation['mean_clear_mae_delta_stop'] and worst <= evaluation['worst_per_case_clear_mae_delta_stop']
        report['views'][view] = {'cases': len(deltas), 'mean_mae_delta': mean,
                                'worst_per_case_mae_delta': worst, 'deltas': deltas, 'pass': passed}
        report['pass'] = report['pass'] and passed
    return report


def blur_quality_gate(candidate, baseline, evaluation=None):
    """Full-support paired53 blur; compare per-case regressions, RAW/PNG apart."""
    evaluation = DEFAULT_PROTOCOL['evaluation'] if evaluation is None else evaluation
    require(set(candidate) == set(baseline) and len(candidate) == evaluation['normal_blur_cases'],
            'Normal-blur control namespace differs')
    report = {'pass': True, 'views': {}, 'qualification_not_implied': True}
    for view in evaluation['clear_views']:
        metrics = {}
        for name in evaluation['blur_photometric_metrics']:
            deltas = {}
            for identifier in sorted(candidate):
                a, b = candidate[identifier][view][name], baseline[identifier][view][name]
                require(math.isfinite(a) and math.isfinite(b), 'Nonfinite normal-blur photometry')
                deltas[identifier] = a - b
            mean, worst = sum(deltas.values()) / len(deltas), max(deltas.values())
            passed = (mean <= evaluation['mean_blur_photometric_delta_stop'] and
                      worst <= evaluation['worst_per_case_blur_photometric_delta_stop'])
            metrics[name] = {'cases': len(deltas), 'mean_delta': mean,
                             'worst_per_case_delta': worst, 'deltas': deltas, 'pass': passed}
            report['pass'] = report['pass'] and passed
        report['views'][view] = metrics
    return report


def validate_training_examples(pairs, selection):
    require(len(pairs) == 4, 'Every update needs exactly four matched slots')
    edge = [24, 28, 32, 40][(selection.update - 1) % 4]
    for slot, pair in enumerate(pairs, start=1):
        record, parameters = pair.record, pair.parameters
        expected_id = selection.id_a if slot <= 2 else selection.id_b
        require(record['reference_id'] == expected_id and record['role'] == 'train' and
                record['update'] == selection.update and record['slot'] == slot and
                record['pair_role'] == ('A' if slot <= 2 else 'B') and
                record['member'] == ('normal' if slot % 2 else 'dark') and
                parameters['behavior_anchor'] == (slot == 3) and
                parameters['image_eligible'] == (edge >= 32 if slot <= 2 else True),
                'Matched TRAIN slot role/eligibility differs')
    for first, second in [(pairs[0], pairs[1]), (pairs[2], pairs[3])]:
        require(first.record['pair_id'] == second.record['pair_id'] and
                first.record['clean512_float32_sha256'] == second.record['clean512_float32_sha256'] and
                first.parameters['noise_fields'] == second.parameters['noise_fields'] and
                first.parameters['exposure_linear'] == 1.0 and
                0 < second.parameters['exposure_linear'] < 1,
                'Normal/dark source/target/noise-field pairing differs')


def compare_prior_baseline(rows, prior):
    """Disclose saved-V1/platform differences; never replace same-VM guards."""
    require(prior.get('complete') is True and prior.get('update') == 0,
            'Prior V1 baseline metrics must be a complete update0 snapshot')
    a, b = {row['id']: row for row in rows}, {row['id']: row for row in prior['rows']}
    require(len(a) == len(b) == 406 and set(a) == set(b), 'Prior V1 baseline cohort differs')
    result = {'cases': 406, 'source_protocol_sha256': SOURCE_PROTOCOL_V1,
              'guard_baseline_remains_same_VM_V2_update0': True,
              'differences_do_not_imply_model_learning': True, 'views': {}}
    for view, key in [('raw', 'raw_sha256'), ('native_png', 'native_png_sha256')]:
        equal = [identifier for identifier in a if a[identifier][key] == b[identifier][key]]
        metric_deltas = {}
        for metric in ['mae', 'mse', 'psnr']:
            deltas = []
            for identifier in a:
                x, y = a[identifier][view].get(metric), b[identifier][view].get(metric)
                require((x is None) == (y is None), 'Prior baseline paired metric eligibility differs')
                if x is not None:
                    require(math.isfinite(x) and math.isfinite(y), 'Nonfinite prior comparison metric')
                    deltas.append(x - y)
            metric_deltas[metric] = {'paired_cases': len(deltas),
                                     'mean_delta': sum(deltas) / len(deltas),
                                     'maximum_absolute_delta': max(abs(value) for value in deltas)}
        result['views'][view] = {'equal_output_file_SHA256_count': len(equal),
                                'different_output_file_SHA256_count': 406 - len(equal),
                                'metrics': metric_deltas}
    return result


def verify_saved_validation(root, protocol, checked, reference_rows, *, resolver=None):
    resolve = (lambda name: bounded(root, name)) if resolver is None else resolver
    saved = protocol['validation_saved']
    manifest = read_json(resolve(saved['manifest']))
    require(manifest.get('format') == 'codeformer-adaptation-saved-validation-v1' and
            manifest.get('source_protocol_sha256') == SOURCE_PROTOCOL_V1 and
            manifest.get('case_count') == 318, 'Saved validation provenance differs')
    original_path = manifest.get('retained_original_manifest_path')
    require(original_path == 'provenance/v1/validation_inputs_v1.json' and
            checked.get(original_path) == manifest.get('retained_original_manifest_sha256'),
            'Retained V1 input manifest must be byte-bound')
    original = read_json(resolve(original_path))
    require(original.get('complete') is True and original.get('cases_count') == 318,
            'Original V1 validation manifest is incomplete')
    validation = {row['id']: row for row in reference_rows
                  if row['historical_role'] == 'validation'}
    expected = {'validation_' + identifier + '_' + profile: (identifier, profile)
                for identifier in validation for profile in protocol['evaluation']['profiles']}
    items = manifest.get('items')
    require(isinstance(items, list) and len(items) == 318 and
            {row['id'] for row in items} == set(expected), 'Saved validation case set differs')
    for row in items:
        identifier, profile = expected[row['id']]
        reference = validation[identifier]
        for path_key, hash_key in [('input_path', 'input_sha256'),
                                   ('pair_record_path', 'pair_record_sha256'),
                                   ('target_path', 'target_sha256')]:
            require(checked.get(row[path_key]) == row[hash_key], 'Saved validation file not bound')
        require(row['target_path'] == reference['existing_target_path'] and
                row['target_sha256'] == reference['existing_target_sha256'],
                'Saved validation target mapping differs')
        pair = read_json(resolve(row['pair_record_path']))
        require(pair.get('reference_id') == identifier and pair.get('role') == 'validation' and
                pair.get('profile') == profile and pair.get('source_sha256') == reference['source_sha256'] and
                pair.get('reference_manifest_sha256') == checked[protocol['data_manifest']],
                'Saved pair role/reference/profile/source mapping differs')
        prior = original['cases'].get(row['id'])
        require(isinstance(prior, dict) and prior.get('sha256') == row['input_sha256'],
                'Saved input is not the exact V1 VM file')
        require(row.get('original_VM_paths') ==
                {'input': prior['path'], 'pair_record': prior['pair_record_path']},
                'Original VM paths must be retained as provenance-only metadata')
    return items


def verify_transfer(root, protocol_sha, *, dependency_root=None, legacy_dependency_root=None):
    """Static bytes/schema checks only: imports no Torch, runs no network/model."""
    require(HEX.fullmatch(protocol_sha) is not None, 'Expected exact protocol SHA256')
    root = Path(root).resolve(strict=True)
    protocol_path = bounded(root, 'protocol.json')
    require(sha256(protocol_path) == protocol_sha, 'Protocol SHA256 differs')
    protocol = validate_protocol(read_json(protocol_path))
    dependencies = protocol.get('dependency_files_sha256')
    require(isinstance(dependencies, dict) and dependencies and all(
            protocol['files_sha256'].get(name) == digest for name, digest in dependencies.items()),
            'Explicit dependency-copy allowlist must match combined file bindings')
    require(legacy_dependency_root is None or dependency_root is not None,
            'Legacy staging resolver requires the explicit exact FP32 parent')
    parent_protocol = legacy_protocol = None
    if dependency_root is not None:
        dependency_root = Path(dependency_root).resolve(strict=True)
        require(dependency_root.name == protocol['parent_dir'] and
                sha256(bounded(dependency_root, 'protocol.json')) == SOURCE_PROTOCOL_FP32,
                'Static dependency review requires the exact retained FP32 parent')
        parent_protocol = read_json(bounded(dependency_root, 'protocol.json'))
        require(all(parent_protocol['files_sha256'].get(name) == digest for name, digest in dependencies.items()),
                'New dependency allowlist differs from exact FP32 parent sources')
    if legacy_dependency_root is not None:
        legacy_dependency_root = Path(legacy_dependency_root).resolve(strict=True)
        require(legacy_dependency_root.name == 'cctv_codeformer_adaptation_v2_vm' and
                sha256(bounded(legacy_dependency_root, 'protocol.json')) == SOURCE_PROTOCOL_V2,
                'Only exact retained V2 dependencies may resolve a local FP32 overlay')
        legacy_protocol = read_json(bounded(legacy_dependency_root, 'protocol.json'))
    verify_packet_namespace(root, protocol, staged_dependencies=dependency_root is not None)
    def resolve(name):
        # Explicit CPU staging review only; main/run never supplies this option.
        if dependency_root is not None and name in dependencies:
            if (dependency_root / name).exists():
                return bounded(dependency_root, name)
            require(legacy_dependency_root is not None and
                    parent_protocol['dependency_files_sha256'].get(name) == dependencies[name] and
                    legacy_protocol['files_sha256'].get(name) == dependencies[name],
                    'Missing FP32 overlay dependency is not explicitly V2-bound: ' + name)
            return bounded(legacy_dependency_root, name)
        return bounded(root, name)
    checked = {}
    for relative, expected in protocol['files_sha256'].items():
        path = resolve(relative)
        require(sha256(path) == expected, 'Packet file differs: ' + relative)
        checked[relative] = expected
    for label, row in protocol['assets'].items():
        path = resolve(row['path'])
        require(path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'],
                'Model asset differs: ' + label)
        checked[row['path']] = row['sha256']
    readback = read_json(resolve(protocol['design']['readback_path']))
    require(readback.get('complete') is True and readback.get('approved') is True and
            readback.get('reviewed_files_sha256', {}).get('CCTV_CODEFORMER_ADAIN_GRADIENT_ISOLATION_V1_DESIGN.md') ==
            protocol['design']['sha256'] and readback.get('training_authorized') is False,
            'Independent design readback is not the approved implementation-only record')
    require(sha256(Path(__file__)) == checked['research/codeformer_adain_gradient_isolation_v1/worker.py'],
            'Executing worker bytes differ from the frozen packet worker')
    verify_source_readiness(protocol, read_json(resolve(protocol['source_readiness']['path'])))
    ledger = read_json(resolve(protocol['resource_ledger']['path']))
    require(ledger.get('complete') is True, 'Independent finite resource ledger is incomplete')
    references = read_json(resolve(protocol['data_manifest']))
    rows = references.get('references', references.get('sources', []))
    require(isinstance(rows, list) and rows, 'Empty reference manifest')
    ids = set()
    roles = {'train': 0, 'validation': 0}
    for row in rows:
        role = row.get('historical_role', row.get('role'))
        require(role in roles and row['id'] not in ids, 'TRAIN/validation role or identity differs')
        require(not any(x in row['id'].lower() for x in ['final', 'test']), 'Reserved-final/test reference forbidden')
        ids.add(row['id']); roles[role] += 1
        relative, digest = row['source_path'], row['source_sha256']
        require(relative in checked and checked[relative] == digest, 'Reference is not packet-bound')
        if role == 'validation':
            require(checked.get(row['existing_target_path']) == row['existing_target_sha256'],
                    'Validation 256 target must remain its exact retained file')
    require(roles == {'train': 391, 'validation': 53}, 'Frozen reference counts differ')
    cases = read_json(resolve(protocol['comparison_cases']))
    cases = cases['cases'] if isinstance(cases, dict) else cases
    require(len(cases) == 88 and len({row['id'] for row in cases}) == 88, 'Comparison cohort differs')
    kinds = {'native_unpaired': 0, 'paired_synthetic': 0}
    for row in cases:
        require(row['evidence'] in kinds, 'Undeclared comparison evidence')
        kinds[row['evidence']] += 1
        for path_key, hash_key in [('prepared_path', 'prepared_sha256'), ('observed_path', 'observed_sha256')]:
            require(checked.get(row[path_key]) == row[hash_key], 'Unbound comparison input/mask')
        if row['evidence'] == 'paired_synthetic':
            require(checked.get(row['target_path']) == row['target_sha256'], 'Unbound paired target')
        else:
            require(not row.get('target_path'), 'Native CCTV must remain unpaired')
    require(kinds == {'native_unpaired': 48, 'paired_synthetic': 40}, 'Comparison evidence counts differ')
    saved = verify_saved_validation(root, protocol, checked, rows, resolver=resolve)
    parity = parity_manifest(root, protocol, checked, resolver=resolve)
    hq = HQ_parity_manifest(root, protocol, checked)
    require(set(hq) == {row['id'] for row in rows if row.get('historical_role', row.get('role')) == 'train'},
            'HQ parity reference namespace differs from TRAIN')
    receipt = {'complete': True, 'verified_files': len(checked), 'references': roles,
               'comparison_cases': kinds, 'saved_V1_validation_cases': len(saved),
               'protocol_sha256': protocol_sha,
               'first_V2_training_observations': len(parity['cases']),
               'retained_behavior_anchors': len(parity['behavior_anchors']), 'retained_HQ_parity_targets': len(hq),
               'staging_review_with_read_only_parent': dependency_root is not None,
               'model_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0}
    return protocol, receipt


@contextmanager
def pid_lock(root, protocol_sha):
    path = Path(root) / 'adain_gradient_isolation_v1.lock'
    payload = {'pid': os.getpid(), 'host': socket.gethostname(), 'protocol_sha256': protocol_sha,
               'created_utc': datetime.now(timezone.utc).isoformat()}
    with path.open('x', encoding='utf-8') as stream:
        json.dump(payload, stream); stream.flush(); os.fsync(stream.fileno())
    try:
        yield
    finally:
        if path.exists() and read_json(path) == payload:
            path.unlink()  # Only this process's exact lock; no stale-lock auto-removal.


def require_vm_path(root):
    require(platform.system() == 'Linux', 'Actual training is Linux L4 VM only')
    base = (Path.home() / 'forensic-dgp').resolve(strict=True)
    root = Path(root).resolve(strict=True)
    require(root == base / PACKAGE_DIR and root != base and root.is_relative_to(base),
            'Training root must be the exact new packet below ~/forensic-dgp')
    require(bool(os.environ.get('TMUX')), 'Enter the human-opened tmux session before training')
    return root


manual_root = require_vm_path


def require_l4(torch):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One available CUDA GPU required')
    require(torch.cuda.get_device_name(0) == 'NVIDIA L4', 'This finite recipe requires NVIDIA L4')
    check = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                           capture_output=True, text=True, check=True, timeout=20)
    others = [int(value.strip()) for value in check.stdout.splitlines()
              if value.strip().isdigit() and int(value.strip()) != os.getpid()]
    require(not others, 'Competing GPU process; retain packet and do not train')
    # An enclosing tmux shell is expected, never treated as a competing task.


def _tensor(array, torch, device='cuda'):
    return torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32)


def _cpu_tree(value, torch):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: _cpu_tree(v, torch) for k, v in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(v, torch) for v in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(v, torch) for v in value)
    return value


def _rng_state(torch, np):
    state = np.random.get_state()
    return {'python': random.getstate(), 'numpy': {'algorithm': state[0], 'keys': state[1].tolist(),
            'position': state[2], 'has_gauss': state[3], 'cached_gaussian': state[4]},
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all()}


def HQ_parity_manifest(root, protocol, checked):
    declared = protocol['hq_cache_parity']
    require(checked.get(declared['path']) == declared['sha256'], 'Unbound HQ parity manifest')
    value = read_json(bounded(root, declared['path']))
    require(value.get('format') == 'codeformer-fp32-HQ-cache-parity-v1' and
            value.get('source_protocol_sha256') == SOURCE_PROTOCOL_FP32,
            'HQ parity provenance differs')
    rows = value.get('cases')
    require(isinstance(rows, list) and len(rows) == 391 and
            len({row.get('reference_id') for row in rows}) == 391, 'Exactly391 HQ TRAIN targets required')
    for row in rows:
        require(re.fullmatch(r'tr_ffhq_[0-9]{5}', row.get('reference_id', '')) is not None,
                'Non-TRAIN HQ cache reference')
        require(row.get('role') == 'train' and checked.get(row.get('source_path')) == row.get('source_sha256') and
                HEX.fullmatch(row.get('source_sha256', '')) is not None,
                'HQ target source/role is not the bound TRAIN photograph')
        for key in ('clean512_float32_sha256', 'teacher_state_sha256',
                    'restoration_codebook_state_sha256', 'indices_array_sha256', 'features_array_sha256'):
            require(HEX.fullmatch(row.get(key, '')) is not None, 'Incomplete exact HQ cache digest: ' + key)
    return {row['reference_id']: row for row in rows}


def verify_HQ_cache_parity(expected, reference_id, clean, teacher, codebook, indices, features,
                           *, source_path, source_sha256):
    actual = {'reference_id': reference_id, 'clean512_float32_sha256': clean,
              'teacher_state_sha256': teacher, 'restoration_codebook_state_sha256': codebook,
              'indices_array_sha256': indices, 'features_array_sha256': features,
              'role': 'train', 'source_path': source_path, 'source_sha256': source_sha256}
    require(all(expected.get(name) == value for name, value in actual.items()),
            'Recomputed HQ targets differ from the exact retained FP32 targets: ' + reference_id)
    return {'pass': True, **actual}


def _restore_rng(state, torch, np):
    random.setstate(state['python'])
    value = state['numpy']
    np.random.set_state((value['algorithm'], np.asarray(value['keys'], dtype=np.uint32),
                         value['position'], value['has_gauss'], value['cached_gaussian']))
    torch.set_rng_state(state['torch_cpu'])
    torch.cuda.set_rng_state_all(state['torch_cuda'])


def _rng_equal(left, right, torch):
    return (left['python'] == right['python'] and left['numpy'] == right['numpy'] and
            torch.equal(left['torch_cpu'], right['torch_cpu']) and
            len(left['torch_cuda']) == len(right['torch_cuda']) and all(
                torch.equal(a, b) for a, b in zip(left['torch_cuda'], right['torch_cuda'])))


def route_witness(root, out, net, teacher, perceptual, model, data, architecture,
                  route, references, clean_cache, retained, protocol, torch, np, Image,
                  reserve, parent_budget, frozen_check):
    """Four standalone VM forwards/eight queries, with no optimizer constructed.

    Full gradients are queried sequentially and immediately reduced/released.
    Nothing writes parameter.grad. CPU/CUDA/Python/NumPy RNG and every module's
    training flag are restored even if a witness fails. Saved norms are runtime
    readouts, not locally reproducible gradients.
    """
    started = time.monotonic()
    def budget():
        parent_budget()
        if time.monotonic() - started > protocol['limits']['witness_seconds']:
            raise PilotStop('Finite180-second pre-Adam gradient witness limit reached')
    directory = out / 'gradient_route_witness'; directory.mkdir()
    before_rng = _rng_state(torch, np)
    before_state = model.state_digest(net)
    before_frozen = frozen_check()
    module_modes = [(module, module.training) for module in net.modules()]
    original_callable = architecture.adaptive_instance_normalization
    parameters = {name: value for name, value in net.named_parameters() if value.requires_grad}
    require(len(parameters) == 350 and all(value.grad is None for value in net.parameters()),
            'Witness requires350 trainable tensors and empty parameter.grad before Adam')
    names, values = tuple(parameters), tuple(parameters.values())
    by_id = {ref.id: ref for ref in references}
    selected = [row for row in retained['cases'] if row['update'] == 3]
    require(len(selected) == 4 and [row['slot'] for row in selected] == [1, 2, 3, 4],
            'Predeclared witness must use update3 slots1/2')
    pairs = data.make_training_update(by_id[selected[0]['reference_id']],
                                      by_id[selected[2]['reference_id']], 3)
    rows, parity = [], []
    report = {'complete': False, 'pass': False, 'policy': route.POLICY,
              'optimizer_constructed': False, 'optimizer_updates': 0, 'backward_calls': 0,
              'source_state_sha256': before_state, 'forwards': rows, 'parity': parity,
              'full_gradient_vectors_exported': False,
              'norms_are_source_bound_VM_readouts_not_independent_gradient_replay': True}

    def save_rng(label, state):
        record = {'python': state['python'], 'numpy': state['numpy'], 'torch': {}}
        for name, value in [('CPU', state['torch_cpu']), *[
                ('CUDA%d' % index, item) for index, item in enumerate(state['torch_cuda'])]]:
            path = directory / ('rng_' + label + '_' + name + '.npy')
            _atomic_binary(path, lambda stream, value=value:
                np.save(stream, value.cpu().numpy(), allow_pickle=False), byte_cap=1024**2)
            record['torch'][name] = {'path': path.relative_to(root).as_posix(), 'sha256': sha256(path)}
        atomic_json(directory / ('rng_' + label + '.json'), record)

    save_rng('before', before_rng)
    try:
        net.train(); net.quantize.eval(); net.generator.eval()
        for slot in (1, 2):
            pair = pairs[slot - 1]
            expected = selected[slot - 1]
            ref = by_id[expected['reference_id']]
            input_parity = verify_pair_parity(root, expected, pair, np, Image)
            require(pair.parameters['image_eligible'] is True,
                    'Witness must preserve original edge32 image eligibility')
            codes = clean_cache[ref.id]
            codes = model.CleanCodes(codes.indices.to('cuda'), codes.features.to('cuda'))
            original_saved = None
            for mode in ('original', 'isolated'):
                budget()
                _restore_rng(before_rng, torch, np)
                case_dir = directory / ('update_0003_slot%d_%s' % (slot, mode)); case_dir.mkdir()
                reserve('route_witness_perceptual_forwards',
                        protocol['training']['call_caps']['route_witness_perceptual_forwards'])
                reserve('route_witness_forwards', protocol['training']['call_caps']['route_witness_forwards'])
                capture = route.isolated_style(architecture) if mode == 'isolated' else nullcontext(None)
                with torch.autocast(device_type='cuda', enabled=False):
                    with capture as route_capture, capture_candidate_path(net, torch, witness_arrays=True) as dtype_capture:
                        total, terms, prediction = model.forward_training_terms(net, teacher, perceptual,
                            _tensor(pair.input256, torch), _tensor(pair.clean512, torch),
                            image_eligible=True, codes=codes, behavior=None)
                    route_record = route.validate_capture(route_capture) if mode == 'isolated' else {
                        'policy': 'unchanged-original-AdaIN', 'original_callable_restored':
                            architecture.adaptive_instance_normalization is original_callable}
                    tokens = dtype_capture.pop('selected_tokens').reshape(256).numpy()
                    saved = {'native': prediction.detach().float().cpu().numpy().copy(),
                        'raw': model.delivered_rgb(prediction).detach().cpu().numpy().copy(),
                        'logits': None, 'latent': None, 'tokens': tokens.copy(),
                        'terms': {name: float(value.detach().float().cpu()) for name, value in terms.items()}}
                    # Capture hook records source hashes for these arrays; the
                    # tensors returned internally by the unchanged model are
                    # captured below using one read-only hook, never a forward.
                    saved['logits'] = dtype_capture.pop('witness_logits')
                    saved['latent'] = dtype_capture.pop('witness_latent')
                    require(saved['native'].shape == (1, 3, 512, 512) and
                            saved['raw'].shape == (1, 3, 256, 256) and
                            saved['logits'].shape == (1, 256, 1024) and
                            saved['latent'].shape == (1, 256, 16, 16) and
                            saved['tokens'].shape == (256,) and saved['tokens'].dtype == np.int64 and
                            all(array.dtype == np.float32 and np.isfinite(array).all() for name, array in saved.items()
                                if name not in {'terms', 'tokens'}), 'Witness output geometry/precision differs')
                    require(dtype_capture['logits_dtype'] == 'torch.float32' and
                            dtype_capture['probabilities_dtype'] == 'torch.float32' and
                            prediction.dtype == torch.float32 and not torch.is_autocast_enabled('cuda'),
                            'Witness requires the unchanged FP32 path')
                    raw_rgb = np.ascontiguousarray(saved['raw'][0].transpose(1, 2, 0), dtype=np.float32)
                    native_rgb = np.ascontiguousarray(((prediction.detach().float() + 1) / 2)[0]
                        .permute(1, 2, 0).cpu().numpy())
                    for name, value in [('native512', native_rgb), ('raw256', raw_rgb), ('actual_tokens', tokens)]:
                        _atomic_binary(case_dir / (name + '.npy'), lambda stream, value=value:
                            np.save(stream, value, allow_pickle=False), byte_cap=value.nbytes + 256)
                    png = np.floor(np.clip(raw_rgb, 0, 1) * 255).astype(np.uint8)
                    _atomic_binary(case_dir / 'native.png', lambda stream:
                        Image.fromarray(png).save(stream, format='PNG'), byte_cap=protocol['limits']['PNG_file_bytes'])
                    image_objective = sum(model.LOSS_WEIGHTS[name] * value for name, value in terms.items()
                                          if name in route.IMAGE_TERMS)
                    supervised_objective = sum(model.LOSS_WEIGHTS[name] * value for name, value in terms.items()
                                               if name in route.SUPERVISED_TERMS)
                    summaries = {}
                    for name, loss, retain in [('image', image_objective, True),
                                               ('CE_HQ', supervised_objective, False)]:
                        budget()
                        reserve('gradient_queries', protocol['training']['call_caps']['gradient_queries'])
                        gradients = torch.autograd.grad(loss, values, allow_unused=True,
                                                       create_graph=False, retain_graph=retain)
                        summaries[name] = route.summarize_gradients(names, gradients, np)
                        del gradients
                    gradient_check = route.validate_witness_gradients(mode, summaries['image'], summaries['CE_HQ'])
                    require(all(value.grad is None for value in net.parameters()) and
                            all(value.grad is None for module in (teacher, perceptual) for value in module.parameters()),
                            'Witness query populated parameter.grad')
                    row = {'slot': slot, 'update': 3, 'mode': mode, 'reference_id': ref.id,
                        'input_and_clean_parity': input_parity, 'route': route_record,
                        'dtypes': dtype_capture, 'terms': saved['terms'], 'gradients': summaries,
                        'gradient_check': gradient_check, 'array_sha256': {name: array_sha(value, np)
                            for name, value in saved.items() if name != 'terms'},
                        'files_sha256': {path.name: sha256(path) for path in sorted(case_dir.iterdir())},
                        'parameter_grad_fields_all_none': True}
                    rows.append(row)
                    atomic_json(case_dir / 'record.json', row)
                    if mode == 'original':
                        original_saved = saved
                    else:
                        parity.append({'slot': slot, **route.validate_forward_parity(original_saved, saved, np)})
                        original_saved = None
                    del total, terms, prediction, image_objective, supervised_objective, saved, summaries
                    budget()
            del codes
        require(len(rows) == 4 and len(parity) == 2 and
                architecture.adaptive_instance_normalization is original_callable,
                'Witness forward/query/callable restoration census differs')
        report['complete'] = report['pass'] = True
    except BaseException as error:
        report.update(error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        architecture.adaptive_instance_normalization = original_callable
        for module, training in module_modes:
            module.training = training
        _restore_rng(before_rng, torch, np)
        restored = _rng_state(torch, np)
        report['CPU_CUDA_Python_NumPy_RNG_restored_exact'] = _rng_equal(before_rng, restored, torch)
        report['terminal_state_sha256'] = model.state_digest(net)
        report['frozen_before'] = before_frozen
        report['frozen_after'] = frozen_check()
        report['model_state_unchanged'] = report['terminal_state_sha256'] == before_state
        report['parameter_grad_fields_all_none'] = all(value.grad is None for value in net.parameters())
        report['original_callable_restored'] = architecture.adaptive_instance_normalization is original_callable
        report['seconds'] = time.monotonic() - started
        save_rng('restored', restored)
        atomic_json(directory / 'witness.json', report)
        require(report['CPU_CUDA_Python_NumPy_RNG_restored_exact'] and report['model_state_unchanged'] and
                report['parameter_grad_fields_all_none'] and report['original_callable_restored'] and
                report['frozen_before'] == report['frozen_after'],
                'Witness failed to restore RNG/model/grad/callable controls')
    return report


def _atomic_torch(path, value, torch, *, byte_cap):
    require(not path.exists(), 'Keep existing checkpoint: ' + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temp.open('xb') as stream:
        torch.save(value, CappedWriter(stream, byte_cap)); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp, path)
    # Any incomplete temporary file is intentionally retained as failure evidence.


def _atomic_binary(path, writer, *, byte_cap=None):
    require(not path.exists(), 'Preserve existing output: ' + str(path))
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('xb') as stream:
        writer(stream if byte_cap is None else CappedWriter(stream, byte_cap))
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def run(root, protocol, protocol_sha):
    root = require_vm_path(root)  # Must run before importing a neural framework.
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    retained_parity = parity_manifest(root, protocol, protocol['files_sha256'])
    parity_cases = {(row['update'], row['slot']): row for row in retained_parity['cases']}
    parity_updates = {row['update']: row for row in retained_parity['updates']}
    parity_anchors = {row['update']: row for row in retained_parity['behavior_anchors']}
    out = bounded(root, protocol['outputs'], must_exist=False)
    require(not out.exists(), 'No automatic repeat/resume; retain previous result directory')
    started = time.monotonic()
    update, pending_microbatches, optimizer_inflight = 0, 0, False
    pending_step_record = None
    state_valid = True
    net = optimizer = teacher = perceptual = behavior_teacher = None
    committed_rng = committed_sampler = None
    sampler = None
    request = {'signal': None}
    previous_handlers = {}
    for sig in [signal.SIGINT, signal.SIGTERM]:
        previous_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, lambda value, frame: request.__setitem__('signal', value))
    counts = {'optimizer_updates': 0, 'microbatches': 0, 'gradient_queries': 0,
              'backward_calls': 0, 'baseline_forwards': 0, 'initializer_parity_candidate_forwards': 0,
              'snapshot_forwards': 0, 'teacher_forwards': 0, 'candidate_training_forwards': 0,
              'perceptual_prerequisite_forwards': 0, 'perceptual_training_forwards': 0,
              'route_witness_forwards': 0, 'route_witness_perceptual_forwards': 0}
    counts['behavior_teacher_forwards'] = 0
    counts['optimizer_step_attempts'] = 0
    counts['training_input_writes'] = 0
    frozen = {}
    initial_trainable = {}
    coverage = {}
    last_safe_checkpoint = None
    training_elapsed = 0.0
    training_start = None
    active_step_start = None
    validation_inputs = {}
    call_index = 0
    current_call = None

    def budget(stage_start=None, stage_cap=None):
        if request['signal'] is not None:
            raise PilotStop('Manual signal requested stop at a safe stage boundary')
        if time.monotonic() - started > protocol['limits']['worker_seconds']:
            raise PilotStop('Worker wall-time cap reached')
        if stage_start is not None and time.monotonic() - stage_start > stage_cap:
            raise PilotStop('Stage wall-time cap reached')
        if training_start is not None:
            current = training_elapsed + (0.0 if active_step_start is None else
                                          time.monotonic() - active_step_start)
            if current > protocol['limits']['training_seconds']:
                raise PilotStop('Training optimization wall-time cap reached at a safe boundary')
        if shutil.disk_usage(root).free < protocol['limits']['reserve_free_bytes']:
            raise PilotStop('4 GiB storage reserve reached')
        accounting = output_accounting(out)
        if accounting['final_bytes'] > protocol['limits']['result_payload_bytes']:
            raise PilotStop('Declared 3.5 GiB final result payload reached')
        if accounting['temporary_bytes'] > protocol['limits']['checkpoint_temporary_bytes']:
            raise PilotStop('Declared 1 GiB incomplete temporary evidence bound reached')
        if accounting['metadata_bytes'] > protocol['limits']['metadata_bytes']:
            raise PilotStop('Declared 128 MiB result metadata bound reached')
        if accounting['regular_files'] > REGULAR_OUTPUT_FILE_CAP:
            raise PilotStop('Declared 7000 retained regular-file limit reached')
        if (torch.cuda.memory_reserved() > protocol['limits']['CUDA_reserved_bytes'] or
                torch.cuda.max_memory_reserved() > protocol['limits']['CUDA_reserved_bytes']):
            raise PilotStop('Declared 22 GiB CUDA reserved-memory bound reached')

    def reserve(name, cap):
        nonlocal call_index, current_call
        reserve_call(counts, name, cap)
        call_index += 1
        current_call = {'event': call_index, 'before_call': name, 'committed_update': update,
            'pending_microbatches': pending_microbatches, 'counts': dict(counts),
            'seconds': time.monotonic() - started}
        atomic_json(out / 'call_trace' / ('%05d.json' % call_index), current_call)

    def frozen_check():
        nonlocal state_valid
        current = {'candidate_frozen': model.frozen_state_digest(net),
                   'teacher': model.state_digest(teacher), 'perceptual': model.state_digest(perceptual),
                   'behavior_teacher': model.state_digest(behavior_teacher)}
        if current != frozen:
            state_valid = False
            raise PilotStop('Frozen codebook/generator/teacher/perceptual mutation')
        require(all(p.grad is None for p in teacher.parameters()) and
                all(p.grad is None for p in perceptual.parameters()) and
                all(p.grad is None for p in behavior_teacher.parameters()), 'Frozen module has gradients')
        require(not teacher.training and not perceptual.training and not behavior_teacher.training and
                not any(p.requires_grad for module in [teacher, perceptual, behavior_teacher]
                        for p in module.parameters()), 'Frozen teachers/perceptual eval status changed')
        return current

    def sampler_state():
        return None if sampler is None else deepcopy(sampler.state_dict())

    def checkpoint(directory, *, stop=False, reason=None):
        nonlocal last_safe_checkpoint
        if net is None or optimizer is None:
            return None
        accounting = output_accounting(out)
        reservation = protocol['limits']['checkpoint_temporary_bytes']
        require(accounting['temporary_bytes'] == 0,
                'Retain incomplete temporary checkpoint; do not create a second temporary state')
        require(accounting['final_bytes'] + reservation <= protocol['limits']['result_payload_bytes'],
                'Checkpoint reservation exceeds final payload bound; retain last safe state')
        require(shutil.disk_usage(root).free >= reservation + protocol['limits']['reserve_free_bytes'],
                'Insufficient checkpoint reservation plus 4 GiB reserve')
        file = directory / ('stopped_checkpoint.pt' if stop else 'checkpoint.pt')
        safe = (not optimizer_inflight and state_valid and
                committed_rng is not None and committed_sampler is not None)
        payload = {'format': 'codeformer-adain-gradient-isolation-state-v1', 'model': _cpu_tree(net.state_dict(), torch),
                   'optimizer': _cpu_tree(optimizer.state_dict(), torch), 'completed_updates': update,
                   'schedule': {'completed_updates': update, 'protocol_training': protocol['training']},
                   'rng': committed_rng, 'sampler': committed_sampler, 'protocol_sha256': protocol_sha,
                   'counts': dict(counts), 'resume_safe': safe, 'partial_microbatches_discarded': pending_microbatches,
                   'last_safe_checkpoint': last_safe_checkpoint, 'stop_reason': reason,
                   'no_automatic_resume_supported': True, 'EMA': False}
        writer_cap = checkpoint_byte_cap(bool(optimizer.state))
        require(writer_cap <= reservation, 'Checkpoint-specific writer cap exceeds temporary reservation')
        _atomic_torch(file, payload, torch, byte_cap=writer_cap)
        if safe:
            last_safe_checkpoint = file.relative_to(root).as_posix()
        return {'path': file.relative_to(root).as_posix(), 'sha256': sha256(file),
                'bytes': file.stat().st_size, 'resume_safe': safe}

    def open_rgb(relative):
        with Image.open(bounded(root, relative)) as image:
            require(image.size == (256, 256), 'Prepared comparison geometry differs')
            return np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0

    def save_case(directory, identifier, input_array, target, observed, evidence, profile, source):
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', identifier), 'Unsafe case ID')
        directory = directory / identifier
        directory.mkdir()
        require_payload_room(out, protocol['limits'], 3 * 256 * 256 * 4 + 256 +
                             protocol['limits']['PNG_file_bytes'] + 8192)
        reserve('snapshot_forwards', protocol['training']['call_caps']['snapshot_forwards'])
        with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
            native = model.infer(net, _tensor(input_array, torch))
            rgb = model.delivered_rgb(native)[0].permute(1, 2, 0).float().cpu().numpy()
        require(rgb.shape == (256, 256, 3) and np.isfinite(rgb).all(), 'Invalid delivered RAW')
        raw_file, png_file = directory / 'raw.npy', directory / 'native.png'
        _atomic_binary(raw_file, lambda stream: np.save(stream,
            np.ascontiguousarray(rgb, dtype=np.float32), allow_pickle=False))
        png = np.floor(np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        png[~observed] = np.rint(input_array[~observed] * 255).astype(np.uint8)
        _atomic_binary(png_file, lambda stream: Image.fromarray(png).save(stream, format='PNG'),
                       byte_cap=protocol['limits']['PNG_file_bytes'])
        row = {'id': identifier, 'evidence': evidence, 'profile': profile, 'source': source,
               'raw_sha256': sha256(raw_file), 'native_png_sha256': sha256(png_file),
               'observed_pixels': int(observed.sum()), 'raw': {'mae': None, 'mse': None, 'psnr': None},
               'native_png': {'mae': None, 'mse': None, 'psnr': None}}
        if target is not None:
            require(evidence == 'paired_synthetic' and observed.any(), 'Invalid paired metric mask')
            for view, prediction in [('raw', rgb), ('native_png', png.astype(np.float32) / 255.0)]:
                error = prediction.astype(np.float64)[observed] - target.astype(np.float64)[observed]
                mse = float(np.mean(error**2))
                row[view] = {'mae': float(np.mean(np.abs(error))), 'mse': mse,
                             'psnr': None if mse == 0 else -10 * math.log10(mse)}
                if profile == 'blur' and source == 'FFHQ validation':
                    require(bool(observed.all()), 'Blur photometry needs complete aligned target support')
                    predicted64 = prediction.astype(np.float64)
                    target64 = target.astype(np.float64)
                    luma_delta = (predicted64 - target64) @ np.asarray([.2126, .7152, .0722])
                    pooled_p = predicted64.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3))
                    pooled_t = target64.reshape(16, 16, 16, 16, 3).mean(axis=(1, 3))
                    row[view]['luminance_bias'] = float(abs(luma_delta.mean()))
                    row[view]['coarse_rgb_mae'] = float(np.abs(pooled_p - pooled_t).mean())
        return row

    def snapshot(at, validation, comparisons, baseline):
        snap_start = time.monotonic()
        directory = out / ('snapshot_%04d' % at)
        directory.mkdir()
        rows, clear, blur = [], {}, {}
        net.eval()
        checkpoint_receipt = checkpoint(directory)
        try:
            for ref in validation:
                for index, profile in enumerate(protocol['evaluation']['profiles']):
                    budget(snap_start, protocol['limits']['snapshot_seconds'])
                    identifier = 'validation_' + ref.id + '_' + profile
                    saved = validation_inputs[identifier]
                    require(sha256(bounded(root, saved['input_path'])) == saved['input_sha256'],
                            'Exact V1 VM-saved validation input changed')
                    target = validation_targets[ref.id]
                    row = save_case(directory, identifier, open_rgb(saved['input_path']), target,
                                    np.ones((256, 256), dtype=bool), 'paired_synthetic', profile, 'FFHQ validation')
                    row['input_path'] = saved['input_path']; row['input_sha256'] = saved['input_sha256']
                    rows.append(row)
                    if profile == 'clear':
                        clear[identifier] = {view: row[view]['mae'] for view in protocol['evaluation']['clear_views']}
                    if profile == 'blur':
                        blur[identifier] = {view: {name: row[view][name]
                            for name in protocol['evaluation']['blur_photometric_metrics']}
                            for view in protocol['evaluation']['clear_views']}
            for case in comparisons:
                budget(snap_start, protocol['limits']['snapshot_seconds'])
                original = open_rgb(case['prepared_path'])
                with Image.open(bounded(root, case['observed_path'])) as image:
                    require(image.size == (256, 256), 'Observed mask geometry differs')
                    observed = np.asarray(image.convert('L')) > 0
                target = open_rgb(case['target_path']) if case.get('target_path') else None
                identifier = 'comparison_' + case['id']
                row = save_case(directory, identifier, original, target, observed,
                                case['evidence'], case['profile'], case['source'])
                rows.append(row)
                if case['evidence'] == 'paired_synthetic' and case['profile'] == 'clear':
                    clear[identifier] = {view: row[view]['mae'] for view in protocol['evaluation']['clear_views']}
            require(len(rows) == 406 and len(clear) == 61 and len(blur) == 53, 'Snapshot/control cohort differs')
            if at == 0:
                atomic_json(out / 'validation_inputs.json', {'complete': True, 'cases': validation_inputs,
                    'saved_V1_VM_pixels_reused_for_all_snapshots': True, 'cases_count': len(validation_inputs),
                    'source_protocol_sha256': SOURCE_PROTOCOL_V1,
                    'portable_manifest_sha256': protocol['validation_saved']['sha256'],
                    'cross_platform_degraded_pixels_not_assumed_equal': True})
                prior_report = compare_prior_baseline(rows,
                    read_json(bounded(root, protocol['prior_baseline']['metrics'])))
                atomic_json(out / 'V1_baseline_comparison.json', prior_report)
            gate = None if baseline is None else clear_quality_gate(clear, baseline['clear'], protocol['evaluation'])
            blur_gate = None if baseline is None else blur_quality_gate(blur, baseline['blur'], protocol['evaluation'])
            atomic_json(directory / 'metrics.json', {'complete': True, 'update': at, 'rows': rows,
                        'clear': clear, 'blur': blur, 'quality_guard': gate,
                        'normal_blur_photometric_guard': blur_gate, 'checkpoint': checkpoint_receipt,
                        'seconds': time.monotonic() - snap_start, 'frozen': frozen_check(),
                        'no_model_qualification_claim': True})
            print(json.dumps({'snapshot': at, 'cases': len(rows), 'clear_controls': len(clear),
                              'quality_guard_pass': gate is None or gate['pass'],
                              'normal_blur_guard_pass': blur_gate is None or blur_gate['pass']}), flush=True)
            if gate is not None and not gate['pass']:
                raise PilotStop('Declared RAW/PNG clear-preservation guard failed; retained snapshot')
            if blur_gate is not None and not blur_gate['pass']:
                raise PilotStop('Declared RAW/PNG normal-blur photometric guard failed; retained snapshot')
            return {'clear': clear, 'blur': blur}
        except BaseException as error:
            if not (directory / 'metrics.json').exists():
                atomic_json(directory / 'partial_metrics.json', {'complete': False, 'rows': rows, 'clear': clear,
                            'error': str(error), 'checkpoint': checkpoint_receipt, 'update': at})
            raise

    with pid_lock(root, protocol_sha):
        out.mkdir(parents=True)
        atomic_json(out / 'run_started.json', {'protocol_sha256': protocol_sha, 'pid': os.getpid(),
                    'manual_launch_required': True, 'counts': counts, 'qualification_not_implied': True})
        try:
            init_start = time.monotonic()
            import torch
            import numpy as np
            from PIL import Image
            from research.codeformer_adaptation_v2 import model, data
            from research.codeformer_adain_gradient_isolation_v1 import gradient_route
            from third_party.codeformer import codeformer_arch as architecture
            from research.codeformer_adaptation_v1.perceptual import load_perceptual
            from pretrained_face_restoration import load_face_restorer
            require(data.DESIGN_SHA256 == SOURCE_DATA_DESIGN_SHA256 and
                    data.CAMERA_POLICY == protocol['training']['pair_policy'] and
                    data.MAX_UPDATES == protocol['training']['data_horizon_updates'],
                    'V2 camera module/protocol design differs')
            require_l4(torch)
            require(shutil.disk_usage(root).free >= protocol['limits']['initial_free_bytes'],
                    'Require 16 GiB free after packet installation for snapshots/export/reserve')
            import cv2
            import PIL
            runtime = {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                       'Pillow': PIL.__version__, 'OpenCV': cv2.__version__, 'CUDA': torch.version.cuda,
                       'device': torch.cuda.get_device_name(0), 'capability': list(torch.cuda.get_device_capability(0)),
                       'initial_free_bytes': shutil.disk_usage(root).free,
                       'cross_platform_degraded_pixels_not_assumed_equal': True}
            try:
                packages = subprocess.run([sys.executable, '-m', 'pip', 'freeze'],
                    capture_output=True, text=True, timeout=30, check=False)
                runtime['pip_freeze'] = packages.stdout[:2 * 1024**2]
                runtime['pip_freeze_returncode'] = packages.returncode
            except subprocess.TimeoutExpired:
                runtime['pip_freeze_stop'] = 'Optional package inventory exceeded 30 seconds'
            runtime_parity = verify_runtime_parity(runtime, retained_parity['runtime_expected'])
            atomic_json(out / 'runtime_parity.json', runtime_parity)
            runtime['candidate_precision'] = 'FP32 with disabled autocast; no precision fallback'
            random.seed(protocol['seed']); np.random.seed(protocol['seed'])
            torch.manual_seed(protocol['seed']); torch.cuda.manual_seed_all(protocol['seed'])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            runtime['algorithm_settings'] = algorithm_metadata(torch)
            atomic_json(out / 'runtime.json', runtime)
            assets = {name: bounded(root, value['path']) for name, value in protocol['assets'].items()}
            net, census = model.load_trainable(assets['initializer'], device='cuda')
            teacher = model.load_teacher(assets['teacher'], device='cuda')
            behavior_teacher = model.load_behavior_teacher(assets['initializer'], device='cuda')
            compatibility = model.assert_shared_codebook(net, teacher)
            perceptual = load_perceptual(assets['perceptual_features'].parent, device='cuda')
            teacher.eval().requires_grad_(False); perceptual.eval().requires_grad_(False)
            perceptual_before = model.state_digest(perceptual)
            prediction_fixture = torch.linspace(-0.8, 0.8, 3 * 64 * 64, device='cuda',
                                                dtype=torch.float32).reshape(1, 3, 64, 64).requires_grad_(True)
            target_fixture = torch.flip(prediction_fixture.detach(), dims=[3]) * 0.5
            with torch.autocast(device_type='cuda', enabled=False):
                reserve('perceptual_prerequisite_forwards',
                             protocol['training']['call_caps']['perceptual_prerequisite_forwards'])
                prerequisite_loss = perceptual(prediction_fixture, target_fixture)
                reserve('gradient_queries', protocol['training']['call_caps']['gradient_queries'])
                prerequisite_gradient, = torch.autograd.grad(prerequisite_loss, prediction_fixture,
                                                             create_graph=False, retain_graph=False)
            require(torch.isfinite(prerequisite_gradient).all().item() and
                    prerequisite_gradient.abs().max().item() > 0, 'LPIPS input gradient is nonfinite or detached')
            require(all(p.grad is None for p in perceptual.parameters()) and
                    model.state_digest(perceptual) == perceptual_before, 'LPIPS prerequisite changed frozen weights')
            perceptual_prerequisite = {'complete': True, 'gradient_queries': 1, 'optimizer_updates': 0,
                'fixture_shape': [1, 3, 64, 64], 'loss': float(prerequisite_loss.detach().cpu()),
                'input_gradient_maximum': float(prerequisite_gradient.abs().max().cpu()),
                'input_gradient_l2': float(prerequisite_gradient.norm().cpu()),
                'frozen_state_sha256': perceptual_before, 'frozen_parameter_gradients_none': True}
            atomic_json(out / 'perceptual_input_gradient_prerequisite.json', perceptual_prerequisite)
            del prediction_fixture, target_fixture, prerequisite_loss, prerequisite_gradient
            require(model.LOSS_WEIGHTS == protocol['training']['loss_weights'], 'Loss declaration differs')
            references = data.load_reference_manifest(root, 'train', root / protocol['data_manifest'],
                expected_manifest_sha256=protocol['files_sha256'][protocol['data_manifest']])
            validation = data.load_reference_manifest(root, 'validation', root / protocol['data_manifest'],
                expected_manifest_sha256=protocol['files_sha256'][protocol['data_manifest']])
            comparisons = read_json(root / protocol['comparison_cases'])
            comparisons = comparisons['cases'] if isinstance(comparisons, dict) else comparisons
            require(len(references) == 391 and len(validation) == 53, 'Frozen TRAIN/validation split differs')
            references = tuple(sorted(references, key=lambda ref: ref.id))
            train_by_id = {ref.id: ref for ref in references}
            sampler = data.TrainSampler(sorted(train_by_id), seed=protocol['seed'])
            reference_ledger = read_json(root / protocol['data_manifest'])
            reference_rows = {row['id']: row for row in reference_ledger['references']}
            validation_targets = {ref.id: open_rgb(reference_rows[ref.id]['existing_target_path']) for ref in validation}
            saved_manifest = read_json(bounded(root, protocol['validation_saved']['manifest']))
            validation_inputs = {row['id']: row for row in saved_manifest['items']}
            require(len(validation_inputs) == 318, 'Exact saved V1 validation inputs missing')
            if hasattr(hashlib, 'file_digest'):
                benchmark, benchmark_provenance = load_face_restorer(assets['initializer'], device='cuda')
                benchmark_hash_compatibility = 'native hashlib.file_digest'
            else:
                from unittest.mock import patch
                from research.codeformer_adaptation_v1.common import file_digest_compat
                with patch.object(hashlib, 'file_digest', file_digest_compat, create=True):
                    benchmark, benchmark_provenance = load_face_restorer(assets['initializer'], device='cuda')
                benchmark_hash_compatibility = 'isolated Python3.10 streamed-hash shim during unchanged benchmark loader only'
            parity = []
            for case in comparisons[:2]:
                value = _tensor(open_rgb(case['prepared_path']), torch)
                reserve('baseline_forwards', protocol['training']['call_caps']['baseline_forwards'])
                reserve('initializer_parity_candidate_forwards',
                             protocol['training']['call_caps']['initializer_parity_candidate_forwards'])
                with torch.inference_mode(), torch.autocast(device_type='cuda', enabled=False):
                    predicted = model.delivered_rgb(model.infer(net, value))
                    original = benchmark(value)
                maximum = float((predicted - original).abs().max().cpu())
                require(maximum <= 1e-6, 'Initializer and untouched benchmark wrapper differ')
                parity.append({'id': case['id'], 'maximum_raw_error': maximum})
            del benchmark
            torch.cuda.empty_cache()
            frozen = {'candidate_frozen': model.frozen_state_digest(net),
                      'teacher': model.state_digest(teacher), 'perceptual': model.state_digest(perceptual),
                      'behavior_teacher': model.state_digest(behavior_teacher)}
            cache_codebook_state = model.state_digest(net.quantize)
            parameters = dict(net.named_parameters())
            groups = []
            for name, lr in [('encoder_token', protocol['training']['encoder_token_lr']),
                             ('fusion', protocol['training']['fusion_lr'])]:
                chosen = [p for n, p in parameters.items() if p.requires_grad and
                          (n.startswith('fuse_convs_dict.') == (name == 'fusion'))]
                require(chosen, 'Empty declared optimizer group')
                groups.append({'params': chosen, 'lr': lr, 'base_lr': lr, 'name': name})
            initial_trainable = {n: p.detach().cpu().clone() for n, p in parameters.items() if p.requires_grad}
            coverage = {n: {'group': n.split('.')[0], 'gradient_observations': 0,
                           'nonzero_gradient_observations': 0, 'maximum_absolute_gradient': 0.0}
                        for n in initial_trainable}
            # Frozen FP32 teacher codes are cached on CPU, never clean RGB labels
            # invented from a restoration output. Only TRAIN targets are cached.
            clean_cache = {}
            hq_expected = HQ_parity_manifest(root, protocol, protocol['files_sha256'])
            for ref in references:
                budget(init_start, protocol['limits']['initialization_seconds'])
                clean = data.derive_clean_target(ref)
                clean = clean[0] if isinstance(clean, tuple) else clean
                reserve('teacher_forwards', protocol['training']['call_caps']['teacher_forwards'])
                codes = model.hq_targets(teacher, _tensor(clean, torch), codebook=net.quantize)
                clean_cache[ref.id] = model.CleanCodes(codes.indices.cpu(), codes.features.cpu())
                cache_dir = out / 'HQ_cache' / ref.id
                cache_dir.mkdir(parents=True)
                cached_indices = clean_cache[ref.id].indices.numpy()
                cached_features = clean_cache[ref.id].features.numpy()
                require(cached_indices.dtype == np.int64 and cached_indices.shape == (1, 256) and
                        cached_features.dtype == np.float32 and cached_features.shape == (1, 256, 16, 16) and
                        np.isfinite(cached_features).all() and cached_indices.min() >= 0 and cached_indices.max() < 1024,
                        'HQ cache tensor geometry/dtype invalid')
                hq_parity = verify_HQ_cache_parity(hq_expected[ref.id], ref.id,
                    array_sha(clean, np), frozen['teacher'], cache_codebook_state,
                    array_sha(cached_indices, np), array_sha(cached_features, np),
                    source_path=ref.source_relative_path, source_sha256=ref.source_sha256)
                require_payload_room(out, protocol['limits'], cached_indices.nbytes + cached_features.nbytes + 16384)
                for name, array in [('indices', cached_indices), ('features', cached_features)]:
                    _atomic_binary(cache_dir / (name + '.npy'), lambda stream, array=array:
                        np.save(stream, array, allow_pickle=False))
                atomic_json(cache_dir / 'record.json', {'reference_id': ref.id, 'role': ref.role,
                    'source_path': ref.source_relative_path, 'source_sha256': ref.source_sha256,
                    'clean512_float32_sha256': array_sha(clean, np),
                    'teacher_state_sha256': frozen['teacher'], 'restoration_codebook_state_sha256': cache_codebook_state,
                    'indices_path': (cache_dir / 'indices.npy').relative_to(root).as_posix(),
                    'indices_sha256': sha256(cache_dir / 'indices.npy'),
                    'features_path': (cache_dir / 'features.npy').relative_to(root).as_posix(),
                    'features_sha256': sha256(cache_dir / 'features.npy'),
                    'exact_retained_FP32_target_parity': hq_parity,
                    'hq_cache_parity_manifest_sha256': protocol['hq_cache_parity']['sha256']})
            budget(init_start, protocol['limits']['initialization_seconds'])
            witness = route_witness(root, out, net, teacher, perceptual, model, data,
                architecture, gradient_route, references, clean_cache, retained_parity,
                protocol, torch, np, Image, reserve, lambda: budget(init_start,
                    protocol['limits']['initialization_seconds']), frozen_check)
            require(witness['complete'] and witness['pass'] and optimizer is None,
                    'Before-Adam route witness did not close')
            optimizer = torch.optim.Adam(groups, betas=tuple(protocol['training']['betas']),
                                         eps=protocol['training']['eps'], weight_decay=0.0)
            committed_rng, committed_sampler = _rng_state(torch, np), sampler_state()
            state_bytes = sum(v.numel() * v.element_size() for v in net.state_dict().values())
            trainable_bytes = sum(p.numel() * p.element_size() for p in parameters.values() if p.requires_grad)
            atomic_json(out / 'initialization.json', {'complete': True, 'parity': parity,
                'benchmark_provenance': benchmark_provenance, 'census': census, 'frozen': frozen,
                'teacher_codebook_compatibility': compatibility,
                'perceptual_input_gradient_prerequisite': perceptual_prerequisite,
                'benchmark_hash_compatibility': benchmark_hash_compatibility,
                'teacher_cache_cases': len(clean_cache), 'teacher_cache_fp32': True,
                'retained_FP32_HQ_cache_parity': True, 'gradient_route_witness': witness,
                'behavior_teacher_FP32_frozen': True,
                'behavior_policy': '50 budgeted FP32 online teacher comparisons; exact retained V2 arrays are loss targets; both sets exported',
                'validation_saved': protocol['validation_saved'],
                'mixed_precision': 'Disabled autocast; FP32 candidate/teacher/code/perceptual; no GradScaler',
                'resource_estimate': {'model_state_bytes': state_bytes, 'Adam_moment_bytes': 2 * trainable_bytes,
                    'gradient_bytes': trainable_bytes, 'activation_and_allocator_bytes_not_measured': True,
                    'L4_training_timing_not_measured_until_benchmark': True},
                'seconds': time.monotonic() - init_start})
            baseline = snapshot(0, validation, comparisons, None)
            training_start = time.monotonic()
            benchmark_start = training_start
            history = out / 'training.jsonl'
            for next_update in range(1, protocol['training']['max_updates'] + 1):
                budget()
                if request['signal'] is not None:
                    raise PilotStop('Manual signal requested stop between optimizer updates')
                if training_elapsed > protocol['limits']['training_seconds']:
                    raise PilotStop('Training optimization wall-time cap reached')
                step_start = time.monotonic()
                active_step_start = step_start
                if next_update <= 20 and step_start - benchmark_start > protocol['limits']['benchmark_seconds']:
                    raise PilotStop('20-update benchmark time cap reached')
                net.train(); net.quantize.eval(); net.generator.eval()
                optimizer.zero_grad(set_to_none=True)
                for group in optimizer.param_groups:
                    group['lr'] = learning_rate(next_update, group['base_lr'], protocol['training'])
                pending_microbatches = 0
                losses = []
                selection = sampler.preview_update()
                require(selection.update == next_update and selection.id_a != selection.id_b,
                        'Transactional TRAIN selection differs')
                require(selection.to_record() == parity_updates[next_update]['TRAIN_selection'] and
                        {group['name']: group['lr'] for group in optimizer.param_groups} ==
                        parity_updates[next_update]['group_lrs'], 'First-fifty selection or LR changed')
                ref_a, ref_b = train_by_id[selection.id_a], train_by_id[selection.id_b]
                pairs = data.make_training_update(ref_a, ref_b, next_update)
                validate_training_examples(pairs, selection)
                selected_refs = [ref_a, ref_a, ref_b, ref_b]
                for slot, (ref, pair) in enumerate(zip(selected_refs, pairs), start=1):
                    budget()
                    parity_receipt = verify_pair_parity(root, parity_cases[(next_update, slot)], pair, np, Image)
                    require(pair.record['reference_id'] == ref.id and ref.role == 'train',
                            'Only the selected approved TRAIN reference may be optimized')
                    require_payload_room(out, protocol['limits'],
                                         protocol['limits']['PNG_file_bytes'] + 65536)
                    input_dir = out / 'training_inputs' / ('update_%04d' % next_update) / ('slot_%d' % slot)
                    input_dir.mkdir(parents=True)
                    input_pixels = np.rint(pair.input256 * 255).astype(np.uint8)
                    require(hashlib.sha256(input_pixels.tobytes()).hexdigest() ==
                            pair.record['input256_rgb8_sha256'], 'TRAIN input PNG roundtrip changed pixels')
                    input_path = input_dir / 'input.png'
                    reserve('training_input_writes',
                                 protocol['training']['call_caps']['training_input_writes'])
                    _atomic_binary(input_path, lambda stream:
                        Image.fromarray(input_pixels).save(stream, format='PNG'),
                        byte_cap=protocol['limits']['PNG_file_bytes'])
                    exposure_record = dict(pair.record,
                        saved_input_path=input_path.relative_to(root).as_posix(),
                        saved_input_sha256=sha256(input_path))
                    atomic_json(input_dir / 'pair_record.json', exposure_record)
                    atomic_json(input_dir / 'parity.json', parity_receipt)
                    if next_update == 1 and slot == 1:
                        example_dir = out / 'first_training_example'; example_dir.mkdir()
                        _atomic_binary(example_dir / 'input256.npy', lambda stream:
                            np.save(stream, pair.input256, allow_pickle=False))
                        _atomic_binary(example_dir / 'input256.png', lambda stream:
                            Image.fromarray(np.rint(pair.input256 * 255).astype(np.uint8)).save(stream, format='PNG'),
                            byte_cap=protocol['limits']['PNG_file_bytes'])
                        _atomic_binary(example_dir / 'clean512.png', lambda stream:
                            Image.fromarray(np.rint(pair.clean512 * 255).astype(np.uint8)).save(stream, format='PNG'),
                            byte_cap=3 * 512 * 512 + 4096)
                        atomic_json(example_dir / 'pair_record.json', pair.record)
                    behavior = None
                    anchor_record = None
                    if slot == protocol['training']['behavior_slot']:
                        # Teacher runs before this candidate graph; all previous
                        # microbatch graphs have already been freed by backward.
                        reserve('behavior_teacher_forwards',
                                     protocol['training']['call_caps']['behavior_teacher_forwards'])
                        online_behavior = model.behavior_targets(behavior_teacher, _tensor(pair.input256, torch))
                        budget()
                        anchor_dir = out / 'behavior_anchors' / ('update_%04d' % next_update)
                        online_dir = out / 'online_behavior_anchors' / ('update_%04d' % next_update)
                        anchor_dir.mkdir(parents=True); online_dir.mkdir(parents=True)
                        require_payload_room(out, protocol['limits'], 2 * (3 * 256 * 256 * 4 +
                                             256 * 1024 * 4 + 2 * 256 + 65536))
                        raw_online = online_behavior.delivered256.detach().cpu().numpy()
                        logits_online = online_behavior.logits.detach().cpu().numpy()
                        retained_targets = load_retained_anchor_arrays(root, parity_anchors[next_update], np)
                        anchor_parity = verify_anchor_parity(root, parity_anchors[next_update], raw_online,
                            logits_online, np, retained=retained_targets)
                        require(parity_anchors[next_update]['teacher_state_sha256'] == frozen['behavior_teacher'],
                                'Retained behavior teacher state differs')
                        raw_anchor, logits_anchor = retained_targets['delivered'], retained_targets['logits']
                        behavior = model.BehaviorTargets(
                            torch.from_numpy(np.ascontiguousarray(raw_anchor).copy()).clone().to('cuda'),
                            torch.from_numpy(np.ascontiguousarray(logits_anchor).copy()).clone().to('cuda'))
                        require(not behavior.delivered256.requires_grad and not behavior.logits.requires_grad and
                                not torch.is_inference(behavior.delivered256) and not torch.is_inference(behavior.logits),
                                'Behavior targets must be ordinary detached exact retained tensors')
                        for name, array in [('delivered256', raw_online), ('logits', logits_online)]:
                            _atomic_binary(online_dir / (name + '.npy'), lambda stream, array=array:
                                np.save(stream, array, allow_pickle=False))
                        atomic_json(online_dir / 'record.json', {'update': next_update,
                            'reference_id': ref.id, 'input256_float32_sha256': pair.record['input256_float32_sha256'],
                            'teacher_state_sha256': frozen['behavior_teacher'], 'teacher_FP32_eval_frozen': True,
                            'delivered_path': (online_dir / 'delivered256.npy').relative_to(root).as_posix(),
                            'delivered_sha256': sha256(online_dir / 'delivered256.npy'),
                            'logits_path': (online_dir / 'logits.npy').relative_to(root).as_posix(),
                            'logits_sha256': sha256(online_dir / 'logits.npy'),
                            'not_used_as_behavior_loss_targets': True})
                        atomic_json(anchor_dir / 'parity.json', anchor_parity)
                        del online_behavior, retained_targets, raw_online, logits_online
                        _atomic_binary(anchor_dir / 'delivered256.npy', lambda stream:
                            np.save(stream, raw_anchor, allow_pickle=False))
                        _atomic_binary(anchor_dir / 'logits.npy', lambda stream:
                            np.save(stream, logits_anchor, allow_pickle=False))
                        anchor_record = {'update': next_update, 'slot': slot,
                            'reference_id': ref.id, 'input256_float32_sha256': pair.record['input256_float32_sha256'],
                            'pair_record': exposure_record, 'teacher_state_sha256': frozen['behavior_teacher'],
                            'teacher_FP32_eval_frozen': True, 'teacher_is_not_clean_identity_truth': True,
                            'target_policy': 'exact retained V2 FP32 arrays; newly computed arrays retained separately',
                            'retained_delivered_sha256': parity_anchors[next_update]['delivered_sha256'],
                            'retained_logits_sha256': parity_anchors[next_update]['logits_sha256'],
                            'delivered_path': (anchor_dir / 'delivered256.npy').relative_to(root).as_posix(),
                            'delivered_sha256': sha256(anchor_dir / 'delivered256.npy'),
                            'logits_path': (anchor_dir / 'logits.npy').relative_to(root).as_posix(),
                            'logits_sha256': sha256(anchor_dir / 'logits.npy')}
                        atomic_json(anchor_dir / 'record.json', anchor_record)
                    codes = clean_cache[ref.id]
                    codes = model.CleanCodes(codes.indices.to('cuda'), codes.features.to('cuda'))
                    eligible = bool(pair.parameters['image_eligible'])
                    if eligible:
                        reserve('perceptual_training_forwards',
                                     protocol['training']['call_caps']['perceptual_training_forwards'])
                    with torch.autocast(device_type='cuda', enabled=False):
                        reserve('candidate_training_forwards',
                                     protocol['training']['call_caps']['candidate_training_forwards'])
                        with gradient_route.isolated_style(architecture) as route_capture, \
                                capture_candidate_path(net, torch) as captured_path:
                            total, terms, prediction = model.forward_training_terms(net, teacher, perceptual,
                                _tensor(pair.input256, torch), _tensor(pair.clean512, torch),
                                image_eligible=eligible, codes=codes, behavior=behavior)
                        selected_tokens = captured_path.pop('selected_tokens').reshape(256).numpy()
                        require(selected_tokens.dtype == np.int64 and selected_tokens.shape == (256,) and
                                selected_tokens.min() >= 0 and selected_tokens.max() < 1024,
                                'Actual selected candidate tokens have invalid geometry or bounds')
                        route_receipt = gradient_route.validate_capture(route_capture)
                        dtype_record = {**captured_path, 'adain_gradient_route': route_receipt,
                            'native_output_dtype': str(prediction.dtype),
                            'autocast_enabled': bool(torch.is_autocast_enabled('cuda')),
                            'actual_native_softmax_and_selected_indices_captured': True}
                        require(dtype_record['logits_dtype'] == 'torch.float32' and
                                dtype_record['probabilities_dtype'] == 'torch.float32' and
                                dtype_record['native_output_dtype'] == 'torch.float32' and
                                dtype_record['autocast_enabled'] is False,
                                'Mixed-precision candidate execution forbidden')
                        _atomic_binary(input_dir / 'actual_tokens.npy', lambda stream:
                            np.save(stream, selected_tokens, allow_pickle=False))
                        atomic_json(input_dir / 'dtypes.json', dtype_record)
                        if next_update == 1 and slot == 3:
                            image_l1 = float(terms['behavior_rgb_l1'].detach().float().cpu())
                            token_kl = float(terms['behavior_token_kl'].detach().float().cpu())
                            prospective = {'update': 1, 'slot': 3, 'image_anchor_l1': image_l1,
                                'token_KL': token_kl, 'bound': 1e-6, 'optimizer_updates_before_check': update,
                                'dtypes': dtype_record, 'tokens_file': (input_dir / 'actual_tokens.npy').relative_to(root).as_posix(),
                                'tokens_sha256': sha256(input_dir / 'actual_tokens.npy'),
                                'pass': math.isfinite(image_l1) and math.isfinite(token_kl) and
                                        0 <= image_l1 <= 1e-6 and abs(token_kl) <= 1e-6}
                            atomic_json(out / 'initial_FP32_path_check.json', prospective)
                            initial_path_gate(image_l1, token_kl, dtype_record)
                            require(update == 0 and counts['optimizer_step_attempts'] == 0,
                                    'Path check must precede first Adam attempt')
                    budget()
                    require(torch.isfinite(total).item(), 'Nonfinite training loss')
                    reserve('backward_calls', protocol['training']['call_caps']['backward_calls'])
                    (total / protocol['training']['accumulate']).backward()
                    counts['microbatches'] += 1; pending_microbatches += 1
                    budget()
                    require(counts['backward_calls'] <= protocol['training']['max_backward_calls'] and
                            counts['gradient_queries'] <= protocol['training']['max_gradient_queries'] and
                            counts['behavior_teacher_forwards'] <= protocol['training']['max_updates'],
                            'Declared finite gradient/backward/teacher count exceeded')
                    losses.append({'id': ref.id, 'slot': slot,
                        'profile': pair.record['profile'], 'image_eligible': eligible,
                        'input_and_clean_parity': parity_receipt, 'candidate_dtypes': dtype_record,
                        'pair_record': exposure_record, 'behavior_anchor': anchor_record,
                        'terms': {n: float(v.detach().float().cpu()) for n, v in terms.items()},
                        'weighted_terms': {n: model.LOSS_WEIGHTS[n] * float(v.detach().float().cpu())
                                           for n, v in terms.items()},
                        'total_before_accumulation_division': float(total.detach().float().cpu())})
                    del total, terms, prediction, behavior
                for name, parameter in parameters.items():
                    if not parameter.requires_grad:
                        require(parameter.grad is None, 'Frozen candidate parameter received gradient')
                        continue
                    grad = parameter.grad
                    if grad is not None:
                        require(torch.isfinite(grad).all().item(), 'Nonfinite gradient: ' + name)
                    if next_update <= 20 and grad is not None:
                        row = coverage[name]; row['gradient_observations'] += 1
                        maximum = float(grad.detach().abs().max().cpu())
                        row['maximum_absolute_gradient'] = max(row['maximum_absolute_gradient'], maximum)
                        row['nonzero_gradient_observations'] += int(maximum > 0)
                pending_step_record = {'update': next_update, 'microbatches': losses,
                    'group_lrs': {g['name']: g['lr'] for g in optimizer.param_groups},
                    'TRAIN_selection': selection.to_record(), 'optimizer_committed': False}
                state_valid = False
                reserve('optimizer_step_attempts', protocol['training']['call_caps']['optimizer_step_attempts'])
                optimizer_inflight = True; optimizer.step(); optimizer_inflight = False
                require(all(torch.isfinite(p).all().item() for p in parameters.values()), 'Nonfinite updated parameter')
                sampler.commit(selection)
                update = next_update; counts['optimizer_updates'] = update; pending_microbatches = 0
                training_elapsed += time.monotonic() - step_start
                active_step_start = None
                committed_rng, committed_sampler = _rng_state(torch, np), sampler_state()
                state_valid = True
                pending_step_record['optimizer_committed'] = True
                pending_step_record['optimization_seconds'] = training_elapsed
                budget()
                history_line = json.dumps(pending_step_record, allow_nan=False) + '\n'
                history_bytes = len(history_line.encode('utf-8'))
                require(history_bytes <= protocol['limits']['training_history_line_bytes'],
                        'Declared bounded trajectory-line size exceeded')
                require_payload_room(out, protocol['limits'], history_bytes)
                with history.open('a', encoding='utf-8', newline='\n') as stream:
                    stream.write(history_line)
                    stream.flush()
                pending_step_record = None
                if update == 20:
                    projection = training_elapsed / 20 * protocol['training']['max_updates']
                    group_coverage = {}
                    for name, row in coverage.items():
                        delta = parameters[name].detach().cpu() - initial_trainable[name]
                        row['actual_change_maximum'] = float(delta.abs().max())
                        row['actual_change_l2'] = float(delta.double().norm())
                        group = group_coverage.setdefault(row['group'], {'finite_nonzero_grad': False, 'actual_change': False})
                        group['finite_nonzero_grad'] |= row['nonzero_gradient_observations'] > 0
                        group['actual_change'] |= row['actual_change_maximum'] > 0
                    intended = set(model.TRAINABLE_GROUPS)
                    passed = set(group_coverage) == intended and all(v['finite_nonzero_grad'] and v['actual_change']
                                                                              for v in group_coverage.values())
                    atomic_json(out / 'benchmark20.json', {'complete': True, 'updates': 20, 'parameters': coverage,
                        'group_coverage': group_coverage, 'group_requirement_pass': passed,
                        'all_parameters_nonzero_not_required': True, 'seconds': training_elapsed,
                        'projected_training_seconds': projection, 'frozen': frozen_check(),
                        'CUDA_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                        'CUDA_peak_reserved_bytes': torch.cuda.max_memory_reserved()})
                    if time.monotonic() - benchmark_start > protocol['limits']['benchmark_seconds']:
                        raise PilotStop('20-update benchmark exceeded 600 seconds')
                    require(passed, 'Intended group finite-gradient/actual-change coverage failed')
                    if projection > protocol['limits']['projected_training_seconds']:
                        raise PilotStop('Projected finite training exceeds 900 seconds')
                    if torch.cuda.max_memory_reserved() > protocol['limits']['CUDA_reserved_bytes']:
                        raise PilotStop('Benchmark CUDA reserved-memory headroom bound failed')
                    initial_trainable.clear()
                if update % 20 == 0:
                    print(json.dumps({'update': update, 'of': protocol['training']['max_updates'], 'optimization_seconds': training_elapsed}), flush=True)
                if update in protocol['training']['snapshots']:
                    snapshot(update, validation, comparisons, baseline)
            frozen_check()
            budget()
            require(counts['optimizer_updates'] == 50 and counts['microbatches'] == 200 and
                    all(counts[name] == cap for name, cap in expected_call_caps(protocol['training']).items()),
                    'Completed finite exposure/forward counts differ')
            atomic_json(out / 'run_results.json', {'complete': True, 'completed_updates': update,
                'counts': counts, 'optimization_seconds': training_elapsed, 'worker_seconds': time.monotonic() - started,
                'protocol_sha256': protocol_sha, 'training_finished_not_qualification': True,
                'model_qualified': False, 'automatic_app_promotion': False})
        except BaseException as error:
            detail = {'complete': False, 'error_type': type(error).__name__, 'error': str(error),
                      'traceback': traceback.format_exc(), 'completed_updates': update, 'counts': counts,
                      'protocol_sha256': protocol_sha, 'optimizer_inflight': optimizer_inflight,
                      'pending_step_record': pending_step_record,
                      'last_call_reservation': current_call,
                      'pending_step_record_preserves_committed_row_before_history_abort': True,
                      'worker_seconds': time.monotonic() - started, 'model_qualified': False}
            try:
                detail['checkpoint'] = checkpoint(out, stop=True, reason=str(error))
            except BaseException as checkpoint_error:
                detail['checkpoint_error'] = repr(checkpoint_error)
                detail['last_safe_checkpoint'] = last_safe_checkpoint
            atomic_json(out / 'failure.json', detail)
            raise
        finally:
            for sig, previous in previous_handlers.items():
                signal.signal(sig, previous)


def export_lock_state(root):
    """Never remove a stale lock; package it after a proven dead Linux PID."""
    path = Path(root) / 'adain_gradient_isolation_v1.lock'
    if not path.exists():
        return {'present': False, 'active': False}
    lock = read_json(bounded(root, 'adain_gradient_isolation_v1.lock'))
    require(type(lock.get('pid')) is int and 0 < lock['pid'] < 2**31, 'Invalid retained PID lock')
    if lock.get('host') != socket.gethostname():
        return {'present': True, 'active': False, 'foreign_host_lock_retained': True, 'lock': lock}
    require(platform.system() == 'Linux', 'An existing local PID lock needs Linux process inspection before export')
    try:
        os.kill(lock['pid'], 0)  # Linux existence query only, never used on Windows.
    except ProcessLookupError:
        return {'present': True, 'active': False, 'dead_PID_lock_retained': True, 'lock': lock}
    except PermissionError:
        raise PilotStop('Cannot inspect retained worker PID; refuse an active-writer export')
    raise PilotStop('Active worker PID still exists; wait before exporting')


def closed_export_companions(root, protocol_sha):
    """Only closed installer/run companions; never include the live export log."""
    names = ('dependency_install_receipt.json', 'dependency_install_journal.jsonl',
             'outputs/supervision/preflight.log', 'outputs/supervision/trainer.log',
             'outputs/supervision/run_receipt.json')
    receipt = read_json(bounded(root, names[0]))
    require(receipt.get('complete') is True and receipt.get('protocol_sha256') == protocol_sha,
            'Closed verified install receipt required for export')
    run = read_json(bounded(root, names[-1]))
    require(run.get('protocol_sha256') == protocol_sha and type(run.get('training_process_exit_code')) is int
            and [stage.get('mode', stage.get('stage')) for stage in run.get('stages', [])] ==
            ['--verify-transfer', '--run'], 'Closed supervisor training stage required for export')
    require(run['stages'][-1]['exit_code'] == run['training_process_exit_code'], 'Supervisor training exit receipt differs')
    return {name: sha256(bounded(root, name)) for name in names}


def _export_results(root, protocol, protocol_sha, *, destination=None):
    """Package existing evidence only. This function imports no neural framework."""
    started = time.monotonic()
    root = Path(root).resolve(strict=True)
    lock_status = export_lock_state(root)
    out = bounded(root, protocol['outputs'], must_exist=False)
    require(out.is_dir(), 'No run evidence to export')
    destination = Path.home() if destination is None else Path(destination).resolve(strict=True)
    archive = destination / (protocol['export_prefix'] + '-results.tar.gz')
    require(not archive.exists(), 'Preserve previous result export; choose a new packet version')
    require(not list(destination.glob(archive.name + '.partial-*')),
            'Retained partial archive exists; no automatic repeated export accumulation')
    files = {}
    incomplete = {}
    export_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8]
    stage = 'inventory'
    for path in sorted(out.rglob('*')):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            bounded(root, relative)
            files[relative] = sha256(path)
            if '.tmp-' in path.name or '.partial-' in path.name:
                incomplete[relative] = {'sha256': files[relative], 'bytes': path.stat().st_size,
                                        'resume_safe': False, 'reason': 'Incomplete temporary write preserved as failure evidence'}
        require(time.monotonic() - started <= protocol['limits']['export_seconds'], 'Export finite cap reached')
    for relative in ['protocol.json', *protocol['export_files']]:
        path = bounded(root, relative)
        digest = sha256(path)
        expected = protocol_sha if relative == 'protocol.json' else protocol['files_sha256'][relative]
        require(digest == expected, 'Declared source/metadata changed before export: ' + relative)
        files[relative] = digest
    operational = closed_export_companions(root, protocol_sha)
    files.update(operational)
    for relative in ['adain_gradient_isolation_v1.lock']:
        if (root / relative).exists():
            files[relative] = sha256(bounded(root, relative))
    retained = {name: expected for name, expected in protocol['files_sha256'].items() if name not in files}
    retained.update({row['path']: row['sha256'] for row in protocol['assets'].values() if row['path'] not in files})
    # A new manifest per attempt preserves failed export history and permits a
    # bounded retry without changing the training protocol or replacing evidence.
    manifest_path = out / ('export_manifest_' + export_id + '.json')
    require(len(files) + 1 <= REGULAR_OUTPUT_FILE_CAP,
            'Export namespace exceeds the declared 7000 regular-file limit')
    atomic_json(manifest_path, {'complete': True, 'protocol_sha256': protocol_sha, 'files_sha256': files,
        'run_results_present': (out / 'run_results.json').exists(), 'failure_present': (out / 'failure.json').exists(),
        'retained_not_exported_sha256': retained, 'immutable_inputs_and_original_models_not_duplicated': True,
        'incomplete_temp_files': incomplete, 'incomplete_temp_files_are_not_resumable_checkpoints': True,
        'retained_worker_lock': lock_status,
        'closed_operational_companions_sha256': operational,
        'training_success_not_implied': True, 'model_qualified': False})
    files[manifest_path.relative_to(root).as_posix()] = sha256(manifest_path)
    upper_payload_bytes = sum(bounded(root, name).stat().st_size for name in files)
    accounting = output_accounting(out)
    require(accounting['final_bytes'] <= protocol['limits']['result_payload_bytes'] and
            accounting['temporary_bytes'] <= protocol['limits']['checkpoint_temporary_bytes'],
            'Result final/temporary byte budget exceeded; retain evidence')
    require(upper_payload_bytes - accounting['temporary_bytes'] <= protocol['limits']['result_payload_bytes'],
            'Final exported result plus source/notice companions exceeds 3.5 GiB payload cap')
    exported_metadata_bytes = sum(bounded(root, name).stat().st_size for name in files
        if Path(name).suffix.lower() in {'.json', '.jsonl', '.log', '.txt', '.md', '.py', '.sh',
                                         '.sha256', '.csv', '.yaml', '.yml'} or
        Path(name).name.upper() in {'LICENSE', 'COPYING', 'NOTICE', 'AUTHORS'})
    require(exported_metadata_bytes <= protocol['limits']['metadata_bytes'],
            'Exported trajectory/log/source metadata exceeds the 128 MiB bound')
    # TAR headers/padding and DEFLATE incompressible overhead, plus a fixed
    # one-MiB end/header allowance. Actual output still uses a strict cap.
    tar_upper = sum(((bounded(root, name).stat().st_size + 511) // 512) * 512 + 4096
                    for name in files) + 1024**2
    archive_upper = tar_upper + (tar_upper // 1000 + 1) + 1024**2
    require(archive_upper <= protocol['limits']['archive_bytes'],
            'Conservative archive size exceeds the declared 3.6 GiB cap')
    require(shutil.disk_usage(destination).free >= archive_upper + protocol['limits']['reserve_free_bytes'],
            'Insufficient conservative export space plus 4 GiB reserve')
    temp = archive.with_name(archive.name + '.partial-' + uuid.uuid4().hex)
    try:
        stage = 'archive'
        with temp.open('xb') as raw:
            capped = CappedWriter(raw, protocol['limits']['archive_bytes'])
            with gzip.GzipFile(fileobj=capped, mode='wb', compresslevel=1, mtime=0) as compressed:
                with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as target:
                    for relative, expected in files.items():
                        require(time.monotonic() - started <= protocol['limits']['export_seconds'], 'Export finite cap reached')
                        path = bounded(root, relative)
                        before_info = path.stat()
                        require(sha256(path) == expected, 'Export source changed')
                        target.add(path, arcname=relative, recursive=False)
                        after_info = path.stat()
                        require((before_info.st_size, before_info.st_mtime_ns, before_info.st_ino) ==
                                (after_info.st_size, after_info.st_mtime_ns, after_info.st_ino),
                                'Export source metadata changed while archiving')
            raw.flush(); os.fsync(raw.fileno())
        os.replace(temp, archive)
    except BaseException as error:
        atomic_json(out / ('export_failure_' + export_id + '.json'),
                    {'complete': False, 'stage': stage, 'error': str(error), 'error_type': type(error).__name__,
                     'protocol_sha256': protocol_sha, 'seconds': time.monotonic() - started,
                     'originals_and_prior_attempts_retained': True, 'model_qualified': False})
        raise
    # A failed bounded partial archive remains labelled, never silently deleted.
    receipt = {'complete': True, 'archive_sha256': sha256(archive), 'bytes': archive.stat().st_size,
               'protocol_sha256': protocol_sha, 'training_success_not_implied': True,
               'run_results_present': (out / 'run_results.json').exists(), 'failure_present': (out / 'failure.json').exists(),
               'manifest_member': manifest_path.relative_to(root).as_posix(),
               'manifest_sha256': sha256(manifest_path), 'retained_not_exported_files': len(retained),
               'incomplete_temp_files': len(incomplete),
               'upper_payload_bytes': upper_payload_bytes, 'conservative_archive_upper_bytes': archive_upper,
               'result_accounting': accounting,
               'exported_metadata_bytes': exported_metadata_bytes,
               'seconds': time.monotonic() - started, 'model_qualified': False}
    atomic_json(destination / (protocol['export_prefix'] + '-export.json'), receipt)
    sidecar = destination / (archive.name + '.sha256')
    with sidecar.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(receipt['archive_sha256'] + '  ' + archive.name + '\n')
    return receipt


def export_results(root, protocol, protocol_sha, *, destination=None):
    """Retain a new structured failure even when export space/prechecks stop."""
    started = time.monotonic()
    try:
        return _export_results(root, protocol, protocol_sha, destination=destination)
    except BaseException as error:
        try:
            out = bounded(root, protocol['outputs'], must_exist=False)
            if out.is_dir():
                name = 'export_stop_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8] + '.json'
                atomic_json(out / name, {'complete': False, 'error_type': type(error).__name__,
                    'error': str(error), 'protocol_sha256': protocol_sha,
                    'seconds': time.monotonic() - started, 'training_success_not_implied': True,
                    'original_run_evidence_unchanged': True, 'model_qualified': False})
        except BaseException:
            pass  # Disk-full/path-denied must not replace the original failure.
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--protocol-sha', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--verify-transfer', action='store_true')
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--export', action='store_true')
    args = parser.parse_args(argv)
    protocol, receipt = verify_transfer(args.root, args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps(receipt), flush=True)
    elif args.export:
        print(json.dumps(export_results(args.root, protocol, args.protocol_sha)), flush=True)
    else:
        run(args.root, protocol, args.protocol_sha)


if __name__ == '__main__':
    main()
