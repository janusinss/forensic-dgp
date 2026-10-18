"""Finite, human-launched CodeFormer adaptation; import/static checks never train.

--run is Linux/L4/home guarded. No resume CLI, app switch or qualification flag
exists. A completed export means only that evidence was packaged successfully.
"""
import argparse
from contextlib import contextmanager
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
    'format': 'codeformer-adaptation-finite-manual-v2',
    'seed': 20261015,
    'data_manifest': 'references.json',
    'comparison_cases': 'comparison_cases.json',
    'outputs': 'outputs/adaptation_v2',
    'export_prefix': 'cctv-codeformer-adaptation-v2',
    'design': {'path': 'instructions/CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md',
               'sha256': 'a7105b577863e22c0bc35ce12d310234a510eec41c8115ebeec2e9b46bb511b9',
               'readback_path': 'provenance/v2/INDEPENDENT_DESIGN_READBACK_R2.json'},
    'training': {
        'max_updates': 600, 'microbatch': 1, 'accumulate': 4,
        'snapshots': [0, 50, 200, 600], 'benchmark_updates': 20,
        'optimizer': 'Adam', 'betas': [0.9, 0.99], 'eps': 1e-8,
        'weight_decay': 0.0, 'encoder_token_lr': 1e-5, 'fusion_lr': 2e-5,
        'warmup_updates': 50, 'cosine_end_lr': 2e-6,
        'autocast': 'bfloat16', 'grad_scaler': False, 'gradient_clipping': None,
        'GAN': False, 'EMA': False, 'resume': False,
        'max_gradient_queries': 1, 'max_backward_calls': 2400,
        'loss_weights': {'token_ce': 0.5, 'latent_mse': 1.0,
                         'pixel_l1_rgb': 1.0, 'lpips_vgg': 1.0,
                         'coarse_rgb_l1': 4.0, 'behavior_rgb_l1': 2.0,
                         'behavior_token_kl': 0.1},
        'pair_policy': 'paired-linear-light-keyed-fields-v2',
        'severe_edges': [24, 28, 32, 40],
        'control_variants': ['clear256', 'clear128', 'clear64', 'light256', 'light128', 'light64'],
        'cycle': 'u=update-1; Aedge=edges[u%4]; Bvariant=variants[(u//4)%6]',
        'behavior_slot': 3, 'distillation_temperature': 2.0,
        'call_caps': {'optimizer_step_attempts': 600, 'backward_calls': 2400,
                      'gradient_queries': 1, 'perceptual_prerequisite_forwards': 1,
                      'candidate_training_forwards': 2400, 'perceptual_training_forwards': 1800,
                      'behavior_teacher_forwards': 600, 'teacher_forwards': 391,
                      'snapshot_forwards': 1624, 'baseline_forwards': 2,
                      'initializer_parity_candidate_forwards': 2, 'training_input_writes': 2400},
    },
    'limits': {
        'initialization_seconds': 600, 'benchmark_seconds': 600,
        'projected_training_seconds': 3600, 'training_seconds': 3600,
        'snapshot_seconds': 600, 'worker_seconds': 7200,
        'export_seconds': 600, 'initial_free_bytes': 20 * 1024**3,
        'reserve_free_bytes': 4 * 1024**3,
        'result_payload_bytes': 7 * 1024**3, 'archive_bytes': 7623566951,
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
    final, temporary, metadata = 0, 0, 0
    if Path(out).exists():
        for path in Path(out).rglob('*'):
            require(not path.is_symlink(), 'Output symlink is not permitted')
            if path.is_file():
                if '.tmp-' in path.name:
                    temporary += path.stat().st_size
                else:
                    final += path.stat().st_size
                if path.suffix.lower() in {'.json', '.jsonl', '.log', '.txt', '.sha256'}:
                    metadata += path.stat().st_size
    return {'final_bytes': final, 'temporary_bytes': temporary, 'metadata_bytes': metadata}


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
            'Prospective write exceeds the declared 7 GiB result payload cap')
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
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


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
    """V2 cannot silently loosen a stop, change data roles, or extend epochs."""
    require(isinstance(protocol, dict), 'Protocol must be an object')
    for key, expected in DEFAULT_PROTOCOL.items():
        require(protocol.get(key) == expected, 'V2 protocol control differs: ' + key)
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
    require('research/codeformer_adaptation_v2/worker.py' in files,
            'Executing worker source is not bound')
    require(files.get(protocol['design']['path']) == protocol['design']['sha256'],
            'Reviewed V2 design bytes must be bound')
    require(protocol['design']['readback_path'] in files,
            'Independent design readback must be bound')
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
    position = (update - warmup) / (training['max_updates'] - warmup)
    return training['cosine_end_lr'] + (base - training['cosine_end_lr']) * (1 + math.cos(math.pi * position)) / 2


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


def verify_saved_validation(root, protocol, checked, reference_rows):
    saved = protocol['validation_saved']
    manifest = read_json(bounded(root, saved['manifest']))
    require(manifest.get('format') == 'codeformer-adaptation-saved-validation-v1' and
            manifest.get('source_protocol_sha256') == SOURCE_PROTOCOL_V1 and
            manifest.get('case_count') == 318, 'Saved validation provenance differs')
    original_path = manifest.get('retained_original_manifest_path')
    require(original_path == 'provenance/v1/validation_inputs_v1.json' and
            checked.get(original_path) == manifest.get('retained_original_manifest_sha256'),
            'Retained V1 input manifest must be byte-bound')
    original = read_json(bounded(root, original_path))
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
        pair = read_json(bounded(root, row['pair_record_path']))
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


def verify_transfer(root, protocol_sha):
    """Static bytes/schema checks only: imports no Torch, runs no network/model."""
    require(HEX.fullmatch(protocol_sha) is not None, 'Expected exact protocol SHA256')
    root = Path(root).resolve(strict=True)
    protocol_path = bounded(root, 'protocol.json')
    require(sha256(protocol_path) == protocol_sha, 'Protocol SHA256 differs')
    protocol = validate_protocol(read_json(protocol_path))
    checked = {}
    for relative, expected in protocol['files_sha256'].items():
        path = bounded(root, relative)
        require(sha256(path) == expected, 'Packet file differs: ' + relative)
        checked[relative] = expected
    for label, row in protocol['assets'].items():
        path = bounded(root, row['path'])
        require(path.stat().st_size == row['bytes'] and sha256(path) == row['sha256'],
                'Model asset differs: ' + label)
        checked[row['path']] = row['sha256']
    readback = read_json(bounded(root, protocol['design']['readback_path']))
    require(readback.get('status') == 'APPROVED_FOR_PROSPECTIVE_IMPLEMENTATION_ONLY' and
            readback.get('design_sha256') == protocol['design']['sha256'] and
            readback.get('implementation_authorized') is True and
            readback.get('training_authorized_by_this_review') is False,
            'Independent design readback is not the approved implementation-only record')
    require(sha256(Path(__file__)) == checked['research/codeformer_adaptation_v2/worker.py'],
            'Executing worker bytes differ from the frozen packet worker')
    references = read_json(bounded(root, protocol['data_manifest']))
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
    cases = read_json(bounded(root, protocol['comparison_cases']))
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
    saved = verify_saved_validation(root, protocol, checked, rows)
    receipt = {'complete': True, 'verified_files': len(checked), 'references': roles,
               'comparison_cases': kinds, 'saved_V1_validation_cases': len(saved),
               'protocol_sha256': protocol_sha,
               'model_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0}
    return protocol, receipt


@contextmanager
def pid_lock(root, protocol_sha):
    path = Path(root) / 'adaptation_v2.lock'
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
    require(root != base and root.is_relative_to(base), 'Training root must be inside ~/forensic-dgp')
    return root


def require_l4(torch):
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One available CUDA GPU required')
    require(torch.cuda.get_device_name(0) == 'NVIDIA L4', 'This finite recipe requires NVIDIA L4')
    require(torch.cuda.is_bf16_supported(), 'Declared BF16 hardware support unavailable')
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
              'perceptual_prerequisite_forwards': 0, 'perceptual_training_forwards': 0}
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
            raise PilotStop('Declared 7 GiB final result payload reached')
        if accounting['temporary_bytes'] > protocol['limits']['checkpoint_temporary_bytes']:
            raise PilotStop('Declared 1 GiB incomplete temporary evidence bound reached')
        if accounting['metadata_bytes'] > protocol['limits']['metadata_bytes']:
            raise PilotStop('Declared 128 MiB result metadata bound reached')
        if torch.cuda.memory_reserved() > protocol['limits']['CUDA_reserved_bytes']:
            raise PilotStop('Declared 22 GiB CUDA reserved-memory bound reached')

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
        payload = {'format': 'codeformer-adaptation-resumable-state-v2', 'model': _cpu_tree(net.state_dict(), torch),
                   'optimizer': _cpu_tree(optimizer.state_dict(), torch), 'completed_updates': update,
                   'schedule': {'completed_updates': update, 'protocol_training': protocol['training']},
                   'rng': committed_rng, 'sampler': committed_sampler, 'protocol_sha256': protocol_sha,
                   'counts': dict(counts), 'resume_safe': safe, 'partial_microbatches_discarded': pending_microbatches,
                   'last_safe_checkpoint': last_safe_checkpoint, 'stop_reason': reason,
                   'no_automatic_resume_supported': True, 'EMA': False}
        _atomic_torch(file, payload, torch, byte_cap=reservation)
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
        reserve_call(counts, 'snapshot_forwards', protocol['training']['call_caps']['snapshot_forwards'])
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
            from research.codeformer_adaptation_v1.perceptual import load_perceptual
            from pretrained_face_restoration import load_face_restorer
            require(data.DESIGN_SHA256 == protocol['design']['sha256'] and
                    data.CAMERA_POLICY == protocol['training']['pair_policy'] and
                    data.MAX_UPDATES == protocol['training']['max_updates'],
                    'V2 camera module/protocol design differs')
            require_l4(torch)
            require(shutil.disk_usage(root).free >= protocol['limits']['initial_free_bytes'],
                    'Require 20 GiB free after packet installation for snapshots/export/reserve')
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
            atomic_json(out / 'runtime.json', runtime)
            random.seed(protocol['seed']); np.random.seed(protocol['seed'])
            torch.manual_seed(protocol['seed']); torch.cuda.manual_seed_all(protocol['seed'])
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
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
                reserve_call(counts, 'perceptual_prerequisite_forwards',
                             protocol['training']['call_caps']['perceptual_prerequisite_forwards'])
                prerequisite_loss = perceptual(prediction_fixture, target_fixture)
                reserve_call(counts, 'gradient_queries', protocol['training']['call_caps']['gradient_queries'])
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
                reserve_call(counts, 'baseline_forwards', protocol['training']['call_caps']['baseline_forwards'])
                reserve_call(counts, 'initializer_parity_candidate_forwards',
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
            parameters = dict(net.named_parameters())
            groups = []
            for name, lr in [('encoder_token', protocol['training']['encoder_token_lr']),
                             ('fusion', protocol['training']['fusion_lr'])]:
                chosen = [p for n, p in parameters.items() if p.requires_grad and
                          (n.startswith('fuse_convs_dict.') == (name == 'fusion'))]
                require(chosen, 'Empty declared optimizer group')
                groups.append({'params': chosen, 'lr': lr, 'base_lr': lr, 'name': name})
            optimizer = torch.optim.Adam(groups, betas=tuple(protocol['training']['betas']),
                                         eps=protocol['training']['eps'], weight_decay=0.0)
            initial_trainable = {n: p.detach().cpu().clone() for n, p in parameters.items() if p.requires_grad}
            coverage = {n: {'group': n.split('.')[0], 'gradient_observations': 0,
                           'nonzero_gradient_observations': 0, 'maximum_absolute_gradient': 0.0}
                        for n in initial_trainable}
            # Frozen FP32 teacher codes are cached on CPU, never clean RGB labels
            # invented from a restoration output. Only TRAIN targets are cached.
            clean_cache = {}
            for ref in references:
                budget(init_start, protocol['limits']['initialization_seconds'])
                clean = data.derive_clean_target(ref)
                clean = clean[0] if isinstance(clean, tuple) else clean
                reserve_call(counts, 'teacher_forwards', protocol['training']['call_caps']['teacher_forwards'])
                codes = model.hq_targets(teacher, _tensor(clean, torch), codebook=net.quantize)
                clean_cache[ref.id] = model.CleanCodes(codes.indices.cpu(), codes.features.cpu())
            budget(init_start, protocol['limits']['initialization_seconds'])
            committed_rng, committed_sampler = _rng_state(torch, np), sampler_state()
            state_bytes = sum(v.numel() * v.element_size() for v in net.state_dict().values())
            trainable_bytes = sum(p.numel() * p.element_size() for p in parameters.values() if p.requires_grad)
            atomic_json(out / 'initialization.json', {'complete': True, 'parity': parity,
                'benchmark_provenance': benchmark_provenance, 'census': census, 'frozen': frozen,
                'teacher_codebook_compatibility': compatibility,
                'perceptual_input_gradient_prerequisite': perceptual_prerequisite,
                'benchmark_hash_compatibility': benchmark_hash_compatibility,
                'teacher_cache_cases': len(clean_cache), 'teacher_cache_fp32': True,
                'behavior_teacher_FP32_frozen': True,
                'validation_saved': protocol['validation_saved'],
                'mixed_precision': 'BF16 autocast; FP32 teacher/code/perceptual; no GradScaler',
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
                ref_a, ref_b = train_by_id[selection.id_a], train_by_id[selection.id_b]
                pairs = data.make_training_update(ref_a, ref_b, next_update)
                validate_training_examples(pairs, selection)
                selected_refs = [ref_a, ref_a, ref_b, ref_b]
                for slot, (ref, pair) in enumerate(zip(selected_refs, pairs), start=1):
                    budget()
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
                    reserve_call(counts, 'training_input_writes',
                                 protocol['training']['call_caps']['training_input_writes'])
                    _atomic_binary(input_path, lambda stream:
                        Image.fromarray(input_pixels).save(stream, format='PNG'),
                        byte_cap=protocol['limits']['PNG_file_bytes'])
                    exposure_record = dict(pair.record,
                        saved_input_path=input_path.relative_to(root).as_posix(),
                        saved_input_sha256=sha256(input_path))
                    atomic_json(input_dir / 'pair_record.json', exposure_record)
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
                        reserve_call(counts, 'behavior_teacher_forwards',
                                     protocol['training']['call_caps']['behavior_teacher_forwards'])
                        behavior = model.behavior_targets(behavior_teacher, _tensor(pair.input256, torch))
                        budget()
                        anchor_dir = out / 'behavior_anchors' / ('update_%04d' % next_update)
                        anchor_dir.mkdir(parents=True)
                        require_payload_room(out, protocol['limits'], 3 * 256 * 256 * 4 +
                                             256 * 1024 * 4 + 2 * 256 + 65536)
                        raw_anchor = behavior.delivered256.detach().cpu().numpy()
                        logits_anchor = behavior.logits.detach().cpu().numpy()
                        _atomic_binary(anchor_dir / 'delivered256.npy', lambda stream:
                            np.save(stream, raw_anchor, allow_pickle=False))
                        _atomic_binary(anchor_dir / 'logits.npy', lambda stream:
                            np.save(stream, logits_anchor, allow_pickle=False))
                        anchor_record = {'update': next_update, 'slot': slot,
                            'reference_id': ref.id, 'input256_float32_sha256': pair.record['input256_float32_sha256'],
                            'pair_record': exposure_record, 'teacher_state_sha256': frozen['behavior_teacher'],
                            'teacher_FP32_eval_frozen': True, 'teacher_is_not_clean_identity_truth': True,
                            'delivered_path': (anchor_dir / 'delivered256.npy').relative_to(root).as_posix(),
                            'delivered_sha256': sha256(anchor_dir / 'delivered256.npy'),
                            'logits_path': (anchor_dir / 'logits.npy').relative_to(root).as_posix(),
                            'logits_sha256': sha256(anchor_dir / 'logits.npy')}
                        atomic_json(anchor_dir / 'record.json', anchor_record)
                    codes = clean_cache[ref.id]
                    codes = model.CleanCodes(codes.indices.to('cuda'), codes.features.to('cuda'))
                    eligible = bool(pair.parameters['image_eligible'])
                    if eligible:
                        reserve_call(counts, 'perceptual_training_forwards',
                                     protocol['training']['call_caps']['perceptual_training_forwards'])
                    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                        reserve_call(counts, 'candidate_training_forwards',
                                     protocol['training']['call_caps']['candidate_training_forwards'])
                        total, terms, prediction = model.forward_training_terms(net, teacher, perceptual,
                            _tensor(pair.input256, torch), _tensor(pair.clean512, torch),
                            image_eligible=eligible, codes=codes, behavior=behavior)
                    budget()
                    require(torch.isfinite(total).item(), 'Nonfinite training loss')
                    reserve_call(counts, 'backward_calls', protocol['training']['call_caps']['backward_calls'])
                    (total / protocol['training']['accumulate']).backward()
                    counts['microbatches'] += 1; pending_microbatches += 1
                    budget()
                    require(counts['backward_calls'] <= protocol['training']['max_backward_calls'] and
                            counts['gradient_queries'] <= protocol['training']['max_gradient_queries'] and
                            counts['behavior_teacher_forwards'] <= protocol['training']['max_updates'],
                            'Declared finite gradient/backward/teacher count exceeded')
                    losses.append({'id': ref.id, 'slot': slot,
                        'profile': pair.record['profile'], 'image_eligible': eligible,
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
                reserve_call(counts, 'optimizer_step_attempts', protocol['training']['call_caps']['optimizer_step_attempts'])
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
                        raise PilotStop('Projected finite training exceeds 3600 seconds')
                    if torch.cuda.max_memory_reserved() > protocol['limits']['CUDA_reserved_bytes']:
                        raise PilotStop('Benchmark CUDA reserved-memory headroom bound failed')
                    initial_trainable.clear()
                if update % 20 == 0:
                    print(json.dumps({'update': update, 'of': protocol['training']['max_updates'], 'optimization_seconds': training_elapsed}), flush=True)
                if update in protocol['training']['snapshots']:
                    snapshot(update, validation, comparisons, baseline)
            frozen_check()
            budget()
            require(counts['optimizer_updates'] == 600 and counts['backward_calls'] == 2400 and
                    counts['candidate_training_forwards'] == 2400 and
                    counts['behavior_teacher_forwards'] == 600 and
                    counts['perceptual_training_forwards'] == 1800 and
                    counts['snapshot_forwards'] == 1624 and counts['training_input_writes'] == 2400,
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
    path = Path(root) / 'adaptation_v2.lock'
    if not path.exists():
        return {'present': False, 'active': False}
    lock = read_json(bounded(root, 'adaptation_v2.lock'))
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
        files[relative] = sha256(path)
    for relative in ['trainer.log', 'adaptation_v2.lock']:
        if (root / relative).exists():
            files[relative] = sha256(bounded(root, relative))
    retained = {name: expected for name, expected in protocol['files_sha256'].items() if name not in files}
    retained.update({row['path']: row['sha256'] for row in protocol['assets'].values() if row['path'] not in files})
    # A new manifest per attempt preserves failed export history and permits a
    # bounded retry without changing the training protocol or replacing evidence.
    manifest_path = out / ('export_manifest_' + export_id + '.json')
    atomic_json(manifest_path, {'complete': True, 'protocol_sha256': protocol_sha, 'files_sha256': files,
        'run_results_present': (out / 'run_results.json').exists(), 'failure_present': (out / 'failure.json').exists(),
        'retained_not_exported_sha256': retained, 'immutable_inputs_and_original_models_not_duplicated': True,
        'incomplete_temp_files': incomplete, 'incomplete_temp_files_are_not_resumable_checkpoints': True,
        'retained_worker_lock': lock_status,
        'training_success_not_implied': True, 'model_qualified': False})
    files[manifest_path.relative_to(root).as_posix()] = sha256(manifest_path)
    upper_payload_bytes = sum(bounded(root, name).stat().st_size for name in files)
    accounting = output_accounting(out)
    require(accounting['final_bytes'] <= protocol['limits']['result_payload_bytes'] and
            accounting['temporary_bytes'] <= protocol['limits']['checkpoint_temporary_bytes'],
            'Result final/temporary byte budget exceeded; retain evidence')
    require(upper_payload_bytes - accounting['temporary_bytes'] <= protocol['limits']['result_payload_bytes'],
            'Final exported result plus source/notice companions exceeds 7 GiB payload cap')
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
            'Conservative archive size exceeds the declared 7.1 GiB cap')
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
