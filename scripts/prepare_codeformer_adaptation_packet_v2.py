"""Stage new portable V2 inputs, then freeze reviewed sources on explicit request.

No neural imports, remote operations, model calls or training occur. An inputs-
only stage has no protocol or runnable supervisor until independent readiness.
"""
import argparse
import ast
import builtins
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
NAME = 'cctv_codeformer_adaptation_v2_vm'
EXECUTION_ARCHIVE = 'cctv-codeformer-adaptation-v2-execution.tar.gz'
EVIDENCE = 'outputs/codeformer_training_readiness_v2/packet_inputs_v1'
DESIGN_SHA = 'a7105b577863e22c0bc35ce12d310234a510eec41c8115ebeec2e9b46bb511b9'
V1_PROTOCOL_SHA = 'e463222441c0930a01a85ee0711a1ea31b0d80ed3487a4a5459c71e65f517d3e'
V1_PACKET = 'outputs/cctv_codeformer_adaptation_v1_vm'
V1_RETURN = 'outputs/codeformer_adaptation_v1_return'
V1_REVIEW = 'outputs/codeformer_adaptation_v1_return_review'
READBACK = 'outputs/codeformer_adaptation_v2_design_review/INDEPENDENT_DESIGN_READBACK_R2.json'
FEEDBACK = 'CCTV_CODEFORMER_ADAPTATION_V1_USER_REVIEW_20261015.md'
HEX = re.compile(r'^[0-9a-f]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)', re.I)
SOURCE_REQUIRED = {
    'research/codeformer_adaptation_v2/__init__.py', 'research/codeformer_adaptation_v2/data.py',
    'research/codeformer_adaptation_v2/model.py', 'research/codeformer_adaptation_v2/worker.py',
    'scripts/prepare_codeformer_adaptation_packet_v2.py', 'scripts/seal_codeformer_adaptation_packet_v2.py',
    'scripts/supervise_codeformer_adaptation_v2.py', 'tests/test_codeformer_adaptation_packet_v2.py',
}
SHARED = ['research/__init__.py', 'research/codeformer_adaptation_v1/__init__.py',
          'research/codeformer_adaptation_v1/common.py', 'research/codeformer_adaptation_v1/data.py',
          'research/codeformer_adaptation_v1/perceptual.py', 'pretrained_face_restoration.py',
          'third_party/__init__.py', *['third_party/codeformer/' + x for x in
          ('__init__.py', 'codeformer_arch.py', 'vqgan_arch.py', 'registry.py', 'LICENSE', 'NOTICE.md')]]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(4 << 20), b''):
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
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as target:
        target.write(data); target.flush(); os.fsync(target.fileno())


def write_json(path, value):
    write_new(path, (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode('utf-8'))


def canonical(relative):
    require(isinstance(relative, str) and bool(relative), 'Empty relative path')
    value = PurePosixPath(relative)
    require(bool(value.parts) and not value.is_absolute() and value.as_posix() == relative and
            all(part not in ('', '.', '..') for part in value.parts), 'Noncanonical or escaping path')
    require(not any(ch in relative for ch in '\\:"<>|?*') and
            all(not part.endswith((' ', '.')) and not DEVICE.match(part) and
                all(ord(ch) >= 32 for ch in part) for part in value.parts), 'Nonportable packet path')
    return relative


def bounded(root, relative, *, exists=True):
    canonical(relative)
    root = Path(root).resolve(strict=True)
    path = root.joinpath(*PurePosixPath(relative).parts)
    require(path.resolve(strict=exists).is_relative_to(root), 'Path escapes root')
    for item in [path, *path.parents]:
        if item == root.parent:
            break
        if item.exists():
            info = item.lstat()
            require(not item.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 1024,
                    'Symlink/reparse point is not a research transfer asset')
    if exists:
        require(stat.S_ISREG(path.lstat().st_mode), 'Expected regular source file')
    return path


def exact_namespace(packet):
    names, folded = set(), set()
    for parent, dirs, files in os.walk(packet, followlinks=False):
        for name in dirs:
            path = Path(parent) / name
            info = path.lstat()
            require(not path.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 1024,
                    'Unexpected packet directory link')
        for name in files:
            path = Path(parent) / name
            relative = path.relative_to(packet).as_posix()
            bounded(packet, relative)
            require(relative.casefold() not in folded, 'Case-colliding packet paths')
            folded.add(relative.casefold()); names.add(relative)
    return names


def export_companion(relative):
    canonical(relative)
    path = PurePosixPath(relative)
    return not relative.startswith(('data/', 'weights/')) and path.suffix.lower() in {
        '.py', '.md', '.txt', '.json', '.csv', '.yaml', '.yml', '.sh', '.sha256'} or \
        (not relative.startswith(('data/', 'weights/')) and path.name.upper() in {'LICENSE', 'COPYING', 'NOTICE', 'AUTHORS'})


def copy_exact(root, packet, source_relative, destination_relative, expected, bindings, copies, journal):
    source = bounded(root, source_relative)
    require(isinstance(expected, str) and HEX.fullmatch(expected), 'Expected source hash missing')
    require(destination_relative not in bindings and destination_relative.casefold() not in
            {x.casefold() for x in bindings}, 'Duplicate/colliding staged destination')
    destination = bounded(packet, destination_relative, exists=False)
    require(not destination.exists(), 'Retain existing staged file')
    info = source.stat()
    require(sha256(source) == expected, 'Original fingerprint differs: ' + source_relative)
    journal.write(json.dumps({'event': 'intent', 'source': source_relative, 'destination': destination_relative,
        'sha256': expected, 'bytes': info.st_size}) + '\n'); journal.flush(); os.fsync(journal.fileno())
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as src, destination.open('xb') as dst:
        shutil.copyfileobj(src, dst, length=4 << 20); dst.flush(); os.fsync(dst.fileno())
    after = source.stat()
    require((info.st_size, info.st_mtime_ns, info.st_ino, info.st_dev) ==
            (after.st_size, after.st_mtime_ns, after.st_ino, after.st_dev), 'Source changed during transfer')
    require(sha256(destination) == expected and destination.stat().st_size == info.st_size, 'Copied bytes differ')
    bindings[destination_relative] = expected
    copies[destination_relative] = {'original': source_relative, 'sha256': expected, 'bytes': info.st_size}
    journal.write(json.dumps({'event': 'copied', 'destination': destination_relative, 'sha256': expected,
                             'bytes': info.st_size}) + '\n'); journal.flush(); os.fsync(journal.fileno())


def stage_inputs(root):
    started = time.monotonic()
    root = Path(root).resolve(strict=True)
    packet, evidence = root / 'outputs' / NAME, root / EVIDENCE
    require(not packet.exists() and not evidence.exists(), 'Preserve prior V2 staging/evidence; no automatic retry')
    require(not (root / 'outputs' / EXECUTION_ARCHIVE).exists(), 'Sealed archive cannot be revised')
    require(shutil.disk_usage(root).free >= 5 * 1024**3, 'Require5GiB local staging headroom')
    v1 = read_json(bounded(root, V1_PACKET + '/protocol.json'))
    require(sha256(root / V1_PACKET / 'protocol.json') == V1_PROTOCOL_SHA, 'Frozen V1 packet protocol changed')
    require(sha256(bounded(root, 'CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md')) == DESIGN_SHA, 'Unapproved design')
    readback = read_json(bounded(root, READBACK))
    require(readback.get('design_sha256') == DESIGN_SHA and readback.get('status') ==
            'APPROVED_FOR_PROSPECTIVE_IMPLEMENTATION_ONLY' and readback.get('implementation_authorized') is True and
            readback.get('training_authorized_by_this_review') is False, 'Independent implementation readback absent')
    refs = read_json(bounded(root, V1_PACKET + '/references.json'))
    rows = refs['references']
    require(len(rows) == 444 and len({r['id'] for r in rows}) == 444 and
            sum(r['historical_role'] == 'train' for r in rows) == 391 and
            sum(r['historical_role'] == 'validation' for r in rows) == 53, 'Frozen reference roles differ')
    imported = read_json(bounded(root, V1_REVIEW + '/archive_review/independent_archive_import_v1.json'))
    require(imported.get('complete') is True and imported.get('protocol_sha256') == V1_PROTOCOL_SHA and
            imported.get('exact_imported_namespace_and_hashes') is True, 'Independent V1 input-return audit missing')
    export = read_json(bounded(root, 'outputs/cctv-codeformer-adaptation-v1-export.json'))
    require(export.get('complete') is True and export.get('archive_sha256') == imported['archive_sha256'] and
            export.get('protocol_sha256') == V1_PROTOCOL_SHA, 'Returned V1 export receipt differs')
    returned_manifest_path = V1_RETURN + '/' + canonical(export['manifest_member'])
    require(sha256(bounded(root, returned_manifest_path)) == export['manifest_sha256'], 'V1 return manifest changed')
    returned_bindings = read_json(root / returned_manifest_path)['files_sha256']
    packet.mkdir(); evidence.mkdir(parents=True)
    bindings, copies, assets = {}, {}, {}
    try:
        with (evidence / 'copy_journal.jsonl').open('x', encoding='utf-8', newline='\n') as journal:
            def copy(old, new, expected=None):
                if expected is None:
                    expected = sha256(bounded(root, old))
                copy_exact(root, packet, old, new, expected, bindings, copies, journal)
            def from_v1(relative):
                require(relative in v1['files_sha256'], 'V1 original binding missing: ' + relative)
                copy(V1_PACKET + '/' + relative, relative, v1['files_sha256'][relative])
            def from_return(relative, destination):
                require(relative in returned_bindings, 'Returned V1 source hash missing')
                copy(V1_RETURN + '/' + relative, destination, returned_bindings[relative])
            for relative in ('references.json', 'comparison_cases.json'):
                from_v1(relative)
            for row in rows:
                from_v1(row['source_path'])
                if row['historical_role'] == 'validation':
                    from_v1(row['existing_target_path'])
            comparisons = read_json(packet / 'comparison_cases.json')['cases']
            require(len(comparisons) == len({r['id'] for r in comparisons}) == 88 and
                    sum(r['evidence'] == 'native_unpaired' for r in comparisons) == 48 and
                    sum(r['evidence'] == 'paired_synthetic' for r in comparisons) == 40, 'Comparison cohort changed')
            for row in comparisons:
                require(row['evidence'] != 'native_unpaired' or 'target_path' not in row, 'Native target forbidden')
                for key in ('prepared_path', 'observed_path', 'target_path'):
                    if key in row and row[key] not in bindings:
                        from_v1(row[key])
            for label, asset in v1['assets'].items():
                from_v1(asset['path'])
                assets[label] = deepcopy(asset)
            for relative in SHARED:
                copy(relative, relative, v1['files_sha256'][relative])
            for relative in ('perceptual/manifest.json', 'perceptual/LICENSE_LPIPS.txt',
                             'perceptual/LICENSE_TORCHVISION.txt', 'perceptual/NOTICE.md',
                             'provenance/restoration_provenance.json', 'provenance/teacher_acquisition.json',
                             'provenance/upstream_CodeFormer_README.md', 'provenance/upstream_CodeFormer_LICENSE'):
                from_v1(relative)
            copy(V1_PACKET + '/protocol.json', 'provenance/v1/protocol.json', V1_PROTOCOL_SHA)
            validation_path = V1_RETURN + '/outputs/adaptation_v1/validation_inputs.json'
            from_return('outputs/adaptation_v1/validation_inputs.json', 'provenance/v1/validation_inputs_v1.json')
            from_return('outputs/adaptation_v1/snapshot_0000/metrics.json', 'provenance/v1/snapshot_0000_metrics.json')
            original = read_json(root / validation_path)
            require(original.get('complete') is True and original.get('cases_count') == 318 and
                    len(original['cases']) == 318, 'Incomplete saved V1 development input manifest')
            reference_by_id = {r['id']: r for r in rows if r['historical_role'] == 'validation'}
            portable = []
            for identifier, prior in sorted(original['cases'].items()):
                pair_source = V1_RETURN + '/' + canonical(prior['pair_record_path'])
                pair = read_json(bounded(root, pair_source))
                ref = reference_by_id.get(pair.get('reference_id'))
                require(ref is not None and pair.get('role') == 'validation' and
                        pair.get('profile') in v1['evaluation']['profiles'] and
                        identifier == 'validation_' + ref['id'] + '_' + pair['profile'] and
                        pair.get('source_sha256') == ref['source_sha256'] and
                        pair.get('reference_manifest_sha256') == bindings['references.json'], 'Saved DEV mapping changed')
                input_destination = 'data/validation_v1/' + identifier + '/input.png'
                pair_destination = 'data/validation_v1/' + identifier + '/pair_record.json'
                require(returned_bindings[canonical(prior['path'])] == prior['sha256'], 'Original VM input hash differs')
                from_return(canonical(prior['path']), input_destination)
                from_return(canonical(prior['pair_record_path']), pair_destination)
                portable.append({'id': identifier, 'input_path': input_destination,
                    'input_sha256': bindings[input_destination], 'pair_record_path': pair_destination,
                    'pair_record_sha256': bindings[pair_destination], 'target_path': ref['existing_target_path'],
                    'target_sha256': ref['existing_target_sha256'],
                    'original_VM_paths': {'input': prior['path'], 'pair_record': prior['pair_record_path']}})
            expected_ids = {'validation_' + ref['id'] + '_' + profile for ref in reference_by_id.values()
                            for profile in v1['evaluation']['profiles']}
            require({r['id'] for r in portable} == expected_ids and len(portable) == 318, 'DEV318 namespace differs')
            manifest = {'format': 'codeformer-adaptation-saved-validation-v1',
                'source_protocol_sha256': V1_PROTOCOL_SHA, 'case_count': 318,
                'retained_original_manifest_path': 'provenance/v1/validation_inputs_v1.json',
                'retained_original_manifest_sha256': bindings['provenance/v1/validation_inputs_v1.json'],
                'items': portable, 'original_VM_paths_are_provenance_only': True,
                'locally_regenerated_development_pixels': False}
            write_json(packet / 'validation_saved_v1.json', manifest)
            bindings['validation_saved_v1.json'] = sha256(packet / 'validation_saved_v1.json')
            explicit = {
                'CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md': 'instructions/CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md',
                READBACK: 'provenance/v2/INDEPENDENT_DESIGN_READBACK_R2.json',
                FEEDBACK: 'instructions/' + FEEDBACK,
                'CCTV_CODEFORMER_ADAPTATION_V1_RETURN_REVIEW.md': 'provenance/v1/CCTV_CODEFORMER_ADAPTATION_V1_RETURN_REVIEW.md',
                'SYSTEM_WORKFLOW_AND_GOAL.md': 'instructions/SYSTEM_WORKFLOW_AND_GOAL.md',
                'PRACTICAL_OUTPUT_SCOPE.md': 'instructions/PRACTICAL_OUTPUT_SCOPE.md',
                V1_REVIEW + '/archive_review/independent_archive_import_v1.json': 'provenance/v1/independent_archive_import_v1.json',
                V1_REVIEW + '/training_review/INDEPENDENT_TRAINING_REVIEW_V1.md': 'provenance/v1/INDEPENDENT_TRAINING_REVIEW_V1.md',
                V1_REVIEW + '/training_review/independent_trajectory_audit_v1.json': 'provenance/v1/independent_trajectory_audit_v1.json',
                V1_REVIEW + '/metrics_review/RETURN_METRICS_REVIEW_V1.md': 'provenance/v1/RETURN_METRICS_REVIEW_V1.md',
                V1_REVIEW + '/metrics_review/actual_return_v1/metrics_audit.json': 'provenance/v1/metrics_audit.json',
                V1_REVIEW + '/metrics_review/secondary_metrics_v1/SSIM_REVIEW_V1.md': 'provenance/v1/SSIM_REVIEW_V1.md',
                V1_REVIEW + '/post_v1_review/POST_V1_DESIGN_REVIEW.md': 'provenance/v1/POST_V1_DESIGN_REVIEW.md',
                'outputs/codeformer_adaptation_v2_design_review/INDEPENDENT_DESIGN_REVIEW_V1.json': 'provenance/v2/INDEPENDENT_DESIGN_REVIEW_V1.json',
            }
            for old, new in explicit.items():
                copy(old, new, DESIGN_SHA if new == 'instructions/CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md' else None)
        receipt = {'complete': True, 'phase': 'inputs_only_not_frozen', 'packet': 'outputs/' + NAME,
            'design_sha256': DESIGN_SHA, 'V1_protocol_sha256': V1_PROTOCOL_SHA, 'files_sha256': bindings,
            'assets': assets, 'copy_bindings': copies, 'TRAIN_references': 391, 'validation_references': 53,
            'saved_validation_cases': 318, 'comparison_cases': 88,
            'generated_files': {'validation_saved_v1.json': bindings['validation_saved_v1.json']},
            'bytes': sum(bounded(packet, p).stat().st_size for p in bindings),
            'protocol_created': False, 'training_started': False, 'model_calls': 0,
            'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0,
            'seconds': time.monotonic() - started}
        require(exact_namespace(packet) == set(bindings), 'Unlisted staged namespace')
        write_json(evidence / 'input_stage.json', receipt)
        print(json.dumps({k: v for k, v in receipt.items() if k not in ('files_sha256', 'assets', 'copy_bindings')}), flush=True)
        return receipt
    except BaseException as error:
        write_json(evidence / 'input_stage_failure.json', {'complete': False, 'error': str(error),
            'traceback': traceback.format_exc(), 'copied_files': len(copies), 'originals_and_stage_retained': True,
            'training_started': False, 'model_calls': 0, 'VM_calls': 0})
        raise


@contextmanager
def no_neural_imports():
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        require(name.split('.')[0] not in {'torch', 'tensorflow', 'numpy', 'PIL', 'cv2', 'lpips', 'torchvision'},
                'Static packaging must not import a neural/image runtime: ' + name)
        return original(name, *args, **kwargs)
    builtins.__import__ = guarded
    try:
        yield
    finally:
        builtins.__import__ = original


def load_worker(packet):
    path = bounded(packet, 'research/codeformer_adaptation_v2/worker.py')
    ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 10))
    with no_neural_imports():
        spec = importlib.util.spec_from_file_location('reviewed_v2_packet_static_worker', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def shell_runner():
    return ('#!/usr/bin/env bash\nset -euo pipefail\n'
            'cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"\n'
            'PROTOCOL_SHA="${1:?Pass the frozen V2 protocol SHA256}"\n'
            'exec python -B -u scripts/supervise_codeformer_adaptation_v2.py '
            '--root . --protocol-sha "$PROTOCOL_SHA"\n').encode('utf-8')


def storage_ledger(protocol, companion_bytes):
    """Shape/finite-count planning bound; runtime still measures actual bytes.

    V2 retains the same model/Adam tensor shapes as audited V1. A retained
    incomplete state replaces one missing completed state: the worker refuses a
    second temporary checkpoint and will not create a stopped checkpoint while
    that first temporary exists. It is never added to five completed states.
    """
    limits = protocol['limits']
    require(companion_bytes <= 16 * 1024**2, 'Export companions exceed independently planned16MiB allowance')
    # V1 complete serializations plus1MiB per checkpoint for added sampler/count
    # metadata. Actual writes remain capped and reservation-checked by worker.
    initial_state = 376629621 + 1024**2
    trained_state = 855224120 + 1024**2
    completed_states = initial_state + 4 * trained_state
    one_partial_state = initial_state + 3 * trained_state + limits['checkpoint_temporary_bytes']
    components = {
        'states_complete_or_one_replacing_incomplete': max(completed_states, one_partial_state),
        'snapshot_RAW_NPY_with128byte_headers': 1624 * (3 * 256 * 256 * 4 + 128),
        'snapshot_PNG_perfile_cap': 1624 * limits['PNG_file_bytes'],
        'all_training_input_PNG_perfile_cap': 2400 * limits['PNG_file_bytes'],
        'teacher_anchor_FP32_arrays_with256byte_header_allowance': 600 *
            (3 * 256 * 256 * 4 + 256 * 1024 * 4 + 2 * 256),
        'first_example_arrays_and_PNG_allowance': 2 * 1024**2,
        'bounded_output_JSON_logs_records_and_exportmanifest': limits['metadata_bytes'] + 4 * 1024**2,
        'source_notice_manifest_companion_allowance': 16 * 1024**2,
        'external_trainer_log_and_protocol_allowance': 16 * 1024**2,
    }
    result_upper = sum(components.values())
    # <10k planned files;12k reserves incomplete/error and explicit companions.
    member_upper = 12000
    tar_upper = result_upper + member_upper * (4096 + 511) + 1024**2
    archive_upper = tar_upper + (tar_upper // 1000 + 1) + 1024**2
    require(result_upper <= limits['result_payload_bytes'], 'Prospective full evidence exceeds7GiB; do not freeze')
    require(archive_upper <= limits['archive_bytes'], 'Prospective TAR/gzip evidence exceeds7.1GiB; do not freeze')
    allocation = limits['result_payload_bytes'] + limits['archive_bytes'] + \
        limits['checkpoint_temporary_bytes'] + limits['reserve_free_bytes']
    require(allocation <= limits['initial_free_bytes'], 'Declared post-install space cannot cover finite allocation')
    return {'format': 'codeformer-v2-storage-ledger-v1', 'complete': True,
        'components_upper_bytes': components, 'prospective_result_upper_bytes': result_upper,
        'prospective_archive_upper_bytes': archive_upper, 'archive_member_upper': member_upper,
        'current_companion_bytes': companion_bytes, 'completed_state_scenario_bytes': completed_states,
        'one_replacing_partial_state_scenario_bytes': one_partial_state,
        'no_second_incomplete_checkpoint_required': True,
        'HQ_teacher_cache': '391 FP32 target-code entries in CPU RAM, no disk cache serialization',
        'V1_serialization_basis': {'initial_bytes': 376629621, 'trained_bytes': 855224120,
            'extra_per_state_metadata_allowance': 1024**2},
        'runtime_actual_byte_checks_remain_required': True,
        'startup_allocation_plus_reserve_bytes': allocation,
        'initial_free_bytes': limits['initial_free_bytes'], 'training_started': False,
        'neural_calls': 0, 'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0}


def finalize(root, readiness_path, readiness_sha):
    started = time.monotonic()
    root = Path(root).resolve(strict=True)
    packet, evidence = root / 'outputs' / NAME, root / EVIDENCE
    require(not (root / 'outputs' / EXECUTION_ARCHIVE).exists() and not (packet / 'protocol.json').exists(),
            'Do not revise a frozen V2 protocol/archive')
    readiness_path = bounded(root, readiness_path)
    require(HEX.fullmatch(readiness_sha) and sha256(readiness_path) == readiness_sha, 'Readiness hash differs')
    readiness = read_json(readiness_path)
    require(readiness.get('complete') is True and readiness.get('approved_for_protocol_freeze') is True and
            readiness.get('design_sha256') == DESIGN_SHA and readiness.get('training_authorized') is False,
            'Independent reviewed sources/tests must approve protocol freeze only')
    source_hashes = readiness.get('source_sha256', {})
    require(SOURCE_REQUIRED <= set(source_hashes), 'Incomplete independently reviewed executable sources')
    staged = read_json(bounded(root, EVIDENCE + '/input_stage.json'))
    bindings, copies, assets = deepcopy(staged['files_sha256']), deepcopy(staged['copy_bindings']), deepcopy(staged['assets'])
    require(exact_namespace(packet) == set(bindings), 'Inputs-only stage has unrecorded changes')
    for name, expected in bindings.items():
        require(sha256(bounded(packet, name)) == expected, 'Staged input changed: ' + name)
    for name, row in copies.items():
        require(sha256(bounded(root, row['original'])) == row['sha256'], 'Retained original changed before freeze')
    require(sha256(bounded(root, 'CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md')) == DESIGN_SHA, 'Design changed before freeze')
    with (evidence / 'finalize_journal.jsonl').open('x', encoding='utf-8', newline='\n') as journal:
        for relative, expected in sorted(source_hashes.items()):
            canonical(relative)
            require(HEX.fullmatch(expected), 'Malformed reviewed source hash')
            if relative in bindings:
                require(expected == bindings[relative] and sha256(bounded(root, relative)) == expected,
                        'Reviewed immutable dependency changed')
                continue
            require(relative.startswith(('research/codeformer_adaptation_v2/', 'tests/')) or relative in SOURCE_REQUIRED,
                    'Unexpected source-readiness executable namespace')
            require(PurePosixPath(relative).suffix == '.py', 'Readiness executable must be Python source')
            ast.parse(bounded(root, relative).read_text(encoding='utf-8'), feature_version=(3, 10))
            copy_exact(root, packet, relative, relative, expected, bindings, copies, journal)
        copy_exact(root, packet, readiness_path.relative_to(root).as_posix(), 'provenance/v2/source_readiness.json',
                   readiness_sha, bindings, copies, journal)
    runner = shell_runner()
    write_new(packet / 'scripts/run_adaptation_v2.sh', runner)
    bindings['scripts/run_adaptation_v2.sh'] = sha256(packet / 'scripts/run_adaptation_v2.sh')
    readme = ('# CodeFormer V2 — prepared manual pilot, not a trained result\n\n'
        'The untouched initializer starts a changed 600-update joint data/loss/schedule pilot.\n'
        '391 TRAIN and 53 validation roles remain separate. All 318 development inputs are\n'
        'the exact saved V1 VM bytes; 88 comparison cases remain fixed, 48 native unpaired.\n'
        'Require 20 GiB free after installation; 4 GiB reserve; 22 GiB reserved CUDA ceiling.\n'
        'Human launch only inside tmux using the frozen protocol hash. No automatic resume,\n'
        'app switch or qualification follows from training or export completion.\n'
        'All V1 stops/checkpoints remain preserved. The user liked many V1 outputs but requested\n'
        'brightness correction; that feedback does not waive preservation or final evaluation.\n'
        'See instructions/CCTV_CODEFORMER_ADAPTATION_V2_DESIGN.md and\n'
        'provenance/v2/source_readiness.json before launch.\n').encode('utf-8')
    write_new(packet / 'READ_ME_FIRST.md', readme)
    bindings['READ_ME_FIRST.md'] = sha256(packet / 'READ_ME_FIRST.md')
    worker = load_worker(packet)
    protocol = deepcopy(worker.DEFAULT_PROTOCOL)
    protocol.update(assets=assets, files_sha256=bindings,
        export_files=[name for name in sorted(bindings) if export_companion(name)],
        validation_saved={'manifest': 'validation_saved_v1.json', 'sha256': bindings['validation_saved_v1.json'],
            'source_protocol_sha256': V1_PROTOCOL_SHA, 'case_count': 318},
        prior_baseline={'metrics': 'provenance/v1/snapshot_0000_metrics.json',
            'sha256': bindings['provenance/v1/snapshot_0000_metrics.json'], 'source_protocol_sha256': V1_PROTOCOL_SHA},
        provenance={'source_readiness_sha256': readiness_sha, 'V1_inputs_preserved': True,
            'source_copy_map_retained_outside_packet': EVIDENCE + '/finalized.json',
            'pretraining_identity_overlap_unknown': True, 'repeated_development_exposure': True},
        scientific_qualification={'training_completion_not_qualification': True,
            'user_acceptance_required': True, 'independent_final_review_required': True,
            'native_unpaired': True, 'no_automatic_app_promotion': True})
    companion_bytes = sum((packet / name).stat().st_size for name in protocol['export_files'])
    ledger = storage_ledger(protocol, companion_bytes)
    write_json(packet / 'provenance/v2/storage_ledger_v1.json', ledger)
    bindings['provenance/v2/storage_ledger_v1.json'] = sha256(packet / 'provenance/v2/storage_ledger_v1.json')
    protocol['export_files'].append('provenance/v2/storage_ledger_v1.json')
    with no_neural_imports():
        worker.validate_protocol(protocol)
    write_json(packet / 'protocol.json', protocol)
    pin = sha256(packet / 'protocol.json')
    with no_neural_imports():
        _, verification = worker.verify_transfer(packet, pin)
    require(exact_namespace(packet) == set(bindings) | {'protocol.json'}, 'Unexpected final staged namespace')
    receipt = {'complete': True, 'phase': 'reviewed_protocol_frozen_not_archive_sealed',
        'packet': 'outputs/' + NAME, 'protocol_sha256': pin, 'readiness_sha256': readiness_sha,
        'readiness_path': readiness_path.relative_to(root).as_posix(), 'design_sha256': DESIGN_SHA,
        'copy_bindings': copies, 'files': len(bindings) + 1,
        'bytes': sum((packet / p).stat().st_size for p in exact_namespace(packet)),
        'verification': verification, 'training_started': False, 'model_calls': 0,
        'storage_ledger': ledger,
        'gradient_queries': 0, 'optimizer_updates': 0, 'VM_calls': 0,
        'seconds': time.monotonic() - started}
    write_json(evidence / 'finalized.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'copy_bindings'}), flush=True)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--finalize', action='store_true')
    parser.add_argument('--readiness')
    parser.add_argument('--readiness-sha')
    args = parser.parse_args()
    if args.finalize:
        require(bool(args.readiness) and bool(args.readiness_sha), 'Finalization requires exact independent readiness receipt')
        finalize(args.root, args.readiness, args.readiness_sha)
    else:
        require(args.readiness is None and args.readiness_sha is None, 'Readiness fields only apply to finalization')
        stage_inputs(args.root)


if __name__ == '__main__':
    main()
