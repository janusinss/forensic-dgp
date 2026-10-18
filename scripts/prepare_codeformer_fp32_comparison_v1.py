"""Stage a finite FP32 comparison overlay; never launch a model or a VM.

Local --stage-inputs creates CPU parity/provenance files only. The separately
reviewed root sealer publishes protocol/archive later. On the manual Linux VM,
--stage-dependencies copies only explicitly pinned immutable sibling inputs;
it never copies old run outputs, overwrites files, uses hardlinks or trains.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'cctv_codeformer_fp32_comparison_v1_vm'
PARENT = 'cctv_codeformer_adaptation_v2_vm'
PARENT_PROTOCOL_SHA = '029291a2d58d7bc8b33fb99b86774e13db4e5c098bb8af5f508fdc5e24b20183'
DESIGN = 'CCTV_CODEFORMER_FP32_COMPARISON_V1_DESIGN.md'
DESIGN_SHA = '44da9c3afa4b0f0209ad762aa0f52c999f83054014bde076e39d46816f8a66fb'
V2_DESIGN_SHA = 'a7105b577863e22c0bc35ce12d310234a510eec41c8115ebeec2e9b46bb511b9'
RETURN_DIR = 'outputs/codeformer_adaptation_v2_return'
REVIEW_DIR = 'outputs/codeformer_adaptation_v2_return_review'
EVIDENCE = 'outputs/codeformer_fp32_comparison_v1_preparation'
PARITY = 'training_parity_v1.json'
FORMAT = 'codeformer-fp32-comparison-finite-manual-v1'
HEX = re.compile(r'^[0-9a-f]{64}$')
DEVICE = re.compile(r'^(con|prn|aux|nul|conin\$|conout\$|com[1-9¹²³]|lpt[1-9¹²³])(?:\.|$)', re.I)
V2_EXPORT_MANIFEST = 'outputs/adaptation_v2/export_manifest_20261015T140245Z-40fec745.json'
V2_EXPORT_MANIFEST_SHA = '7ed6bd2bcda8cc245c9229c5568a3264bc01fd8034f599f5aca4a3462bdcd169'
DESIGN_READBACK = 'outputs/codeformer_path_diagnostic_v1_return_review/independent_report_design_readback_state_v1.json'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(name):
    require(isinstance(name, str) and bool(name), 'Empty dependency path')
    p = PurePosixPath(name)
    require(p.parts and not p.is_absolute() and p.as_posix() == name and
            all(x not in ('', '.', '..') for x in p.parts), 'Noncanonical dependency path')
    require(not any(x in name for x in '\\:"<>|?*') and all(
        not DEVICE.match(x) and not x.endswith((' ', '.')) and all(ord(c) >= 32 for c in x)
        for x in p.parts), 'Nonportable dependency path')
    return name


def no_links(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists() or item.is_symlink():
            info = item.lstat()
            require(not item.is_symlink() and not getattr(info, 'st_file_attributes', 0) & 1024,
                    'Symlink/reparse path forbidden: ' + str(item))
    return path


def bounded(root, name, exists=True):
    path = no_links(Path(root) / canonical(name))
    require(path.is_relative_to(no_links(root)), 'Dependency escapes root')
    if exists:
        info = path.lstat(); require(stat.S_ISREG(info.st_mode), 'Expected regular file: ' + name)
    return path


def sha(path):
    path = no_links(path); before = path.stat()
    require(stat.S_ISREG(before.st_mode), 'Expected regular hashing input')
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024**2), b''):
            digest.update(block)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns, before.st_ino) ==
            (after.st_size, after.st_mtime_ns, after.st_ino), 'Input changed during hash')
    return digest.hexdigest()


def read_json(path):
    def pairs(rows):
        d = {}
        for k, v in rows:
            require(k not in d, 'Duplicate JSON key'); d[k] = v
        return d
    payload = no_links(path).read_bytes()
    require(len(payload) <= 16 * 1024**2, 'Metadata exceeds finite16MiB bound')
    return json.loads(payload.decode('utf-8-sig'), object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def write_bytes(path, payload):
    path = no_links(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as output:
        output.write(payload); output.flush(); os.fsync(output.fileno())


def write_json(path, row):
    write_bytes(path, (json.dumps(row, indent=2, allow_nan=False) + '\n').encode('utf-8'))


def dependency_names(parent_protocol):
    """Immutable inputs only: no historical worker, launcher, tests or outputs."""
    names = set(parent_protocol['files_sha256'])
    return sorted(name for name in names if name != 'READ_ME_FIRST.md' and
                  name != 'research/codeformer_adaptation_v2/worker.py' and
                  not name.startswith(('scripts/', 'tests/', 'outputs/')))


def exact_files(root):
    result, folded = set(), set()
    for folder, dirs, names in os.walk(root, followlinks=False):
        for name in dirs:
            no_links(Path(folder) / name)
        for name in names:
            relative = (Path(folder) / name).relative_to(root).as_posix()
            bounded(root, relative)
            require(relative.casefold() not in folded, 'Case-colliding input namespace')
            result.add(relative); folded.add(relative.casefold())
    return result


def protocol_template(parent, staged, *, parity_sha):
    p = deepcopy(parent)
    p.update(format=FORMAT, parent_dir=PARENT, parent_protocol_sha256=PARENT_PROTOCOL_SHA,
             outputs='outputs/fp32_comparison_v1', export_prefix='cctv-codeformer-fp32-comparison-v1')
    p['design'] = {'path': 'instructions/' + DESIGN, 'sha256': DESIGN_SHA,
                  'readback_path': 'provenance/fp32_comparison_v1/INDEPENDENT_DESIGN_READBACK_V1.json'}
    t = p['training']
    t.update(max_updates=50, schedule_horizon_updates=600, data_horizon_updates=600,
             snapshots=[0, 50], autocast='disabled-FP32', max_backward_calls=200)
    t['call_caps'].update(optimizer_step_attempts=50, backward_calls=200, candidate_training_forwards=200,
        perceptual_training_forwards=148, behavior_teacher_forwards=50, snapshot_forwards=812,
        training_input_writes=200)
    p['limits'].update(projected_training_seconds=900, training_seconds=900, worker_seconds=3000,
        initial_free_bytes=16 * 1024**3, result_payload_bytes=7 * 1024**3 // 2,
        archive_bytes=3865470566)
    p['training_parity'] = {'path': PARITY, 'sha256': parity_sha, 'cases': 200, 'updates': 50,
                            'behavior_anchors': 50, 'source_protocol_sha256': PARENT_PROTOCOL_SHA}
    p['source_data_design_sha256'] = V2_DESIGN_SHA
    p['dependency_files_sha256'] = deepcopy(staged['dependency_files_sha256'])
    p['dependency_files_bytes'] = deepcopy(staged['dependency_files_bytes'])
    p['parent_files_sha256'] = deepcopy(staged['dependency_files_sha256'])
    p['files_sha256'] = {**staged['dependency_files_sha256'], **staged['files_sha256']}
    p['export_files'] = sorted(name for name in parent['export_files'] if name in p['dependency_files_sha256'])
    # Input PNG/NPY and full historical training JSONL remain immutable packet
    # inputs, referenced by the exported parity manifest rather than re-exported.
    p['export_files'] += sorted(name for name in staged['files_sha256'] if
        Path(name).suffix.lower() in {'.py', '.md', '.txt', '.json', '.csv', '.yaml', '.yml', '.sh', '.sha256'})
    p['provenance'].update(parent_protocol_sha256=PARENT_PROTOCOL_SHA,
        source_design_sha256=V2_DESIGN_SHA, old_V2_stopped_checkpoint_not_initializer=True,
        exact_HQ_cache_equality_to_unexported_old_cache_not_claimed=True,
        stage_receipt_external=EVIDENCE + '/stage_inputs.json')
    return p


def stage_inputs(root=ROOT):
    root = no_links(root).resolve(strict=True)
    parent = root / 'outputs' / PARENT; returned = root / RETURN_DIR
    packet = root / 'outputs' / PACKAGE; evidence = root / EVIDENCE
    require(not packet.exists() and not (evidence / 'stage_inputs.json').exists() and
            not (evidence / 'stage_failure.json').exists(), 'Preserve prior stage; no automatic repeat')
    require(sha(parent / 'protocol.json') == sha(returned / 'protocol.json') == PARENT_PROTOCOL_SHA,
            'Exact parent/returned V2 protocol differs')
    require(sha(root / DESIGN) == DESIGN_SHA, 'Reviewed FP32 design changed')
    p = read_json(parent / 'protocol.json')
    require(sha(returned / V2_EXPORT_MANIFEST) == V2_EXPORT_MANIFEST_SHA, 'Retained V2 export manifest changed')
    export = read_json(returned / V2_EXPORT_MANIFEST)
    import_receipt = read_json(root / REVIEW_DIR / 'archive_import_v1.json')
    trajectory = read_json(root / REVIEW_DIR / 'trajectory_v2/trajectory_audit.json')
    require(import_receipt['complete'] is True and trajectory['audit_finished'] is True and
            trajectory['evidence_consistency_pass'] is True and trajectory['completed_updates'] == 50,
            'Closed successful V2 transport/trajectory audit required')
    readback = read_json(root / DESIGN_READBACK)
    require(readback['complete'] is True and readback['approved'] is True and
            readback['reviewed_files_sha256'][DESIGN] == DESIGN_SHA, 'Approved new design readback required')
    history_path = returned / 'outputs/adaptation_v2/training.jsonl'
    runtime_path = returned / 'outputs/adaptation_v2/runtime.json'
    require(sha(history_path) == export['files_sha256']['outputs/adaptation_v2/training.jsonl'] and
            sha(runtime_path) == export['files_sha256']['outputs/adaptation_v2/runtime.json'], 'V2 history/runtime changed')
    history = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines()]
    require(len(history) == 50 and [r['update'] for r in history] == list(range(1, 51)) and
            all(row['optimizer_committed'] is True and len(row['microbatches']) == 4 for row in history),
            'Exactly50 committed parent updates required')
    refs = read_json(parent / p['data_manifest'])['references']; by_id = {r['id']: r for r in refs}
    dependencies = dependency_names(p)
    dep_hashes, dep_sizes = {}, {}
    # Verify the complete dependency namespace before staging any new file.
    for name in dependencies:
        source = bounded(parent, name); expected = p['files_sha256'][name]
        require(sha(source) == expected, 'Frozen parent dependency changed: ' + name)
        dep_hashes[name] = expected; dep_sizes[name] = source.stat().st_size
    copied = []
    for row in history:
        for slot, sample in enumerate(row['microbatches'], 1):
            pair = sample['pair_record']; ref = by_id[sample['id']]
            require(pair['role'] == ref['historical_role'] == 'train' and pair['reference_id'] == ref['id'] and
                    pair['update'] == row['update'] and pair['slot'] == sample['slot'] == slot and
                    pair['source_sha256'] == ref['source_sha256'] and
                    pair['reference_manifest_sha256'] == p['files_sha256'][p['data_manifest']],
                    'Exact TRAIN/source/role/slot binding differs')
            record_path = str(PurePosixPath(pair['saved_input_path']).with_name('pair_record.json'))
            source = bounded(returned, pair['saved_input_path']); record = bounded(returned, record_path)
            require(sha(source) == pair['saved_input_sha256'] == export['files_sha256'][pair['saved_input_path']] and
                    sha(record) == export['files_sha256'][record_path] and read_json(record) == pair,
                    'Retained TRAIN image/record mismatch')
            if slot == 3:
                a = sample['behavior_anchor']
                require(a['update'] == row['update'] and a['reference_id'] == ref['id'] and a['pair_record'] == pair,
                        'Behavior anchor input binding mismatch')
                for label in ('delivered', 'logits'):
                    path = bounded(returned, a[label + '_path'])
                    require(sha(path) == a[label + '_sha256'] == export['files_sha256'][a[label + '_path']],
                            'Retained behavior anchor mismatch')
            copied.append((row, sample, ref, source, record, record_path))
    require(len(copied) == 200 and len({r['id'] for _, _, r, _, _, _ in copied}) == 100,
            'Expected200 observations/100 reference IDs, not an identity claim')
    packet.mkdir(); evidence.mkdir(parents=True, exist_ok=True)
    files, sizes, copy_map, cases, anchors, updates = {}, {}, {}, [], [], []
    def copy(source, name, expected):
        require(name not in files and name not in dep_hashes, 'Overlay/dependency collision')
        require(sha(source) == expected, 'Overlay source changed')
        target = bounded(packet, name, exists=False); write_bytes(target, source.read_bytes())
        require(sha(target) == expected, 'Overlay destination mismatch')
        files[name] = expected; sizes[name] = target.stat().st_size
        copy_map[name] = {'source': source.relative_to(root).as_posix(), 'sha256': expected, 'bytes': sizes[name]}
    try:
        for row, sample, ref, source, record, old_record in copied:
            update, slot, pair = row['update'], sample['slot'], sample['pair_record']
            base = f'parity/training_inputs/update_{update:04d}/slot_{slot}/'
            inp, rec = base + 'input.png', base + 'pair_record.json'
            copy(source, inp, pair['saved_input_sha256']); copy(record, rec, export['files_sha256'][old_record])
            cases.append({'update': update, 'slot': slot, 'reference_id': ref['id'], 'input_path': inp,
                'input_sha256': files[inp], 'pair_record_path': rec, 'pair_record_sha256': files[rec],
                'input256_rgb8_sha256': pair['input256_rgb8_sha256'],
                'input256_float32_sha256': pair['input256_float32_sha256'],
                'clean512_rgb8_sha256': pair['clean512_rgb8_sha256'],
                'clean512_float32_sha256': pair['clean512_float32_sha256'],
                'clean_source_path': ref['source_path'], 'source_sha256': ref['source_sha256'],
                'image_eligible': sample['image_eligible'], 'profile': pair['profile']})
            if slot == 3:
                anchor = sample['behavior_anchor']; item = {'update': update, 'slot': 3, 'reference_id': ref['id'],
                    'input256_float32_sha256': anchor['input256_float32_sha256'], 'pair_record_path': rec,
                    'pair_record_sha256': files[rec], 'teacher_state_sha256': anchor['teacher_state_sha256'],
                    'teacher_FP32_eval_frozen': anchor['teacher_FP32_eval_frozen']}
                for label, filename in (('delivered', 'delivered256.npy'), ('logits', 'logits.npy')):
                    name = f'parity/behavior/update_{update:04d}/' + filename
                    copy(bounded(returned, anchor[label + '_path']), name, anchor[label + '_sha256'])
                    item[label + '_path'] = name; item[label + '_sha256'] = files[name]
                anchors.append(item)
        updates = [{'update': row['update'], 'group_lrs': row['group_lrs'], 'TRAIN_selection': row['TRAIN_selection']}
                   for row in history]
        provenance = {
            'history': ('provenance/fp32_comparison_v1/V2_training_first50.jsonl', history_path),
            'runtime': ('provenance/fp32_comparison_v1/V2_runtime.json', runtime_path),
            'protocol': ('provenance/fp32_comparison_v1/V2_protocol.json', parent / 'protocol.json'),
            'export_manifest': ('provenance/fp32_comparison_v1/V2_export_manifest.json', returned / V2_EXPORT_MANIFEST),
            'import_audit': ('provenance/fp32_comparison_v1/V2_archive_import.json', root / REVIEW_DIR / 'archive_import_v1.json'),
            'trajectory_audit': ('provenance/fp32_comparison_v1/V2_trajectory_audit.json', root / REVIEW_DIR / 'trajectory_v2/trajectory_audit.json'),
            'publication_readback': ('provenance/fp32_comparison_v1/V2_publication_readback.json', root / REVIEW_DIR / 'publication_readback_v1.json'),
        }
        for name, source in provenance.values():
            copy(source, name, sha(source))
        for relative, name in (
            ('outputs/cctv_native_development_v2/frozen_subset.json',
             'provenance/native/qmul_frozen_subset.json'),
            ('outputs/cctv_chokepoint_native_development_v1/selection_plan.json',
             'provenance/native/chokepoint_selection_plan.json')):
            source = bounded(root, relative)
            copy(source, name, sha(source))
        parity = {'format': 'codeformer-v2-first50-training-parity-v1', 'source_protocol_sha256': PARENT_PROTOCOL_SHA,
            'source_export_manifest_sha256': V2_EXPORT_MANIFEST_SHA,
            'source_history': provenance['history'][0], 'source_history_sha256': files[provenance['history'][0]],
            'source_runtime': provenance['runtime'][0], 'source_runtime_sha256': files[provenance['runtime'][0]],
            'cases': cases, 'behavior_anchors': anchors, 'updates': updates,
            'runtime_expected': read_json(runtime_path), 'reference_ID_count': 100,
            'reference_ID_count_is_not_unique_people_count': True, 'TRAIN_only': True,
            'HQ_cache_old_new_tensor_equality_unproven': True,
            'no_local_neural_gradient_optimizer_or_VM_calls': True}
        write_json(packet / PARITY, parity); files[PARITY] = sha(packet / PARITY); sizes[PARITY] = (packet / PARITY).stat().st_size
        copy(root / DESIGN, 'instructions/' + DESIGN, DESIGN_SHA)
        copy(root / DESIGN_READBACK, 'provenance/fp32_comparison_v1/INDEPENDENT_DESIGN_READBACK_V1.json', sha(root / DESIGN_READBACK))
        staged = {'complete': True, 'phase': 'inputs_only_not_sealed_not_runnable',
            'package_dir': PACKAGE, 'parent_dir': PARENT, 'parent_protocol_sha256': PARENT_PROTOCOL_SHA,
            'design_sha256': DESIGN_SHA, 'training_parity_path': PARITY, 'training_parity_sha256': files[PARITY],
            'files_sha256': files, 'files_bytes': sizes, 'copy_bindings': copy_map,
            'dependency_files_sha256': dep_hashes, 'dependency_files_bytes': dep_sizes,
            'overlay_files': len(files), 'overlay_bytes': sum(sizes.values()),
            'dependency_files': len(dep_hashes), 'dependency_bytes': sum(dep_sizes.values()),
            'cases': 200, 'reference_IDs_not_people': 100, 'behavior_anchors': 50,
            'optimizer_updates': 0, 'neural_calls': 0, 'VM_calls': 0,
            'new_actual_training_authorized': False}
        require(exact_files(packet) == set(files), 'Unlisted staged overlay')
        write_json(evidence / 'stage_inputs.json', staged)
        write_json(evidence / 'protocol_template_unsealed.json', protocol_template(p, staged, parity_sha=files[PARITY]))
        return staged
    except BaseException as exc:
        write_json(evidence / 'stage_failure.json', {'complete': False, 'message': repr(exc),
            'staging_retained': True, 'no_retry': True, 'local_neural_VM_optimizer_calls': 0})
        raise


def manual_dependency_roots(root, *, system=None, home=None):
    system = platform.system() if system is None else system
    require(system == 'Linux', 'Dependency installation is manual on the Linux VM')
    home = Path.home() if home is None else Path(home)
    root = no_links(root).resolve(strict=True)
    expected = no_links(home / 'forensic-dgp' / PACKAGE)
    require(root == expected.resolve(strict=True), 'Use exact new comparison directory under ~/forensic-dgp')
    parent = no_links(root.parent / PARENT).resolve(strict=True)
    require(parent.parent == root.parent and parent != root, 'Sibling dependency root differs')
    return root, parent


def validate_dependency_plan(root, parent, protocol_sha):
    require(HEX.fullmatch(protocol_sha) and sha(root / 'protocol.json') == protocol_sha,
            'Pinned new comparison protocol required')
    p = read_json(root / 'protocol.json')
    require(p['format'] == FORMAT and p['parent_dir'] == PARENT and
            p['parent_protocol_sha256'] == PARENT_PROTOCOL_SHA and
            sha(parent / 'protocol.json') == PARENT_PROTOCOL_SHA, 'Sibling original protocol mismatch')
    pp = read_json(parent / 'protocol.json')
    deps, sizes = p['dependency_files_sha256'], p['dependency_files_bytes']
    require(set(deps) == set(sizes) == set(dependency_names(pp)), 'Complete dependency plan differs')
    files = p['files_sha256']; require(set(deps) <= set(files), 'Dependencies missing from complete file map')
    folded = set()
    for name, expected in files.items():
        canonical(name); require(HEX.fullmatch(expected) and name.casefold() not in folded, 'Malformed/colliding map')
        folded.add(name.casefold())
        if name in deps:
            source = bounded(parent, name)
            require(expected == deps[name] == pp['files_sha256'][name] and
                    type(sizes[name]) is int and source.stat().st_size == sizes[name] and sha(source) == expected,
                    'Frozen dependency content differs: ' + name)
            require(not bounded(root, name, exists=False).exists(), 'Refuse dependency overwrite: ' + name)
        else:
            require(sha(bounded(root, name)) == expected, 'Overlay content differs: ' + name)
    require(not any(name.startswith('outputs/') for name in deps), 'Historical outputs cannot be copied')
    return p


def stage_dependencies(root, protocol_sha, *, system=None, home=None):
    started = time.monotonic()
    root, parent = manual_dependency_roots(root, system=system, home=home)
    require(not (root / 'outputs').exists(), 'Existing run evidence: preserve and do not install/relaunch')
    receipt = root / 'dependency_install_receipt.json'; journal_path = root / 'dependency_install_journal.jsonl'
    require(not receipt.exists() and not journal_path.exists(), 'Preserve prior/partial dependency install; no retry')
    p = validate_dependency_plan(root, parent, protocol_sha)
    dependency_bytes = sum(p['dependency_files_bytes'].values())
    require(shutil.disk_usage(root).free >= dependency_bytes + p['limits']['initial_free_bytes'],
            'Require dependency copy bytes plus16GiB free before installation')
    with journal_path.open('x', encoding='utf-8', newline='\n') as log:
        def event(row):
            log.write(json.dumps(row) + '\n'); log.flush(); os.fsync(log.fileno())
        event({'event': 'validated_all_dependencies_before_copy', 'files': len(p['dependency_files_sha256']),
               'bytes': dependency_bytes, 'protocol_sha256': protocol_sha})
        try:
            for name, expected in sorted(p['dependency_files_sha256'].items()):
                require(time.monotonic() - started < 900, 'Finite dependency installation900s exceeded')
                require(shutil.disk_usage(root).free >= p['limits']['initial_free_bytes'], 'Retain16GiB install reserve')
                source, destination = bounded(parent, name), bounded(root, name, exists=False)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with source.open('rb') as inp, destination.open('xb') as out:
                    shutil.copyfileobj(inp, out, 1024**2); out.flush(); os.fsync(out.fileno())
                require(destination.stat().st_size == p['dependency_files_bytes'][name] and
                        sha(destination) == sha(source) == expected, 'Copied dependency differs: ' + name)
                require(destination.stat().st_ino != source.stat().st_ino, 'Hardlink installation forbidden')
                event({'event': 'copied_verified_regular_file', 'path': name, 'sha256': expected,
                       'bytes': destination.stat().st_size})
            require(shutil.disk_usage(root).free >= p['limits']['initial_free_bytes'], 'Require16GiB after install')
            result = {'complete': True, 'protocol_sha256': protocol_sha, 'files': len(p['dependency_files_sha256']),
                'bytes': dependency_bytes, 'parent_protocol_sha256': PARENT_PROTOCOL_SHA,
                'copy_policy': 'verified regular copies; no hardlinks; parent inputs only; no old outputs',
                'seconds': time.monotonic() - started, 'neural_calls': 0, 'optimizer_updates': 0}
            write_json(receipt, result); event({'event': 'dependency_install_complete'})
            return result
        except BaseException as exc:
            event({'event': 'dependency_install_failed', 'error': repr(exc), 'partial_files_retained': True})
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    flags = parser.add_mutually_exclusive_group(required=True)
    flags.add_argument('--stage-inputs', action='store_true')
    flags.add_argument('--stage-dependencies', action='store_true')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--protocol-sha')
    args = parser.parse_args()
    if args.stage_inputs:
        require(args.protocol_sha is None, 'Inputs-only stage has no sealed protocol')
        result = stage_inputs(args.root)
    else:
        require(args.protocol_sha is not None, 'Pinned protocol required for dependency installation')
        result = stage_dependencies(args.root, args.protocol_sha)
    print(json.dumps({k: v for k, v in result.items() if k not in
        {'files_sha256', 'files_bytes', 'copy_bindings', 'dependency_files_sha256', 'dependency_files_bytes'}}, indent=2))


if __name__ == '__main__':
    main()
