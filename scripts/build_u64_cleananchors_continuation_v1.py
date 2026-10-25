"""Seal a new manual-only packet after exact source/test reviews.

This builder performs only file operations and static checks. It does not load
checkpoints, import Torch/NumPy, connect to a VM or run any training/inference.
"""
import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import stat
import os
import re
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'research/codeformer_u64_cleananchors_continuation_v1'
PARENT = ROOT / 'outputs/cctv_codeformer_u96_decoder_readout_v1_vm'
INITIALIZERS = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm'
PREP = ROOT / 'outputs/codeformer_u64_cleananchors_continuation_v1_preparation'
DESIGN = ROOT / 'outputs/codeformer_u64_continuation_diagnosis_v1'
REVIEW = PREP / 'runner_review/complete_runner_source_review_v1.json'
TESTS = PREP / 'root_runner_source_tests_v1.json'
STAGE = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PREFIX = 'cctv-codeformer-u64-cleananchors-continuation-v1'
CACHE_RECEIPT = PREP / 'cached_controls_materialization_v1.json'
CACHE_PEER = PREP / 'independent_cached_controls_readback_v1.json'
ADMISSION = PREP / 'prospective_data_reuse_admission_v1.json'
VISUAL_PLAN = PREP / 'planned_visual_review_v1.json'
SHARED_SOURCES = {
    'research/codeformer_u96_postfusion_v1/__init__.py': (66, 'b294dfcc71d5358daccc4a7e02d475eae39c3c0a24952ef522062653a8ade54a'),
    'research/codeformer_u96_postfusion_v1/model.py': (23240, 'ca2b466161932068c01980bb41fcc08e12ee8f0120f17c8fdf47898f35934aa8'),
    'research/codeformer_u96_spatial_feasibility_v1/__init__.py': (84, '9b6cd2566bb68204c76948c35b8e66c9a83cdc43a198735a8fe8bdd79ed0ea7e'),
    'research/codeformer_u96_spatial_feasibility_v1/worker.py': (72086, 'fae271100642604ae0682ebf52ec163e768fa6882eced87e0c34e587980126b6'),
    'research/codeformer_u96_spatial_feasibility_v1/metrics.py': (17480, '1ec77a3cf3621494d0dbe36f8f46a2a8db06c4006bfab41be639399156e08b05'),
}
MAX_STATIC_BYTES = 512 * 1024**2


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def pin(path, base=ROOT):
    path, base = Path(path).absolute(), Path(base).absolute()
    need(path == path.resolve(strict=True) and base == base.resolve(strict=True) and
         path.is_relative_to(base), 'Canonical source must remain inside its declared base')
    for ancestor in (path.parent, *path.parent.parents, base):
        info = ancestor.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode) and
             not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary source ancestry required')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode) and
         not getattr(before, 'st_file_attributes', 0) & 1024 and before.st_nlink == 1
         and 0 <= before.st_size <= 2*1024**3, 'Exact bounded single-link regular source required')
    digest = sha(path)
    after = path.lstat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed during hash')
    return {'path': path.relative_to(base).as_posix(), 'bytes': after.st_size, 'sha256': digest}


def doc(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON evidence key refused')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON evidence refused')
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def write_new(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def validate_review(review, tests, source_files):
    """No source freezing from a stale green receipt or omitted executable."""
    need(review['complete'] is True and review['source_review_pass'] is True and
         review['unresolved_blockers'] == [], 'Independent complete-runner review required')
    need(tests['complete'] is True and tests['all_tests_pass'] is True and
         tests['neural_gradient_optimizer_training_VM_calls'] == 0,
         'Pure current-source tests required')
    approved = {}
    for record in (review, tests):
        pins = record['sources']
        need(type(pins) is list and len({row['path'] for row in pins}) == len(pins),
             'Unique exact reviewed/tested source bindings required')
        bound = {row['path'] for row in pins}
        need(source_files <= bound, 'Every executable and builder must be both reviewed and tested')
        for row in pins:
            need(pin(ROOT / row['path']) == row, 'Reviewed or tested source changed: ' + row['path'])
            need(row['path'] not in approved or approved[row['path']] == row,
                 'Review/test source bindings disagree')
            approved[row['path']] = row
    return approved


def approved_copy_binding(source, approved, *, expected_content=None):
    """Rejoin the actual copy bytes to approval, not merely a fresh hash."""
    before = pin(source)
    if source.is_relative_to(PACKAGE) or before['path'] in SHARED_SOURCES:
        need(before['path'] in approved and before == approved[before['path']],
             'Executable changed or appeared after review: ' + before['path'])
    if expected_content is not None:
        need((before['bytes'], before['sha256']) ==
             (expected_content['bytes'], expected_content['sha256']),
             'Parent-bound content changed before copy: ' + before['path'])
    return before


def validate_cache(materialized, peer, selection):
    """Bind physical controls without importing producer code or array libraries."""
    need(materialized['complete'] is True and materialized['cache_payload_hashes_exact'] is True
         and materialized['physical_cache_files'] == 180,
         'Complete materialized 180-file cache required')
    need(peer['complete'] is True and peer['independent_cached_controls_readback_pass'] is True
         and peer['materialization_receipt'] == pin(CACHE_RECEIPT)
         and peer['all_actual_cache_bytes_equal_fixed_archive_index'] is True,
         'Independent current physical cache readback required')
    table = materialized['physical_cache_inventory']
    need(type(table) is dict and len(table) == 180
         and sum(row['bytes'] for row in table.values()) == materialized['physical_cache_bytes'],
         'Exact physical-cache namespace and byte accounting required')
    source_scope = doc(ROOT / materialized['scope']['path'])
    need(pin(ROOT / materialized['scope']['path']) == materialized['scope'] and source_scope['complete'] is True,
         'Current retained cache-scope receipt required')
    groups = (('cached_frozen_U96_controls', selection['monitors'] + selection['guards']),
              ('terminal_u64_parity_witnesses', selection['monitors'][:4]))
    selected = set()
    for name, expected in groups:
        rows = materialized[name]
        need([row['id'] for row in rows] == [row['id'] for row in expected]
             and [row['id'] for row in rows] == [row['id'] for row in source_scope[name]],
             'Exact ordered56 controls and four parity witnesses required')
        for row, original, case in zip(rows, source_scope[name], expected):
            need({key: value for key, value in row.items() if key not in ('cache_record', 'cache_outputs', 'cache_materialized')}
                 == {key: value for key, value in original.items() if key != 'cache_materialized'}
                 and row['cache_materialized'] is True and row['split'] == 'TRAIN'
                 and all(row[key] == case[key] for key in ('id', 'reference_id', 'role', 'split')),
                 'Cached row must retain exact admitted case and historical reference')
            for binding, old in ((row['cache_record'], row['record_member']),
                                 (row['cache_outputs']['raw'], row['outputs']['raw']),
                                 (row['cache_outputs']['png'], row['outputs']['png'])):
                need(table.get(old['path']) == binding and pin(ROOT / binding['path']) == binding
                     and (binding['bytes'], binding['sha256']) == (old['bytes'], old['sha256']),
                     'Exact physical cached file differs from retained selected member')
                selected.add(old['path'])
    need(selected == set(table), 'No unbound extra or omitted cached payload')
    cache_root = PREP / 'cached_controls'
    actual = {path.relative_to(ROOT).as_posix() for path in cache_root.rglob('*') if path.is_file()}
    need(actual == {row['path'] for row in table.values()}, 'Physical cache contains an unexpected file')
    return {name: materialized[name] for name, _ in groups}


def included_archive_files(files, omitted):
    """Omit only the two large retained weights; terminal heads stay included."""
    need(type(omitted) is set and len(omitted) == 2 and omitted <= set(files), 'Exact two initializer omissions required')
    included = {name: row for name, row in files.items() if name not in omitted}
    need('weights/terminal_u0064_heads.npz' in included, 'Execution archive must carry learned terminal-u64 heads')
    return included


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true', help='Create the exclusive independently reviewed execution packet')
    args = parser.parse_args()
    need(args.build, 'Use --build only after current independent source review and pure tests')
    need('torch' not in sys.modules and 'numpy' not in sys.modules, 'Static builder only')
    archive = ROOT / 'outputs' / (PREFIX + '-execution.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    need(not any(path.exists() for path in (STAGE, archive, checksum, PREP / 'build_receipt_v1.json')),
         'Prior stage/archives/receipt immutable; retain any partial build')
    need(shutil.disk_usage(ROOT).free >= 3 * 1024**3, 'At least3GiB local build headroom')
    namespace = list(PACKAGE.iterdir())
    need(all(path.is_file() and path.suffix in ('.py', '.sh') for path in namespace), 'Exact ordinary executable package namespace required')
    required = {path.relative_to(ROOT).as_posix() for path in namespace} | set(SHARED_SOURCES)
    required.add(Path(__file__).resolve().relative_to(ROOT).as_posix())
    review, tests = doc(REVIEW), doc(TESTS)
    approved = validate_review(review, tests, required)
    admission = doc(ADMISSION)
    need(admission['complete'] is True and admission['new_continuation_data_reuse_admission'] is True
         and review['continuation_reuse'] == pin(ADMISSION),
         'Independent source review must bind the explicit prospective TRAIN reuse admission')
    visual = doc(VISUAL_PLAN)
    need(visual['complete'] is True and review['visual_plan'] == pin(VISUAL_PLAN),
         'Independent source review must bind the prospective visual-review plan')
    for relative, fixed in SHARED_SOURCES.items():
        actual = pin(ROOT / relative)
        need((actual['bytes'], actual['sha256']) == fixed, 'Pinned retained architecture/storage/metrics helper changed')
    sys.path.insert(0, str(ROOT))
    from research.codeformer_u64_cleananchors_continuation_v1 import contract, packet
    for path in PACKAGE.glob('*.py'):
        ast.parse(path.read_bytes(), feature_version=(3, 10))
    need(all(b'\r' not in path.read_bytes() for path in PACKAGE.iterdir()), 'New executable sources require LF')
    parent_path = PARENT / 'protocol.json'
    parent_raw = parent_path.read_bytes()
    selection = contract.select_parent_cases(parent_raw)
    parent = doc(parent_path)
    materialized, cache_peer = doc(CACHE_RECEIPT), doc(CACHE_PEER)
    cache_rows = validate_cache(materialized, cache_peer, selection)
    need(admission['parent_protocol'] == pin(parent_path), 'Prospective reuse must bind the exact parent protocol')
    for row in parent['model'].values():
        need(pin(INITIALIZERS / row['path'], INITIALIZERS) == row, 'Exact retained R2/U96 initializer bytes required')
    head_source = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_inference_export/terminal_u0064_heads.npz'
    head_expected = {'bytes': contract.HEAD_BYTES, 'sha256': contract.HEAD_FILE_SHA256}
    need({key: pin(head_source)[key] for key in ('bytes', 'sha256')} == head_expected, 'Exact terminal-u64 head bytes required')
    historical_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit/saved_pixel_audit_v1.json'
    need((pin(historical_path)['bytes'], pin(historical_path)['sha256']) == packet.HISTORICAL_PIXEL_AUDIT_FIXED,
         'Exact retained u64 pixel/gate audit required')
    historical_index_path = ROOT / 'outputs/codeformer_u96_postfusion_smoothanchors_lr_v1_return_audit/archive_index_v1.json'
    historical_index_pin = pin(historical_index_path)
    need((historical_index_pin['bytes'], historical_index_pin['sha256']) == packet.HISTORICAL_ARCHIVE_INDEX_FIXED,
         'Exact retained u64 archive index required')
    STAGE.mkdir()
    copies = []

    def copy(source, relative, *, expected_content=None):
        need(type(relative) is str and packet.relative(relative).as_posix() == relative, 'Canonical build destination required')
        before = approved_copy_binding(source, approved, expected_content=expected_content)
        target = STAGE / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        need(not target.exists(), 'Exclusive build target')
        shutil.copyfile(source, target)
        installed = pin(target, STAGE)
        need((before['bytes'], before['sha256']) == (installed['bytes'], installed['sha256'])
             and pin(source) == before, 'Source changed during copy')
        copies.append({'source': before, 'installed': installed})
        return installed

    package_relative = PACKAGE.relative_to(ROOT).as_posix()
    for path in sorted(PACKAGE.iterdir()):
        if path.name not in ('script_entry.py', 'run_continuation.sh'):
            copy(path, package_relative + '/' + path.name)
    copy(PACKAGE / 'script_entry.py', 'scripts/run_u64_cleananchors_continuation_v1.py')
    copy(PACKAGE / 'run_continuation.sh', 'scripts/run_continuation.sh')
    for relative in SHARED_SOURCES:
        copy(ROOT / relative, relative)
    for name, row in parent['files'].items():
        if name.startswith(('assets/', 'evidence/', 'third_party/')):
            need(pin(PARENT / name, PARENT) == {'path': name, **row}, 'Parent-bound source bytes differ')
            copy(PARENT / name, name, expected_content=row)
    research_init = STAGE / 'research/__init__.py'
    with research_init.open('xb') as stream:
        stream.write(b'"""Inert research namespace."""\n'); stream.flush(); os.fsync(stream.fileno())
    parent_pin = copy(parent_path, 'evidence/u64_parent_protocol.json', expected_content={
        'bytes': contract.PARENT_BYTES, 'sha256': contract.PARENT_SHA256})
    heads = copy(head_source, 'weights/terminal_u0064_heads.npz', expected_content=head_expected)
    historical = copy(historical_path, 'evidence/u64_historical_pixel_audit.json', expected_content={
        'bytes': packet.HISTORICAL_PIXEL_AUDIT_FIXED[0], 'sha256': packet.HISTORICAL_PIXEL_AUDIT_FIXED[1]})
    historical_index = copy(historical_index_path, 'evidence/u64_historical_archive_index.json',
                            expected_content=historical_index_pin)
    reuse = copy(ADMISSION, 'evidence/u64_continuation_reuse.json')
    visual_plan = copy(VISUAL_PLAN, 'evidence/u64_planned_visual_review_v1.json')
    controls, parity = [], []
    for key, destination, table in (('cached_frozen_U96_controls', 'controls', controls),
                                    ('terminal_u64_parity_witnesses', 'u64_parity', parity)):
        for row in cache_rows[key]:
            need(re.fullmatch(r'[A-Za-z0-9_\-]{1,120}', row['id']), 'Safe unique cached case ID required')
            base = 'assets/' + destination + '/' + row['id'] + '/'
            table.append({'id': row['id'],
                'raw': copy(ROOT / row['cache_outputs']['raw']['path'], base + 'raw.npy', expected_content=row['cache_outputs']['raw']),
                'png': copy(ROOT / row['cache_outputs']['png']['path'], base + 'result.png', expected_content=row['cache_outputs']['png']),
                'record': copy(ROOT / row['cache_record']['path'], base + 'record.json', expected_content=row['cache_record'])})
    copy(REVIEW, 'evidence/u64_runner_source_review_v1.json')
    copy(TESTS, 'evidence/u64_root_runner_tests_v1.json')
    proof_paths = (CACHE_RECEIPT, CACHE_PEER, ROOT / materialized['scope']['path'])
    for source in proof_paths:
        copy(source, 'evidence/u64_readiness/' + source.name)
    for name in ('continuation_source_review_v1.json', 'continuation_source_review_v1.md',
                 'continuation_objective_review_v1.json', 'continuation_objective_review_v1.md',
                 'continuation_contract_design_review_v1.json', 'continuation_contract_design_review_v1.md'):
        copy(DESIGN / name, 'evidence/u64_design/' + name)
    for name in ('SYSTEM_WORKFLOW_AND_GOAL.md', 'PRACTICAL_OUTPUT_SCOPE.md', 'CCTV_CODEFORMER_FINETUNING_PLAN_V1.md',
                 'CCTV_CODEFORMER_U64_CLEANANCHORS_CONTINUATION_V1_PREPARATION.md'):
        copy(ROOT / name, 'evidence/u64_instructions/' + name)
    prototype = {key: pin(STAGE / (package_relative + '/' + key + '.py'), STAGE)
                 for key in ('model', 'objective', 'contract')}
    files = {path.relative_to(STAGE).as_posix(): {'bytes': path.stat().st_size, 'sha256': sha(path)}
             for path in sorted(STAGE.rglob('*')) if path.is_file()}
    omitted = {row['path'] for row in parent['model'].values()}
    for row in parent['model'].values():
        files[row['path']] = {key: row[key] for key in ('bytes', 'sha256')}
    current_required = {path.relative_to(ROOT).as_posix() for path in PACKAGE.iterdir() if path.is_file()}
    current_required |= set(SHARED_SOURCES) | {Path(__file__).resolve().relative_to(ROOT).as_posix()}
    need(required == current_required and validate_review(doc(REVIEW), doc(TESTS), required) == approved,
         'Executable namespace or approval changed during build')
    need(doc(REVIEW)['continuation_reuse'] == pin(ADMISSION) and doc(CACHE_PEER)['materialization_receipt'] == pin(CACHE_RECEIPT),
         'Prospective reuse/cache approval changed during copy')
    protocol = {
        'format': packet.FORMAT, 'stage_name': packet.STAGE_NAME, 'created_UTC': datetime.now(timezone.utc).isoformat(),
        'files': files, 'model': parent['model'], 'u64_head': heads, 'prototype': prototype,
        'data': parent['data'], 'parent_protocol': parent_pin, 'cases': parent['cases'], 'guards': parent['guards'],
        'historical_pixel_audit': historical, 'historical_archive_index': historical_index,
        'continuation_reuse': reuse, 'visual_plan': visual_plan, 'controls': controls, 'u64_parity': parity,
        'inherited_DEV_evidence': parent['inherited_DEV_evidence'], 'quality_policy': parent['quality_policy'],
        'continuation': contract.prospective_contract(), 'limits': contract.LIMITS,
        'manual_launch_only': True, 'automatic_retry_resume': False, 'model_qualified': False,
        'training_success_not_implied': True, 'source_status': 'PREPARED_UNRUN_EXACT_U64_WARM_START_CHANGED_CLEAN_TARGET_SUPERVISION',
        'only_R2_U96_files_omitted_from_execution_upload': True,
    }
    write_new(STAGE / 'protocol.json', protocol)
    protocol_sha = sha(STAGE / 'protocol.json')
    packet.verify_static(STAGE, protocol_sha, missing_weights_for_install=True)
    expected = included_archive_files(files, omitted)
    expected['protocol.json'] = {'bytes': (STAGE / 'protocol.json').stat().st_size, 'sha256': protocol_sha}
    need(sum(row['bytes'] for row in expected.values()) <= MAX_STATIC_BYTES, 'Finite512MiB static execution payload required')
    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for name in sorted(expected):
                    tar.add(STAGE / name, arcname=STAGE.name + '/' + name, recursive=False)
        raw.flush(); os.fsync(raw.fileno())
    need(archive.stat().st_size <= MAX_STATIC_BYTES, 'Finite512MiB static archive required')
    digest = sha(archive)
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + archive.name + '\n'); stream.flush(); os.fsync(stream.fileno())
    seen = set()
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            need(member.isfile() and member.name.startswith(STAGE.name + '/'), 'Regular namespaced members only')
            name = member.name[len(STAGE.name) + 1:]
            need(name in expected and name not in seen, 'Exact unique reviewed members')
            seen.add(name)
            payload = tar.extractfile(member).read()
            need(len(payload) == expected[name]['bytes'] and hashlib.sha256(payload).hexdigest() == expected[name]['sha256'],
                 'Archive payload differs')
    expanded = 0
    with gzip.open(archive, 'rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            expanded += len(block)
            need(expanded <= MAX_STATIC_BYTES + 16*1024**2, 'Bounded complete gzip readback required')
    need(seen == set(expected) and 'weights/terminal_u0064_heads.npz' in seen
         and not omitted & seen and 'torch' not in sys.modules and 'numpy' not in sys.modules,
         'Complete static readback with terminal heads included and only two large weights omitted')
    need(checksum.read_bytes() == (digest + '  ' + archive.name + '\n').encode('ascii'), 'Exact LF checksum')
    need(validate_review(doc(REVIEW), doc(TESTS), required) == approved
         and doc(REVIEW)['continuation_reuse'] == pin(ADMISSION), 'Approval/source changed during archive readback')
    receipt = {'complete': True, 'status': 'PREPARED_UNRUN_MANUAL_U64_CONTINUATION_PACKET',
        'builder': pin(Path(__file__).resolve()), 'protocol': pin(STAGE / 'protocol.json'), 'archive': pin(archive),
        'checksum': pin(checksum), 'source_review': pin(REVIEW), 'continuation_reuse': pin(ADMISSION),
        'cached_controls_readback': pin(CACHE_PEER), 'static_files': len(files), 'archive_members': len(seen),
        'copies': copies, 'full_stream_payload_readback': True, 'full_gzip_CRC_footer_readback': True,
        'two_large_initializer_files_omitted': sorted(omitted), 'terminal_u64_heads_included': True,
        'neural_gradient_optimizer_training_VM_calls': 0, 'model_qualified': False}
    write_new(PREP / 'build_receipt_v1.json', receipt)
    print(json.dumps({key: receipt[key] for key in ('complete', 'status', 'protocol', 'archive', 'archive_members')}))


if __name__ == '__main__':
    main()


