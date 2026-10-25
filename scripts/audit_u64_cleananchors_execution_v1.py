"""Independent bounded execution-packet byte audit; never extracts or runs it.

The trusted complete-runner approval is an external local input, not authority
created by resealing an archive. Only exact reviewed local inert contract code
is imported; archived executable, model and array bytes are never executed.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
import stat
import sys
import tarfile
import time
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]
STAGE = 'cctv_codeformer_u64_cleananchors_continuation_v1_vm'
PACKAGE = 'research/codeformer_u64_cleananchors_continuation_v1'
PREFIX = 'cctv-codeformer-u64-cleananchors-continuation-v1'
FORMAT = 'complete-u64-cleananchors-continuation-manual-L4-v1'
BUILDER = 'scripts/build_u64_cleananchors_continuation_v1.py'
PREP = ROOT / 'outputs/codeformer_u64_cleananchors_continuation_v1_preparation'
TRUSTED_REVIEW = PREP / 'runner_review/complete_runner_source_review_v1.json'
REVIEW_MEMBER = 'evidence/u64_runner_source_review_v1.json'
TESTS_MEMBER = 'evidence/u64_root_runner_tests_v1.json'
TESTS_SOURCE = 'outputs/codeformer_u64_cleananchors_continuation_v1_preparation/root_runner_source_tests_v1.json'
MAX_ARCHIVE = 128 * 1024**2
MAX_EXPANDED = 256 * 1024**2
MAX_MEMBER = 32 * 1024**2
MAX_MEMBERS = 1500
MAX_SECONDS = 180
HELPER = ROOT / 'scripts/audit_u96_decoder_readout_transport_v1.py'
HELPER_FIXED = (20828, 'dc3d05237b0eb163579d4034ae5490d8ecfb6005bafba478f243f9f0c219dcfc')
PARENT_MEMBER = 'evidence/u64_parent_protocol.json'
PIXEL_MEMBER = 'evidence/u64_historical_pixel_audit.json'
INDEX_MEMBER = 'evidence/u64_historical_archive_index.json'
HEAD_MEMBER = 'weights/terminal_u0064_heads.npz'
FIXED = {
    PARENT_MEMBER: (197079, 'a9b7899cb23ada04779f350dfd37f11818533bf9bf462404e342b8d7e575294e'),
    PIXEL_MEMBER: (2677742, '8b73cbb901452faff8b3679bcf9d077d4d8c98a90e461c18af260565fd04c18d'),
    INDEX_MEMBER: (1103555, '0f376b6bff438914fb9e1a9f1c00ae0ea236ee06aaa00496e333a60cd9013356'),
    HEAD_MEMBER: (685818, '77c0e1f236eb70d37f3f81059b75580e92d606bb2f8b5dd5eeb00da0305dca17'),
}
MODEL_FIXED = {
    'r2': (376609465, 'd4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'),
    'u96': (212827, '958e8338d4bc1a0204c314c3af239ab0ab3d783c67e3e532d9fe23c58f8a73d0'),
}
CORE_FIXED = {
    'model.py': (21294, '6a67468fd6b94e938efe8548205282c048635fcc63e8e4efd8389a2910129829'),
    'objective.py': (8205, 'cffafed3feae10ce91890cfd68ee5de96c8db8edad00041420615cd071f2b7ad'),
    'contract.py': (10844, '99e1bbef911750c5ef0622d350e203bcafed58559b3c2a2c500744ef76102bb0'),
}
SHARED_FIXED = {
    'research/codeformer_u96_postfusion_v1/__init__.py':
        (66, 'b294dfcc71d5358daccc4a7e02d475eae39c3c0a24952ef522062653a8ade54a'),
    'research/codeformer_u96_postfusion_v1/model.py':
        (23240, 'ca2b466161932068c01980bb41fcc08e12ee8f0120f17c8fdf47898f35934aa8'),
    'research/codeformer_u96_spatial_feasibility_v1/__init__.py':
        (84, '9b6cd2566bb68204c76948c35b8e66c9a83cdc43a198735a8fe8bdd79ed0ea7e'),
    'research/codeformer_u96_spatial_feasibility_v1/metrics.py':
        (17480, '1ec77a3cf3621494d0dbe36f8f46a2a8db06c4006bfab41be639399156e08b05'),
    'research/codeformer_u96_spatial_feasibility_v1/worker.py':
        (72086, 'fae271100642604ae0682ebf52ec163e768fa6882eced87e0c34e587980126b6'),
}
ENTRY_MAP = {
    PACKAGE + '/script_entry.py': 'scripts/run_u64_cleananchors_continuation_v1.py',
    PACKAGE + '/run_continuation.sh': 'scripts/run_continuation.sh',
}
OLD_RUN = 'outputs/u96_postfusion_smoothanchors_lr_v1/'
OLD_ARCHIVE_FIXED = (1166628159, '7e5208d2ff69713c834c21fb466020ae54d7ee68b84795e2c044ffc11847041e')
OLD_PROTOCOL_SHA = '8d4e4805e0cfa09d5c1a2c55c1b89855542a48c6768befb4a745b4df211319d9'


def need(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    need(type(value) is str and bool(value) and value != '.' and '\\' not in value
         and ':' not in value, 'Canonical relative POSIX path required')
    path = PurePosixPath(value)
    need(not path.is_absolute() and path.as_posix() == value
         and all(part not in ('', '.', '..') for part in path.parts), 'Confined relative path required')
    return value


def relative(name):
    need(type(name) is str and name.startswith(STAGE + '/'), 'Exact new execution prefix required')
    return canonical(name[len(STAGE) + 1:])


def filepin(path):
    """Bounded ordinary input with unchanged metadata and workspace ancestry."""
    path = Path(path).absolute()
    need(path == path.resolve(strict=True) and path.is_relative_to(ROOT), 'Canonical workspace input required')
    for parent in path.parents:
        info = parent.lstat()
        need(stat.S_ISDIR(info.st_mode) and not stat.S_ISLNK(info.st_mode)
             and not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary input ancestry required')
    before = path.lstat()
    need(stat.S_ISREG(before.st_mode) and not stat.S_ISLNK(before.st_mode)
         and before.st_nlink == 1 and not getattr(before, 'st_file_attributes', 0) & 1024
         and 0 < before.st_size <= 512 * 1024**2, 'Bounded single-link ordinary input required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            digest.update(block)
    after = path.lstat()
    need((before.st_ino, before.st_size, before.st_mtime_ns)
         == (after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed during hashing')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': after.st_size, 'sha256': digest.hexdigest()}


def load_helper():
    pin = filepin(HELPER)
    need((pin['bytes'], pin['sha256']) == HELPER_FIXED, 'Exact reviewed R1 framing primitives required')
    spec = importlib.util.spec_from_file_location('_u64_execution_R1_framing', HELPER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Only these fixed R1 primitives are reused. No old operation/return audit runs.
_helper = load_helper()


def inspect_stream(raw, *, guard=lambda: None):
    """Complete one-gzip CRC/ISIZE, bounded tar inventory and zero EOF padding."""
    stream = None
    def bounded_guard():
        guard()
        if stream is not None:
            need(stream.expanded <= MAX_EXPANDED, 'Execution expanded-byte cap before tar parsing')
    class ExecutionGzip(_helper.SingleGzip):
        def read(self, size):
            value = super().read(size)
            need(self.expanded <= MAX_EXPANDED, 'Execution expanded-byte cap before tar parsing')
            return value
    stream = ExecutionGzip(raw, bounded_guard)
    members, contents, last = {}, {}, 0
    with tarfile.open(fileobj=stream, mode='r|') as archive:
        for member in archive:
            guard()
            need(member.isfile() and member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
                 and not member.linkname and 0 <= member.size <= MAX_MEMBER,
                 'Only bounded regular source/data members required')
            name = relative(member.name)
            need(name not in members and len(members) < MAX_MEMBERS, 'Unique bounded member namespace required')
            value = archive.extractfile(member).read(MAX_MEMBER + 1)
            need(len(value) == member.size, 'Complete declared member content required')
            members[name] = {'bytes': len(value), 'sha256': sha(value)}
            if name.endswith(('.json', '.py', '.sh')):
                contents[name] = value
            last = ((member.offset_data + member.size + 511) // 512) * 512
    while stream.read(65536):
        guard()
    padding = _helper.verify_tar_end(stream, last)
    return members, contents, {'gzip_single_frame_complete': stream.done,
        'gzip_CRC_and_ISIZE_verified': True, 'expanded_bytes': stream.expanded,
        'complete_zero_tar_padding_bytes': padding}


def content_binding(value):
    need(type(value) is dict and set(value) == {'bytes', 'sha256'}
         and type(value['bytes']) is int and 0 <= value['bytes'] <= 512 * 1024**2
         and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
         'Exact finite content binding required')
    return value


def binding(value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'Exact path/content binding required')
    canonical(value['path'])
    return content_binding({key: value[key] for key in ('bytes', 'sha256')})


def bound_json_member(name, members, contents):
    need(name in members and name in contents
         and members[name] == {'bytes': len(contents[name]), 'sha256': sha(contents[name])},
         'Actual archived JSON byte binding required: ' + name)
    return _helper.strict_json(contents[name])


def fixed_binding(p, key, name, members):
    value = p[key]
    expected = {'bytes': FIXED[name][0], 'sha256': FIXED[name][1]}
    need(binding(value) == expected and value['path'] == name and members.get(name) == expected
         and p['files'].get(name) == expected, 'Exact included immutable binding required: ' + name)


def approved_sources(review):
    need(type(review) is dict and review.get('complete') is True
         and review.get('source_review_pass') is True and review.get('unresolved_blockers') == []
         and review.get('new_continuation_data_reuse_admission') is True
         and type(review.get('neural_gradient_optimizer_training_VM_calls')) is int
         and review['neural_gradient_optimizer_training_VM_calls'] == 0,
         'Trusted complete-runner review and new TRAIN-reuse admission required')
    rows = review['sources']
    need(type(rows) is list and 0 < len(rows) <= 1000, 'Bounded approved source list required')
    approved = {}
    for row in rows:
        binding(row)
        need(row['path'] not in approved, 'Unique approved source paths required')
        approved[row['path']] = row
    need(BUILDER in approved, 'Builder must be independently source-reviewed')
    return approved


def bind_trusted_review(members, contents, trusted_bytes):
    need(type(trusted_bytes) is bytes and contents.get(REVIEW_MEMBER) == trusted_bytes,
         'Archived approval must match independently trusted local review bytes')
    return approved_sources(bound_json_member(REVIEW_MEMBER, members, contents))


def parent_readback(p, members, contents):
    fixed_binding(p, 'parent_protocol', PARENT_MEMBER, members)
    parent = bound_json_member(PARENT_MEMBER, members, contents)
    for field in ('model', 'data', 'cases', 'guards', 'quality_policy', 'inherited_DEV_evidence'):
        need(p[field] == parent[field], 'Exact archived immutable parent field required: ' + field)
    return parent


def reviewed_plans_readback(p, members, contents, review):
    """Bind actual archived plans to approval; installed/workspace paths differ."""
    for key, name in (('continuation_reuse', 'evidence/u64_continuation_reuse.json'),
                      ('visual_plan', 'evidence/u64_planned_visual_review_v1.json')):
        expected = binding(review[key])
        need(binding(p[key]) == expected and p[key]['path'] == name
             and members.get(name) == expected and p['files'].get(name) == expected,
             'Archived prospective plan must match independently reviewed bytes: ' + key)
        plan = bound_json_member(name, members, contents)
        need(plan.get('complete') is True, 'Complete reviewed prospective plan required')
        if key == 'continuation_reuse':
            need(plan.get('new_continuation_data_reuse_admission') is True,
                 'Actual archived explicit new TRAIN data-reuse admission required')


def reviewed_tests_readback(p, members, contents, approved):
    """A replaced green test receipt cannot authorize freshly resealed sources."""
    need(TESTS_SOURCE in approved, 'Trusted complete-runner review must pin current root test receipt')
    expected = binding(approved[TESTS_SOURCE])
    need(members.get(TESTS_MEMBER) == expected and p['files'].get(TESTS_MEMBER) == expected,
         'Actual archived root test receipt must equal independently reviewed bytes')
    tests = bound_json_member(TESTS_MEMBER, members, contents)
    need(tests.get('complete') is True and tests.get('all_tests_pass') is True
         and type(tests.get('neural_gradient_optimizer_training_VM_calls')) is int
         and tests['neural_gradient_optimizer_training_VM_calls'] == 0,
         'Complete actual archived pure source-test evidence required')
    rows = tests['sources']
    need(type(rows) is list and len(rows) <= 1000, 'Bounded tested source bindings required')
    tested = {}
    for row in rows:
        binding(row)
        need(row['path'] not in tested, 'Unique tested source paths required')
        tested[row['path']] = row
    required = {name for name in approved if name.startswith(PACKAGE + '/')} | set(SHARED_FIXED) | {BUILDER}
    need(required <= set(tested) and all(tested[name] == approved[name] for name in required),
         'Every new/shared executable and builder must join the actual reviewed pure test receipt')
    return {'path': TESTS_MEMBER, **expected}


def reviewed_namespace_readback(p, members, contents, approved):
    """Close all installed research/script sources and exactly two omitted weights."""
    need(type(p['model']) is dict and set(p['model']) == set(MODEL_FIXED), 'Exactly two original model bindings required')
    omitted = set()
    for key, fixed in MODEL_FIXED.items():
        row = p['model'][key]
        value = binding(row)
        need(row['path'].startswith('weights/') and (value['bytes'], value['sha256']) == fixed
             and p['files'].get(row['path']) == value, 'Exact original R2/U96 model binding required')
        omitted.add(row['path'])
    need(len(omitted) == 2, 'Distinct original model paths required')
    fixed_binding(p, 'u64_head', HEAD_MEMBER, members)
    need({name for name in p['files'] if name.startswith('weights/')} == omitted | {HEAD_MEMBER},
         'Only R2/U96 may be omitted; the retained u64 head must be included')
    need(type(p['prototype']) is dict and set(p['prototype']) == {'model', 'objective', 'contract'},
         'Exact three core prototype bindings required')
    for name, fixed in CORE_FIXED.items():
        source = PACKAGE + '/' + name
        row = p['prototype'][name[:-3]]
        expected = {'bytes': fixed[0], 'sha256': fixed[1]}
        need(binding(row) == expected and row['path'] == source and members.get(source) == expected
             and p['files'].get(source) == expected, 'Exact protocol prototype field required: ' + source)
    installed = {ENTRY_MAP.get(name, name): name for name in approved if name.startswith(PACKAGE + '/')}
    need(set(ENTRY_MAP) <= set(approved), 'Both reviewed entry bridges required')
    for name, fixed in SHARED_FIXED.items():
        need(name in approved and (approved[name]['bytes'], approved[name]['sha256']) == fixed,
             'Reviewed exact retained helper/factory required: ' + name)
        installed[name] = name
    actual = {name for name in members if name.startswith(('research/', 'scripts/'))
              and name != 'research/__init__.py'}
    need(actual == set(installed), 'Exact complete reviewed installed source namespace required')
    need(contents.get('research/__init__.py') == b'"""Inert research namespace."""\n',
         'Exact inert research namespace initializer required')
    joined = []
    for name, source in sorted(installed.items()):
        expected = binding(approved[source])
        need(members.get(name) == expected, 'Executable bytes must join independent reviewed source: ' + name)
        if name.endswith(('.py', '.sh')):
            need(name in contents and b'\r' not in contents[name], 'Actual LF executable bytes required')
        joined.append({'installed': name, 'reviewed_source': source, **expected})
    need(PACKAGE + '/worker.py' in installed and PACKAGE + '/model.py' in installed,
         'New worker and complete-u64 initializer approvals required')
    return omitted, joined


def load_inert_contract(approved, contents):
    """Import fresh reviewed LOCAL arithmetic only, never any archived module."""
    local = {}
    for name, fixed in CORE_FIXED.items():
        source = PACKAGE + '/' + name
        need(source in approved and (approved[source]['bytes'], approved[source]['sha256']) == fixed,
             'Exact admitted core source required: ' + source)
        local[name] = ROOT / source
        raw = local[name].read_bytes()
        need((len(raw), sha(raw)) == fixed and raw == contents.get(source),
             'Reviewed local core must equal actual archived source')
    namespace = '_u64_execution_audit_local_inert'
    names = (namespace, namespace + '.objective', namespace + '.contract')
    need(all(name not in sys.modules for name in names), 'Fresh local inert namespace required')
    package = ModuleType(namespace)
    package.__path__ = []
    sys.modules[namespace] = package
    try:
        loaded = {}
        for name in ('objective', 'contract'):
            spec = importlib.util.spec_from_file_location(namespace + '.' + name, local[name + '.py'])
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            loaded[name] = module
        return loaded['contract']
    finally:
        for name in names:
            sys.modules.pop(name, None)


def retained_namespace_readback(p, members, contents, cache_paths):
    parent = bound_json_member(PARENT_MEMBER, members, contents)
    retained = {name: row for name, row in parent['files'].items()
                if name.startswith(('assets/', 'evidence/', 'third_party/'))}
    for name, value in retained.items():
        need(p['files'].get(name) == value and members.get(name) == value,
             'Immutable retained parent file changed or omitted: ' + name)
    installed_sources = {name for name in members if name.startswith(('research/', 'scripts/'))}
    for name in members:
        if name in retained or name in installed_sources or name in cache_paths or name in (
                'protocol.json', HEAD_MEMBER):
            continue
        need(name.startswith('evidence/u64_') and name.endswith(('.json', '.md')),
             'Unexpected new executable/data namespace: ' + name)
    need(not set(retained) & cache_paths, 'Cached controls cannot replace immutable parent assets')
    return len(retained)


def historical_cache_readback(p, members, contents, selection):
    """Join all180 actual bytes to the fixed old index, pixel audit and records."""
    fixed_binding(p, 'historical_pixel_audit', PIXEL_MEMBER, members)
    fixed_binding(p, 'historical_archive_index', INDEX_MEMBER, members)
    pixel = bound_json_member(PIXEL_MEMBER, members, contents)
    index = bound_json_member(INDEX_MEMBER, members, contents)
    need(pixel.get('complete') is True and pixel.get('independent_saved_pixel_integrity_pass') is True
         and pixel.get('pilot_quality_gates_pass') is True and pixel.get('model_qualified') is False
         and pixel['protocol']['sha256'] == OLD_PROTOCOL_SHA,
         'Exact historical finite-screen pixel evidence required')
    need(index.get('complete') is True
         and (index['archive_bytes'], index['archive_SHA256']) == OLD_ARCHIVE_FIXED,
         'Exact old return-index provenance required')
    bank = selection['monitors'] + selection['guards']
    need((len(selection['monitors']), len(selection['guards'])) == (32, 24), 'Exact32 adaptive TRAIN monitors/24 guards required')
    paths, joins = set(), []
    for label, field, expected, kind, update, directory in (
            ('frozen_U96', 'controls', bank, 'reference', 0, 'reference/'),
            ('terminal_u64_parity', 'u64_parity', selection['monitors'][:4], 'snapshot', 64, 'snapshots/u0064/')):
        rows = p[field]
        need(type(rows) is list and all(type(row) is dict for row in rows)
             and [row.get('id') for row in rows] == [row['id'] for row in expected],
             'Exact ordered historical ' + label + ' membership required')
        for row, reference in zip(rows, expected):
            need(set(row) == {'id', 'raw', 'png', 'record'}, 'Exact cached record/RAW/PNG row required')
            key = row['id']
            original = OLD_RUN + directory + key + '/'
            found = [item for item in pixel['all_saved_pixel_metrics'] if item['id'] == key
                     and item['kind'] == kind and item['update'] == update]
            need(len(found) == 1, 'Unique historical pixel record required')
            measured = found[0]
            need(measured['role'] == reference['role'] and measured['record'] == original + 'record.json',
                 'Historical role/record path must match immutable TRAIN case')
            record = None
            for part, old_suffix, new_suffix in (('raw', 'raw.npy', 'raw.npy'),
                                                ('png', 'output.png', 'result.png'),
                                                ('record', 'record.json', 'record.json')):
                value = binding(row[part])
                source = original + old_suffix
                installed = ('assets/controls/' if field == 'controls' else 'assets/u64_parity/') + key + '/' + new_suffix
                need(row[part]['path'] == installed and installed not in paths
                     and members.get(installed) == value and p['files'].get(installed) == value,
                     'Distinct exact installed cached artifact required: ' + installed)
                need(index['members'].get(source) == {'path': source, **value},
                     'Actual cached bytes must join original independently indexed record/output: ' + source)
                paths.add(installed)
                if part == 'record':
                    record = bound_json_member(installed, members, contents)
                else:
                    saved = measured['raw_file' if part == 'raw' else 'png_file']
                    need(saved == {'path': source, **value}, 'Cached bytes must join fixed historical pixel output')
            need(record.get('complete') is True and record.get('model_qualified') is False
                 and record['kind'] == kind and record['update'] == update
                 and record['id'] == key and record['role'] == reference['role']
                 and record['reference_id'] == reference['reference_id'] and record['split'] == 'TRAIN'
                 and record.get('cohort') == reference.get('cohort', 'legacy_guard')
                 and record['input'] == reference['input'] and record.get('target') == reference.get('target'),
                 'Actual archived cached record must join immutable case and historical phase')
            for part, field_name in (('raw', 'raw_file'), ('png', 'png_file')):
                need(record['outputs'][part] == measured[field_name], 'Record output pin must equal historical audited output')
            need(record['metrics'] == measured['metrics']
                 and record['regional_metrics'] == measured['regional_metrics'],
                 'Actual record metrics must equal independently audited saved pixel metrics')
            joins.append({'id': key, 'kind': label, 'original_record': original + 'record.json',
                          'installed_record': row['record']['path'], 'actual_index_and_pixel_joins': True})
    need(len(paths) == 180 and len(joins) == 60, 'All180 cached artifacts/60 record joins required')
    return paths, joins


def audit(archive_path, protocol_path, checksum_path, destination):
    """Read one sealed execution archive; call only after its builder is approved."""
    start = time.monotonic()
    need('torch' not in sys.modules, 'No framework may precede execution-file audit')
    need(not destination.exists(), 'Preserve prior independent execution audits')
    def guard():
        need(time.monotonic() - start < MAX_SECONDS, 'Independent execution-audit180s cap')
    def pins():
        return {'archive': filepin(archive_path), 'protocol': filepin(protocol_path),
                'checksum': filepin(checksum_path), 'trusted_runner_source_review': filepin(TRUSTED_REVIEW),
                'builder_source': filepin(ROOT / BUILDER), 'retained_gzip_tar_primitives': filepin(HELPER)}
    need(0 < archive_path.stat().st_size <= MAX_ARCHIVE, 'Execution archive128MiB cap')
    initial = pins()
    trusted_bytes = TRUSTED_REVIEW.read_bytes()
    need(sha(trusted_bytes) == initial['trusted_runner_source_review']['sha256'], 'Stable trusted review read required')
    protocol_bytes, checksum = protocol_path.read_bytes(), checksum_path.read_bytes()
    need(checksum == (initial['archive']['sha256'] + '  ' + archive_path.name + '\n').encode('ascii'),
         'Exact LF companion checksum required')
    need(b'\r' not in protocol_bytes and protocol_bytes.endswith(b'\n'), 'LF sealed protocol required')
    p = _helper.strict_json(protocol_bytes)
    need(type(p) is dict and p['format'] == FORMAT and p['stage_name'] == STAGE, 'Distinct complete-u64 continuation protocol required')
    files = p['files']
    need(type(files) is dict and 0 < len(files) <= MAX_MEMBERS and 'protocol.json' not in files,
         'Bounded nonrecursive sealed file table required')
    for name, value in files.items():
        canonical(name); content_binding(value)
    with archive_path.open('rb') as stream:
        members, contents, framing = inspect_stream(stream, guard=guard)
    approved = bind_trusted_review(members, contents, trusted_bytes)
    need(initial['builder_source'] == approved[BUILDER], 'Current builder must equal independently reviewed builder source')
    omitted, executable_joins = reviewed_namespace_readback(p, members, contents, approved)
    expected = {name: value for name, value in files.items() if name not in omitted}
    expected['protocol.json'] = {'bytes': len(protocol_bytes), 'sha256': sha(protocol_bytes)}
    need(members == expected and contents.get('protocol.json') == protocol_bytes,
         'All sealed execution bytes included; only original R2/U96 model payloads omitted')
    parent_readback(p, members, contents)
    review = bound_json_member(REVIEW_MEMBER, members, contents)
    reviewed_plans_readback(p, members, contents, review)
    tests_join = reviewed_tests_readback(p, members, contents, approved)
    contract = load_inert_contract(approved, contents)
    selection = contract.select_parent_cases(contents[PARENT_MEMBER])
    need(p['continuation'] == contract.prospective_contract() and p['limits'] == contract.LIMITS
         and all(type(value) is int for value in p['limits'].values()), 'Exact292/64/16 finite contract required')
    plan = contract.operation_plan(selection)
    need(len(plan) == 292 and sum(row['backward'] for row in plan) == 64
         and sum(row['optimizer_application_after_this_forward'] for row in plan) == 16
         and (len(selection['optimization']), len(selection['monitors']), len(selection['guards'])) == (64, 32, 24),
         'Exact admitted TRAIN optimization/monitor/guard schedule required')
    cache_paths, control_joins = historical_cache_readback(p, members, contents, selection)
    retained = retained_namespace_readback(p, members, contents, cache_paths)
    need(p['manual_launch_only'] is True and p['automatic_retry_resume'] is False
         and p['model_qualified'] is False and p.get('training_success_not_implied') is True,
         'Manual-only packet verification cannot imply actual execution or qualification')
    guard()
    final = pins()
    need(final == initial and (len(protocol_bytes), sha(protocol_bytes)) == (final['protocol']['bytes'], final['protocol']['sha256'])
         and (len(checksum), sha(checksum)) == (final['checksum']['bytes'], final['checksum']['sha256']),
         'Audited inputs changed during verification')
    need('torch' not in sys.modules, 'No framework in execution-file audit')
    result = {'complete': True, 'independent_archive_audit_pass': True,
        'recorded_UTC': datetime.now(timezone.utc).isoformat(), **final,
        'audited_input_pins_unchanged': True, 'members': len(members),
        'payload_bytes': sum(row['bytes'] for row in members.values()), 'framing': framing,
        'reviewed_executable_join': executable_joins, 'omitted_only_original_model_paths': sorted(omitted),
        'included_u64_head': {'path': HEAD_MEMBER, **members[HEAD_MEMBER]},
        'immutable_retained_parent_files': retained, 'cached_control_members': len(cache_paths),
        'cached_record_index_and_pixel_joins': control_joins,
        'cached_frozen_U96_controls': 56, 'terminal_u64_parity_witnesses': 4,
        'actual_archived_parent_and_evidence_bytes_verified': True,
        'actual_archived_reviewed_root_tests': tests_join,
        'auditor': filepin(Path(__file__).resolve()), 'seconds': time.monotonic() - start,
        'Torch_imported': False, 'neural_gradient_optimizer_training_VM_calls': 0,
        'arrays_or_head_deserialized': False, 'actual_L4_parity_gradient_quality_verified': False,
        'model_qualified': False}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    args = parser.parse_args()
    need(args.audit, 'Explicit --audit required')
    result = audit(ROOT / 'outputs' / (PREFIX + '-execution.tar.gz'),
        ROOT / 'outputs' / STAGE / 'protocol.json',
        ROOT / 'outputs' / (PREFIX + '-execution.tar.gz.sha256'),
        PREP / 'independent_execution_archive_audit_v1.json')
    print(json.dumps({key: result[key] for key in ('complete', 'members', 'framing', 'archive')}))


if __name__ == '__main__':
    main()
