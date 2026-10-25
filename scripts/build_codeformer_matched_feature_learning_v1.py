"""Source-only gated builder for an independently reviewed manual L4 packet.

Import and pure tests create no stage/archive and load no numerical framework,
model, historical worker or execution entry. --build is an explicit local file
operation gate; it never connects to a VM or launches any training operation.
"""
import argparse
import ast
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'research/codeformer_matched_feature_learning_v1'
STAGE_NAME = 'cctv_codeformer_matched_feature_learning_v1_vm'
PREFIX = 'cctv-codeformer-matched-feature-learning-v1'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
STAGE = ROOT / 'outputs' / STAGE_NAME
PREP = ROOT / 'outputs/codeformer_matched_feature_learning_v1_source_preparation'
DATA = ROOT / 'outputs/codeformer_matched_feature_learning_v1_preparation'
REVIEW = PREP / 'runner_review/complete_runner_source_review_v1.json'
TESTS = PREP / 'root_runner_source_tests_v1.json'
VISUAL = PREP / 'visual_plan_v1.json'
BUILDER = 'scripts/build_codeformer_matched_feature_learning_v1.py'
BOOTSTRAP = 'scripts/run_matched_feature_learning_v1.py'
WORKSPACE_PARENTS = frozenset(('research/__init__.py', 'third_party/__init__.py',
                              'third_party/codeformer/__init__.py'))
PARENT_FIXED = (279879, '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419')
MAX_SOURCE = 512 * 1024 ** 2
MAX_ARCHIVE = 1024 ** 3
MAX_EXPANDED = MAX_ARCHIVE + 32 * 1024 ** 2
MODEL_PATHS = frozenset(('weights/codeformer_r2.pth', 'weights/codeformer_u96_adapter.pth'))
CORE_PROOFS = frozenset((
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_factory_graph_review_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_contract_review_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_objective_quality_repair_review_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_records_review_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_execution_repair_review_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_packet_bootstrap_repair_review_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_entry_review_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_worker_review_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_visual_plan_review_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/independent_builder_review_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_preparation/data_admission_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_preparation/independent_data_audit_v1.json',
    'outputs/codeformer_matched_feature_learning_v1_source_preparation/visual_plan_v1.json',
))


def need(condition, message):
    if not condition:
        raise ValueError(message)


def relative(value):
    need(type(value) is str and bool(value) and '\\' not in value and ':' not in value and
         all(ord(c) >= 32 for c in value), 'Canonical relative POSIX namespace required')
    parsed = PurePosixPath(value)
    need(not parsed.is_absolute() and parsed.as_posix() == value and value != '.' and
         '..' not in parsed.parts, 'Path escape/alias refused')
    return parsed


def ordinary(path, *, directory=False):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        expected = stat.S_ISDIR(info.st_mode) if item != path or directory else stat.S_ISREG(info.st_mode)
        need(expected and not stat.S_ISLNK(info.st_mode) and
             not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary nonlinked ancestry required')
        if item == path and not directory:
            need(info.st_nlink == 1 and 0 <= info.st_size <= MAX_SOURCE,
                 'Bounded single-link ordinary source required')
    return path.resolve(strict=True)


def content_pin(path):
    path = ordinary(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''):
            digest.update(block)
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed during hashing')
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def pin(path, base=ROOT):
    path, base = ordinary(path), ordinary(base, directory=True)
    need(path.is_relative_to(base), 'Source pin escapes declared root')
    return {'path': path.relative_to(base).as_posix(), **content_pin(path)}


def strict_json(raw):
    need(type(raw) is bytes and len(raw) <= 32 * 1024 ** 2, 'Bounded exact JSON bytes required')
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key refused')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('Nonfinite JSON token refused: ' + value)
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    def finite(item, depth=0):
        need(depth <= 64, 'Bounded JSON depth required')
        if type(item) is float:
            need(math.isfinite(item), 'Finite JSON scalar required')
        elif type(item) is dict:
            for child in item.values():
                finite(child, depth + 1)
        elif type(item) is list:
            for child in item:
                finite(child, depth + 1)
    finite(value)
    return value


def read_json(path):
    before = content_pin(path)
    raw = ordinary(path).read_bytes()
    need(len(raw) == before['bytes'] and hashlib.sha256(raw).hexdigest() == before['sha256'] and
         content_pin(path) == before, 'JSON source changed before decode')
    return strict_json(raw)


def write_new(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    need(path.read_bytes() == raw, 'New JSON readback differs')


def _pin_schema(value):
    need(type(value) is dict and set(value) == {'bytes', 'sha256'} and
         type(value['bytes']) is int and 0 <= value['bytes'] <= MAX_SOURCE and
         type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
         'Exact finite content pin required')


def required_sources(root=ROOT):
    """All new executable/package/test files, plus the explicitly run bootstrap."""
    root = ordinary(root, directory=True)
    package = ordinary(root / PACKAGE, directory=True)
    sources = {BUILDER, BOOTSTRAP} | set(WORKSPACE_PARENTS)
    for path in package.iterdir():
        ordinary(path)
        need(path.suffix in ('.py', '.sh'), 'New package caches/directories/unexpected files refused')
        sources.add(path.relative_to(root).as_posix())
    tests = list((root / 'tests').glob('test_matched_feature_*_v1.py'))
    need(bool(tests), 'All matched source tests, including builder tests, required')
    for path in tests:
        ordinary(path)
        sources.add(path.relative_to(root).as_posix())
    need('tests/test_matched_feature_builder_v1.py' in sources,
         'Builder itself must be independently reviewed and tested')
    for name in sources:
        ordinary(root / name)
    return sources


def inherited_sources(parent):
    prefix = PARENT.relative_to(ROOT).as_posix() + '/'
    return {prefix + name: row for name, row in parent['files'].items()
            if name.startswith(('research/', 'third_party/')) and name.endswith(('.py', '.sh'))}


def validate_reviews(root, review, tests, required, inherited):
    """Current exact pins must join two external receipts; never self-approve."""
    need(review.get('complete') is True and review.get('independent_review') is True and
         review.get('source_review_pass') is True and review.get('blockers') == [],
         'Complete independent joined source review without blockers required')
    need(tests.get('complete') is True and tests.get('passed') is True and
         type(tests.get('failures')) is int and tests['failures'] == 0 and
         type(tests.get('errors')) is int and tests['errors'] == 0 and
         type(tests.get('tests_run')) is int and tests['tests_run'] > 0 and
         type(tests.get('passed_count')) is int and type(tests.get('skipped')) is int and
         tests['passed_count'] >= 0 and tests['skipped'] >= 0 and
         tests['passed_count'] + tests['skipped'] == tests['tests_run'],
         'Complete passing root source tests with explicit skips required')
    expected_names = set(required) | set(inherited)
    for receipt in (review, tests):
        need(receipt.get('exact_current_sources') is True and
             type(receipt.get('neural_model_gradient_optimizer_VM_calls')) is int and
             receipt['neural_model_gradient_optimizer_VM_calls'] == 0,
             'Exact current source-only receipt required')
        table = receipt.get('source_pins')
        need(type(table) is dict and set(table) == expected_names,
             'Every and only required new and inherited source must be reviewed/tested')
        for name, expected in table.items():
            relative(name)
            _pin_schema(expected)
            need(content_pin(root / name) == expected, 'Reviewed/tested source changed: ' + name)
            if name in inherited:
                need(expected == inherited[name], 'Inherited source differs from fixed old protocol')
    need(review['source_pins'] == tests['source_pins'], 'Review/test source pin tables disagree')
    for receipt in (review, tests):
        proofs = receipt.get('evidence_files')
        need(type(proofs) is dict and CORE_PROOFS <= set(proofs),
             'Joined graph/contract/data/source proofs and visual plan required')
        for name, expected in proofs.items():
            relative(name)
            need(name.endswith(('.json', '.md')) and name.startswith('outputs/'),
                 'Readiness evidence stays in bounded outputs JSON/Markdown scope')
            _pin_schema(expected)
            need(content_pin(root / name) == expected, 'Readiness evidence changed: ' + name)
    need(review['evidence_files'] == tests['evidence_files'], 'Review/test evidence pins disagree')
    return {'sources': review['source_pins'], 'evidence': review['evidence_files']}


def copy_verified(source, target, expected, *, source_base, target_base):
    """Exact source authentication before exclusive streaming copy and readback."""
    _pin_schema(expected)
    before = pin(source, source_base)
    need({key: before[key] for key in ('bytes', 'sha256')} == expected,
         'Source differs from authorized pin before copy')
    target_base = ordinary(target_base, directory=True)
    target = Path(target).absolute()
    need(target.is_relative_to(target_base) and target != target_base, 'Copy target escapes stage')
    target.parent.mkdir(parents=True, exist_ok=True)
    ordinary(target.parent, directory=True)
    with ordinary(source).open('rb') as original, target.open('xb') as installed:
        for block in iter(lambda: original.read(1024 ** 2), b''):
            installed.write(block)
        installed.flush()
        os.fsync(installed.fileno())
    need(content_pin(target) == expected and pin(source, source_base) == before,
         'Source/copy content changed; partial build retained')
    return pin(target, target_base)


def included_archive_files(files):
    need(MODEL_PATHS <= set(files) and 'weights/terminal_u0064_heads.npz' in files,
         'Exact two model omissions and included trained heads required')
    return {name: dict(row) for name, row in files.items() if name not in MODEL_PATHS}


def audit_tar_terminator(stream):
    """Read complete gzip-expanded tar blocks; reject hidden data after EOF.

    Semantic headers, namespaces and payload hashes are checked independently
    by tarfile. This second scan accounts for every raw header (including PAX)
    and payload block, requires zero padding and at least two zero end blocks,
    and consumes the gzip footer. No concatenated second tar can hide after
    the first end marker. Only the writer's ordinary octal size fields occur.
    """
    expanded = headers = zero_blocks = 0
    def block():
        nonlocal expanded
        parts = []
        size = 0
        while size < 512:
            chunk = stream.read(512 - size)
            if not chunk:
                break
            parts.append(chunk)
            size += len(chunk)
            expanded += len(chunk)
            need(expanded <= MAX_EXPANDED, 'Bounded complete gzip/tar readback required')
        raw = b''.join(parts)
        need(not raw or len(raw) == 512, 'Truncated raw tar block')
        return raw
    while True:
        header = block()
        if not header:
            need(zero_blocks >= 2, 'Two complete tar end blocks required')
            return {'gzip_expanded_bytes': expanded, 'raw_tar_headers': headers,
                    'nonzero_suffix_after_tar_EOF_refused': True}
        if header == b'\0' * 512:
            zero_blocks += 1
            continue
        need(zero_blocks == 0, 'Nonzero payload after tar EOF refused')
        raw_size = header[124:136].rstrip(b'\0 ').lstrip(b' ')
        need(bool(raw_size) and all(character in b'01234567' for character in raw_size),
             'Ordinary bounded octal tar size required')
        size = int(raw_size, 8)
        need(0 <= size <= MAX_SOURCE, 'Raw tar member size cap exceeded')
        headers += 1
        remaining = size
        while remaining:
            payload = block()
            need(bool(payload), 'Truncated raw tar payload')
            used = min(remaining, 512)
            need(not payload[used:].strip(b'\0'), 'Nonzero tar member padding refused')
            remaining -= used


def audit_archive(path, expected, *, stage_name=STAGE_NAME):
    """Stream every payload; enforce exact regular namespace, hashes and gzip EOF."""
    path = ordinary(path)
    need(path.stat().st_size <= MAX_ARCHIVE, 'Execution archive cap1GiB exceeded')
    need(type(expected) is dict and bool(expected), 'Exact archive inventory required')
    for name, row in expected.items():
        relative(name)
        _pin_schema(row)
    need(sum(row['bytes'] for row in expected.values()) <= MAX_ARCHIVE,
         'Execution uncompressed payload cap1GiB exceeded')
    before = content_pin(path)
    seen = set()
    with tarfile.open(path, mode='r|gz') as archive:
        for member in archive:
            need(member.isfile() and not member.issym() and not member.islnk() and
                 member.name.startswith(stage_name + '/'), 'Only regular stage-qualified archive members allowed')
            name = member.name[len(stage_name) + 1:]
            relative(name)
            need(name in expected and name not in seen and member.size == expected[name]['bytes'],
                 'Unexpected/duplicate/wrong-size archive member')
            need(not member.linkname and not member.sparse and
                 all(key == 'path' and value == member.name for key, value in member.pax_headers.items()),
                 'Archive links/sparse/custom metadata refused')
            seen.add(name)
            stream = archive.extractfile(member)
            need(stream is not None, 'Complete regular archive payload required')
            digest = hashlib.sha256()
            remaining = member.size
            while remaining:
                block = stream.read(min(1024 ** 2, remaining))
                need(bool(block), 'Truncated archive member')
                digest.update(block)
                remaining -= len(block)
            need(stream.read(1) == b'' and digest.hexdigest() == expected[name]['sha256'],
                 'Archive payload differs from exact manifest')
    with gzip.open(path, 'rb') as stream:
        wire = audit_tar_terminator(stream)
    need(seen == set(expected) and content_pin(path) == before,
         'Archive inventory incomplete or archive changed during audit')
    return {'members': len(seen), 'payload_bytes': sum(row['bytes'] for row in expected.values()),
            **wire, 'archive': before,
            'all_payloads_readback': True, 'full_gzip_CRC_footer_readback': True}


def _parent():
    path = PARENT / 'protocol.json'
    need(tuple(content_pin(path)[key] for key in ('bytes', 'sha256')) == PARENT_FIXED,
         'Exact retained279879-byte continuation protocol required before decode')
    parent = read_json(path)
    need(type(parent.get('files')) is dict and len(parent['files']) == 469,
         'Exact old469-file manifest required')
    for name, row in parent['files'].items():
        relative(name)
        _pin_schema(row)
    need(set(row['path'] for row in parent['model'].values()) == MODEL_PATHS,
         'Only retained R2/U96 model initializer paths required')
    return parent


def build():
    """Called only by explicit --build after a separately authored current join."""
    need(not any(name.split('.')[0] in {'torch', 'numpy', 'PIL', 'tensorflow', 'jax'} or
                 name.endswith('.worker') for name in sys.modules), 'Static file-only builder required')
    archive = ROOT / 'outputs' / (PREFIX + '-execution.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    receipt_path = PREP / 'build_receipt_v1.json'
    need(not any(path.exists() for path in (STAGE, archive, checksum, receipt_path)),
         'Existing stage/archive/receipt immutable; retain partial builds')
    need(shutil.disk_usage(ROOT).free >= 3 * 1024 ** 3, 'At least3GiB local build headroom required')
    parent = _parent()
    inherited = inherited_sources(parent)
    required = required_sources()
    review, tests = read_json(REVIEW), read_json(TESTS)
    approved = validate_reviews(ROOT, review, tests, required, inherited)
    join_receipt_pins = {'review': content_pin(REVIEW), 'tests': content_pin(TESTS)}
    guard_sources = {name: content_pin(ROOT / name) for name in required | set(inherited)}
    for name in required:
        raw = ordinary(ROOT / name).read_bytes()
        need(name in WORKSPACE_PARENTS or b'\r' not in raw,
             'New executable/test sources require LF: ' + name)
        if name.endswith('.py'):
            ast.parse(raw, feature_version=(3, 10))
    # All transitive custom source pins have been joined before these pure imports.
    sys.path.insert(0, str(ROOT))
    contract = importlib.import_module('research.codeformer_matched_feature_learning_v1.contract')
    packet = importlib.import_module('research.codeformer_matched_feature_learning_v1.packet')
    for module, name in ((contract, 'contract'), (packet, 'packet')):
        need(Path(module.__file__).resolve() == ROOT / PACKAGE / (name + '.py'),
             'Pure helper belongs to another source root')
    selection = contract.select_cases((DATA / 'data_admission_v1.json').read_bytes(),
                                     (DATA / 'candidate_data_v1.json').read_bytes(),
                                     (PARENT / parent['parent_protocol']['path']).read_bytes())
    visual = read_json(VISUAL)
    need(visual.get('complete') is True, 'Complete prospective visual plan required')
    # Resolve the two local weights without decoding them. Missing bytes stop
    # before stage creation and name every missing fixed artifact explicitly.
    initializer_sources = {}
    missing = []
    for row in parent['model'].values():
        choices = (PARENT / row['path'], ROOT / row['path'], ROOT /
                   'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm' / row['path'])
        source = next((path for path in choices if path.exists()), None)
        if source is None:
            missing.append(row['path'])
        else:
            need(content_pin(source) == {key: row[key] for key in ('bytes', 'sha256')},
                 'Available retained initializer content differs')
            initializer_sources[row['path']] = source
    need(not missing, 'Missing fixed local initializer bytes; no stage created: ' + ','.join(missing))
    STAGE.mkdir()
    copies = []
    def copy(source, name, expected, source_base=ROOT):
        relative(name)
        installed = copy_verified(source, STAGE / name, expected,
                                  source_base=source_base, target_base=STAGE)
        copies.append({'source': pin(source, source_base), 'installed': installed})
        return installed
    # Every retained listed byte is rehashed at copy time. Old scripts are
    # excluded; old research files remain inert dependencies/evidence only.
    for name, expected in sorted(parent['files'].items()):
        if name.startswith('scripts/') or name in MODEL_PATHS:
            continue
        copy(PARENT / name, name, expected, PARENT)
    for name, source in initializer_sources.items():
        copy(source, name, {key: parent['model'][
            'r2' if name.endswith('codeformer_r2.pth') else 'u96'][key] for key in ('bytes', 'sha256')})
    for name in sorted(required):
        destination = ('evidence/matched_workspace_parents/' + name
                       if name in WORKSPACE_PARENTS else name)
        copy(ROOT / name, destination, approved['sources'][name])
    copied_data = {}
    for bank in ('new_optimization', 'new_monitors'):
        for row in selection[bank]:
            for key in ('input', 'target'):
                destination = row[key]['path']
                expected = {field: row[key][field] for field in ('bytes', 'sha256')}
                if destination not in copied_data:
                    source = ROOT / row['historical_candidate_workspace_' + key]['path']
                    copied_data[destination] = copy(source, destination, expected)
                need(copied_data[destination] == row[key], 'Repeated photograph target binding differs')
    need(len(copied_data) == 192, 'Exact160 inputs plus32 photographic targets required')
    evidence = {}
    fixed_sources = {
        'admission': DATA / 'data_admission_v1.json', 'candidate_data': DATA / 'candidate_data_v1.json',
        'parent_protocol': PARENT / parent['parent_protocol']['path'],
        'historical_pixel_audit': PARENT / parent['historical_pixel_audit']['path'],
        'historical_archive_index': PARENT / parent['historical_archive_index']['path'],
        'retained_continuation': PARENT / 'protocol.json',
    }
    for key, (name, size, digest) in packet.EVIDENCE_FIXED.items():
        expected = {'bytes': size, 'sha256': digest}
        if (STAGE / name).exists():
            need(content_pin(STAGE / name) == expected, 'Inherited evidence collision differs')
            evidence[key] = pin(STAGE / name, STAGE)
        else:
            evidence[key] = copy(fixed_sources[key], name, expected)
    proof_bindings = []
    for name, expected in sorted(approved['evidence'].items()):
        destination = 'evidence/matched_readiness/' + name
        proof_bindings.append(copy(ROOT / name, destination, expected))
    source_review = copy(REVIEW, 'evidence/matched_runner_source_review_v1.json', join_receipt_pins['review'])
    source_tests = copy(TESTS, 'evidence/matched_root_runner_tests_v1.json', join_receipt_pins['tests'])
    visual_plan = copy(VISUAL, 'evidence/matched_visual_plan_v1.json', approved['evidence'][VISUAL.relative_to(ROOT).as_posix()])
    need(required_sources() == required and
         validate_reviews(ROOT, read_json(REVIEW), read_json(TESTS), required, inherited) == approved and
         guard_sources == {name: content_pin(ROOT / name) for name in guard_sources} and
         join_receipt_pins == {'review': content_pin(REVIEW), 'tests': content_pin(TESTS)},
         'Reviewed sources, executable namespace or approval changed during copy')
    files = {path.relative_to(STAGE).as_posix(): content_pin(path)
             for path in sorted(STAGE.rglob('*')) if path.is_file()}
    protocol = {
        'format': packet.FORMAT, 'stage_name': packet.STAGE_NAME, 'output': packet.OUTPUT_RELATIVE,
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'files': files,
        'comparison': contract.prospective_contract(), 'limits': dict(contract.LIMITS),
        'selection': selection, 'operation_plan': contract.operation_plan(selection),
        'model': parent['model'], 'u64_head': parent['u64_head'], **evidence,
        'controls': parent['controls'], 'u64_parity': parent['u64_parity'],
        'quality_policy': parent['quality_policy'], 'visual_plan': visual_plan,
        'source_review': source_review, 'source_tests': source_tests, 'readiness_proofs': proof_bindings,
        'manual_launch_only': True, 'automatic_retry_resume': False, 'model_qualified': False,
        'training_success_not_implied': True,
        'source_status': 'PREPARED_UNRUN_MATCHED_TRAIN_COMPARISON_NOT_QUALIFIED',
        'only_R2_U96_files_omitted_from_execution_upload': True,
    }
    write_new(STAGE / 'protocol.json', protocol)
    protocol_sha = content_pin(STAGE / 'protocol.json')['sha256']
    # A fresh process is essential: packet module-origin checks bind staged
    # source rather than already loaded workspace modules. This guarded mode
    # verifies only transfer bytes and never imports the worker or frameworks.
    check = subprocess.run([sys.executable, '-B', str(STAGE / BOOTSTRAP), '--root', str(STAGE),
                            '--protocol-sha', protocol_sha, '--verify-transfer'],
                           cwd=STAGE, capture_output=True, text=True, timeout=180, check=False)
    need(check.returncode == 0, 'Fresh staged transfer-only verification failed: ' + check.stderr[-4000:])
    transfer = strict_json(check.stdout.strip().encode('utf-8'))
    need(transfer.get('complete') is True and transfer.get('neural_gradient_optimizer_training_calls') == 0,
         'Exact successful transfer-only verification receipt required')
    expected = included_archive_files(files)
    expected['protocol.json'] = content_pin(STAGE / 'protocol.json')
    need(sum(row['bytes'] for row in expected.values()) <= MAX_ARCHIVE, 'Execution payload cap1GiB exceeded')
    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.PAX_FORMAT) as tar:
                for name, row in sorted(expected.items()):
                    need(content_pin(STAGE / name) == row, 'Stage changed before archive member copy')
                    info = tarfile.TarInfo(STAGE_NAME + '/' + name)
                    info.size, info.mode, info.mtime = row['bytes'], 0o644, 0
                    with ordinary(STAGE / name).open('rb') as stream:
                        tar.addfile(info, stream)
        raw.flush()
        os.fsync(raw.fileno())
    audit = audit_archive(archive, expected)
    digest = audit['archive']['sha256']
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(digest + '  ' + archive.name + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    need(checksum.read_bytes() == (digest + '  ' + archive.name + '\n').encode('ascii'),
         'Exact LF SHA256 sidecar required')
    need(validate_reviews(ROOT, read_json(REVIEW), read_json(TESTS), required_sources(), inherited) == approved and
         join_receipt_pins == {'review': content_pin(REVIEW), 'tests': content_pin(TESTS)},
         'Approval or source changed during archive readback')
    receipt = {'complete': True, 'status': 'PREPARED_UNRUN_MANUAL_MATCHED_PACKET',
        'builder': pin(ROOT / BUILDER), 'protocol': pin(STAGE / 'protocol.json'),
        'archive': pin(archive), 'checksum': pin(checksum), 'copies': copies,
        'review': pin(REVIEW), 'tests': pin(TESTS), 'static_files': len(files), 'archive_audit': audit,
        'local_stage_retained_with_both_exact_model_weights': True,
        'missing_local_initializer_files': [], 'omitted_archive_files': sorted(MODEL_PATHS),
        'trained_u64_heads_included': True, 'transfer_only': transfer,
        'neural_model_gradient_optimizer_VM_calls': 0, 'model_qualified': False}
    write_new(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args(argv)
    need(args.build is True, 'Explicit --build required after current independent join and root tests')
    result = build()
    print(json.dumps({key: result[key] for key in ('complete', 'status', 'protocol', 'archive')}))


if __name__ == '__main__':
    main()
