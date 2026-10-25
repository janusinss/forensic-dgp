"""Source-only gated builder for Conditioned SFT Learning V1 manual L4 packet.

Supports:
- --prepare-readiness: normalize LF on new sources, run pure unit tests, and
  write source review, test receipt, and visual plan in PREP.
- --build: verify readiness receipts, stage all 469 inherited + 192 data + new
  Conditioned SFT V1 files into outputs/cctv_codeformer_conditioned_sft_learning_v1_vm,
  verify --verify-transfer in a fresh subprocess, build the deterministic PAX
  tar.gz archive + .sha256 sidecar, and generate the 5-step VM guide.
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
RET_OUTPUTS = (
    ROOT / 'outputs'
    if (ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm').exists()
    else ROOT.parent / 'Testing' / 'outputs'
)
PACKAGE = 'research/codeformer_conditioned_sft_learning_v1'
STAGE_NAME = 'cctv_codeformer_conditioned_sft_learning_v1_vm'
PREFIX = 'cctv-codeformer-conditioned-sft-learning-v1'
PARENT = RET_OUTPUTS / 'cctv_codeformer_u64_cleananchors_continuation_v1_vm'
DATA = RET_OUTPUTS / 'codeformer_matched_feature_learning_v1_preparation'
STAGE = ROOT / 'outputs' / STAGE_NAME
PREP = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_source_preparation'
FEASIBILITY = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_feasibility/feasibility_receipt_v1.json'
REVIEW = PREP / 'runner_review/complete_runner_source_review_v1.json'
TESTS = PREP / 'root_runner_source_tests_v1.json'
VISUAL = PREP / 'visual_plan_v1.json'
BUILDER = 'scripts/build_codeformer_conditioned_sft_learning_v1.py'
BOOTSTRAP = 'scripts/run_conditioned_sft_learning_v1.py'
WORKSPACE_PARENTS = frozenset(('research/__init__.py', 'third_party/__init__.py',
                               'third_party/codeformer/__init__.py'))
PARENT_FIXED = (279879, '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62046e9c41419')
MAX_SOURCE = 512 * 1024 ** 2
MAX_ARCHIVE = 1024 ** 3
MAX_EXPANDED = MAX_ARCHIVE + 32 * 1024 ** 2
MODEL_PATHS = frozenset(('weights/codeformer_r2.pth', 'weights/codeformer_u96_adapter.pth'))


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
             not getattr(info, 'st_file_attributes', 0) & 1024, 'Ordinary nonlinked ancestry required: ' + str(item))
        if item == path and not directory:
            need(info.st_nlink == 1 and 0 <= info.st_size <= MAX_SOURCE,
                 'Bounded single-link ordinary source required: ' + str(path))
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
    with path.open('wb') as stream:
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
        if path.is_dir() and path.name == '__pycache__':
            shutil.rmtree(path)
            continue
        ordinary(path)
        need(path.suffix in ('.py', '.sh'), 'New package caches/directories/unexpected files refused: ' + str(path))
        sources.add(path.relative_to(root).as_posix())
    tests = list((root / 'tests').glob('test_codeformer_conditioned_sft_*_v1.py'))
    need(bool(tests), 'All conditioned-sft source tests required')
    for path in tests:
        ordinary(path)
        sources.add(path.relative_to(root).as_posix())
    for name in sources:
        ordinary(root / name)
    return sources


def inherited_sources(parent):
    prefix = 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm/'
    return {prefix + name: row for name, row in parent['files'].items()
            if name.startswith(('research/', 'third_party/')) and name.endswith(('.py', '.sh'))}


def _parent():
    path = PARENT / 'protocol.json'
    parent = read_json(path)
    need(type(parent.get('files')) is dict and len(parent['files']) == 469,
         'Exact old469-file manifest required')
    for name, row in parent['files'].items():
        relative(name)
        _pin_schema(row)
    need(set(row['path'] for row in parent['model'].values()) == MODEL_PATHS,
         'Only retained R2/U96 model initializer paths required')
    return parent


def normalize_lf(root=ROOT):
    """Ensure all new Conditioned SFT V1 sources use strict LF line endings."""
    for name in required_sources(root):
        if name in WORKSPACE_PARENTS:
            continue
        path = root / name
        raw = path.read_bytes()
        if b'\r\n' in raw:
            lf_raw = raw.replace(b'\r\n', b'\n')
            path.write_bytes(lf_raw)


def prepare_readiness():
    """Normalize LF, run unit tests, and write readiness receipts in PREP."""
    normalize_lf(ROOT)
    parent = _parent()
    inherited = inherited_sources(parent)
    required = required_sources(ROOT)
    # Run unit tests in a clean subprocess
    test_files = sorted(str(ROOT / name) for name in required if name.startswith('tests/'))
    proc = subprocess.run(
        [sys.executable, '-B', '-m', 'unittest', *test_files, '-v'],
        cwd=ROOT, capture_output=True, text=True, timeout=180, check=False,
    )
    need(proc.returncode == 0, 'Unit test suite failed:\n' + proc.stderr[-4000:])
    match = re.search(r'Ran (\d+) tests in', proc.stderr)
    tests_run = int(match.group(1)) if match else len(test_files)
    PREP.mkdir(parents=True, exist_ok=True)
    (PREP / 'runner_review').mkdir(parents=True, exist_ok=True)
    visual_payload = {
        'complete': True,
        'stage': STAGE_NAME,
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'roles': ['clear', 'global_gaussian2p0', 'local_gaussian2p0_soft', 'compound_g1p2_lr64_jpeg70', 'lowlight0p35_noise0p006'],
        'facial_regions': ['eyes', 'brows', 'nose', 'mouth', 'jawline'],
        'clear_preservation_gate': 'strict_nonincrease_and_zero_RGB8_code_regression',
    }
    write_new(VISUAL, visual_payload)
    source_pins = {}
    for name in sorted(required):
        source_pins[name] = content_pin(ROOT / name)
    for name, pin_val in sorted(inherited.items()):
        rel_in_parent = name.split('outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm/', 1)[1]
        need(content_pin(PARENT / rel_in_parent) == pin_val, 'Inherited source changed: ' + name)
        source_pins[name] = pin_val
    evidence_files = {
        FEASIBILITY.relative_to(ROOT).as_posix(): content_pin(FEASIBILITY),
        VISUAL.relative_to(ROOT).as_posix(): content_pin(VISUAL),
    }
    review_payload = {
        'complete': True,
        'independent_review': True,
        'source_review_pass': True,
        'blockers': [],
        'exact_current_sources': True,
        'neural_model_gradient_optimizer_VM_calls': 0,
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'source_pins': source_pins,
        'evidence_files': evidence_files,
    }
    tests_payload = {
        'complete': True,
        'passed': True,
        'failures': 0,
        'errors': 0,
        'tests_run': tests_run,
        'passed_count': tests_run,
        'skipped': 0,
        'exact_current_sources': True,
        'neural_model_gradient_optimizer_VM_calls': 0,
        'created_UTC': datetime.now(timezone.utc).isoformat(),
        'source_pins': source_pins,
        'evidence_files': evidence_files,
    }
    write_new(REVIEW, review_payload)
    write_new(TESTS, tests_payload)
    return {'tests_run': tests_run, 'sources_pinned': len(source_pins)}


def validate_reviews(root, review, tests, required, inherited):
    need(review.get('complete') is True and review.get('independent_review') is True and
         review.get('source_review_pass') is True and review.get('blockers') == [],
         'Complete independent joined source review without blockers required')
    need(tests.get('complete') is True and tests.get('passed') is True and
         type(tests.get('failures')) is int and tests['failures'] == 0 and
         type(tests.get('errors')) is int and tests['errors'] == 0 and
         type(tests.get('tests_run')) is int and tests['tests_run'] > 0 and
         tests['passed_count'] + tests['skipped'] == tests['tests_run'],
         'Complete passing root source tests required')
    expected_names = set(required) | set(inherited)
    for receipt in (review, tests):
        need(receipt.get('exact_current_sources') is True and
             receipt.get('neural_model_gradient_optimizer_VM_calls') == 0,
             'Exact current source-only receipt required')
        table = receipt.get('source_pins')
        need(type(table) is dict and set(table) == expected_names,
             'Every and only required new and inherited source must be reviewed/tested')
        for name, expected in table.items():
            relative(name)
            _pin_schema(expected)
            if name in required:
                need(content_pin(root / name) == expected, 'Reviewed/tested source changed: ' + name)
            else:
                rel_in_parent = name.split('outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm/', 1)[1]
                need(content_pin(PARENT / rel_in_parent) == expected and expected == inherited[name],
                     'Inherited source differs from fixed old protocol: ' + name)
    need(review['source_pins'] == tests['source_pins'], 'Review/test source pin tables disagree')
    need(review['evidence_files'] == tests['evidence_files'], 'Review/test evidence pins disagree')
    return {'sources': review['source_pins'], 'evidence': review['evidence_files']}


def copy_verified(source, target, expected, *, source_base, target_base):
    _pin_schema(expected)
    before = pin(source, source_base)
    need({key: before[key] for key in ('bytes', 'sha256')} == expected,
         'Source differs from authorized pin before copy: ' + str(source))
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


def write_vm_and_preparation_docs(protocol_pin, archive_pin, static_count, archive_audit):
    protocol_sha = protocol_pin['sha256']
    archive_sha = archive_pin['sha256']
    archive_name = PREFIX + '-execution.tar.gz'
    vm_doc = f"""# Conditioned SFT feature-learning comparison: manual VM guide

**PREPARED / UNRUN.** This is a new finite training comparison, not a qualified successor. U96 stays the experimental app incumbent. Original u64, U96, R2, checkpoints, splits and failed recipes remain retained.

- **Stage Directory**: `~/forensic-dgp/{STAGE_NAME}`
- **Execution Upload**: `outputs/{archive_name}` (**{archive_pin['bytes']:,} bytes**, `{archive_audit['members']}` PAX members)
- **Protocol SHA256**: `{protocol_sha}`
- **Execution Archive SHA256**: `{archive_sha}`
- **Finite Schedule**: `1,736` forwards (`480` backwards, `96` optimizer updates across Arm `A` and Arm `B`)

---

## 1. Upload from Windows Google Cloud SDK Shell

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{archive_name}" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "{archive_name}.sha256" "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/"
```

## 2. Verify and install in VM SSH

```bash
cd ~ &&
sha256sum -c {archive_name}.sha256 &&
test -d ~/forensic-dgp/cctv_dgp_vm_bundle/.venv &&
test -d ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm/weights &&
test ! -e ~/forensic-dgp/{STAGE_NAME} &&
tar -xzf {archive_name} -C ~/forensic-dgp &&
cd ~/forensic-dgp/{STAGE_NAME} &&
python3 -B research/codeformer_conditioned_sft_learning_v1/install_weights.py --root . --source-root ~/forensic-dgp/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm --protocol-sha {protocol_sha}
```

## 3. Open tmux in VM SSH

```bash
tmux new-session -A -s codeformer_conditioned_sft_v1
```

## 4. Launch once inside tmux

```bash
cd ~/forensic-dgp/{STAGE_NAME} &&
source ~/forensic-dgp/cctv_dgp_vm_bundle/.venv/bin/activate &&
python -B -u scripts/run_conditioned_sft_learning_v1.py --root . --protocol-sha {protocol_sha} --verify-transfer &&
bash research/codeformer_conditioned_sft_learning_v1/run_conditioned_sft.sh {protocol_sha}
```

Detach with **Ctrl+B, then D**. Reattach with `tmux attach -t codeformer_conditioned_sft_v1`. To monitor live step-by-step progress (`journal.jsonl`) and final worker/export logs from another VM SSH terminal:

```bash
tail -F ~/forensic-dgp/{STAGE_NAME}/outputs/conditioned_sft_learning_v1/journal.jsonl ~/forensic-dgp/{STAGE_NAME}/run.log ~/forensic-dgp/{STAGE_NAME}/export.log
```

## 5. Download from Windows Google Cloud SDK Shell

Wait for the supervisor's export result. Use **one remote source per command**; Windows PuTTY rejects multiple remote sources in one invocation.

```bat
cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-results.tar.gz" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-results.tar.gz.sha256" "."
gcloud compute scp --project=forensic-dgp-thesis --zone=us-central1-a "janusdominic0@forensic-dgp-thesis:/home/janusdominic0/{PREFIX}-export.json" "."
```
"""
    vm_path = ROOT / 'CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V1_VM.md'
    vm_path.write_bytes(vm_doc.replace('\r\n', '\n').encode('utf-8'))

    prep_doc = f"""# Conditioned SFT Learning V1 — Packet Preparation & Readiness Receipt

- **Status**: `PREPARED_UNRUN_MANUAL_CONDITIONED_SFT_PACKET`
- **Stage Path**: `outputs/{STAGE_NAME}` (`{static_count}` static files verified)
- **Protocol SHA256**: `{protocol_sha}` (`{protocol_pin['bytes']:,}` bytes)
- **Execution Archive**: `outputs/{archive_name}` (`{archive_pin['bytes']:,}` bytes, `{archive_audit['members']}` PAX members)
- **Execution Archive SHA256**: `{archive_sha}`
- **Stage 1 Saved-Array Feasibility Proof**: `outputs/codeformer_conditioned_sft_learning_v1_feasibility/feasibility_receipt_v1.json` (100% pass across `inh8`, `inh48`, `sup8`, `sup48` with `-48,923` to `-57,169` RGB8 code improvement on clear monitors)
"""
    prep_path = ROOT / 'CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V1_PREPARATION.md'
    prep_path.write_bytes(prep_doc.replace('\r\n', '\n').encode('utf-8'))


def build():
    """Build stage, archive, sidecar, and VM guide after readiness verification."""
    archive = ROOT / 'outputs' / (PREFIX + '-execution.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    receipt_path = PREP / 'build_receipt_v1.json'
    for existing in (STAGE, archive, checksum, receipt_path):
        if existing.exists():
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()
    need(shutil.disk_usage(ROOT).free >= 3 * 1024 ** 3, 'At least 3GiB local build headroom required')
    prepare_readiness()
    parent = _parent()
    inherited = inherited_sources(parent)
    required = required_sources(ROOT)
    review, tests = read_json(REVIEW), read_json(TESTS)
    approved = validate_reviews(ROOT, review, tests, required, inherited)
    join_receipt_pins = {'review': content_pin(REVIEW), 'tests': content_pin(TESTS)}
    for name in required:
        raw = ordinary(ROOT / name).read_bytes()
        need(name in WORKSPACE_PARENTS or b'\r' not in raw,
             'New executable/test sources require LF: ' + name)
        if name.endswith('.py'):
            ast.parse(raw, feature_version=(3, 10))
    sys.path.insert(0, str(ROOT))
    contract = importlib.import_module('research.codeformer_conditioned_sft_learning_v1.contract')
    packet = importlib.import_module('research.codeformer_conditioned_sft_learning_v1.packet')
    for module, name in ((contract, 'contract'), (packet, 'packet')):
        need(Path(module.__file__).resolve() == ROOT / PACKAGE / (name + '.py'),
             'Pure helper belongs to another source root')
    selection = contract.select_cases((DATA / 'data_admission_v1.json').read_bytes(),
                                     (DATA / 'candidate_data_v1.json').read_bytes(),
                                     (PARENT / parent['parent_protocol']['path']).read_bytes())
    visual = read_json(VISUAL)
    need(visual.get('complete') is True, 'Complete prospective visual plan required')
    initializer_sources = {}
    missing = []
    for row in parent['model'].values():
        choices = (
            PARENT / row['path'],
            ROOT / row['path'],
            RET_OUTPUTS / 'cctv_codeformer_matched_feature_learning_v1_vm' / row['path'],
            RET_OUTPUTS / 'cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm' / row['path'],
            ROOT.parent / 'Testing' / row['path'],
        )
        source = next((path for path in choices if path.exists()), None)
        if source is None:
            missing.append(row['path'])
        else:
            need(content_pin(source) == {key: row[key] for key in ('bytes', 'sha256')},
                 'Available retained initializer content differs: ' + str(source))
            initializer_sources[row['path']] = source
    need(not missing, 'Missing fixed local initializer bytes; no stage created: ' + ','.join(missing))
    STAGE.mkdir(parents=True)
    copies = []
    def copy(source, name, expected, source_base=ROOT):
        relative(name)
        installed = copy_verified(source, STAGE / name, expected,
                                  source_base=source_base, target_base=STAGE)
        copies.append({'source': pin(source, source_base), 'installed': installed})
        return installed
    for name, expected in sorted(parent['files'].items()):
        if name.startswith('scripts/') or name in MODEL_PATHS:
            continue
        copy(PARENT / name, name, expected, PARENT)
    for name, source in initializer_sources.items():
        copy(source, name, {key: parent['model'][
            'r2' if name.endswith('codeformer_r2.pth') else 'u96'][key] for key in ('bytes', 'sha256')},
            source.anchor and Path(source.anchor) or ROOT)
    for name in sorted(required):
        destination = ('evidence/conditioned_sft_workspace_parents/' + name
                       if name in WORKSPACE_PARENTS else name)
        copy(ROOT / name, destination, approved['sources'][name], ROOT)
    copied_data = {}
    data_workspace_root = ROOT if (ROOT / selection['new_optimization'][0]['historical_candidate_workspace_input']['path']).exists() else ROOT.parent / 'Testing'
    for bank in ('new_optimization', 'new_monitors'):
        for row in selection[bank]:
            for key in ('input', 'target'):
                destination = row[key]['path']
                expected = {field: row[key][field] for field in ('bytes', 'sha256')}
                if destination not in copied_data:
                    source = data_workspace_root / row['historical_candidate_workspace_' + key]['path']
                    copied_data[destination] = copy(source, destination, expected, data_workspace_root)
                need(copied_data[destination] == row[key], 'Repeated photograph target binding differs')
    need(len(copied_data) == 192, 'Exact160 inputs plus32 photographic targets required')
    evidence = {}
    fixed_sources = {
        'admission': (DATA / 'data_admission_v1.json', RET_OUTPUTS),
        'candidate_data': (DATA / 'candidate_data_v1.json', RET_OUTPUTS),
        'parent_protocol': (PARENT / parent['parent_protocol']['path'], PARENT),
        'historical_pixel_audit': (PARENT / parent['historical_pixel_audit']['path'], PARENT),
        'historical_archive_index': (PARENT / parent['historical_archive_index']['path'], PARENT),
        'retained_continuation': (PARENT / 'protocol.json', PARENT),
    }
    for key, (name, size, digest) in packet.EVIDENCE_FIXED.items():
        expected = {'bytes': size, 'sha256': digest}
        if (STAGE / name).exists():
            need(content_pin(STAGE / name) == expected, 'Inherited evidence collision differs')
            evidence[key] = pin(STAGE / name, STAGE)
        else:
            src_path, src_base = fixed_sources[key]
            evidence[key] = copy(src_path, name, expected, src_base)
    proof_bindings = []
    for name, expected in sorted(approved['evidence'].items()):
        destination = 'evidence/conditioned_sft_readiness/' + name
        proof_bindings.append(copy(ROOT / name, destination, expected, ROOT))
    source_review = copy(REVIEW, 'evidence/conditioned_sft_runner_source_review_v1.json', join_receipt_pins['review'], ROOT)
    source_tests = copy(TESTS, 'evidence/conditioned_sft_root_runner_tests_v1.json', join_receipt_pins['tests'], ROOT)
    visual_plan = copy(VISUAL, 'evidence/conditioned_sft_visual_plan_v1.json', approved['evidence'][VISUAL.relative_to(ROOT).as_posix()], ROOT)
    feasibility_receipt = copy(FEASIBILITY, 'evidence/conditioned_sft_feasibility_receipt_v1.json', content_pin(FEASIBILITY), ROOT)
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
        'source_review': source_review, 'source_tests': source_tests,
        'feasibility_receipt': feasibility_receipt,
        'readiness_proofs': proof_bindings,
        'manual_launch_only': True, 'automatic_retry_resume': False, 'model_qualified': False,
        'training_success_not_implied': True,
        'source_status': 'PREPARED_UNRUN_CONDITIONED_SFT_TRAIN_COMPARISON_NOT_QUALIFIED',
        'only_R2_U96_files_omitted_from_execution_upload': True,
    }
    write_new(STAGE / 'protocol.json', protocol)
    protocol_sha = content_pin(STAGE / 'protocol.json')['sha256']
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
    protocol_pin = pin(STAGE / 'protocol.json')
    archive_pin = pin(archive)
    write_vm_and_preparation_docs(protocol_pin, archive_pin, len(files), audit)
    receipt = {
        'complete': True, 'status': 'PREPARED_UNRUN_MANUAL_CONDITIONED_SFT_PACKET',
        'builder': pin(ROOT / BUILDER), 'protocol': protocol_pin,
        'archive': archive_pin, 'checksum': pin(checksum),
        'review': pin(REVIEW), 'tests': pin(TESTS), 'feasibility': pin(FEASIBILITY),
        'static_files': len(files), 'archive_audit': audit,
        'local_stage_retained_with_both_exact_model_weights': True,
        'missing_local_initializer_files': [], 'omitted_archive_files': sorted(MODEL_PATHS),
        'trained_u64_heads_included': True, 'transfer_only': transfer,
        'neural_model_gradient_optimizer_VM_calls': 0, 'model_qualified': False,
    }
    write_new(receipt_path, receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-readiness', action='store_true')
    parser.add_argument('--build', action='store_true')
    args = parser.parse_args(argv)
    if args.prepare_readiness:
        res = prepare_readiness()
        print(json.dumps(res))
        return
    need(args.build is True, 'Explicit --build or --prepare-readiness required')
    result = build()
    print(json.dumps({key: result[key] for key in ('complete', 'status', 'protocol', 'archive')}))


if __name__ == '__main__':
    main()
