"""Seal reviewed sources for a distinct four-update manual L4 pilot.

File/source work only. This builder imports no Torch, constructs no model or
optimizer, performs no gradient operation, and never connects to a VM.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
REL = 'research/codeformer_u64_gate_aligned_pilot_v1'
PREP = ROOT / 'outputs/codeformer_u64_gate_aligned_pilot_v1_preparation'
PARENT = ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm'
STAGE_NAME = 'cctv_codeformer_u64_gate_aligned_pilot_v1_vm'
STAGE = ROOT / 'outputs' / STAGE_NAME
PREFIX = 'cctv-codeformer-u64-gate-aligned-pilot-v1'
PARENT_SHA = '4db752a04222210a9fbccf176cf422c5fadeef443d39a15ae7b62086e9c41419'
SOURCES = tuple(REL + '/' + name for name in (
    '__init__.py','contract.py','arithmetic.py','quality.py','runtime.py','worker.py','run_gate_aligned.sh')) + (
    'research/__init__.py',
    'research/codeformer_u64_role_gradients_v1/__init__.py',
    'research/codeformer_u64_role_gradients_v1/contract.py',
    'research/codeformer_u64_gate_aligned_v1/__init__.py',
    'research/codeformer_u64_gate_aligned_v1/solver.py',
    'research/codeformer_u64_gate_aligned_v1/contract.py',
    'scripts/run_u64_gate_aligned_pilot_v1.py',
    'scripts/build_u64_gate_aligned_pilot_v1.py',)
REPORTS = (
    'CCTV_CODEFORMER_U64_ROLE_GRADIENTS_V1_RETURN_REVIEW.md',
    'CCTV_CODEFORMER_U64_GATE_ALIGNED_V1_DESIGN.md',
    'CCTV_CODEFORMER_U64_CLEANANCHORS_CONTINUATION_V1_RETURN_REVIEW.md',
    'outputs/codeformer_u64_gate_aligned_v1_design/independent_source_review_v1.json',)
GRADIENT_PROTOCOL = ROOT / 'outputs/cctv_codeformer_u64_role_gradients_v1_vm/protocol.json'
GRADIENT_PROTOCOL_PIN = {'bytes':184633,'sha256':'07d00785b4aef82c2e915a032fa2808064d0f728972f8c518694698a13f5d5e3'}
MODEL_ALIASES = {'weights/codeformer_r2.pth','weights/codeformer_u96_adapter.pth'}
HELPER_PINS = {
 'research/codeformer_u64_gate_aligned_v1/__init__.py': (87, '9eb734f64e4e7871f2357f8af11cabaf3a4b4efea1bf4d148bfb6455eaafadca'),
 'research/codeformer_u64_role_gradients_v1/__init__.py': (82, 'b13cc93cf4d889fde3d255356e1c9e7e483aa845de75e2ea4a94a2c562554baf'),
 'research/codeformer_u64_role_gradients_v1/contract.py': (11189, '90a11da6be2e531648e4c5051471946469e287d2bc151177006554932395074e'),
 'research/codeformer_u64_gate_aligned_v1/solver.py': (7511, '0e68b9ab6d7fbd019dbdee6119530b718fa29f38e0ea80aefc1891ecccea91ba'),
 'research/codeformer_u64_gate_aligned_v1/contract.py': (10905, '18bfd58795380e1ff2b3c1fab2179c87e7dfb19bf4a46cd0a03e7f6c10dcf4ca')}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def pin(path, base=ROOT):
    path, base = Path(path).absolute(), Path(base).absolute()
    need(path.is_relative_to(base), 'Source escapes declared base')
    for current in (path, *path.parents):
        info = current.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Linked/reparse ancestry refused')
        need(stat.S_ISREG(info.st_mode) if current == path else stat.S_ISDIR(info.st_mode),
             'Ordinary file/directory required')
    before = path.stat()
    need(before.st_nlink == 1 and 0 <= before.st_size <= 512 * 1024**2, 'Bounded single-link file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'Source changed during hashing')
    return {'path': path.relative_to(base).as_posix(), 'bytes': after.st_size,
            'sha256': digest.hexdigest()}


def doc(path):
    def pairs(items):
        result = {}
        for key, value in items:
            need(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON refused')
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_constant=invalid)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def admission(path, *, tests=False):
    value = doc(path)
    need(value.get('complete') is True, 'Complete reviewed receipt required')
    if tests:
        need(value.get('all_tests_pass') is True and
             value.get('neural_gradient_optimizer_training_VM_calls') == 0,
             'Current pure tests with zero model/VM operations required')
    else:
        need(value.get('source_review_pass') is True and value.get('unresolved_blockers') == [],
             'Independent source approval with no unresolved blockers required')
    rows = value['sources']
    need(type(rows) is list and len({r['path'] for r in rows}) == len(rows), 'Unique source bindings required')
    approved = {r['path']: r for r in rows}
    need(set(SOURCES) <= set(approved), 'Every runtime/builder source needs review and pure tests')
    for name in SOURCES:
        need(pin(ROOT / name) == approved[name], 'Source changed after approval: ' + name)
    return value


def parent_readback():
    binding = pin(PARENT / 'protocol.json', PARENT)
    need(binding['bytes'] == 279879 and binding['sha256'] == PARENT_SHA, 'Exact executed parent protocol required')
    parent = doc(PARENT / 'protocol.json')
    model_paths = {v['path'] for v in parent['model'].values()}
    aliases = {}
    consumed = []
    need(len(parent['files']) == 469, 'Retained 469-file parent namespace required')
    for name, expected in sorted(parent['files'].items()):
        candidate = PARENT / name
        if not candidate.exists():
            need(name in MODEL_ALIASES and name in model_paths, 'Missing retained non-model parent file: ' + name)
            candidate = ROOT / 'outputs/cctv_codeformer_u96_postfusion_smoothanchors_lr_v1_vm' / name
            aliases[name] = candidate.relative_to(ROOT).as_posix()
        actual = pin(candidate)
        need({k: actual[k] for k in ('bytes', 'sha256')} == expected, 'Parent bytes differ: ' + name)
        consumed.append({'parent_relative': name, 'actual_local': actual})
    return parent, {'complete': True, 'files': consumed, 'local_only_model_aliases': aliases,
        'VM_requires_all469_at_actual_parent_stage': True, 'VM_inventory_performed': False}


def copy_bound(source, destination, expected=None):
    before = pin(source)
    if expected is not None:
        need(before == expected, 'Source appeared/changed after review')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open('rb') as incoming, destination.open('xb') as outgoing:
        shutil.copyfileobj(incoming, outgoing, 1024**2)
        outgoing.flush(); os.fsync(outgoing.fileno())
    need(pin(source) == before, 'Source changed while copying')
    actual = pin(destination, STAGE)
    need(all(actual[k] == before[k] for k in ('bytes', 'sha256')), 'Copied source mismatch')
    return actual


def build():
    need(not STAGE.exists(), 'Fresh pilot stage required; preserve partial stage, do not retry')
    for ending in ('-execution.tar.gz', '-execution.tar.gz.sha256'):
        need(not (ROOT / 'outputs' / (PREFIX + ending)).exists(), 'Existing archive/checksum immutable')
    review_path, test_path = PREP / 'source_review_v1.json', PREP / 'source_tests_v1.json'
    review = admission(review_path)
    tests = admission(test_path, tests=True)
    need({r['path']: r for r in review['sources']} == {r['path']: r for r in tests['sources']},
         'Reviewed/tested current sources disagree')
    approved = {r['path']: r for r in review['sources']}
    for name, (size, digest) in HELPER_PINS.items():
        need(approved[name] == {'path':name,'bytes':size,'sha256':digest}, 'Exact reviewed helper source required')
    parent, parent_proof = parent_readback()
    sys.path.insert(0, str(ROOT))
    contract = importlib.import_module('research.codeformer_u64_gate_aligned_pilot_v1.contract')
    need(Path(contract.__file__).resolve() == ROOT / REL / 'contract.py', 'Own inert contract required')
    parent_raw = (PARENT / 'protocol.json').read_bytes()
    standard_raw = (PARENT / parent['parent_protocol']['path']).read_bytes()
    selection = contract.select_cases(parent_raw, standard_raw)
    plan = contract.operation_plan(selection)
    need(len(plan) == 544 and sum(row['kind']=='forward' for row in plan)==540 and sum(row['kind']=='parameter_application' for row in plan)==4 and sum(len(row.get('gradient_queries',[])) for row in plan)==512, 'Exactly540 forwards512 queries4 applications required')
    gradient_pin = pin(GRADIENT_PROTOCOL)
    need({k:gradient_pin[k] for k in ('bytes','sha256')} == GRADIENT_PROTOCOL_PIN, 'Exact reviewed gradient protocol required')
    pilot = contract.prospective_contract(parent_raw,standard_raw,GRADIENT_PROTOCOL.read_bytes())
    STAGE.mkdir()
    files = {}
    for relative in SOURCES:
        row = copy_bound(ROOT / relative, STAGE / relative, approved[relative])
        files[relative] = {k: row[k] for k in ('bytes', 'sha256')}
    # The user-facing scripts location is an exact, separately bound shell alias.
    alias = 'scripts/run_gate_aligned.sh'
    row = copy_bound(ROOT / REL / 'run_gate_aligned.sh', STAGE / alias,
                     approved[REL + '/run_gate_aligned.sh'])
    files[alias] = {k: row[k] for k in ('bytes', 'sha256')}
    for index, relative in enumerate(REPORTS):
        target = 'evidence/review_' + str(index) + '_' + Path(relative).name
        row = copy_bound(ROOT / relative, STAGE / target)
        files[target] = {k: row[k] for k in ('bytes', 'sha256')}
    evidence = {'source_review': (review_path, 'evidence/source_review.json'),
                'source_tests': (test_path, 'evidence/source_tests.json'),
                'parent_protocol_copy': (PARENT / 'protocol.json', 'evidence/parent_protocol.json'),
                'standard_protocol_copy': (PARENT / parent['parent_protocol']['path'], 'evidence/standard_protocol.json'),
                'gradient_protocol_copy': (GRADIENT_PROTOCOL, 'evidence/gradient_protocol.json')}
    evidence_pins = {}
    for key, (source, target) in evidence.items():
        row = copy_bound(source, STAGE / target)
        files[target] = {k: row[k] for k in ('bytes', 'sha256')}
        evidence_pins[key] = row
    write_new(STAGE / 'evidence/local_parent_readback.json', parent_proof)
    row = pin(STAGE / 'evidence/local_parent_readback.json', STAGE)
    files[row['path']] = {k: row[k] for k in ('bytes', 'sha256')}
    protocol = {'format': 'u64-gate-aligned-pilot-manual-L4-v1', 'stage_name': STAGE_NAME,
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'files': files,
        'parent_protocol': {'bytes': 279879, 'sha256': PARENT_SHA,
                            'stage_name': PARENT.name},
        'limits': contract.LIMITS, 'pilot': pilot,
        'selection': selection, 'plan': plan, **evidence_pins,
        'manual_launch_only': True, 'automatic_retry_resume': False,
        'model_qualified': False, 'training_success_not_implied': True,
        'original_u64_is_retained': True, 'app_incumbent_unchanged': True}
    write_new(STAGE / 'protocol.json', protocol)
    protocol_pin = pin(STAGE / 'protocol.json', STAGE)
    archive = ROOT / 'outputs' / (PREFIX + '-execution.tar.gz')
    expected_staged = {'protocol.json':protocol_pin,
                       **{name:{'path':name,**value} for name,value in files.items()}}
    class Reader:
        def __init__(self, source):
            self.source=source; self.digest=hashlib.sha256()
        def read(self, size):
            data=self.source.read(size); self.digest.update(data); return data
    with archive.open('xb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
                for source in sorted(STAGE.rglob('*')):
                    if not source.is_file():
                        continue
                    relative=source.relative_to(STAGE).as_posix()
                    need(pin(source, STAGE) == expected_staged[relative], 'Staged source changed before archive')
                    info = tar.gettarinfo(str(source), arcname=STAGE_NAME + '/' + source.relative_to(STAGE).as_posix())
                    info.uid = info.gid = 0; info.uname = info.gname = ''; info.mtime = 0
                    info.mode = 0o644
                    with source.open('rb') as stream:
                        reader=Reader(stream); tar.addfile(info, reader)
                        need(reader.digest.hexdigest() == expected_staged[relative]['sha256'], 'Archived source payload changed')
                    need(pin(source, STAGE) == expected_staged[relative], 'Staged source changed while archiving')
        raw.flush(); os.fsync(raw.fileno())
    archive_pin = pin(archive)
    for name,binding in expected_staged.items():
        need(pin(STAGE/name, STAGE) == binding, 'Staged payload changed after archive')
    need(archive_pin['bytes'] < 16 * 1024**2, 'Small source-only transfer must remain below16MiB')
    checksum = archive.with_name(archive.name + '.sha256')
    with checksum.open('x', encoding='ascii', newline='\n') as stream:
        stream.write(archive_pin['sha256'] + '  ' + archive.name + '\n')
        stream.flush(); os.fsync(stream.fileno())
    receipt = {'complete': True, 'status': 'SEALED_UNRUN_PENDING_INDEPENDENT_ARCHIVE_GUIDE_REVIEW',
        'archive': archive_pin, 'checksum': pin(checksum), 'protocol': protocol_pin,
        'sources': [approved[name] for name in SOURCES], 'parent_static_files_rehashed': len(parent['files']),
        'local_alias_count': len(parent_proof['local_only_model_aliases']),
        'stage_static_files': len(files), 'full_forwards_planned': 540, 'gradient_queries_planned': 512, 'parameter_applications_planned': 4,
        'optimizer_objects': 0, 'actual_parameter_applications': 0, 'neural_gradient_optimizer_training_VM_calls': 0,
        'model_qualified': False}
    write_new(PREP / 'build_receipt_v1.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', action='store_true', required=True)
    parser.parse_args()
    build()
