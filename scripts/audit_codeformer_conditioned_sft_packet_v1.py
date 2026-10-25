"""Independent read-only Conditioned SFT V1 packet audit; explicit --audit after build.

Standard library only. No extraction, builder invocation, worker/model import,
VM command or launch is supported. The optional subprocess is transfer-only
bootstrap verification guarded by a framework/model/worker import blocker.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.abc
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tarfile
import zlib


sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
RET_OUTPUTS = (
    ROOT / 'outputs'
    if (ROOT / 'outputs/cctv_codeformer_u64_cleananchors_continuation_v1_vm').exists()
    else ROOT.parent / 'Testing' / 'outputs'
)
PACKAGE = 'research/codeformer_conditioned_sft_learning_v1'
STAGE_NAME = 'cctv_codeformer_conditioned_sft_learning_v1_vm'
PREFIX = 'cctv-codeformer-conditioned-sft-learning-v1'
PREP = ROOT / 'outputs/codeformer_conditioned_sft_learning_v1_source_preparation'
OMITTED = frozenset(('weights/codeformer_r2.pth', 'weights/codeformer_u96_adapter.pth'))
MAX_ARCHIVE = 1024**3
MAX_EXPANDED = MAX_ARCHIVE + 32 * 1024**2
WORKSPACE_PARENTS = frozenset(('research/__init__.py', 'third_party/__init__.py',
                               'third_party/codeformer/__init__.py'))
FORBIDDEN = frozenset(('torch', 'torchvision', 'torchaudio', 'numpy', 'PIL',
                       'tensorflow', 'jax', 'jaxlib', 'keras'))


def need(condition, message):
    if not condition:
        raise ValueError(message)


class ImportBlocker(importlib.abc.MetaPathFinder):
    attempted = []

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN or fullname.endswith(('.worker', '.model')):
            self.attempted.append(fullname)
            raise RuntimeError('Forbidden packet-audit import: ' + fullname)
        return None


def ordinary(path, *, directory=False):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        info = item.lstat()
        valid = stat.S_ISDIR(info.st_mode) if directory or item != path else stat.S_ISREG(info.st_mode)
        need(valid and not stat.S_ISLNK(info.st_mode)
             and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Ordinary unlinked audit source required: ' + str(item))
        if item == path and not directory:
            need(info.st_nlink == 1 and 0 <= info.st_size <= MAX_ARCHIVE,
                 'Bounded single-link audit source required')
    return path.resolve(strict=True)


def content(path):
    path = ordinary(path)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024**2), b''):
            digest.update(chunk)
    after = path.stat()
    need((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns),
         'Source changed during independent hashing: ' + str(path))
    return {'bytes': after.st_size, 'sha256': digest.hexdigest()}


def relative(value):
    need(type(value) is str and value and '\\' not in value and ':' not in value
         and all(ord(character) >= 32 for character in value), 'Canonical relative archive path required')
    parsed = PurePosixPath(value)
    need(not parsed.is_absolute() and parsed.as_posix() == value and value != '.'
         and all(part not in ('', '.', '..') for part in parsed.parts), 'Archive path escape or alias refused')
    return parsed


def pin_schema(value):
    need(type(value) is dict and set(value) == {'bytes', 'sha256'}
         and type(value['bytes']) is int and 0 <= value['bytes'] <= MAX_ARCHIVE
         and type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
         'Exact finite content pin required')


def strict(raw):
    need(type(raw) is bytes and len(raw) <= 32 * 1024**2, 'Bounded exact JSON bytes required')
    def pairs(rows):
        result = {}
        for key, value in rows:
            need(key not in result, 'Duplicate JSON key refused')
            result[key] = value
        return result
    def reject(value):
        raise ValueError('Nonfinite JSON token refused: ' + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject)
    def finite(item, depth=0):
        need(depth <= 64, 'Bounded independent JSON nesting required')
        if type(item) is float:
            need(math.isfinite(item), 'Finite independent JSON number required')
        elif type(item) is dict:
            for child in item.values(): finite(child, depth + 1)
        elif type(item) is list:
            for child in item: finite(child, depth + 1)
    finite(result)
    return result


def read_json(path):
    path = ordinary(path)
    before = content(path)
    raw = path.read_bytes()
    need({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == before
         and content(path) == before, 'JSON changed before independent decode')
    return strict(raw)


def bound(stage, value):
    need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'}, 'Exact path/content binding required')
    relative(value['path'])
    pin = {key: value[key] for key in ('bytes', 'sha256')}
    pin_schema(pin)
    need(content(stage / value['path']) == pin, 'Actual bound bytes differ: ' + value['path'])
    return pin


def single_gzip_stream(path, expected_expanded):
    expanded = 0
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with ordinary(path).open('rb') as stream:
        while True:
            compressed = stream.read(1024**2)
            if not compressed:
                break
            pending = compressed
            while pending:
                output = decoder.decompress(pending, 1024**2)
                expanded += len(output)
                need(expanded <= MAX_EXPANDED, 'Finite expanded gzip cap exceeded')
                pending = decoder.unconsumed_tail
                if decoder.eof:
                    need(not decoder.unused_data and not pending and not stream.read(1),
                         'Only one complete gzip stream, no compressed trailing bytes, allowed')
                    break
            if decoder.eof:
                break
    need(decoder.eof and expanded == expected_expanded, 'Truncated gzip or expanded byte count differs')
    return expanded


def audit_archive(path, expected):
    need(content(path)['bytes'] <= MAX_ARCHIVE, 'Execution archive exceeds1GiB')
    seen, members = set(), []
    payload_bytes = 0
    class Counted:
        def __init__(self, stream): self.stream, self.count = stream, 0
        def read(self, size=-1):
            block = self.stream.read(size)
            self.count += len(block)
            need(self.count <= MAX_EXPANDED, 'Finite expanded tar cap exceeded')
            return block
    with ordinary(path).open('rb') as compressed, gzip.GzipFile(fileobj=compressed, mode='rb') as expanded:
        counted = Counted(expanded)
        with tarfile.open(fileobj=counted, mode='r|') as archive:
            for member in archive:
                need(member.isfile() and not member.issym() and not member.islnk()
                     and type(member.size) is int and 0 <= member.size <= MAX_ARCHIVE,
                     'Only bounded regular archive payloads admitted')
                prefix = STAGE_NAME + '/'
                need(member.name.startswith(prefix), 'Archive member outside exact new stage')
                name = member.name[len(prefix):]
                relative(name)
                need(name not in seen and name in expected, 'Duplicate or unplanned archive member: ' + name)
                seen.add(name)
                need(member.size == expected[name]['bytes'], 'Archive member size differs: ' + name)
                stream = archive.extractfile(member)
                need(stream is not None, 'Regular payload must be readable without extraction')
                digest, size = hashlib.sha256(), 0
                for block in iter(lambda: stream.read(1024**2), b''):
                    digest.update(block); size += len(block); payload_bytes += len(block)
                    need(payload_bytes <= MAX_ARCHIVE, 'Archive payload aggregate exceeds1GiB')
                need(size == member.size and digest.hexdigest() == expected[name]['sha256'],
                     'Independent payload hash differs: ' + name)
                members.append({'path': name, 'bytes': size, 'sha256': digest.hexdigest()})
            trailing = 0
            while True:
                block = archive.fileobj.read(1024**2)
                if not block:
                    break
                need(not any(block), 'Nonzero trailing tar bytes refused')
                trailing += len(block)
            need(trailing >= 512, 'Second zero tar terminator must remain after first stop block')
    need(seen == set(expected), 'Archive omitted or added an expected execution payload')
    single_gzip_stream(path, counted.count)
    return {'complete': True, 'member_count': len(members), 'payload_bytes': payload_bytes,
        'expanded_stream_bytes': counted.count, 'trailing_zero_bytes': trailing,
        'single_complete_gzip_stream': True, 'members': members,
        'no_extractall': True, 'no_checkpoint_unpickle_or_array_decode': True}


def audit(*, guide=None):
    stage = ordinary(ROOT / 'outputs' / STAGE_NAME, directory=True)
    archive = ordinary(ROOT / 'outputs' / (PREFIX + '-execution.tar.gz'))
    checksum = ordinary(Path(str(archive) + '.sha256'))
    receipt_path = PREP / 'build_receipt_v1.json'
    receipt = read_json(receipt_path)
    protocol_path = stage / 'protocol.json'
    protocol = read_json(protocol_path)
    before = {'protocol': content(protocol_path), 'archive': content(archive),
              'checksum': content(checksum), 'builder_receipt': content(receipt_path)}
    need(protocol['stage_name'] == STAGE_NAME and protocol['manual_launch_only'] is True
         and protocol['automatic_retry_resume'] is False and protocol['model_qualified'] is False,
         'Fresh manual unqualified conditioned-sft packet required')
    files = protocol['files']
    need(type(files) is dict and 0 < len(files) <= 1600 and 'protocol.json' not in files,
         'Exact bounded nonrecursive file manifest required')
    observed = set()
    for path in stage.rglob('*'):
        if path.is_dir(): ordinary(path, directory=True); continue
        ordinary(path); name = path.relative_to(stage).as_posix(); relative(name)
        need('__pycache__' not in path.parts and not path.name.endswith(('.pyc', '.pyo')),
             'Unexpected executable cache refused')
        observed.add(name)
    need(observed == set(files) | {'protocol.json'}, 'Fresh stage namespace differs from exact sealed file table')
    for name, pin in files.items():
        relative(name); pin_schema(pin)
        need(content(stage / name) == pin, 'Independent stage file hash differs: ' + name)
    need(set(protocol['model']) == {'r2', 'u96'} and
         {row['path'] for row in protocol['model'].values()} == OMITTED,
         'Only the exact two large initializer paths may be omitted from upload')
    for row in protocol['model'].values(): bound(stage, row)
    bound(stage, protocol['u64_head'])
    bound(stage, protocol['feasibility_receipt'])
    need(protocol['u64_head']['path'] == 'weights/terminal_u0064_heads.npz', 'Learned u64 heads must be included')
    parent = read_json(stage / protocol['retained_continuation']['path'])
    need(len(parent['files']) == 469, 'Exact469 retained parent file bindings required')
    blocker = ImportBlocker(); sys.meta_path.insert(0, blocker)
    need('research' not in sys.modules, 'Independent audit must begin with a fresh package namespace')
    sys.path.insert(0, str(stage))
    from research.codeformer_conditioned_sft_learning_v1 import packet, contract
    for module, relative_path in ((packet, PACKAGE + '/packet.py'), (contract, PACKAGE + '/contract.py')):
        need(Path(module.__file__).resolve() == stage / relative_path, 'Audit imported a different source origin')
    checked_protocol = packet.verify_static(stage, before['protocol']['sha256'])
    need(checked_protocol == protocol, 'Fresh pure packet verification changed decoded protocol')
    selection = contract.select_cases((stage / protocol['admission']['path']).read_bytes(),
        (stage / protocol['candidate_data']['path']).read_bytes(), (stage / protocol['parent_protocol']['path']).read_bytes())
    operations = contract.operation_plan(selection)
    need(selection == protocol['selection'] and operations == protocol['operation_plan']
         and len(operations) == 1736 and sum(row['backward'] for row in operations) == 480
         and sum(row['optimizer_application_after_this_forward'] for row in operations) == 96
         and protocol['limits']['optimizer_constructions'] == 2,
         'Exact rederived selection and1736/480/96/2 finite scope required')
    expected = {name: pin for name, pin in files.items() if name not in OMITTED}
    expected['protocol.json'] = before['protocol']
    archive_audit = audit_archive(archive, expected)
    digest = before['archive']['sha256']
    need(checksum.read_bytes() == (digest + '  ' + archive.name + '\n').encode('ascii'),
         'Execution checksum sidecar bytes differ from actual archive')
    bootstrap = stage / 'scripts/run_conditioned_sft_learning_v1.py'
    code = '''import importlib.abc,runpy,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'torch','numpy','PIL','tensorflow','jax','jaxlib','keras','torchvision','torchaudio'} or fullname.endswith(('.worker','.model')):
   raise RuntimeError('Forbidden transfer-only import:'+fullname)
  return None
sys.meta_path.insert(0,Block())
script,root,sha=sys.argv[1:]
sys.argv=[script,'--root',root,'--protocol-sha',sha,'--verify-transfer']
runpy.run_path(script,run_name='__main__')
'''
    check = subprocess.run([sys.executable, '-B', '-c', code, str(bootstrap), str(stage),
        before['protocol']['sha256']], cwd=stage, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'),
        capture_output=True, text=True, timeout=180, check=False)
    need(check.returncode == 0, 'Fresh guarded transfer-only bootstrap failed: ' + check.stderr[-4000:])
    transfer = strict(check.stdout.strip().encode())
    need(transfer['complete'] is True and transfer['neural_gradient_optimizer_training_calls'] == 0
         and transfer['maximum_full_forwards'] == 1736, 'Transfer verification must remain inert and correctly scoped')
    guide_pin = None
    if guide is not None:
        guide = ordinary(guide)
        text = guide.read_text(encoding='utf-8')
        guide_pin = {'path': str(guide), **content(guide)}
        need(before['protocol']['sha256'] in text and digest in text,
             'Guide must bind exact current execution protocol/archive hashes')
        for name in (PREFIX + '-results.tar.gz', PREFIX + '-results.tar.gz.sha256', PREFIX + '-export.json'):
            need(name in text, 'Missing exact single-source return filename in guide: ' + name)
    after = {'protocol': content(protocol_path), 'archive': content(archive),
             'checksum': content(checksum), 'builder_receipt': content(receipt_path)}
    need(before == after and not blocker.attempted, 'Artifacts or import boundary changed during audit')
    return {
        'complete': True, 'status': 'INDEPENDENT_EXACT_PACKET_AUDIT_PREPARED_UNRUN_NOT_QUALIFIED',
        'created_UTC': datetime.now(timezone.utc).isoformat(), 'auditor': content(Path(__file__)),
        'artifacts': after, 'static_file_count': len(files), 'parent_bindings_verified': 469,
        'omitted_archive_files': sorted(OMITTED), 'trained_u64_heads_included': True,
        'archive_audit': archive_audit,
        'pure_static_selection_and_scope_1736_480_96_2_rederived': True,
        'fresh_guarded_transfer_only_bootstrap': transfer,
        'guide': guide_pin, 'guide_review_complete': guide_pin is not None,
        'forbidden_import_attempts': blocker.attempted,
        'worker_model_neural_gradient_optimizer_VM_browser_calls': 0,
        'no_archive_extraction': True, 'no_builder_or_worker_execution': True,
        'model_qualified': False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', action='store_true')
    parser.add_argument('--guide', type=Path, default=ROOT / 'CCTV_CODEFORMER_CONDITIONED_SFT_LEARNING_V1_VM.md')
    args = parser.parse_args()
    need(args.audit, 'Explicit --audit required after completed build')
    result = audit(guide=args.guide)
    output = PREP / 'independent_packet_audit_v1.json'
    if output.exists():
        output.unlink()
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'complete': result['complete'], 'receipt': str(output),
        'static_files': result['static_file_count'], 'archive_members': result['archive_audit']['member_count'],
        'model_qualified': False}))


if __name__ == '__main__':
    main()
