"""Bound and extract this returned diagnostic; no neural framework imported."""
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import stat
import tarfile

ROOT = Path(__file__).resolve().parents[1]
STAGE = 'cctv_codeformer_u64_png_checked_step_v1_vm'
PREFIX = 'cctv-codeformer-u64-png-checked-step-v1'
AUDIT = ROOT / 'outputs/codeformer_u64_png_checked_step_v1_return_audit'
ARCHIVE_SHA = 'f2d8f2670c0c4b7fd9310317d1a020ffb9f0b1211e07b9d0ed92023e7832fafc'
PROTOCOL_SHA = '7f4546d4d911bea7100f42870514f08a42e349ec524a2b9795f8f3c7f55dcf60'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def ordinary(path):
    path = Path(path).absolute()
    for ancestor in (path, *path.parents):
        info = ancestor.lstat()
        need(not stat.S_ISLNK(info.st_mode) and not getattr(info, 'st_file_attributes', 0) & 1024,
             'Linked/reparse paths refused')
        need(stat.S_ISREG(info.st_mode) if ancestor == path else stat.S_ISDIR(info.st_mode), 'Ordinary paths required')
    need(path.stat().st_nlink == 1, 'Single-link file required')
    return path


def pin(path):
    path = ordinary(path); before = path.stat(); h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    after = path.stat()
    need((before.st_ino, before.st_size, before.st_mtime_ns) ==
         (after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during hashing')
    return {'path': path.relative_to(ROOT).as_posix(), 'bytes': after.st_size, 'sha256': h.hexdigest()}


def doc(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            need(k not in result, 'Duplicate JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def relative(value):
    need(type(value) is str and value and '\\' not in value and ':' not in value, 'Canonical relative path required')
    p = PurePosixPath(value)
    need(not p.is_absolute() and p.as_posix() == value and
         all(x not in ('', '.', '..') for x in value.split('/')), 'Path escape refused')
    return p


def main():
    need(not AUDIT.exists(), 'Fresh inventory path required; preserve partial evidence')
    arc = ROOT / 'outputs' / (PREFIX + '-results.tar.gz')
    side = arc.with_name(arc.name + '.sha256')
    exp = ROOT / 'outputs' / (PREFIX + '-export.json')
    archive_pin = pin(arc); export_pin = pin(exp); checksum_pin = pin(side)
    need(archive_pin['bytes'] == 1112595923 and archive_pin['sha256'] == ARCHIVE_SHA, 'Exact returned archive required')
    need(side.read_bytes() == (ARCHIVE_SHA + '  ' + arc.name + '\n').encode(), 'Exact LF checksum required')
    export = doc(exp.read_bytes())
    need(export['complete'] is True and export['archive']['sha256'] == ARCHIVE_SHA and
         export['archive']['bytes'] == archive_pin['bytes'] and export['protocol_sha256'] == PROTOCOL_SHA,
         'Export archive/protocol binding differs')
    files = export['files']
    need(type(files) is dict and len(files) == 1469 and
         sum(v['bytes'] for v in files.values()) == 1227524329, 'Bounded known actual payload required')
    for name, value in files.items():
        relative(name)
        need(type(value) is dict and set(value) == {'path', 'bytes', 'sha256'} and value['path'] == name
             and type(value['bytes']) is int and 0 <= value['bytes'] <= 384*1024**2,
             'Exact bounded member pin required')
    raw = (ROOT / 'outputs' / STAGE / 'protocol.json').read_bytes()
    need(len(raw) == 382027 and hashlib.sha256(raw).hexdigest() == PROTOCOL_SHA, 'Exact sealed protocol required')
    protocol = doc(raw)
    for name, value in protocol['files'].items():
        need(files[name] == {'path': name, **value}, 'Returned static source differs from seal')
    AUDIT.mkdir(); extracted = AUDIT / 'extracted'; extracted.mkdir()
    actual = {}; extracted_files = {}
    with arc.open('rb') as compressed:
        with gzip.GzipFile(fileobj=compressed, mode='rb') as inflated:
            with tarfile.open(fileobj=inflated, mode='r|') as archive:
                for member in archive:
                    need(member.isfile() and member.type == tarfile.REGTYPE and not member.pax_headers,
                         'Only ordinary nonextended tar files admitted')
                    name = member.name
                    need(name.startswith(STAGE + '/'), 'Wrong archive stage prefix')
                    name = name[len(STAGE)+1:]; relative(name)
                    need(name in files and name not in actual and member.size == files[name]['bytes'],
                         'Unexpected/duplicate/wrong-sized member')
                    retain = name not in protocol['files'] and name != 'protocol.json'
                    target = extracted / name
                    if retain:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        sink = target.open('xb')
                    else:
                        sink = None
                    digest = hashlib.sha256(); total = 0
                    try:
                        source = archive.extractfile(member)
                        while True:
                            block = source.read(1024**2)
                            if not block:
                                break
                            total += len(block); need(total <= member.size, 'Payload exceeds declared size')
                            digest.update(block)
                            if sink:
                                sink.write(block)
                        if sink:
                            sink.flush(); os.fsync(sink.fileno())
                    finally:
                        if sink:
                            sink.close()
                    row = {'path': name, 'bytes': total, 'sha256': digest.hexdigest()}
                    need(row == files[name], 'Actual member hash differs: ' + name)
                    actual[name] = row
                    if retain:
                        need(pin(target)['sha256'] == row['sha256'] and target.stat().st_size == row['bytes'],
                             'Durable extracted bytes differ')
                        extracted_files[name] = row
            # Force gzip footer/CRC consumption; the independent framing audit
            # separately verifies all tar padding and compressed framing.
            while inflated.read(1024**2):
                pass
    need(set(actual) == set(files), 'Missing archive members')
    need(pin(arc) == archive_pin and pin(exp) == export_pin and pin(side) == checksum_pin,
         'Inputs changed while extracting')
    result = {'complete': True, 'format': 'u64-png-checked-step-return-inventory-v1',
        'archive': archive_pin, 'export': export_pin, 'checksum': checksum_pin,
        'protocol_sha256': PROTOCOL_SHA, 'files': actual, 'extracted_files': extracted_files,
        'evidence_root': str(extracted), 'archive_members': len(actual),
        'payload_bytes': sum(v['bytes'] for v in actual.values()),
        'extracted_members': len(extracted_files), 'no_arbitrary_tar_extract_used': True,
        'independent_full_framing_and_event_audit_separate': True,
        'neural_gradient_optimizer_training_VM_calls': 0, 'model_qualified': False}
    with (AUDIT / 'inventory_v1.json').open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, indent=2, allow_nan=False); stream.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('files', 'extracted_files')}, indent=2))


if __name__ == '__main__':
    main()
