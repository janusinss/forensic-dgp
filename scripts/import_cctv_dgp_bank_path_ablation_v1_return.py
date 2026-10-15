"""Hash-check the thin frozen diagnostic return without importing VM code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT/'outputs'
NAME = 'cctv_dgp_bank_path_ablation_v1_vm'
PREFIX = 'cctv_dgp_bank_path_ablation_v1'
STEM = 'cctv-dgp-bank-path-ablation-v1'
PIN = 'c3cdd781858dbfc0e9e73a4165f63fc34806fc4a2b74284bdcd59da390b7331f'
RETURN = OUTPUT/(PREFIX+'_return')
ANALYSIS = OUTPUT/(PREFIX+'_analysis')
PARENT = OUTPUT/'cctv_dgp_bank_photometric_guard_v1_return'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False)+'\n')


def main():
    start = time.monotonic()
    assert not RETURN.exists() and not ANALYSIS.exists(), 'Preserve prior imports'
    prepared = OUTPUT/NAME
    assert sha(prepared/'protocol.json') == PIN
    p = read(prepared/'protocol.json')
    transfer_path = OUTPUT/(PREFIX+'_transfer_audit.json')
    transfer = read(transfer_path)
    assert transfer['complete'] and transfer['protocol_sha256'] == PIN
    packet = {**p['assets_sha256'], 'protocol.json': PIN}
    for name, expected in packet.items():
        assert sha(prepared/name) == expected, name
    parent_bytes = 0
    for name, expected in p['parent_required_sha256'].items():
        path = PARENT/name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(PARENT)
        assert sha(path) == expected, name
        parent_bytes += path.stat().st_size
    print(dict(parent_assets_verified=len(p['parent_required_sha256']), parent_bytes=parent_bytes), flush=True)
    protected_path = OUTPUT/'cctv_dgp_bank_photometric_guard_v1_analysis/local_protected_before.json'
    protected = read(protected_path)['files_sha256']
    assert len(protected) == 12
    for name, expected in protected.items():
        assert sha(ROOT/name) == expected, name
    archive = OUTPUT/(STEM+'-results.tar.gz')
    checksum = OUTPUT/(STEM+'-results.tar.gz.sha256')
    exported_path = OUTPUT/(STEM+'-export.json')
    exported = read(exported_path)
    line = checksum.read_text(encoding='ascii').strip().split()
    assert len(line) == 2 and line[1] == archive.name
    assert exported['complete'] and exported['diagnostic_success_not_implied']
    assert exported['optimizer_updates'] == 0
    assert sha(archive) == line[0] == exported['archive_sha256']
    assert archive.stat().st_size == exported['bytes']
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) <= 1500
        names, folded = {}, set()
        reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{s}{i}' for s in ['COM', 'LPT'] for i in range(1, 10)}
        for member in members:
            path = PurePosixPath(member.name)
            assert member.isfile() and not member.islnk() and not member.issym()
            assert not path.is_absolute() and '..' not in path.parts and path.parts[0] == NAME
            assert len(path.parts) >= 2 and path.as_posix() == member.name
            assert ':' not in member.name and '\\' not in member.name
            assert all(c == c.rstrip(' .') and c.split('.')[0].upper() not in reserved for c in path.parts)
            name = PurePosixPath(*path.parts[1:]).as_posix()
            assert name.casefold() not in folded
            names[name] = member; folded.add(name.casefold())
        assert all(not any(ancestor.as_posix().casefold() in folded for ancestor in PurePosixPath(n).parents
                          if ancestor.as_posix() != '.') for n in names)
        with tar.extractfile(names['protocol.json']) as stream:
            assert hashlib.sha256(stream.read()).hexdigest() == PIN
        with tar.extractfile(names['export_manifest.json']) as stream:
            manifest = json.load(stream)
        assert manifest['protocol_sha256'] == PIN
        assert manifest['parent_protocol_sha256'] == p['parent_protocol_sha256']
        assert manifest['parent_required_sha256'] == p['parent_required_sha256']
        assert manifest['diagnostic_only'] and manifest['optimizer_updates'] == 0
        assert set(names) == set(manifest['files']) | {'export_manifest.json'}
        assert all(manifest['files'][name] == value for name, value in packet.items())
        assert sum(names[n].size for n in manifest['files']) == manifest['uncompressed_bytes']
        uncompressed = sum(m.size for m in members)
        assert uncompressed <= p['output_byte_cap']+32*1024**2
        assert shutil.disk_usage(OUTPUT).free >= uncompressed+2*1024**3
        ANALYSIS.mkdir(); RETURN.mkdir()
        write(ANALYSIS/'local_protected_before.json', dict(complete=True, files_sha256=protected,
              UTC=datetime.now(timezone.utc).isoformat(), freshly_rechecked=True))
        for index, (name, member) in enumerate(names.items(), 1):
            target = RETURN.joinpath(*PurePosixPath(name).parts)
            assert target.resolve().is_relative_to(RETURN.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with tar.extractfile(member) as source, target.open('xb') as sink:
                for chunk in iter(lambda: source.read(4*1024**2), b''):
                    h.update(chunk); sink.write(chunk)
            assert target.stat().st_size == member.size
            if name != 'export_manifest.json':
                assert h.hexdigest() == manifest['files'][name], name
            if index % 400 == 0:
                print(dict(returned_files=index, of=len(names)), flush=True)
    for name, expected in protected.items():
        assert sha(ROOT/name) == expected
    write(ANALYSIS/'import.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), archive_sha256=line[0], archive_bytes=archive.stat().st_size,
        checksum_sha256=sha(checksum), export_receipt_sha256=sha(exported_path),
        protocol_sha256=PIN, export_manifest_sha256=sha(RETURN/'export_manifest.json'),
        returned_file_bindings=len(names), uncompressed_bytes=uncompressed,
        parent_file_bindings=len(p['parent_required_sha256']), parent_bytes=parent_bytes,
        every_member_path_size_and_hash_checked=True, prepared_packet_unchanged=True,
        protected_app_checkpoint_examples_unchanged=True, transfer_audit_sha256=sha(transfer_path),
        complete_diagnostic_reported=manifest['complete_diagnostic'], failure_reported=manifest['failure_present'],
        VM_worker_imported_or_executed=False, local_neural_gradient_or_optimizer_calls=0,
        independent_audit_pending=True, model_qualified=False, goal_complete=False,
        seconds=time.monotonic()-start))
    print(dict(complete=True, returned_files=len(names), complete_diagnostic_reported=manifest['complete_diagnostic'],
               failure_reported=manifest['failure_present']), flush=True)


if __name__ == '__main__':
    main()
