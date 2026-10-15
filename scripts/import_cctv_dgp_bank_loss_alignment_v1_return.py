"""Import a hash-bound measurement or retained failure without VM code imports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs'
NAME = 'cctv_dgp_bank_loss_alignment_v1_vm'
STEM = 'cctv-dgp-bank-loss-alignment-v1'
PIN = '67838e1a9add105baf01a34fb14a83562d687540de61fc6338384143651241e1'
RETURN = OUT / 'cctv_dgp_bank_loss_alignment_v1_return'
ANALYSIS = OUT / 'cctv_dgp_bank_loss_alignment_v1_analysis'
PARENT = OUT / 'cctv_dgp_bank_photometric_guard_v1_return'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024**2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    start = time.monotonic()
    assert not RETURN.exists() and not ANALYSIS.exists(), 'Preserve previous imports'
    prepared = OUT / NAME
    assert sha(prepared / 'protocol.json') == PIN
    p = read(prepared / 'protocol.json')
    transfer_path = OUT / 'cctv_dgp_bank_loss_alignment_v1_transfer_audit.json'
    transfer = read(transfer_path)
    assert transfer['complete'] and transfer['protocol_sha256'] == PIN
    packet = {**p['assets_sha256'], 'protocol.json': PIN}
    for name, expected in packet.items():
        assert sha(prepared / name) == expected
    parent_bytes = 0
    for name, expected in p['parent_required_sha256'].items():
        path = PARENT / name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(PARENT)
        assert sha(path) == expected, name
        parent_bytes += path.stat().st_size
    protected = read(OUT / 'cctv_dgp_bank_path_ablation_v1_analysis/local_protected_before.json')['files_sha256']
    for name, expected in protected.items():
        assert sha(ROOT / name) == expected, name
    archive = OUT / (STEM + '-results.tar.gz')
    checksum = Path(str(archive) + '.sha256')
    exported_path = OUT / (STEM + '-export.json')
    exported = read(exported_path)
    line = checksum.read_text(encoding='ascii').strip().split()
    assert len(line) == 2 and line[1] == archive.name
    assert exported['complete'] and exported['diagnostic_success_not_implied']
    assert exported['optimizer_updates'] == 0
    assert sha(archive) == line[0] == exported['archive_sha256']
    assert archive.stat().st_size == exported['bytes']
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        assert len(members) <= 100
        names, folded = {}, set()
        reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{s}{i}' for s in ['COM', 'LPT'] for i in range(1, 10)}
        for member in members:
            path = PurePosixPath(member.name)
            assert member.isfile() and not member.islnk() and not member.issym()
            assert not path.is_absolute() and '..' not in path.parts
            assert path.parts[0] == NAME and len(path.parts) >= 2 and path.as_posix() == member.name
            assert ':' not in member.name and '\\' not in member.name
            assert all(c == c.rstrip(' .') and c.split('.')[0].upper() not in reserved for c in path.parts)
            name = PurePosixPath(*path.parts[1:]).as_posix()
            assert name.casefold() not in folded
            names[name] = member
            folded.add(name.casefold())
        assert all(not any(a.as_posix().casefold() in folded for a in PurePosixPath(n).parents
                           if a.as_posix() != '.') for n in names)
        with tar.extractfile(names['export_manifest.json']) as stream:
            manifest = json.load(stream)
        assert manifest['protocol_sha256'] == PIN
        assert manifest['parent_protocol_sha256'] == p['parent_protocol_sha256']
        assert manifest['parent_required_sha256'] == p['parent_required_sha256']
        assert manifest['optimizer_updates'] == 0
        assert set(names) == set(manifest['files']) | {'export_manifest.json'}
        assert all(manifest['files'][name] == value for name, value in packet.items())
        uncompressed = sum(m.size for m in members)
        assert uncompressed <= p['output_byte_cap'] + 4 * 1024**2
        assert shutil.disk_usage(OUT).free >= uncompressed + 2 * 1024**3
        ANALYSIS.mkdir()
        RETURN.mkdir()
        write(ANALYSIS / 'local_protected_before.json', dict(complete=True,
              files_sha256=protected, UTC=datetime.now(timezone.utc).isoformat(), freshly_rechecked=True))
        for name, member in names.items():
            target = RETURN.joinpath(*PurePosixPath(name).parts)
            assert target.resolve().is_relative_to(RETURN.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with tar.extractfile(member) as source, target.open('xb') as sink:
                for chunk in iter(lambda: source.read(4 * 1024**2), b''):
                    digest.update(chunk)
                    sink.write(chunk)
            assert target.stat().st_size == member.size
            if name != 'export_manifest.json':
                assert digest.hexdigest() == manifest['files'][name], name
    for name, expected in protected.items():
        assert sha(ROOT / name) == expected
    record = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), archive_sha256=line[0], archive_bytes=archive.stat().st_size,
        checksum_sha256=sha(checksum), export_receipt_sha256=sha(exported_path), protocol_sha256=PIN,
        export_manifest_sha256=sha(RETURN / 'export_manifest.json'), returned_file_bindings=len(names),
        uncompressed_bytes=uncompressed, parent_file_bindings=len(p['parent_required_sha256']),
        parent_bytes=parent_bytes, every_member_path_size_and_hash_checked=True,
        prepared_packet_unchanged=True, protected_app_checkpoint_examples_unchanged=True,
        transfer_audit_sha256=sha(transfer_path),
        complete_diagnostic_reported=manifest['complete_diagnostic'], failure_reported=manifest['failure_present'],
        VM_worker_imported_or_executed=False, local_neural_gradient_or_optimizer_calls=0,
        independent_audit_pending=True, model_qualified=False, goal_complete=False,
        seconds=time.monotonic() - start)
    write(ANALYSIS / 'import.json', record)
    print(json.dumps({k: record[k] for k in ['complete', 'returned_file_bindings', 'parent_file_bindings',
                     'complete_diagnostic_reported', 'failure_reported']}), flush=True)


if __name__ == '__main__':
    main()
