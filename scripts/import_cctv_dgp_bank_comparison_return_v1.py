"""Verify and safely import a manual return without executing its VM trainer."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'outputs'
NAME = 'cctv_dgp_bank_comparison_v1_vm'
STEM = 'cctv-dgp-bank-comparison-v1'
PIN = '0e4f9645abc99f6d980f3b20240b196f65f3e130472795faa2f964df1a5a3b3e'
RETURN = OUTPUT / 'cctv_dgp_bank_comparison_v1_return'
ANALYSIS = OUTPUT / 'cctv_dgp_bank_comparison_v1_analysis'
PREPARED = OUTPUT / NAME


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def member_name(member):
    p = PurePosixPath(member.name)
    assert member.isfile() and not member.islnk() and not member.issym()
    assert not p.is_absolute() and '..' not in p.parts and p.parts[0] == NAME and len(p.parts) >= 2
    assert p.as_posix() == member.name and ':' not in member.name and '\\' not in member.name
    return PurePosixPath(*p.parts[1:]).as_posix()


def main():
    started = time.monotonic()
    assert not RETURN.exists() and not ANALYSIS.exists(), 'Retain previous imports and evidence'
    archive = OUTPUT / (STEM + '-results.tar.gz')
    checksum = OUTPUT / (STEM + '-results.tar.gz.sha256')
    export_path = OUTPUT / (STEM + '-export.json')
    exported = read(export_path)
    assert exported['complete'] and exported['protocol_sha256'] == PIN
    line = checksum.read_bytes().decode('ascii').strip().split()
    assert len(line) == 2 and line[1] == archive.name and line[0] == exported['archive_sha256']
    assert archive.stat().st_size == exported['bytes'] and sha(archive) == line[0]
    assert sha(PREPARED / 'protocol.json') == PIN
    p = read(PREPARED / 'protocol.json')
    original_bindings = {**p['assets_sha256'], 'protocol.json': PIN}
    for name, expected in original_bindings.items():
        assert sha(PREPARED / name) == expected, name
    print(dict(return_archive_SHA256_verified=True, bytes=exported['bytes']), flush=True)
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers(); names = {}
        for member in members:
            name = member_name(member)
            assert name not in names
            names[name] = member
        assert 'export_manifest.json' in names and 'protocol.json' in names
        with tar.extractfile(names['protocol.json']) as stream:
            raw_protocol = stream.read()
        assert hashlib.sha256(raw_protocol).hexdigest() == PIN
        with tar.extractfile(names['export_manifest.json']) as stream:
            manifest = json.load(stream)
        assert manifest['protocol_sha256'] == PIN
        assert set(names) == set(manifest['files']) | {'export_manifest.json'}
        assert sum(names[name].size for name in manifest['files']) == manifest['uncompressed_bytes']
        assert all(manifest['files'][name] == expected for name, expected in original_bindings.items())
        uncompressed = sum(member.size for member in members)
        bound = p['budgets']['return_uncompressed_bytes'] + sum((PREPARED / name).stat().st_size for name in p['assets_sha256']) + 32*1024**2
        assert uncompressed <= bound and len(members) <= 100000
        assert shutil.disk_usage(OUTPUT).free >= uncompressed + 2*1024**3
        ANALYSIS.mkdir(); RETURN.mkdir()
        for n, (name, member) in enumerate(names.items(), 1):
            target = RETURN.joinpath(*PurePosixPath(name).parts)
            assert target.resolve().is_relative_to(RETURN.resolve())
            target.parent.mkdir(parents=True, exist_ok=True)
            h = hashlib.sha256()
            with tar.extractfile(member) as source, target.open('xb') as sink:
                for chunk in iter(lambda: source.read(4*1024**2), b''):
                    h.update(chunk); sink.write(chunk)
            if name != 'export_manifest.json':
                assert h.hexdigest() == manifest['files'][name], name
            assert target.stat().st_size == member.size
            if n % 2000 == 0:
                print(dict(returned_members_verified=n, of=len(names)), flush=True)
    for name, expected in original_bindings.items():
        assert sha(PREPARED / name) == expected and sha(RETURN / name) == expected
    result = dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(), checker_sha256=sha(Path(__file__)),
        archive_sha256=line[0], archive_bytes=archive.stat().st_size, checksum_sha256=sha(checksum),
        export_receipt_sha256=sha(export_path), protocol_sha256=PIN, export_manifest_sha256=sha(RETURN / 'export_manifest.json'),
        returned_file_bindings=len(names), uncompressed_bytes=uncompressed,
        prepared_packet_bindings_unchanged=len(original_bindings), every_member_path_and_byte_hash_checked=True,
        returned_optimizer_updates_reported=exported['optimizer_updates'], return_execution_success_not_implied=True,
        VM_trainer_imported_or_run=False, local_model_gradient_or_training_calls=0, independent_output_audit_pending=True,
        quality_qualified=False, goal_complete=False, seconds=time.monotonic()-started)
    with (ANALYSIS / 'import.json').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(dict(complete=True, returned_files=len(names), reported_updates=exported['optimizer_updates']), flush=True)


if __name__ == '__main__':
    main()
