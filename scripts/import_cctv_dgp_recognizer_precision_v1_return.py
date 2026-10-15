"""Immutable precision return import; never executes returned code."""
from datetime import datetime, timezone
from pathlib import PurePosixPath
import hashlib
import shutil
import tarfile

from import_cctv_dgp_appearance_input_response_v1_return import ROOT, OUT, BANK, SOURCE, RETURN as INPUT, read, sha, write

NAME = 'cctv_dgp_recognizer_precision_v1_vm'
STEM = 'cctv-dgp-recognizer-precision-v1'
PIN = 'ca8d2fd69dffb19e040a36eb059e4f4124f00f05755cad976d24f8a00f4dbbe7'
RETURN = OUT / 'cctv_dgp_recognizer_precision_v1_return'
ANALYSIS = OUT / 'cctv_dgp_recognizer_precision_v1_analysis'


def member_names(members):
    assert len(members) <= 1000
    names = {}; folded = set()
    reserved = {'CON', 'PRN', 'AUX', 'NUL'} | {f'{s}{i}' for s in ['COM', 'LPT'] for i in range(1, 10)}
    for m in members:
        path = PurePosixPath(m.name)
        assert m.isfile() and not m.islnk() and not m.issym()
        assert not path.is_absolute() and '..' not in path.parts and len(path.parts) >= 2
        assert path.parts[0] == NAME and path.as_posix() == m.name and ':' not in m.name and '\\' not in m.name
        assert not any(ord(c) < 32 or c in '<>"|?*' for c in m.name)
        assert all(n == n.rstrip(' .') and n.split('.')[0].upper() not in reserved for n in path.parts)
        name = PurePosixPath(*path.parts[1:]).as_posix(); assert name.casefold() not in folded
        names[name] = m; folded.add(name.casefold())
    assert all(not any(v.as_posix().casefold() in folded for v in PurePosixPath(n).parents if v.as_posix() != '.') for n in names)
    return names


def main():
    assert len(PIN) == 64 and not RETURN.exists() and not ANALYSIS.exists()
    p = read(OUT / NAME / 'protocol.json'); assert sha(OUT / NAME / 'protocol.json') == PIN
    for folder, field in [(OUT / NAME, 'assets_sha256'), (INPUT, 'input_required_sha256'),
                          (SOURCE, 'response_required_sha256'), (BANK, 'bank_required_sha256')]:
        assert all(sha(folder / n) == h for n, h in p[field].items())
    protected = read(OUT / 'cctv_dgp_appearance_input_response_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    archive = OUT / (STEM + '-results.tar.gz'); checksum = read(OUT / (STEM + '-export.json'))
    line = (OUT / (STEM + '-results.tar.gz.sha256')).read_text(encoding='ascii').strip().split()
    assert len(line) == 2 and line[1] == archive.name and checksum['complete'] and checksum['training_success_not_implied']
    assert sha(archive) == line[0] == checksum['archive_sha256'] and archive.stat().st_size == checksum['bytes']
    assert checksum['optimizer_updates'] == checksum['accepted_updates'] == 0
    with tarfile.open(archive, 'r:gz') as tar:
        names = member_names(tar.getmembers())
        with tar.extractfile(names['export_manifest.json']) as f:
            import json
            manifest = json.load(f)
        assert manifest['protocol_sha256'] == PIN and manifest['training_success_not_implied'] and not manifest['complete_pilot']
        assert manifest['optimizer_updates'] == manifest['accepted_updates'] == 0
        assert set(names) == set(manifest['files']) | {'export_manifest.json'}
        assert all(manifest['files'][n] == h for n, h in {**p['assets_sha256'], 'protocol.json': PIN}.items())
        total = sum(m.size for m in names.values())
        assert total <= p['output_byte_cap'] + 16 * 1024**2 and shutil.disk_usage(OUT).free >= total + 1024**3
        RETURN.mkdir(); ANALYSIS.mkdir()
        write(ANALYSIS / 'local_protected_before.json', dict(complete=True, files_sha256=protected, freshly_rechecked=True))
        for n, m in names.items():
            target = RETURN.joinpath(*PurePosixPath(n).parts); assert target.resolve().is_relative_to(RETURN.resolve())
            target.parent.mkdir(parents=True, exist_ok=True); digest = hashlib.sha256()
            with tar.extractfile(m) as src, target.open('xb') as dst:
                for b in iter(lambda: src.read(4 * 1024**2), b''):
                    dst.write(b); digest.update(b)
            assert target.stat().st_size == m.size
            if n != 'export_manifest.json':
                assert digest.hexdigest() == manifest['files'][n]
    assert all(sha(ROOT / n) == h for n, h in protected.items())
    write(ANALYSIS / 'import.json', dict(complete=True, UTC=datetime.now(timezone.utc).isoformat(),
        checker_sha256=sha(__file__), protocol_sha256=PIN, archive_sha256=sha(archive),
        archive_bytes=archive.stat().st_size, returned_file_bindings=len(manifest['files']),
        dependency_bindings={field: len(p[field]) for field in ['input_required_sha256', 'response_required_sha256', 'bank_required_sha256']},
        reported_diagnostic_complete=manifest['complete_diagnostic'], failure_reported=manifest['failure_present'],
        returned_source_executed=False, local_neural_or_training_calls=0, model_qualified=False, goal_complete=False))
    print(dict(complete=True, returned_bindings=len(manifest['files']), diagnostic_reported=manifest['complete_diagnostic']))


if __name__ == '__main__':
    main()
