"""Hash and namespace verification precede extraction; saved-data audit follows."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

from cctv_dgp_decoder_activation_v1_contract import NAME, STEM, STAGE, read, write, sha, verify_transfer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs'
PREPARATION = OUT/'cctv_dgp_decoder_activation_v1_preparation'
RETURN = OUT/'cctv_dgp_decoder_activation_v1_return'
ANALYSIS = OUT/'cctv_dgp_decoder_activation_v1_analysis'
PIN = 'b93c7a8ce3ab730f1d20f8ebc8786db18469e8652db26554cf0b116dc5785481'


def safe_name(name):
    path = PurePosixPath(name)
    assert isinstance(name, str) and name and ':' not in name and '\\' not in name
    assert not any(char in '<>"|?*' for char in name), 'Windows-invalid filename punctuation'
    assert path.parts and not path.is_absolute() and '..' not in path.parts and str(path) == name
    assert not any(ord(char) < 32 or ord(char) == 127 for char in name)
    forbidden = {'CON','PRN','AUX','NUL'} | {kind+str(i) for kind in ['COM','LPT'] for i in range(1,10)}
    assert all(part and not part.endswith((' ','.')) and part.split('.')[0].upper() not in forbidden for part in path.parts)
    return path


def import_verified(archive, sidecar, export_path, target, protocol, pin):
    assert not target.exists(), 'Retain any previous complete or partial return'
    digest = sha(archive)
    assert sidecar.read_text(encoding='utf-8-sig').strip().split() == [digest, archive.name]
    exported = read(export_path)
    assert exported['complete'] and exported['archive_sha256'] == digest
    assert exported['bytes'] == archive.stat().st_size and exported['protocol_sha256'] == pin
    assert exported['optimizer_updates'] == 0 and exported['model_qualified'] is False
    assert exported['training_success_not_implied'] is True
    maximum = protocol['budgets']['output_byte_cap']+16*1024**2
    assert 0 < archive.stat().st_size <= maximum+1024**2
    with tarfile.open(archive,'r:gz') as stream:
        members, total = [], 0
        for member in stream:
            safe_name(member.name)
            assert member.isreg() and 0 <= member.size <= maximum
            members.append(member); total += member.size
            assert len(members) <= 4096 and total <= maximum
        names = [member.name for member in members]
        assert len(names) == len(set(names)) == len({name.casefold() for name in names})
        folded_names = {name.casefold() for name in names}
        for name in names:
            for ancestor in PurePosixPath(name).parents:
                assert str(ancestor) == '.' or ancestor.as_posix().casefold() not in folded_names, 'Regular file cannot also be a namespace directory'
        assert names.count('result_manifest.json') == 1
        manifest_member = next(member for member in members if member.name == 'result_manifest.json')
        assert manifest_member.size <= 4*1024**2
        manifest = json.load(stream.extractfile(manifest_member))
        assert set(names) == set(manifest['files']) | {'result_manifest.json'}
        assert manifest['complete'] and manifest['protocol_sha256'] == pin
        assert manifest['optimizer_updates'] == 0 and manifest['model_qualified'] is False
        required = set(protocol['assets_sha256']) | {'protocol.json'}
        assert required <= set(names)
        allowed_logs = {'run.log','run_exit.json','saved_audit.log','saved_audit_exit.json'}
        for member in members:
            name = member.name
            assert name in required | allowed_logs | {'result_manifest.json'} or name.startswith(STAGE+'/')
            expected = manifest['files'].get(name)
            if name != 'result_manifest.json':
                assert isinstance(expected,dict) and expected['bytes'] == member.size
            counted, hashed = 0, hashlib.sha256()
            with stream.extractfile(member) as source:
                for block in iter(lambda:source.read(4*1024**2),b''):
                    counted += len(block); assert counted <= member.size
                    hashed.update(block)
            assert counted == member.size
            if expected is not None: assert hashed.hexdigest() == expected['sha256']
            if name in protocol['assets_sha256']: assert hashed.hexdigest() == protocol['assets_sha256'][name]
            if name == 'protocol.json': assert hashed.hexdigest() == pin
        target.mkdir()
        for member in members:
            path = target/member.name
            path.parent.mkdir(parents=True,exist_ok=True)
            assert path.resolve().is_relative_to(target.resolve()) and not path.is_symlink()
            with stream.extractfile(member) as source, path.open('xb') as destination:
                shutil.copyfileobj(source,destination,4*1024**2)
            assert path.stat().st_size == member.size
            if member.name != 'result_manifest.json': assert sha(path) == manifest['files'][member.name]['sha256']
    return dict(complete=True, archive_sha256=digest, archive_bytes=archive.stat().st_size,
        archive_members=len(members), uncompressed_bytes=total, all_hashes_before_extraction=True,
        extracted_payload_rehashed=True, casefold_unique_namespace=True, protocol_sha256=pin,
        Windows_safe_names=True, file_parent_namespace_disjoint=True,
        partial_terminal_without_worker_receipt=exported['partial_terminal_without_worker_receipt'],
        exact_gradient_queries=exported['gradient_queries'], model_qualified=False)


def main():
    started = time.monotonic()
    prepared = read(PREPARATION/'preparation.json')
    assert prepared['protocol_sha256'] == PIN
    assert read(PREPARATION/'independent_preparation_audit.json')['complete']
    packet = OUT/NAME
    assert sha(packet/'protocol.json') == PIN
    assert not ANALYSIS.exists() and not RETURN.exists()
    protocol = read(packet/'protocol.json')
    record = import_verified(OUT/(STEM+'-results.tar.gz'),OUT/(STEM+'-results.tar.gz.sha256'),
        OUT/(STEM+'-export.json'), RETURN, protocol, PIN)
    repair,source,parent,replay = [OUT/name for name in ['cctv_dgp_decoder_hinge_repair_v1_return',
        'cctv_dgp_spatial_readout_v1_return','cctv_dgp_bank_photometric_guard_v1_return','cctv_dgp_spatial_path_replay_v1_return']]
    p,rp,old,data = verify_transfer(RETURN,repair,source,parent,replay,PIN)
    ANALYSIS.mkdir()
    write(ANALYSIS/'import.json',dict(record,UTC=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic()-started, neural_calls=0,gradient_queries=0,optimizer_updates=0,VM_calls=0,
        numerical_audit_pending=True))
    from audit_cctv_dgp_decoder_activation_v1_return import audit_saved
    audit_saved(RETURN,repair,source,parent,replay,p,rp,old,data,ANALYSIS/'independent_audit.json')


if __name__ == '__main__':
    main()
