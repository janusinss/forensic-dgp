"""Strict saved-file import; no returned executable or neural model is loaded."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs'
NAME='cctv_dgp_conditioning_response_v1_vm'
STEM='cctv-dgp-conditioning-response-v1'
PIN='e72d76e3f2330420ee93b43cf528d0b49da132f356a30c336a1ec578c8f82f44'
RETURN=OUT/'cctv_dgp_conditioning_response_v1_return'
ANALYSIS=OUT/'cctv_dgp_conditioning_response_v1_analysis'
PARENT=OUT/'cctv_dgp_bank_photometric_guard_v1_return'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as stream:stream.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def member_names(members):
    names={};folded=set()
    reserved={'CON','PRN','AUX','NUL'}|{f'{s}{i}' for s in ['COM','LPT'] for i in range(1,10)}
    assert len(members)<=2400
    for m in members:
        path=PurePosixPath(m.name)
        assert m.isfile() and not m.islnk() and not m.issym()
        assert not path.is_absolute() and '..' not in path.parts
        assert len(path.parts)>=2 and path.parts[0]==NAME and path.as_posix()==m.name
        assert ':' not in m.name and '\\' not in m.name
        assert not any(ord(c)<32 or c in '<>"|?*' for c in m.name)
        assert all(n==n.rstrip(' .') and n.split('.')[0].upper() not in reserved for n in path.parts)
        name=PurePosixPath(*path.parts[1:]).as_posix()
        assert name.casefold() not in folded
        names[name]=m;folded.add(name.casefold())
    assert all(not any(v.as_posix().casefold() in folded for v in PurePosixPath(n).parents if v.as_posix()!='.') for n in names)
    return names


def main():
    start=time.monotonic();assert len(PIN)==64 and not RETURN.exists() and not ANALYSIS.exists()
    prepared=OUT/NAME;assert sha(prepared/'protocol.json')==PIN
    p=read(prepared/'protocol.json');prep=read(OUT/'cctv_dgp_conditioning_response_v1_preparation.json')
    transfer=read(OUT/'cctv_dgp_conditioning_response_v1_transfer_audit.json')
    assert prep['complete'] and transfer['complete'] and prep['protocol_sha256']==transfer['protocol_sha256']==PIN
    packet={**p['assets_sha256'],'protocol.json':PIN}
    assert all(sha(prepared/n)==h for n,h in packet.items())
    parent_bytes=0
    for name,h in p['parent_required_sha256'].items():
        path=PARENT/name
        assert path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(PARENT)
        assert sha(path)==h,name
        parent_bytes+=path.stat().st_size
    protected=read(OUT/'cctv_dgp_raw_conditioning_learning_v1_r1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    archive=OUT/(STEM+'-results.tar.gz');checksum=Path(str(archive)+'.sha256');exported_path=OUT/(STEM+'-export.json')
    exported=read(exported_path);line=checksum.read_text(encoding='ascii').strip().split()
    assert len(line)==2 and line[1]==archive.name
    assert exported['complete'] and exported['training_success_not_implied'] and exported['accepted_updates']==exported['optimizer_updates']==0
    assert sha(archive)==line[0]==exported['archive_sha256'] and archive.stat().st_size==exported['bytes']
    with tarfile.open(archive,'r:gz') as tar:
        names=member_names(tar.getmembers())
        with tar.extractfile(names['export_manifest.json']) as stream:manifest=json.load(stream)
        assert manifest['protocol_sha256']==PIN and manifest['parent_protocol_sha256']==p['parent_protocol_sha256']
        assert manifest['parent_required_sha256']==p['parent_required_sha256'] and not manifest['complete_pilot']
        assert manifest['accepted_updates']==manifest['optimizer_updates']==0 and manifest['training_success_not_implied']
        assert set(names)==set(manifest['files'])|{'export_manifest.json'}
        assert all(manifest['files'][n]==h for n,h in packet.items())
        total=sum(m.size for m in names.values())
        assert total<=p['output_byte_cap']+64*1024**2 and shutil.disk_usage(OUT).free>=total+2*1024**3
        ANALYSIS.mkdir();RETURN.mkdir()
        write(ANALYSIS/'local_protected_before.json',dict(complete=True,files_sha256=protected,freshly_rechecked=True))
        for name,m in names.items():
            path=RETURN.joinpath(*PurePosixPath(name).parts);assert path.resolve().is_relative_to(RETURN.resolve())
            path.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
            with tar.extractfile(m) as src,path.open('xb') as dst:
                for b in iter(lambda:src.read(4*1024**2),b''):dst.write(b);h.update(b)
            assert path.stat().st_size==m.size
            if name!='export_manifest.json':assert h.hexdigest()==manifest['files'][name],name
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(ANALYSIS/'import.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        archive_sha256=line[0],archive_bytes=archive.stat().st_size,checksum_sha256=sha(checksum),export_receipt_sha256=sha(exported_path),
        protocol_sha256=PIN,export_manifest_sha256=sha(RETURN/'export_manifest.json'),returned_file_bindings=len(names),uncompressed_bytes=total,
        parent_file_bindings=len(p['parent_required_sha256']),parent_bytes=parent_bytes,
        complete_diagnostic_reported=manifest['complete_diagnostic'],failure_reported=manifest['failure_present'],
        exact_paths_sizes_hashes_and_protected_files_checked=True,VM_worker_imported_or_executed=False,local_neural_or_training_calls=0,
        independent_audit_pending=True,model_qualified=False,goal_complete=False,seconds=time.monotonic()-start))
    print(dict(complete=True,returned_file_bindings=len(names),diagnostic_complete_reported=manifest['complete_diagnostic'],failure=manifest['failure_present']))


if __name__=='__main__':main()
