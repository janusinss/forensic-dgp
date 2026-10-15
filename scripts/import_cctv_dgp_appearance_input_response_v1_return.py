"""Strict immutable return import. No returned executable or neural model is loaded."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs'
NAME='cctv_dgp_appearance_input_response_v1_vm';STEM='cctv-dgp-appearance-input-response-v1'
PIN='3ba11275bc5680117ce40438b96987172fd0c772d70a049c7aca38a36f597f12'
RETURN=OUT/'cctv_dgp_appearance_input_response_v1_return'
ANALYSIS=OUT/'cctv_dgp_appearance_input_response_v1_analysis'
SOURCE=OUT/'cctv_dgp_conditioning_response_v1_return'
BANK=OUT/'cctv_dgp_bank_photometric_guard_v1_return'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def member_names(members):
    assert len(members)<=500
    names={};folded=set();reserved={'CON','PRN','AUX','NUL'}|{f'{s}{i}' for s in ['COM','LPT'] for i in range(1,10)}
    for m in members:
        path=PurePosixPath(m.name)
        assert m.isfile() and not m.islnk() and not m.issym()
        assert not path.is_absolute() and '..' not in path.parts and len(path.parts)>=2
        assert path.parts[0]==NAME and path.as_posix()==m.name and ':' not in m.name and '\\' not in m.name
        assert not any(ord(c)<32 or c in '<>"|?*' for c in m.name)
        assert all(n==n.rstrip(' .') and n.split('.')[0].upper() not in reserved for n in path.parts)
        name=PurePosixPath(*path.parts[1:]).as_posix();assert name.casefold() not in folded
        names[name]=m;folded.add(name.casefold())
    assert all(not any(v.as_posix().casefold() in folded for v in PurePosixPath(n).parents if v.as_posix()!='.') for n in names)
    return names


def main():
    assert len(PIN)==64 and not RETURN.exists() and not ANALYSIS.exists()
    packet=OUT/NAME;p=read(packet/'protocol.json');assert sha(packet/'protocol.json')==PIN
    prep=read(OUT/'cctv_dgp_appearance_input_response_v1_preparation.json')
    check=read(OUT/'cctv_dgp_appearance_input_response_v1_transfer_audit.json')
    assert prep['complete'] and check['complete'] and prep['protocol_sha256']==check['protocol_sha256']==PIN
    for folder,bindings in [(packet,p['assets_sha256']),(SOURCE,p['response_required_sha256']),(BANK,p['bank_required_sha256'])]:
        assert all(sha(folder/n)==h for n,h in bindings.items())
    protected=read(OUT/'cctv_dgp_conditioning_response_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    archive=OUT/(STEM+'-results.tar.gz');checksum=Path(str(archive)+'.sha256');export=OUT/(STEM+'-export.json')
    e=read(export);line=checksum.read_text(encoding='ascii').strip().split()
    assert len(line)==2 and line[1]==archive.name and e['complete'] and e['training_success_not_implied']
    assert e['optimizer_updates']==e['accepted_updates']==0
    assert sha(archive)==line[0]==e['archive_sha256'] and archive.stat().st_size==e['bytes']
    with tarfile.open(archive,'r:gz') as tar:
        names=member_names(tar.getmembers())
        with tar.extractfile(names['export_manifest.json']) as f:manifest=json.load(f)
        assert manifest['protocol_sha256']==PIN and manifest['response_protocol_sha256']==p['response_protocol_sha256']
        assert manifest['bank_protocol_sha256']==p['bank_protocol_sha256'] and not manifest['complete_pilot']
        assert manifest['optimizer_updates']==manifest['accepted_updates']==0 and manifest['training_success_not_implied']
        assert set(names)==set(manifest['files'])|{'export_manifest.json'}
        assert all(manifest['files'][n]==h for n,h in {**p['assets_sha256'],'protocol.json':PIN}.items())
        total=sum(m.size for m in names.values())
        assert total<=p['output_byte_cap']+64*1024**2 and shutil.disk_usage(OUT).free>=total+1024**3
        RETURN.mkdir();ANALYSIS.mkdir()
        write(ANALYSIS/'local_protected_before.json',dict(complete=True,files_sha256=protected,freshly_rechecked=True))
        for name,m in names.items():
            destination=RETURN.joinpath(*PurePosixPath(name).parts);assert destination.resolve().is_relative_to(RETURN.resolve())
            destination.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
            with tar.extractfile(m) as src,destination.open('xb') as dst:
                for chunk in iter(lambda:src.read(4*1024**2),b''):dst.write(chunk);h.update(chunk)
            assert destination.stat().st_size==m.size
            if name!='export_manifest.json':assert h.hexdigest()==manifest['files'][name],name
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(ANALYSIS/'import.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        protocol_sha256=PIN,archive_sha256=line[0],archive_bytes=archive.stat().st_size,checksum_sha256=sha(checksum),export_receipt_sha256=sha(export),
        export_manifest_sha256=sha(RETURN/'export_manifest.json'),returned_file_bindings=len(names),uncompressed_bytes=total,
        response_bindings=len(p['response_required_sha256']),bank_bindings=len(p['bank_required_sha256']),
        complete_diagnostic_reported=manifest['complete_diagnostic'],failure_reported=manifest['failure_present'],
        exact_paths_sizes_hashes_and_protected_files_checked=True,VM_worker_imported_or_executed=False,
        local_neural_or_training_calls=0,independent_audit_pending=True,model_qualified=False,goal_complete=False))
    print(dict(complete=True,returned_file_bindings=len(names),diagnostic_complete_reported=manifest['complete_diagnostic'],failure=manifest['failure_present']))


if __name__=='__main__':main()
