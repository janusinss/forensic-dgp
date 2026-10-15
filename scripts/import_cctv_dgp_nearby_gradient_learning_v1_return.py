"""Strict immutable nearby-training return import, without executing returned code."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'outputs'
NAME='cctv_dgp_nearby_gradient_learning_v1_vm'; STEM='cctv-dgp-nearby-gradient-learning-v1'
PIN='f0212a8e9e12396285c381e34503202d6e4f7bc1e143b5614f105d105e163e23'
RETURN=OUT/'cctv_dgp_nearby_gradient_learning_v1_return'
ANALYSIS=OUT/'cctv_dgp_nearby_gradient_learning_v1_analysis'
PARENT=OUT/'cctv_dgp_bank_photometric_guard_v1_return'
RESPONSE=OUT/'cctv_dgp_conditioning_response_v1_return'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    with Path(path).open('x',encoding='utf-8',newline='\n') as f:f.write(json.dumps(value,indent=2,allow_nan=False)+'\n')


def member_names(members):
    assert len(members)<=2400
    names={};folded=set();reserved={'CON','PRN','AUX','NUL'}|{f'{s}{i}' for s in ['COM','LPT'] for i in range(1,10)}
    for m in members:
        path=PurePosixPath(m.name)
        assert m.isfile() and not m.islnk() and not m.issym()
        assert not path.is_absolute() and '..' not in path.parts and len(path.parts)>=2 and path.parts[0]==NAME
        assert path.as_posix()==m.name and ':' not in m.name and '\\' not in m.name
        assert not any(ord(c)<32 or c in '<>"|?*' for c in m.name)
        assert all(v==v.rstrip(' .') and v.split('.')[0].upper() not in reserved for v in path.parts)
        name=PurePosixPath(*path.parts[1:]).as_posix();assert name.casefold() not in folded
        names[name]=m;folded.add(name.casefold())
    assert all(not any(v.as_posix().casefold() in folded for v in PurePosixPath(n).parents if v.as_posix()!='.') for n in names)
    return names


def main():
    assert len(PIN)==64 and not RETURN.exists() and not ANALYSIS.exists()
    packet=OUT/NAME;p=read(packet/'protocol.json');assert sha(packet/'protocol.json')==PIN
    prepared=read(OUT/'cctv_dgp_nearby_gradient_learning_v1_preparation.json')
    audit=read(OUT/'cctv_dgp_nearby_gradient_learning_v1_transfer_audit.json')
    assert prepared['complete'] and audit['complete'] and prepared['protocol_sha256']==audit['protocol_sha256']==PIN
    for folder,hashes in [(packet,p['assets_sha256']),(PARENT,p['parent_required_sha256']),(RESPONSE,p['response_required_sha256'])]:
        assert all(sha(folder/n)==h for n,h in hashes.items())
    protected=read(OUT/'cctv_dgp_recognizer_precision_v1_analysis/local_protected_before.json')['files_sha256']
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    archive=OUT/(STEM+'-results.tar.gz'); exported=read(OUT/(STEM+'-export.json'))
    line=Path(str(archive)+'.sha256').read_text(encoding='ascii').strip().split()
    assert len(line)==2 and line[1]==archive.name and exported['complete'] and exported['training_success_not_implied']
    assert exported['accepted_updates'] in [None,0,1]
    assert sha(archive)==line[0]==exported['archive_sha256'] and archive.stat().st_size==exported['bytes']
    with tarfile.open(archive,'r:gz') as tar:
        names=member_names(tar.getmembers())
        with tar.extractfile(names['export_manifest.json']) as f:manifest=json.load(f)
        assert manifest['protocol_sha256']==PIN and manifest['parent_protocol_sha256']==p['parent_protocol_sha256']
        assert manifest['parent_required_sha256']==p['parent_required_sha256']
        assert manifest['accepted_updates']==exported['accepted_updates'] and manifest['training_success_not_implied']
        assert set(names)==set(manifest['files'])|{'export_manifest.json'}
        assert all(manifest['files'][n]==h for n,h in {**p['assets_sha256'],'protocol.json':PIN}.items())
        size=sum(m.size for m in names.values());assert size<=p['output_byte_cap']+64*1024**2
        assert shutil.disk_usage(OUT).free>=size+2*1024**3
        RETURN.mkdir();ANALYSIS.mkdir()
        write(ANALYSIS/'local_protected_before.json',dict(complete=True,files_sha256=protected,freshly_rechecked=True))
        for n,m in names.items():
            destination=RETURN.joinpath(*PurePosixPath(n).parts);assert destination.resolve().is_relative_to(RETURN.resolve())
            destination.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
            with tar.extractfile(m) as src,destination.open('xb') as dst:
                for b in iter(lambda:src.read(4*1024**2),b''):dst.write(b);h.update(b)
            assert destination.stat().st_size==m.size
            if n!='export_manifest.json':assert h.hexdigest()==manifest['files'][n]
    assert all(sha(ROOT/n)==h for n,h in protected.items())
    write(ANALYSIS/'import.json',dict(complete=True,UTC=datetime.now(timezone.utc).isoformat(),checker_sha256=sha(__file__),
        protocol_sha256=PIN,archive_sha256=line[0],archive_bytes=archive.stat().st_size,returned_file_bindings=len(manifest['files']),
        parent_file_bindings=len(p['parent_required_sha256']),response_file_bindings=len(p['response_required_sha256']),
        reported_complete_pilot=manifest['complete_pilot'],reported_failure=manifest['failure_present'],accepted_updates_reported=exported['accepted_updates'],
        every_member_path_size_and_hash_checked=True,returned_source_executed=False,local_neural_or_training_calls=0,
        independent_audit_pending=True,model_qualified=False,more_epochs_qualified=False,goal_complete=False))
    print(dict(complete=True,returned_bindings=len(manifest['files']),accepted_updates=exported['accepted_updates'],audit_pending=True))


if __name__=='__main__':main()
