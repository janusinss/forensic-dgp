"""Restore absent selected Windows output paths from the immutable Drive TAR."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--plan',type=Path,required=True)
    parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--group',action='append',required=True,help='Exact old output folder basename; repeat to select several')
    parser.add_argument('--restore',action='store_true',help='Without this flag verify selected payloads only')
    a=parser.parse_args();assert digest(a.plan)==a.plan_sha
    p=json.loads(a.plan.read_text());assert p['format']=='existing-drive-tar-local-output-offload-v1'
    assert digest(a.archive)==p['archive']['sha256'],'Archive must match before reading members'
    groups=set(a.group);available={PurePosixPath(r['path']).parts[1] for r in p['candidates']}
    assert groups and groups<=available
    selected=[r for r in p['candidates'] if PurePosixPath(r['path']).parts[1] in groups]
    root=a.root.resolve(strict=True);assert root.is_dir()
    if a.restore:
        missing=[r for r in selected if not root.joinpath(*PurePosixPath(r['path']).parts).exists()]
        assert shutil.disk_usage(root).free>=sum(r['bytes'] for r in missing)+1024**3
    restored=0
    with tarfile.open(a.archive,'r:') as tar:
        members=tar.getmembers();mapping={m.name:m for m in members};assert len(mapping)==len(members)
        assert all(m.isfile() and not m.issym() and not m.islnk() for m in members)
        for row in selected:
            rel=PurePosixPath(row['path']);assert rel.parts[0]=='outputs' and '..' not in rel.parts and ':' not in row['path']
            target=root.joinpath(*rel.parts);assert target.resolve().is_relative_to(root)
            for parent in [target,*target.parents]:
                if parent==root.parent:break
                assert not parent.is_symlink()
                if parent.exists():assert not parent.lstat().st_file_attributes&0x400
            m=mapping[row['recovery_member']];assert m.size==row['bytes'];h=hashlib.sha256()
            with tar.extractfile(m) as src:
                for b in iter(lambda:src.read(1024**2),b''):h.update(b)
            assert h.hexdigest()==row['sha256']
            if target.exists():assert target.is_file() and digest(target)==row['sha256'];continue
            if a.restore:
                target.parent.mkdir(parents=True,exist_ok=True)
                with target.open('xb') as dst,tar.extractfile(m) as src:shutil.copyfileobj(src,dst)
                assert digest(target)==row['sha256']
                os.utime(target,ns=(row['mtime_ns'],row['mtime_ns']))
                restored+=1
    print(json.dumps({'complete':True,'selected':len(selected),'restored':restored,'overwrite':False}))

if __name__=='__main__':main()
