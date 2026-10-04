"""Verify and extract a fresh successful V12 return, then audit it locally."""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tarfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from cctv_dgp_face_code_fit_v12 import sha,require,read,write,verify
PROTOCOL='06f056ea4b1b04e6d8c631e5e78df129345c78279230c7946f6b5deb2e226846'


def import_return(archive,completion,destination):
    require(archive.resolve().is_relative_to(ROOT/'outputs') and destination.resolve().is_relative_to(ROOT/'outputs'), 'Return must stay within project outputs')
    require(not destination.exists(),'Preserve prior/partial extraction')
    terminal=read(completion);require(terminal['complete'] and terminal['protocol_sha256']==PROTOCOL and terminal['optimizer_updates']==300,'Not a successful fitting return')
    require(archive.stat().st_size==terminal['bytes'] and sha(archive)==terminal['archive_sha256'],'Return archive differs')
    checksum=Path(str(archive)+'.sha256').read_text(encoding='ascii').strip().split()
    require(checksum==[terminal['archive_sha256'],archive.name],'Checksum receipt differs')
    bundle=ROOT/'outputs/cctv_dgp_face_code_fit_vm_v12_r2';p=verify(bundle,PROTOCOL)
    with tarfile.open(archive,'r:gz') as stream:
        members=stream.getmembers();require(len({m.name for m in members})==len(members),'Repeated archive member')
        require(sum(m.size for m in members if m.isfile())<2*1024**3,'Unexpected return size')
        for member in members:
            name=PurePosixPath(member.name)
            require(bool(name.parts) and not name.is_absolute() and '..' not in name.parts and
                    '\\' not in member.name and ':' not in member.name and (member.isfile() or member.isdir()),'Unsafe/link member')
            require((destination/member.name).resolve().is_relative_to(destination.resolve()),'Member escapes extraction')
        destination.mkdir();stream.extractall(destination,members=members)
    require(sha(destination/'face_code_fit_protocol_v12.json')==PROTOCOL,'Returned protocol differs')
    sources=0
    for name,pin in p['assets_sha256'].items():
        path=destination/name
        if path.is_file():require(sha(path)==pin,'Returned execution source differs: '+name);sources+=1
    write(destination/'return_import_receipt.json',{'complete':True,'archive_sha256':terminal['archive_sha256'],
        'protocol_sha256':PROTOCOL,'members':len(members),'returned_pinned_sources_checked':sources,
        'completion_sha256':sha(completion),'local_backward_calls':0,'local_optimizer_updates':0})
    out=destination/'outputs/cctv_dgp_face_code_fit_v12'
    subprocess.run([sys.executable,'-X','utf8','-u',str(ROOT/'scripts/audit_cctv_dgp_face_code_fit_v12.py'),
        '--root',str(bundle),'--expected-protocol-sha',PROTOCOL,'--results',str(out),
        '--receipt',str(destination/'local_independent_audit.json')],cwd=ROOT,check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--completion',type=Path,required=True);parser.add_argument('--extract-to',type=Path,required=True)
    args=parser.parse_args();import_return(args.archive.resolve(),args.completion.resolve(),args.extract_to.resolve())
