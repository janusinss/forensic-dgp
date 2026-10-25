"""Seal and fully read back an inert manual-VM packet; no Torch or runtime launch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.codeformer_spatial_objectives_v1 import packet as p


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol-sha',required=True)
    a=parser.parse_args();stage=ROOT/'outputs'/p.VM_NAME
    protocol=p.verify(stage,a.protocol_sha);files=p.inventory(stage)
    p.require(set(files)==set(protocol['files'])|{'protocol.json'},'Only pristine static packet may be sealed')
    archive=ROOT/'outputs/cctv-codeformer-spatial-objectives-v1-execution.tar.gz'
    temporary=Path(str(archive)+'.partial');checksum=Path(str(archive)+'.sha256')
    receipt=ROOT/'outputs/codeformer_spatial_objectives_packet_review_v1/execution_archive_v1.json'
    p.require(all(not f.exists() for f in (archive,temporary,checksum,receipt)),'Preserve prior seal or partial; no overwrite')
    pins={name:p.pin(stage,path) for name,path in files.items()};start=time.monotonic()
    with tarfile.open(temporary,'x:gz',compresslevel=1) as tar:
        for name,path in sorted(files.items()):
            before=p.pin(stage,path);p.require(before==pins[name],'Source changed before archive')
            info=tarfile.TarInfo(p.VM_NAME+'/'+name);info.size=before['bytes'];info.mode=0o644;info.mtime=0
            with path.open('rb') as f:tar.addfile(info,f)
            p.require(p.pin(stage,path)==before,'Source changed during archive')
    observed={};payload=0
    with tarfile.open(temporary,'r|gz') as tar:
        for member in tar:
            p.require(member.isfile() and not member.issparse() and member.name.startswith(p.VM_NAME+'/'),
                      'Only exact regular files beneath new VM root allowed')
            name=member.name[len(p.VM_NAME)+1:];p.canonical(name)
            p.require(name in pins and name not in observed and member.size==pins[name]['bytes'],
                      'Archive namespace or size differs')
            stream=tar.extractfile(member);digest=hashlib.sha256();count=0
            for block in iter(lambda:stream.read(p.MiB),b''):
                count+=len(block);digest.update(block)
            p.require(count==pins[name]['bytes'] and digest.hexdigest()==pins[name]['sha256'],'Archive member bytes differ')
            observed[name]=pins[name];payload+=count
    p.require(set(observed)==set(pins),'Archive missing static assets')
    # Final file is created exclusively. Failed copying retains both partial and
    # incomplete final as evidence; nothing here deletes a previous artifact.
    with temporary.open('rb') as src,archive.open('xb') as dst:
        import shutil
        shutil.copyfileobj(src,dst,p.MiB);dst.flush();os.fsync(dst.fileno())
    digest=p.sha256(archive)
    p.require(digest==p.sha256(temporary),'Published archive differs from verified partial')
    # Only this newly verified temporary exact file is removed; not research data.
    temporary.unlink()
    with checksum.open('xb') as f:f.write((digest+'  '+archive.name+'\n').encode('ascii'))
    r={'format':'codeformer-spatial-objectives-execution-archive-v1','complete':True,
       'archive':{'path':archive.relative_to(ROOT).as_posix(),'bytes':archive.stat().st_size,'sha256':digest},
       'checksum':{'path':checksum.relative_to(ROOT).as_posix(),'bytes':checksum.stat().st_size,'sha256':p.sha256(checksum)},
       'checksum_ASCII_LF_no_CR':b'\r' not in checksum.read_bytes(),
       'protocol_sha256':a.protocol_sha,'regular_members':len(observed),'payload_bytes':payload,
       'all_members_fully_streamed_and_hashed':True,'seconds':time.monotonic()-start,
       'Torch_neural_optimizer_VM_calls':0,'actual_training_unrun':True,'model_qualified':False,
       'stage_source_pins':pins}
    with receipt.open('xb') as f:f.write(p.json_bytes(r))
    print(json.dumps({k:v for k,v in r.items() if k!='stage_source_pins'}))


if __name__=='__main__':main()
