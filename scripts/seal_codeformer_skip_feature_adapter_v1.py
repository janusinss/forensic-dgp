"""Seal and independently stream-read a reviewed new manual training packet."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.codeformer_skip_feature_adapter_v1 import transport as t

STAGE=ROOT/'outputs'/t.VM_NAME
PREP=ROOT/'outputs/codeformer_skip_feature_adapter_v1_preparation'
ARCHIVE=ROOT/'outputs/cctv-codeformer-skip-feature-adapter-v1-execution.tar.gz'
RECEIPT=PREP/'sealed_packet_v1.json'
START=time.monotonic()


def check():
    t.require(time.monotonic()-START < 1200,'Finite1200s local seal/readback cap')


def pin(path):
    check(); path=Path(path)
    before=path.stat()
    with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
    after=path.stat()
    t.require((before.st_ino,before.st_size,before.st_mtime_ns)==
        (after.st_ino,after.st_size,after.st_mtime_ns),'File changed while sealing')
    return {'bytes':before.st_size,'sha256':sha}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-review-sha',required=True);args=parser.parse_args()
    ready=PREP/'source_readiness_v1.json';stage_receipt=PREP/'packet_stage_v1.json'
    t.require(pin(ready)['sha256']==args.source_review_sha,'Exact independent review SHA required')
    review=t.read_json(ready)
    t.require(review['complete'] is True and review['ready_for_seal'] is True and
        review['actual_training_neural_calls']==0,'Reviewed inert source readiness required')
    for name,p in review['source_files'].items():
        t.require(pin(ROOT/name)==p,'Reviewed source changed before sealing')
    stage=t.read_json(stage_receipt);sha=stage['protocol_sha256']
    verified=t.verify_transfer(STAGE,sha)
    expected={**verified['protocol']['files'],'protocol.json':pin(STAGE/'protocol.json')}
    t.require(expected==stage['actual_stage_pin_map'],'Physical stage differs from its receipt')
    t.require(not ARCHIVE.exists() and not RECEIPT.exists() and
        not Path(str(ARCHIVE)+'.sha256').exists(),'Preserve prior/partial archive')
    with ARCHIVE.open('xb') as dest:
        with gzip.GzipFile(filename='',mode='wb',fileobj=dest,mtime=0,compresslevel=3) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.USTAR_FORMAT) as archive:
                for name,p in sorted(expected.items()):
                    check();path=t.regular(STAGE,name);t.require(pin(path)==p,'Stage changed during canonical archive')
                    info=tarfile.TarInfo(t.VM_NAME+'/'+name);info.size=p['bytes'];info.mode=0o644
                    info.uid=info.gid=info.mtime=0;info.uname=info.gname=''
                    with path.open('rb') as source:archive.addfile(info,source)
        dest.flush();os.fsync(dest.fileno())
    archive_pin=pin(ARCHIVE);seen={};casefold=set()
    with tarfile.open(ARCHIVE,'r|gz') as archive:
        for member in archive:
            check();t.require(member.isfile() and not member.pax_headers,'Only canonical regular file members allowed')
            prefix=t.VM_NAME+'/'
            t.require(member.name.startswith(prefix),'Wrong VM archive root')
            name=member.name[len(prefix):];t.canonical(name)
            t.require(name in expected and name not in seen and name.casefold() not in casefold,
                'Duplicate, unexpected or colliding archive member')
            t.require(member.size==expected[name]['bytes'] and member.mode==0o644 and
                member.uid==member.gid==member.mtime==0,'Canonical archive header differs')
            stream=archive.extractfile(member);digest=hashlib.sha256();size=0
            while True:
                block=stream.read(4*1024**2)
                if not block:break
                check();size+=len(block);digest.update(block)
            value={'bytes':size,'sha256':digest.hexdigest()}
            t.require(value==expected[name],'Actual compressed payload differs')
            seen[name]=value;casefold.add(name.casefold())
    t.require(seen==expected and pin(ARCHIVE)==archive_pin,'Archive coverage or bytes changed')
    checksum=archive_pin['sha256']+'  '+ARCHIVE.name+'\n'
    with Path(str(ARCHIVE)+'.sha256').open('xb') as out:out.write(checksum.encode('ascii'));out.flush();os.fsync(out.fileno())
    t.require(b'\r' not in Path(str(ARCHIVE)+'.sha256').read_bytes(),'ASCII/LF checksum required')
    t.write_new_json(RECEIPT,{'complete':True,'archive':{'path':str(ARCHIVE.relative_to(ROOT)).replace('\\','/'),**archive_pin},
        'protocol_sha256':sha,'source_review':pin(ready),'source_review_sha256':args.source_review_sha,
        'stage_files':len(expected),'uncompressed_bytes':sum(p['bytes'] for p in expected.values()),
        'every_compressed_member_payload_independently_verified':True,'ASCII_LF_checksum':True,
        'local_training_neural_gradient_optimizer_VM_calls':0,'training_run':False,'model_qualified':False,
        'automatic_launch_authorized':False,'seconds':time.monotonic()-START,'source':pin(Path(__file__))})
    print(json.dumps({'complete':True,'archive':str(ARCHIVE),'archive_sha256':archive_pin['sha256'],
        'protocol_sha256':sha,'archive_bytes':archive_pin['bytes'],'files':len(expected),'training_run':False}))


if __name__=='__main__':main()
