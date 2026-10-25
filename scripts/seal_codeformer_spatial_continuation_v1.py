"""Local inert archive sealing and full readback; never launches a VM experiment."""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.codeformer_spatial_continuation_v1 import packet as p
from research.codeformer_spatial_continuation_v1 import audit


class Limited:
    def __init__(self, stream):self.stream=stream;self.count=0
    def write(self, data):
        p.require(self.count+len(data)<=1024*p.MiB,'Execution archive1GiB cap')
        count=self.stream.write(data);p.require(count==len(data),'Short archive write')
        self.count+=count;return count
    def flush(self):return self.stream.flush()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol-sha',required=True)
    args=parser.parse_args();stage=ROOT/'outputs'/p.VM_NAME
    protocol=p.verify(stage,args.protocol_sha);files=p.inventory(stage)
    p.require(set(files)==set(protocol['files'])|{'protocol.json'},'Only pristine static stage may be sealed')
    checked=audit.audit_static(stage,protocol);p.require(checked['audit_pass'] is True,'Independent static stage audit')
    p.require('torch' not in sys.modules,'Sealing must remain non-neural')
    archive=ROOT/'outputs/cctv-codeformer-spatial-continuation-v1-execution.tar.gz'
    temporary=Path(str(archive)+'.partial');checksum=Path(str(archive)+'.sha256')
    receipt=ROOT/'outputs/codeformer_spatial_continuation_packet_review_v1/execution_archive_v1.json'
    p.require(all(not f.exists() for f in (archive,temporary,checksum,receipt)),'Preserve existing seal/partial')
    p.require(shutil.disk_usage(archive.parent).free>=6*1024**3,'Local4GiB reserve plus2bounded archive copies')
    pins={name:p.pin(stage,path) for name,path in files.items()};started=time.monotonic()
    with temporary.open('xb') as destination:
        with gzip.GzipFile(filename='',mode='wb',fileobj=Limited(destination),compresslevel=1,mtime=0) as compressed:
            with tarfile.open(fileobj=compressed,mode='w|') as tar:
                for name,path in sorted(files.items()):
                    p.require(time.monotonic()-started<300,'Local seal300s cap')
                    p.require(p.pin(stage,path)==pins[name],'Source changed before archive')
                    info=tarfile.TarInfo(p.VM_NAME+'/'+name)
                    info.size=pins[name]['bytes'];info.mode=0o644;info.mtime=0
                    with path.open('rb') as source:tar.addfile(info,source)
                    p.require(p.pin(stage,path)==pins[name],'Source changed during archive')
        destination.flush();os.fsync(destination.fileno())
    observed={};payload=0
    with tarfile.open(temporary,'r|gz') as tar:
        for member in tar:
            p.require(time.monotonic()-started<300,'Local readback300s cap')
            p.require(member.isfile() and not member.issparse() and member.name.startswith(p.VM_NAME+'/'),
                      'Regular sealed VM members only')
            name=member.name[len(p.VM_NAME)+1:];p.canonical(name)
            p.require(name in pins and name not in observed and member.size==pins[name]['bytes'],'Member namespace/size')
            digest=hashlib.sha256();count=0
            with tar.extractfile(member) as source:
                for b in iter(lambda:source.read(p.MiB),b''):digest.update(b);count+=len(b)
            p.require(count==pins[name]['bytes'] and digest.hexdigest()==pins[name]['sha256'],'Member byte hash')
            observed[name]=pins[name];payload+=count
    p.require(set(observed)==set(pins),'Missing sealed members')
    with temporary.open('rb') as source,archive.open('xb') as destination:
        shutil.copyfileobj(source,destination,p.MiB);destination.flush();os.fsync(destination.fileno())
    digest=p.sha256(archive)
    p.require(digest==p.sha256(temporary),'Published archive differs')
    p.require(temporary.resolve().parent==(ROOT/'outputs').resolve(),'Temporary path must stay in workspace outputs')
    temporary.unlink()  # Only this invocation's verified temporary container.
    with checksum.open('xb') as f:f.write((digest+'  '+archive.name+'\n').encode('ascii'))
    result={'complete':True,'archive':{'path':archive.relative_to(ROOT).as_posix(),
          'bytes':archive.stat().st_size,'sha256':digest},'checksum':p.pin(ROOT,checksum),
          'protocol_sha256':args.protocol_sha,'regular_members':len(pins),'payload_bytes':payload,
          'all_member_bytes_streamed_and_hash_verified':True,'checksum_ASCII_LF_no_CR':b'\r' not in checksum.read_bytes(),
          'compression':'explicit gzip.GzipFile + uncompressed tar stream; Python3.10-compatible APIs',
          'stage_source_pins':pins,'seconds':time.monotonic()-started,
          'Torch_neural_gradient_optimizer_training_VM_calls':0,'actual_continuation_unrun':True,
          'model_qualified':False,'goal_complete':False}
    with receipt.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='stage_source_pins'}))


if __name__=='__main__':main()
