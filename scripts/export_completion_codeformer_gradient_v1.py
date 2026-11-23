"""Bounded export of the manual L4 diagnostic; never exports model weights."""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import tarfile
import time

STEM='cctv-completion-codeformer-gradient-v1'
MAX_BYTES=160*1024**2
MAX_SECONDS=180

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def need(ok,message):
    if not ok:raise ValueError(message)

class LimitedWriter:
    def __init__(self,file,start):self.file=file;self.start=start;self.count=0
    def write(self,data):
        need(time.monotonic()-self.start<MAX_SECONDS,'Export exceeded180seconds')
        need(self.count+len(data)<=MAX_BYTES,'Export exceeds160MiB')
        n=self.file.write(data);self.count+=n;return n
    def flush(self):return self.file.flush()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--protocol-sha',required=True)
    a=parser.parse_args();root=a.root.resolve(strict=True)
    need(os.name=='posix' and root==(Path.home()/'forensic-dgp/cctv_completion_codeformer_gradient_v1_vm').resolve(),'Export only the named VM diagnostic root')
    need(sha(root/'protocol.json')==a.protocol_sha,'Protocol changed before export')
    need((root/'supervisor.json').is_file(),'Diagnostic has no terminal supervisor receipt')
    supervisor=json.loads((root/'supervisor.json').read_text());need('child_exit'in supervisor,'Diagnostic terminal status missing')
    started=time.monotonic();files=[]
    for name in ('protocol.json','supervisor.json','diagnostic.log'):
        path=root/name;need(path.is_file() and not path.is_symlink(),'Required export file missing');files.append(path)
    if (root/'outputs').is_dir():
        for path in sorted((root/'outputs').rglob('*')):
            need(not path.is_symlink(),'Output symlink refused')
            if path.is_file():files.append(path)
    need(sum(p.stat().st_size for p in files)<=MAX_BYTES,'Raw output export exceeds160MiB')
    manifest={'protocol_sha256':a.protocol_sha,'files':[{'path':p.relative_to(root).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}for p in files],'diagnostic_child_exit':supervisor['child_exit'],'optimizer_updates':0,'training_success_not_implied':True,'quality_qualified':False}
    archive=Path.home()/(STEM+'-results.tar.gz');partial=Path(str(archive)+'.partial')
    for path in (archive,partial,Path(str(archive)+'.sha256'),Path.home()/(STEM+'-export.json')):need(not path.exists(),'Retain existing export; refusing overwrite')
    with partial.open('xb') as raw:
        limited=LimitedWriter(raw,started)
        with gzip.GzipFile(filename='',mode='wb',fileobj=limited,compresslevel=1,mtime=0) as zipped:
            with tarfile.open(fileobj=zipped,mode='w|') as tar:
                for path,item in zip(files,manifest['files']):
                    need(sha(path)==item['sha256'],'Evidence changed during export')
                    tar.add(path,arcname='cctv_completion_codeformer_gradient_v1_results/'+item['path'],recursive=False)
                body=(json.dumps(manifest,indent=2)+'\n').encode()
                info=tarfile.TarInfo('cctv_completion_codeformer_gradient_v1_results/export_manifest.json');info.size=len(body);info.mtime=0
                tar.addfile(info,io.BytesIO(body))
    os.replace(partial,archive)
    digest=sha(archive)
    with Path(str(archive)+'.sha256').open('x',encoding='ascii',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
    receipt={'complete':True,'archive_sha256':digest,'bytes':archive.stat().st_size,'seconds':time.monotonic()-started,'protocol_sha256':a.protocol_sha,'diagnostic_child_exit':supervisor['child_exit'],'training_success_not_implied':True,'optimizer_updates':0}
    with (Path.home()/(STEM+'-export.json')).open('x',encoding='utf-8',newline='\n')as stream:json.dump(receipt,stream,indent=2);stream.write('\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
