"""Bounded return export for one manual-L4 CodeFormer completion pilot.

Includes all retained outputs, including a failed terminal .partial, but excludes
installed source/assets/base weights. Archive completion is not model quality.
No neural imports, model loading, training, cleanup or overwrite operation.
"""
import argparse
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import stat
import tarfile
import time

STEM='cctv-completion-codeformer-finetune-v1'
VM_NAME='cctv_completion_codeformer_finetune_v1_vm'
MEMBER_ROOT='cctv_completion_codeformer_finetune_v1_results'
MAX_BYTES=1536*1024**2
MAX_SECONDS=900

def need(ok,message):
    if not ok:raise ValueError(message)

def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def regular(path):
    info=path.lstat();need(stat.S_ISREG(info.st_mode) and not path.is_symlink(),'Only regular evidence files may be exported')
    return info

class LimitedWriter:
    def __init__(self,file,started):self.file=file;self.started=started;self.count=0
    def write(self,data):
        need(time.monotonic()-self.started<MAX_SECONDS,'Export exceeded900seconds')
        need(self.count+len(data)<=MAX_BYTES,'Compressed export exceeds1536MiB')
        n=self.file.write(data);need(n==len(data),'Short archive write');self.count+=n;return n
    def flush(self):return self.file.flush()

def export(root,expected,home):
    root=root.resolve(strict=True);home=home.resolve(strict=True)
    need(os.name=='posix' and root==(home/'forensic-dgp'/VM_NAME).resolve(),'Export only the named manualL4 pilot root')
    started=time.monotonic()
    def guard():need(time.monotonic()-started<MAX_SECONDS,'Export exceeded900seconds')
    required=('protocol.json','supervisor.json','trainer.log')
    files=[]
    for name in required:
        path=root/name;regular(path);files.append(path)
    need(sha(root/'protocol.json')==expected,'Protocol differs before export')
    supervisor=json.loads((root/'supervisor.json').read_text(encoding='utf-8-sig'))
    need(supervisor.get('complete') is True and type(supervisor.get('worker_exit')) is int,'Terminal supervisor receipt required')
    output=root/'outputs'
    if output.exists():
        need(output.is_dir() and not output.is_symlink(),'Invalid output evidence root')
        for path in sorted(output.rglob('*')):
            need(not path.is_symlink(),'Output symlink refused')
            need(path.resolve().is_relative_to(output.resolve()),'Output escaped declared evidence directory')
            if path.is_dir():continue
            regular(path);files.append(path)
    need(sum(regular(p).st_size for p in files)<=MAX_BYTES,'Uncompressed evidence exceeds1536MiB')
    entries=[]
    for path in files:
        guard();before=regular(path);digest=sha(path);after=regular(path)
        need((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns),'Evidence changed while hashing')
        entries.append({'path':path.relative_to(root).as_posix(),'bytes':after.st_size,'sha256':digest})
    partial_states=[row['path'] for row in entries if row['path'].endswith('.partial')]
    committed=[row for row in entries if row['path']=='outputs/terminal_training_state.pth']
    manifest={'format':'completion-codeformer-finetune-return-manifest-v1','protocol_sha256':expected,'files':entries,'worker_exit':supervisor['worker_exit'],'worker_timeout':supervisor.get('timeout'),'worker_interruption':supervisor.get('interruption'),'terminal_state_committed':bool(committed),'terminal_state_pin':committed[0] if committed else None,'partial_files_retained':partial_states,'state_recoverability':'Pending independent safe-load/state/optimizer/RNG audit. A .partial is failed serialization evidence, not a resumable checkpoint.','original_base_weights_exported':False,'model_qualified':False,'training_success_not_implied':True}
    archive=home/(STEM+'-results.tar.gz');partial=Path(str(archive)+'.partial')
    checksum=Path(str(archive)+'.sha256');receipt_path=home/(STEM+'-export.json')
    for path in (archive,partial,checksum,receipt_path):need(not path.exists(),'Preserve prior export; refusing overwrite')
    with partial.open('xb') as raw:
        limited=LimitedWriter(raw,started)
        with gzip.GzipFile(filename='',mode='wb',fileobj=limited,compresslevel=1,mtime=0) as zipped:
            with tarfile.open(fileobj=zipped,mode='w|') as tar:
                for path,row in zip(files,entries):
                    guard();need(regular(path).st_size==row['bytes'] and sha(path)==row['sha256'],'Evidence changed before archive member')
                    info=tar.gettarinfo(str(path),arcname=MEMBER_ROOT+'/'+row['path'])
                    need(info.isreg() and info.size==row['bytes'],'Archive evidence type/size changed')
                    with path.open('rb') as stream:tar.addfile(info,stream)
                    need(sha(path)==row['sha256'],'Evidence changed during archive member')
                body=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode('utf-8')
                info=tarfile.TarInfo(MEMBER_ROOT+'/export_manifest.json');info.size=len(body);info.mtime=0;info.mode=0o644
                tar.addfile(info,io.BytesIO(body))
        raw.flush();os.fsync(raw.fileno())
    guard();need(partial.stat().st_size<=MAX_BYTES,'Final archive byte cap')
    digest=sha(partial);guard();need(not archive.exists(),'Archive appeared before publication')
    os.replace(partial,archive)
    with checksum.open('x',encoding='ascii',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
    receipt={'format':'completion-codeformer-finetune-export-receipt-v1','complete':True,'archive_sha256':digest,'bytes':archive.stat().st_size,'seconds':time.monotonic()-started,'protocol_sha256':expected,'worker_exit':supervisor['worker_exit'],'terminal_state_committed':bool(committed),'partial_files_retained':partial_states,'state_recoverability_pending_independent_audit':True,'training_success_not_implied':True,'model_qualified':False,'payload_files':len(entries),'export_only_neural_calls':0}
    with receipt_path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(receipt),flush=True);return receipt

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    args=parser.parse_args();export(args.root,args.protocol_sha,Path.home())
if __name__=='__main__':main()
