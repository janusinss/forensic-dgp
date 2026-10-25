"""Inert CLI, finite subprocess supervisor and bounded export; never auto launch."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import tarfile
import time

from . import packet

PREFIX='cctv-codeformer-u96-postfusion-preflight-v1'


def write(path,value):
    with path.open('x',encoding='utf-8') as stream:
        json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def export(root,p,expected_sha):
    packet.verify(root,expected_sha)
    packet.need(platform.system()=='Linux','VM export only; use inert local archive verification for transfers')
    archive=Path.home()/(PREFIX+'-results.tar.gz')
    checksum=Path(str(archive)+'.sha256');receipt=Path.home()/(PREFIX+'-export.json')
    packet.need(not any(path.exists() for path in (archive,checksum,receipt,root/'export.json')),'Previous exports are immutable; do not rerun')
    packet.need(shutil.disk_usage(root).free>=p['limits']['continuing_disk_reserve'],'4GiB export reserve required')
    files=[]
    for path in sorted(root.rglob('*')):
        if path.is_dir():continue
        path=packet.regular(path);relative=path.relative_to(root).as_posix()
        if relative.startswith('weights/'):continue
        packet.need(relative in p['files'] or relative=='protocol.json' or relative.startswith('outputs/') or relative in {'run.log','supervisor.json'},'Unexpected return file: '+relative)
        files.append(path)
    total=sum(path.stat().st_size for path in files)
    packet.need(total<=160*1024**2,'Return payload exceeds160MiB')
    start=time.monotonic();deadline=start+330
    class CappedWriter:
        def __init__(self,stream):self.stream=stream;self.bytes=0
        def write(self,data):
            packet.need(time.monotonic()<deadline,'Finite export wall cap reached')
            packet.need(shutil.disk_usage(root).free>=p['limits']['continuing_disk_reserve'],'Disk export reserve reached')
            self.bytes+=len(data);packet.need(self.bytes<=160*1024**2,'Return archive exceeds160MiB')
            return self.stream.write(data)
        def flush(self):return self.stream.flush()
    try:
        with archive.open('xb') as raw:
            capped=CappedWriter(raw)
            with gzip.GzipFile(filename='',fileobj=capped,mode='wb',compresslevel=1,mtime=0) as compressed:
                with tarfile.open(fileobj=compressed,mode='w|',format=tarfile.PAX_FORMAT) as tar:
                    for path in files:
                        packet.need(time.monotonic()<deadline,'Finite export wall cap reached')
                        tar.add(path,arcname=p['stage_name']+'/'+path.relative_to(root).as_posix(),recursive=False)
        digest=packet.sha256(archive)
        with checksum.open('x',encoding='ascii',newline='\n') as stream:stream.write(digest+'  '+archive.name+'\n')
        result={'complete':True,'archive_sha256':digest,'bytes':archive.stat().st_size,'payload_files':len(files),
                'payload_bytes':total,'seconds':time.monotonic()-start,'protocol_sha256':expected_sha,
                'result_present':(root/'outputs/u96_postfusion_preflight_v1/result.json').is_file(),
                'failure_present':(root/'outputs/u96_postfusion_preflight_v1/failure.json').is_file(),
                'training_success_not_implied':True,'optimizer_updates':0,'model_qualified':False}
        write(root/'export.json',result);write(receipt,result)
        print(json.dumps(result),flush=True)
    except BaseException:
        # A partial archive remains diagnostic evidence; no overwrite/retry.
        raise


def supervise(root,p,expected_sha):
    packet.require_vm(root,p)
    packet.need(not (root/'run.log').exists() and not (root/'supervisor.json').exists(),'Prior launch is immutable')
    command=[sys.executable,'-B','-u',str(root/'scripts/run_u96_postfusion_preflight_v1.py'),
             '--root',str(root),'--protocol-sha',expected_sha,'--run']
    start=time.monotonic();timed_out=False
    with (root/'run.log').open('xb') as log:
        child=subprocess.Popen(command,cwd=str(root),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(json.dumps({'child_pid':child.pid,'log':str(root/'run.log'),'external_cap_seconds':p['limits']['external_seconds']}),flush=True)
        try:code=child.wait(timeout=p['limits']['external_seconds'])
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            code=124
    write(root/'supervisor.json',{'complete':True,'worker_exit':code,'timeout':timed_out,'seconds':time.monotonic()-start,
        'protocol_sha256':expected_sha,'optimizer_updates':0,'training_success_not_implied':True,'model_qualified':False})
    # Export is an independently bounded, fresh subprocess even after worker failure.
    command[-1]='--export'
    try:export_exit=subprocess.run(command,cwd=str(root),timeout=345,check=False).returncode
    except subprocess.TimeoutExpired:export_exit=124
    print(json.dumps({'worker_exit':code,'export_exit':export_exit,'optimizer_updates':0,'model_qualified':False}),flush=True)
    return 0 if code==export_exit==0 else (code or export_exit)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--protocol-sha',required=True)
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify-transfer',action='store_true');group.add_argument('--run',action='store_true')
    group.add_argument('--supervise',action='store_true');group.add_argument('--export',action='store_true')
    args=parser.parse_args();root=packet.directory(args.root)
    p=packet.verify(root,args.protocol_sha)
    if args.verify_transfer:
        print(json.dumps({'complete':True,'static_files':len(p['files']),'witnesses':len(p['witnesses']),
            'neural_or_gradient_calls':0,'optimizer_updates':0,'VM_runtime_qualified':False}),flush=True);return 0
    if args.run:
        from . import worker
        worker.run(root,p,args.protocol_sha);return 0
    if args.supervise:return supervise(root,p,args.protocol_sha)
    export(root,p,args.protocol_sha);return 0
