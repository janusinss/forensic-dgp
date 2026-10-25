"""Explicit verify/run/export CLI. Only --run can enter manual L4 training."""
import argparse
import io
import json
import os
from pathlib import Path
import platform
import signal
import sys
import tarfile
import time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from research.codeformer_spatial_objectives_v1 import packet as p

PREFIX='cctv-codeformer-spatial-objectives-v1'


def export(root, protocol, digest):
    """Fresh explicit archive; no model, Torch or optimizer imported."""
    p.require(platform.system()=='Linux','VM export requires Linux')
    p.require((root/'supervisor.json').is_file(),'Terminal supervisor receipt required before export')
    terminal=p.read_json(root/'supervisor.json')
    p.require(terminal.get('complete') is True and terminal.get('worker_terminal') is True and
              terminal.get('owned_child_group_terminal') is True,
              'Worker must be terminal before export')
    home=p.directory(Path.home());archive=home/(PREFIX+'-results.tar.gz')
    temp=home/(archive.name+'.partial');checksum=home/(archive.name+'.sha256')
    companion=home/(PREFIX+'-export.json')
    p.require(all(not f.exists() for f in (archive,temp,checksum,companion)),
              'Preserve previous export/partial; no overwrite')
    observed=p.inventory(root)
    names=sorted(n for n in observed if n in protocol['return_static'] or n=='protocol.json' or
                 (n in protocol['allowed_outputs'] and not n.startswith('cache/') and n!='export.json'))
    total=sum(observed[n].stat().st_size for n in names)
    p.require(total <= p.LIMITS['return_bytes']-2*p.MiB,'Return archive logical cap exceeded')
    writer=p.Writer(root,protocol)
    receipt={'format':'codeformer-spatial-objectives-export-manifest-v1','complete':True,
        'protocol_sha256':digest,'worker_terminal':True,'worker_exit':terminal['worker_exit'],
        'training_success_not_implied':True,'model_qualified':False,
        'VM_only_static_and_observed_features_not_returned':True,
        'checkpoint_pickle_contents_not_audited_by_export':True,
        'members':[p.pin(root,observed[n]) for n in names]}
    writer.json('export.json',receipt);names.append('export.json');names.sort()
    p.require(sum((root/n).stat().st_size for n in names)<=p.LIMITS['return_bytes'],'Final return payload cap')
    started=time.monotonic()
    old=signal.getsignal(signal.SIGALRM)
    try:
        import shutil
        p.require(shutil.disk_usage(home).free>=p.LIMITS['disk_reserve_bytes']+p.LIMITS['archive_bytes'],
                  'Export would cross4GiB disk reserve')
        class Limited:
            def __init__(self,f):self.f=f;self.count=0
            def write(self,b):
                p.require(self.count+len(b)<=p.LIMITS['archive_bytes'],'Archive byte cap')
                n=self.f.write(b);self.count+=n;return n
            def flush(self):return self.f.flush()
        with temp.open('xb') as raw:
            with tarfile.open(fileobj=Limited(raw),mode='w|gz',compresslevel=1) as tar:
                for name in names:
                    path=p.regular(root/name);before=p.pin(root,path)
                    info=tarfile.TarInfo(name);info.size=path.stat().st_size;info.mode=0o644;info.mtime=0
                    with path.open('rb') as f:tar.addfile(info,f)
                    p.require(p.pin(root,path)==before,'Artifact changed during export')
            raw.flush();os.fsync(raw.fileno())
        # No-overwrite atomic final publication; unsupported links retain partial.
        os.link(temp,archive);temp.unlink()
        digest_archive=p.sha256(archive)
        with checksum.open('xb') as f:f.write((digest_archive+'  '+archive.name+'\n').encode('ascii'))
        out={'complete':True,'archive_sha256':digest_archive,'bytes':archive.stat().st_size,
             'protocol_sha256':digest,'seconds':time.monotonic()-started,
             'training_success_not_implied':True,'model_qualified':False,
             'worker_exit':terminal['worker_exit'],'members':len(names),'return_payload_bytes':sum((root/n).stat().st_size for n in names)}
        with companion.open('xb') as f:f.write(p.json_bytes(out))
        print(json.dumps(out),flush=True);return out
    finally:signal.signal(signal.SIGALRM,old)


def actions():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT);parser.add_argument('--protocol-sha',required=True)
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--verify-transfer',action='store_true');action.add_argument('--run',action='store_true')
    action.add_argument('--export',action='store_true')
    a=parser.parse_args();root=p.directory(a.root);protocol=p.verify(root,a.protocol_sha)
    if a.verify_transfer:
        from research.codeformer_spatial_objectives_v1 import audit
        r=audit.audit_static(root,protocol)
        print(json.dumps({'complete':r['audit_pass'],'static_files':r['static_files'],
            'TRAIN_observations':10,'branches':2,'maximum_optimizer_updates':48,
            'actual_neural_or_optimizer_calls':0,'ready_for_manual_runtime_preflight':True}),flush=True)
    elif a.run:
        from research.codeformer_spatial_objectives_v1 import worker
        worker.run(root,protocol,a.protocol_sha)
    else:export(root,protocol,a.protocol_sha)


def main():
    if '--export' not in sys.argv:
        return actions()
    p.require(platform.system()=='Linux','Manual Linux export required')
    def expired(sig,frame):raise TimeoutError('Whole export300s ceiling exceeded')
    old=signal.signal(signal.SIGALRM,expired);signal.alarm(300)
    try:return actions()
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old)


if __name__=='__main__':main()
