"""Exact-file local removal after independent audit and immutable Drive backup."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
import msvcrt
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from prepare_google_drive_offload_20261015_v1 import ROOT,OUT,read,sha,bounded,metadata

def save(name,data):
    with (OUT/name).open('x',encoding='utf-8') as f:json.dump(data,f,indent=2);f.write('\n')

def guard(plan):
    cmd='Get-CimInstance Win32_Process | Select-Object Name,ProcessId,ParentProcessId,CommandLine | ConvertTo-Json -Depth 3'
    run=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',cmd],capture_output=True,text=True,check=True,timeout=30)
    rows=json.loads(run.stdout);rows=rows if isinstance(rows,list) else [rows]
    groups={Path(r['path']).parts[1].lower() for r in plan['candidates']}
    relevant=[]
    for r in rows:
        if not re.match(r'^(python.*|rclone|gcloud|tar|7z|gzip|robocopy|DriveFS|gsutil|pscp)\.exe$',r['Name'],re.I):continue
        text=r.get('CommandLine');assert text,('Uninspectable process',r['Name'],r['ProcessId'])
        normalized=text.replace('\\','/').lower()
        assert not any(g in normalized for g in groups),('Active output dependency',r)
        if r['Name'].lower().startswith('python'):
            allowed=any(s in normalized for s in [' -b app.py','serve_restoration_checkpoint_comparison_v1.py',
                        'apply_google_drive_offload_20261015_v1.py','snapshot_google_drive_offload_20261015_v1.py',
                        'google_drive_offload_20261015_v1/reviewer/review_plan.py'])
            assert allowed,('Unexpected research process',r)
        else:raise RuntimeError('Competing archive/transfer process: '+json.dumps(r))
        relevant.append(r)
    return relevant

def remove(row):
    path=bounded(row['path'])
    expected={k:row[k] for k in ['bytes','mtime_ns','inode','dev','nlink','mode','attributes']}
    assert metadata(path)==expected and row['nlink']==1
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    create=k.CreateFileW;create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];create.restype=wintypes.HANDLE
    handle=create(str(path),0x80000000|0x00010000,0,None,3,0x08000000,None)
    assert handle!=ctypes.c_void_p(-1).value,ctypes.WinError(ctypes.get_last_error())
    fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
    with os.fdopen(fd,'rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==row['sha256'],row['path']
        assert metadata(path)==expected
        class Disposition(ctypes.Structure):_fields_=[('DeleteFile',ctypes.c_ubyte)]
        op=k.SetFileInformationByHandle;op.argtypes=[wintypes.HANDLE,ctypes.c_int,wintypes.LPVOID,wintypes.DWORD];op.restype=wintypes.BOOL
        flag=Disposition(1)
        assert op(handle,4,ctypes.byref(flag),ctypes.sizeof(flag)),ctypes.WinError(ctypes.get_last_error())
    assert not path.exists()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan-sha',required=True)
    parser.add_argument('--validate-only',action='store_true');a=parser.parse_args()
    start=time.monotonic();assert sha(OUT/'plan.json')==a.plan_sha;p=read(OUT/'plan.json')
    assert p['root']==str(ROOT) and len(p['candidates'])==76973
    independent=read(OUT/'reviewer/independent_before_audit.json')
    assert independent['complete'] and independent['plan_sha256']==a.plan_sha
    assert read(OUT/'metadata_transfer_receipt.json')['complete']
    assert read(OUT/'metadata_transfer_receipt.json')['plan_sha256']==a.plan_sha
    for name,digest in p['protected_hashes'].items():assert sha(bounded(name))==digest,name
    save('validation_process_guard.json' if a.validate_only else 'apply_process_guard.json',
         dict(complete=True,processes=guard(p)))
    if a.validate_only:
        print(json.dumps({'complete':True,'validation_only':True,'files_removed':0}));return
    assert not (OUT/'deletion_ledger.jsonl').exists() and not (OUT/'apply_receipt.json').exists()
    before_free=shutil.disk_usage(ROOT).free;count=0;removed=0
    with (OUT/'deletion_ledger.jsonl').open('x',encoding='utf-8') as ledger:
        def event(data):
            ledger.write(json.dumps(data,separators=(',',':'))+'\n');ledger.flush();os.fsync(ledger.fileno())
        for row in p['candidates']:
            assert time.monotonic()-start<1800,'Finite 30-minute apply cap; retain partial ledger'
            if count and count%5000==0:
                guard(p);print({'removed':count,'of':len(p['candidates']),'bytes':removed,'seconds':round(time.monotonic()-start,1)},flush=True)
            base=dict(path=row['path'],sha256=row['sha256'],bytes=row['bytes'],archive_id=p['archive']['id'])
            event(dict(event='removal_intent',**base));remove(row)
            event(dict(event='removed',exclusive_handle_hash_verified=True,native_exact_file_removal=True,**base));count+=1;removed+=row['bytes']
    assert count==len(p['candidates']) and removed==p['candidate_bytes']
    for name,digest in p['protected_hashes'].items():assert sha(bounded(name))==digest,name
    code=subprocess.run([str(ROOT/'venv/Scripts/python.exe'),'-X','utf8','-B',str(ROOT/'scripts/snapshot_google_drive_offload_20261015_v1.py'),'--name','namespace_after'],check=True,timeout=650)
    after=read(OUT/'namespace_after_receipt.json');assert after['complete']
    receipt=dict(complete=True,plan_sha256=a.plan_sha,files_removed=count,bytes_removed=removed,
                 deletion_ledger_sha256=sha(OUT/'deletion_ledger.jsonl'),
                 inventory_after_file='outputs/google_drive_offload_20261015_v1/namespace_after.json.gz',
                 inventory_after_sha256=sha(OUT/'namespace_after.json.gz'),
                 free_bytes_before=before_free,free_bytes_after=shutil.disk_usage(ROOT).free,
                 observed_free_increase_bytes=shutil.disk_usage(ROOT).free-before_free,
                 protected_hashes_verified=len(p['protected_hashes']),VM_calls=0,training_calls=0,
                 seconds=time.monotonic()-start)
    save('apply_receipt.json',receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
