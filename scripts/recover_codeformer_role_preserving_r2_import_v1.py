"""Preserve interrupted inert import; exact rename, no delete or model work."""
from datetime import datetime, timezone
import ctypes
from ctypes import wintypes
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUTPUTS=ROOT/'outputs'
OLD=OUTPUTS/'codeformer_role_preserving_training_v1_r2_return'
SAVED=OUTPUTS/'codeformer_role_preserving_training_v1_r2_return_interrupted_20261018_v1'
REVIEW=OUTPUTS/'codeformer_role_preserving_training_v1_r2_return_review'
SCRIPT=ROOT/'scripts/import_codeformer_role_preserving_training_return_v1_r2.py'
EXPECTED_SCRIPT='4f0a7703c813535110b534d8f792ac488b98fc40ad1258932306ad7c84a5bb39'


def need(value,message):
    if not value:raise ValueError(message)


def safe(path,kind=None):
    path=Path(path).absolute();need(path.resolve(strict=False).is_relative_to(OUTPUTS.resolve()),'Exact workspace outputs scope required')
    for part in (path,*path.parents):
        if part.exists()or part.is_symlink():
            s=part.lstat();need(not stat.S_ISLNK(s.st_mode)and not getattr(s,'st_file_attributes',0)&1024,'No linked/reparse recovery paths')
    if kind:
        s=path.lstat();need((stat.S_ISDIR(s.st_mode)if kind=='dir'else stat.S_ISREG(s.st_mode)and s.st_nlink==1),'Expected ordinary recovery type required')
    return path


def python_processes():
    """Read Windows native process snapshot; admit only this exact wrapper PID."""
    need(os.name=='nt','Windows-native recovery inventory required')
    class ENTRY(ctypes.Structure):
        _fields_=[('dwSize',wintypes.DWORD),('cntUsage',wintypes.DWORD),('th32ProcessID',wintypes.DWORD),
            ('th32DefaultHeapID',ctypes.c_size_t),('th32ModuleID',wintypes.DWORD),('cntThreads',wintypes.DWORD),
            ('th32ParentProcessID',wintypes.DWORD),('pcPriClassBase',wintypes.LONG),('dwFlags',wintypes.DWORD),
            ('szExeFile',wintypes.WCHAR*260)]
    api=ctypes.WinDLL('kernel32',use_last_error=True)
    api.CreateToolhelp32Snapshot.argtypes=[wintypes.DWORD,wintypes.DWORD];api.CreateToolhelp32Snapshot.restype=wintypes.HANDLE
    api.Process32FirstW.argtypes=[wintypes.HANDLE,ctypes.POINTER(ENTRY)];api.Process32FirstW.restype=wintypes.BOOL
    api.Process32NextW.argtypes=[wintypes.HANDLE,ctypes.POINTER(ENTRY)];api.Process32NextW.restype=wintypes.BOOL
    api.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=api.CreateToolhelp32Snapshot(2,0);need(handle not in(None,ctypes.c_void_p(-1).value),'Native process snapshot unavailable')
    rows=[]
    try:
        row=ENTRY();row.dwSize=ctypes.sizeof(row);ok=api.Process32FirstW(handle,ctypes.byref(row));need(bool(ok),'Native process inventory unreadable')
        while ok:
            if row.szExeFile.casefold().startswith(('python','pypy')):
                rows.append({'pid':int(row.th32ProcessID),'parent_pid':int(row.th32ParentProcessID),'executable':row.szExeFile})
            ok=api.Process32NextW(handle,ctypes.byref(row))
        need(ctypes.get_last_error()==18,'Native process iteration ended unexpectedly')
    finally:api.CloseHandle(handle)
    # Windows venv python.exe is a launcher and creates this real interpreter
    # as its child. Admit only that exact direct parent executable, never a
    # sibling/background Python process.
    accepted={os.getpid()}
    parent=next((row for row in rows if row['pid']==os.getppid()),None)
    if parent:
        api.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];api.OpenProcess.restype=wintypes.HANDLE
        api.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)]
        api.QueryFullProcessImageNameW.restype=wintypes.BOOL
        handle=api.OpenProcess(0x1000,False,parent['pid']);need(handle,'Cannot inspect direct launcher executable')
        try:
            text=ctypes.create_unicode_buffer(32768);width=wintypes.DWORD(len(text))
            need(api.QueryFullProcessImageNameW(handle,0,text,ctypes.byref(width)),'Cannot read direct launcher image path')
            need(Path(text.value).resolve()==(ROOT/'venv/Scripts/python.exe').resolve(),'Unexpected Python parent launcher')
            parent['exact_workspace_venv_launcher_path']=text.value;accepted.add(parent['pid'])
        finally:api.CloseHandle(handle)
    need(not any(row['pid']not in accepted for row in rows),'Another Python process exists; preserve interrupted import untouched')
    return rows


def census(root):
    rows=[]
    for parent,dirs,files in os.walk(root,followlinks=False):
        safe(Path(parent),'dir')
        for name in dirs:safe(Path(parent)/name,'dir')
        for name in files:
            path=safe(Path(parent)/name,'file');before=path.stat();h=hashlib.sha256()
            with path.open('rb')as stream:
                for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
            after=path.stat();need((before.st_size,before.st_ino,before.st_mtime_ns)==(after.st_size,after.st_ino,after.st_mtime_ns),'Partial file changed during recovery census')
            rows.append({'path':path.relative_to(root).as_posix(),'bytes':before.st_size,'sha256':h.hexdigest(),
                'inode':before.st_ino,'mtime_ns':before.st_mtime_ns})
    return sorted(rows,key=lambda row:row['path'])


def write_new(path,value):
    safe(path)
    with path.open('x',encoding='utf-8',newline='\n')as stream:json.dump(value,stream,indent=2);stream.write('\n')


def main():
    started=time.monotonic();safe(OLD,'dir');safe(SAVED)
    need(not SAVED.exists()and not(REVIEW/'return_import_v1.json').exists()and
        not(REVIEW/'import_failure_v1.json').exists(),'Exclusive preserved partial/no terminal receipt required')
    need(hashlib.sha256(SCRIPT.read_bytes()).hexdigest()==EXPECTED_SCRIPT,'Previously reviewed importer source changed')
    processes=python_processes();rows=census(OLD)
    need(len(rows)==1378 and sum(row['bytes']for row in rows)==923202037 and
        rows[-1]['path']=='outputs/role_preserving_training_v1_r2/TRAIN/train_u0039_slot2/candidate_native.npy'and
        rows[-1]['bytes']==0,'Exact observed interrupted partial census differs')
    processes_again=python_processes()
    before={'complete_inventory':True,'UTC':datetime.now(timezone.utc).isoformat(),'old_absolute_path':str(OLD),
        'preserved_absolute_path':str(SAVED),'native_process_snapshot_before':processes,
        'native_process_snapshot_after_hashing':processes_again,'only_wrapper_python_PID':os.getpid(),
        'files':rows,'regular_files':len(rows),'payload_bytes':sum(row['bytes']for row in rows),
        'interrupted_zero_length_partial_retained':True,'delete_performed':False,'rename_not_yet_performed':True,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'reviewed_importer_sha256':EXPECTED_SCRIPT}
    write_new(REVIEW/'import_recovery_before_rename_v1.json',before)
    OLD.rename(SAVED)
    need(not OLD.exists()and SAVED.is_dir(),'Atomic exact recovery rename did not close')
    after=census(SAVED);need(after==rows,'Preserved renamed partial bytes/identities differ')
    write_new(REVIEW/'import_recovery_rename_v1.json',{'complete':True,'UTC':datetime.now(timezone.utc).isoformat(),
        'source_absolute':str(OLD),'preserved_absolute':str(SAVED),'all1378file_bytes_hashes_inodes_mtimes_exact':True,
        'preserved_files':1378,'preserved_bytes':923202037,'no_delete':True,'seconds':time.monotonic()-started,
        'new_exclusive_original_importer_pending':True,'actual_local_neural_gradient_optimizer_VM_calls':0})
    print(json.dumps({'preserved_partial':str(SAVED),'files':1378,'bytes':923202037,'starting_fresh_exclusive_import':True}),flush=True)
    result=subprocess.run([sys.executable,'-X','utf8','-B',str(SCRIPT)],cwd=ROOT,check=False)
    need(result.returncode==0,'Fresh importer terminal failure; all partial evidence preserved')
    print(json.dumps({'recovery_complete':True,'fresh_import_exit_code':result.returncode,'seconds':time.monotonic()-started}),flush=True)


if __name__=='__main__':main()
