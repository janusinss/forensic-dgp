"""Remove exactly three disposable R2 test copies after fresh exclusive checks."""
import ctypes,hashlib,json,msvcrt,os,stat,subprocess,time
from ctypes import wintypes
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/local_cleanup_20261028_v1'
DIGEST='d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
ACTIVE=ROOT/'outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth'
NAMES=('implementation-fresh-clone','implementation-r1-integration','r1-fresh-install-verification')
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 assert digest(ACTIVE)==DIGEST
 census=subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command',
  'Get-CimInstance Win32_Process | Select-Object Name,ProcessId,CommandLine | ConvertTo-Json -Depth 3 -Compress'],
  capture_output=True,text=True,check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
 rows=json.loads(census.stdout);targets=[x.lower() for x in NAMES]
 for row in rows:
  name=row['Name'].lower();line=(row.get('CommandLine') or '').lower()
  if name.startswith(('python','node','rclone','git','powershell','pwsh')):
   assert line,'Relevant process command unavailable'
   assert not any(x in line for x in targets),'Scratch copy in active process'
 plan=[]
 for name in NAMES:
  p=ROOT/'scratch'/name/'models/implementation/codeformer_r2.pth'
  assert p.resolve()==p and p.is_relative_to(ROOT/'scratch')
  for part in (p,*p.parents):
   s=part.lstat();assert not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&1024
  s=p.stat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size==376609465
  assert digest(p)==DIGEST and p.stat().st_ino==s.st_ino and p.stat().st_mtime_ns==s.st_mtime_ns
  plan.append({'path':p.relative_to(ROOT).as_posix(),'bytes':s.st_size,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'sha256':DIGEST})
 evidence={'complete':True,'scope':'three exact redundant scratch R2 model copies','active_model':ACTIVE.relative_to(ROOT).as_posix(),
  'active_sha256':DIGEST,'files':plan,'process_count':len(rows),'git_repositories_and_other_files_retained':True}
 with (OUT/'scratch_r2_duplicate_plan_v1.json').open('x',encoding='utf-8') as f:json.dump(evidence,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 k=ctypes.WinDLL('kernel32',use_last_error=True);create=k.CreateFileW
 create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];create.restype=wintypes.HANDLE
 before=__import__('shutil').disk_usage(ROOT).free
 with (OUT/'scratch_r2_duplicate_journal_v1.jsonl').open('x',encoding='utf-8',newline='\n') as journal:
  for row in plan:
   p=ROOT/row['path'];s=p.stat();assert (s.st_ino,s.st_mtime_ns,s.st_size,s.st_nlink)==(row['inode'],row['mtime_ns'],row['bytes'],1)
   journal.write(json.dumps({'intent':row})+'\n');journal.flush();os.fsync(journal.fileno())
   handle=create(str(p),0x80000000|0x00010000,0,None,3,0x08000000,None);assert handle!=ctypes.c_void_p(-1).value
   fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
   with os.fdopen(fd,'rb') as stream:
    h=hashlib.sha256()
    for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
    assert h.hexdigest()==DIGEST and p.stat().st_ino==row['inode']
    class Disposition(ctypes.Structure):_fields_=[('DeleteFile',ctypes.c_ubyte)]
    op=k.SetFileInformationByHandle;op.argtypes=[wintypes.HANDLE,ctypes.c_int,wintypes.LPVOID,wintypes.DWORD];op.restype=wintypes.BOOL
    flag=Disposition(1);assert op(handle,4,ctypes.byref(flag),ctypes.sizeof(flag))
   assert not p.exists();journal.write(json.dumps({'removed':row['path'],'bytes':row['bytes']})+'\n');journal.flush();os.fsync(journal.fileno())
 assert digest(ACTIVE)==DIGEST
 after=__import__('shutil').disk_usage(ROOT).free
 result={'complete':True,'removed_files':len(plan),'logical_bytes_removed':sum(x['bytes'] for x in plan),'free_before':before,'free_after':after,
  'active_R2_sha256_after':DIGEST,'exact_files_only':True,'no_directory_recursive_delete':True,'training_or_VM_calls':0}
 with (OUT/'scratch_r2_duplicate_closure_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(result))
if __name__=='__main__':main()
