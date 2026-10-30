"""Complete a partial exact-file scratch cleanup after independent rechecks."""
import ctypes,hashlib,json,msvcrt,os,stat,subprocess,shutil
from ctypes import wintypes
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'outputs/local_cleanup_20261028_v1'
ACTIVE=ROOT/'outputs/codeformer_role_preserving_training_v1_r2_model/codeformer_role_preserving_r2.pth'
SHA='d4abda08831681007fd0c351904ec5dd416a3a5ff6e16d06b25cd8b70b0fc896'
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main():
 plan=json.loads((OUT/'scratch_r2_duplicate_plan_v1.json').read_bytes())['files']
 journal=[json.loads(x) for x in (OUT/'scratch_r2_duplicate_journal_v1.jsonl').read_text().splitlines()]
 assert len(plan)==3 and len(journal)==5
 for i in range(2):assert journal[2*i]=={'intent':plan[i]} and journal[2*i+1]=={'removed':plan[i]['path'],'bytes':plan[i]['bytes']} and not (ROOT/plan[i]['path']).exists()
 assert journal[4]=={'intent':plan[2]} and digest(ACTIVE)==SHA
 row=plan[2];p=ROOT/row['path'];assert p.resolve()==p and p.is_relative_to(ROOT/'scratch')
 for q in (p,*p.parents):
  s=q.lstat();assert not stat.S_ISLNK(s.st_mode) and not getattr(s,'st_file_attributes',0)&1024
 s=p.stat();assert (s.st_size,s.st_ino,s.st_mtime_ns,s.st_nlink)==(row['bytes'],row['inode'],row['mtime_ns'],1)
 rows=json.loads(subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command','Get-CimInstance Win32_Process | Select-Object Name,ProcessId,CommandLine | ConvertTo-Json -Depth 3 -Compress'],capture_output=True,text=True,check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW).stdout)
 for v in rows:
  if v['Name'].lower().startswith(('python','node','rclone','git','powershell','pwsh')):
   assert v.get('CommandLine') and 'r1-fresh-install-verification' not in v['CommandLine'].lower()
 assert digest(p)==SHA
 k=ctypes.WinDLL('kernel32',use_last_error=True);create=k.CreateFileW
 create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];create.restype=wintypes.HANDLE
 before=shutil.disk_usage(ROOT).free;h=create(str(p),0x80000000|0x10000,0,None,3,0x08000000,None)
 if h==ctypes.c_void_p(-1).value:raise OSError(ctypes.get_last_error(),'Exclusive access unavailable; no removal')
 with os.fdopen(msvcrt.open_osfhandle(h,os.O_RDONLY|os.O_BINARY),'rb') as f:
  value=hashlib.sha256()
  for b in iter(lambda:f.read(8*1024**2),b''):value.update(b)
  assert value.hexdigest()==SHA and p.stat().st_ino==row['inode']
  class Disposition(ctypes.Structure):_fields_=[('DeleteFile',ctypes.c_ubyte)]
  op=k.SetFileInformationByHandle;op.argtypes=[wintypes.HANDLE,ctypes.c_int,wintypes.LPVOID,wintypes.DWORD];op.restype=wintypes.BOOL
  flag=Disposition(1);assert op(h,4,ctypes.byref(flag),ctypes.sizeof(flag))
 assert not p.exists() and digest(ACTIVE)==SHA
 result={'complete':True,'initial_partial_journal_retained':True,'removed_exact_copies':plan,'files_removed_total':3,
  'logical_bytes_removed':sum(x['bytes'] for x in plan),'free_before_final_file':before,'free_after':shutil.disk_usage(ROOT).free,
  'retained_active_R2_sha256':SHA,'git_repositories_other_files_retained':True,'VM_training_calls':0}
 with (OUT/'scratch_r2_duplicate_reconciled_closure_v1.json').open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 print(json.dumps({'complete':True,'files_removed':3,'bytes':result['logical_bytes_removed'],'free_GiB':result['free_after']/1024**3}))
if __name__=='__main__':main()
