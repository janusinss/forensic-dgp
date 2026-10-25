"""Exact local cold restoration offload; keep U96, comparisons/data/code/completion.

Copy is immutable, then full SHA256 download verified. Old raw/checkpoint copies
require a freshly streamed archive and exact per-member hashes. No VM/training.
"""
from __future__ import annotations
import argparse,ctypes,gzip,hashlib,inspect,io,json,msvcrt,os,shutil,stat,subprocess,tarfile,time
from ctypes import wintypes
from pathlib import Path
import storage_offload_20261018_v1 as base

ROOT=base.ROOT
OUT=ROOT/'outputs/u96_inactive_restoration_offload_20261021_v1'
DEST='thesis_archive:FORENSIC_DGP_ARCHIVE/Testing/20261021_u96_inactive_v1'
OLD_DEST='thesis_archive:FORENSIC_DGP_ARCHIVE/Testing'
MARKER='scripts/offload_u96_inactive_restoration_20261021_v1.py'
R2_RESUME='outputs/codeformer_role_preserving_training_v1_r2_return/outputs/role_preserving_training_v1_r2/snapshot_0050/checkpoint.pt'
OLD_RUNS={
 'adain-gradient-isolation-v1':'adain_gradient_isolation_v1',
 'adaptation-v1':'adaptation_v1','adaptation-v2':'adaptation_v2',
 'bounded-fusion-feasibility-v1':'bounded_fusion_feasibility_v1',
 'filter-fusion-training-v1':'filter_fusion_training_v1',
 'fp32-comparison-v1':'fp32_comparison_v1','loss-response-v1':'loss_response_v1',
 'normal-tone-training-v1':'normal_tone_training_v1',
 'role-preserving-training-v1-r2':'role_preserving_training_v1_r2',
 'tone-control-training-v1':'tone_control_training_v1'}
require=base.require

def save(name,value):
 p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x',encoding='utf-8',newline='\n') as f:
  json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 return base.pin(p)
def read(name):return json.loads((OUT/name).read_text(encoding='utf-8-sig'))
def identity(path):return base.metadata(path)
def protected_hash(p):
 before=p.stat();h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 after=p.stat();require((before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns),'Protected file changed')
 return {'path':p.relative_to(ROOT).as_posix(),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns,'sha256':h.hexdigest()}
def guard(candidates):
 source=inspect.getsource(base.process_guard).replace('scripts/storage_offload_20261018_v1.py',MARKER)
 ns=dict(vars(base));ns['ARCHIVES']={name:{'retained_root':name} for name in candidates}
 exec(compile(source,__file__+'::process_guard','exec'),ns)
 return ns['process_guard']()

def command(args,name,cap=14400):
 require(not (OUT/(name+'_receipt.json')).exists(),'Preserve previous operation')
 start=time.monotonic()
 with (OUT/(name+'_stdout.txt')).open('x',encoding='utf-8') as stdout,(OUT/(name+'_stderr.txt')).open('x',encoding='utf-8') as stderr:
  child=subprocess.Popen([str(base.EXE),'--config',str(base.CONFIG),*args],stdout=stdout,stderr=stderr,creationflags=subprocess.CREATE_NO_WINDOW)
  print(json.dumps({'operation':name,'pid':child.pid,'cap_seconds':cap}),flush=True)
  timeout=False
  try:code=child.wait(timeout=cap)
  except subprocess.TimeoutExpired:child.kill();code=child.wait();timeout=True
 receipt={'complete':code==0 and not timeout,'exit_code':code,'timeout':timeout,'seconds':time.monotonic()-start,
  'stdout':base.pin(OUT/(name+'_stdout.txt')),'stderr':base.pin(OUT/(name+'_stderr.txt'))}
 save(name+'_receipt.json',receipt);require(receipt['complete'],'Rclone failed; preserved '+name)
 return receipt

def prepare():
 require(not (OUT/'plan_v1.json').exists(),'Plan already exists')
 conn=base.connection();remote=read('remote_inventory_v1.json');oldplan=base.read(ROOT/'outputs/storage_offload_20261019_v1/plan.json')
 oldby={Path(v['path']).name:v for v in oldplan['files']}
 archives=[]
 for p in sorted((ROOT/'outputs').iterdir()):
  if not (p.is_file() and p.name.startswith(('cctv-dgp-','cctv-codeformer-')) and p.name.endswith('.tar.gz') and
    any(k in p.name for k in ('-execution.','-results.','-failure.','-results-recovery-')) and 'spatial-continuation' not in p.name):continue
  archives.append(base.pin(p))
 save('guard_prepare_v1.json',guard([v['path'] for v in archives]))
 members=[];runs=[]
 for slug,run in OLD_RUNS.items():
  root=ROOT/('outputs/codeformer_'+run+'_return');name='cctv-codeformer-'+slug+'-results.tar.gz'
  matches=[x for x in remote if x['Name']==name];require(len(matches)==1,'Unique old remote archive required '+name)
  server=matches[0];expected=oldby[name];hashes={k.lower():v for k,v in server.get('Hashes',{}).items()}
  require(server['Size']==expected['bytes'] and hashes.get('sha256')==expected['sha256'] and server.get('ID'),'Prior archive hash/ID differs')
  group=[]
  for folder,dirs,files in os.walk(root,followlinks=False):
   for fn in files:
    p=Path(folder)/fn;rel=p.relative_to(ROOT).as_posix();member=p.relative_to(root).as_posix();parts=member.lower().split('/')
    # Archive fitted run state and bulk output RAW only; keep all input data,
    # code, JSON/provenance/gates/logs, PNGs, metrics and comparison exports.
    if not member.startswith('outputs/'+run+'/') or rel==R2_RESUME:continue
    if any(x in ('cache','caches','data','weights','targets','features','inputs','training_cache') for x in parts):continue
    if not (p.suffix=='.npy' or (p.suffix=='.pt' and 'checkpoint' in p.name.lower())):continue
    row=base.pin(p);row['member']=member;row['archive']=server['Path'];members.append(row);group.append(row['path'])
  require(group,'No eligible archived outputs '+run)
  runs.append({'run':run,'retained_root':root.relative_to(ROOT).as_posix(),'remote':server,'source_expected':expected,'candidate_paths':group})
 protected=[]
 selected=base.read(ROOT/'outputs/codeformer_u96_preference_v1/selected_model_v1.json')
 protected_names=set(selected['model_parts'])|{selected['source_resumable_checkpoint']['path'],R2_RESUME}
 for d in ['codeformer_spatial_continuation_v1_models','codeformer_spatial_continuation_return_review_v1','codeformer_role_preserving_training_v1_r2_model']:
  protected_names.update(p.relative_to(ROOT).as_posix() for p in (ROOT/'outputs'/d).rglob('*.json'))
 protected_names.update(['checkpoints/codeformer_inpainting.pth','checkpoints/dgp_zamboanga_final.pth',
  'outputs/codeformer_restoration_pretrained_v1/codeformer.pth','outputs/codeformer_spatial_continuation_v1_models/codeformer_spatial_s_u96.pth',
  'PROJECT_HANDOFF.md','SYSTEM_WORKFLOW_AND_GOAL.md','PRACTICAL_OUTPUT_SCOPE.md','CCTV_CODEFORMER_U96_FOCUS_V1_PLAN.md'])
 for name in sorted(protected_names):protected.append(protected_hash(ROOT/name))
 allpaths=[x['path'] for x in archives+members];require(len(allpaths)==len(set(allpaths)) and not set(allpaths)&protected_names,'Disjoint exact local paths required')
 p={'complete':True,'version':'u96-inactive-restoration-local-only-v1','destination':DEST,'archives':archives,'members':members,'old_runs':runs,
  'protected':protected,'protected_runtime_and_resume':selected,'connection':conn,'source':base.pin(Path(__file__)),
  'archive_bytes':sum(x['bytes'] for x in archives),'member_bytes':sum(x['bytes'] for x in members),
  'preserve':['datasets/caches','source code','comparison exports and PNGs/metrics','U96 base/adapter/resume','completion and removal assets'],
  'VM_training_inference_calls':0,'free_bytes_before':shutil.disk_usage(ROOT).free}
 save('plan_v1.json',p)
 (OUT/'files_v1.txt').write_text(''.join(x['path']+'\n' for x in archives),encoding='utf-8',newline='\n')
 (OUT/'sha256_v1.txt').write_text(''.join(x['sha256']+'  '+x['path']+'\n' for x in archives),encoding='ascii',newline='\n')
 print(json.dumps({'plan_sha256':base.sha(OUT/'plan_v1.json'),'archives':len(archives),'archive_GiB':p['archive_bytes']/1024**3,'members':len(members),'member_GiB':p['member_bytes']/1024**3,'protected':len(protected)}),flush=True)

def plan(expected):
 require(base.sha(OUT/'plan_v1.json')==expected,'Exact plan SHA required')
 p=read('plan_v1.json');require(p['destination']==DEST and base.pin(Path(__file__))==p['source'],'Frozen destination/source required');return p
def verify_protected(p):
 for row in p['protected']:require(protected_hash(ROOT/row['path'])==row,'Protected file changed '+row['path'])
def verify_local(row):
 actual=base.pin(ROOT/row['path']);require(all(actual.get(k)==v for k,v in row.items() if k not in ('member','archive')),'Local identity/content differs '+row['path'])

def snapshot(label):
 start=time.monotonic();stack=[ROOT];count=0;links=[];models=[]
 target=OUT/(label+'_namespace.jsonl.gz')
 with target.open('xb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as z:
  while stack:
   require(time.monotonic()-start<900 and count<2000000,'Finite snapshot cap')
   d=stack.pop()
   with os.scandir(d) as entries:
    for e in entries:
     path=Path(e.path)
     if path==OUT:continue
     s=e.stat(follow_symlinks=False);rel=path.relative_to(ROOT).as_posix()
     if stat.S_ISLNK(s.st_mode) or getattr(s,'st_file_attributes',0)&1024:links.append(rel);continue
     if stat.S_ISDIR(s.st_mode):stack.append(path)
     elif stat.S_ISREG(s.st_mode):
      row={'path':rel,'bytes':s.st_size,'mtime_ns':s.st_mtime_ns};z.write((json.dumps(row,separators=(',',':'))+'\n').encode());count+=1
      if path.suffix.lower() in ('.pt','.pth','.ckpt','.safetensors'):models.append(protected_hash(path))
 result={'complete':True,'files':count,'models':models,'links_not_followed':links,'namespace':base.pin(target),'seconds':time.monotonic()-start}
 save(label+'_snapshot_v1.json',result);print(json.dumps({'snapshot':label,'files':count,'models':len(models),'seconds':result['seconds']}),flush=True);return result

def remote_check(p,name):
 command(['lsjson',DEST,'--recursive','--files-only','--hash','--max-depth','2','--retries','1','--timeout','1m'],name,600)
 rows=json.loads((OUT/(name+'_stdout.txt')).read_text(encoding='utf-8-sig'));by={x['Path']:x for x in rows}
 require(len(rows)==len(by)==len(p['archives']),'Remote namespace differs')
 for row in p['archives']:
  obj=by.get(row['path'],{});h={k.lower():v for k,v in obj.get('Hashes',{}).items()}
  require(obj.get('Size')==row['bytes'] and h.get('sha256')==row['sha256'] and obj.get('ID'),'Remote SHA256/size/ID mismatch')
 require(len({x['ID'] for x in rows})==len(rows),'Duplicate remote IDs');return rows

class HashReader(io.RawIOBase):
 def __init__(self,stream,expected):self.stream=stream;self.expected=expected;self.h=hashlib.sha256();self.count=0;self.start=time.monotonic()
 def readable(self):return True
 def readinto(self,b):
  require(time.monotonic()-self.start<7200,'Finite archive download cap')
  n=self.stream.readinto(b)
  if n:self.h.update(memoryview(b)[:n]);self.count+=n;require(self.count<=self.expected,'Remote archive exceeded exact byte count')
  return n

def verify_members(p):
 for i,run in enumerate(p['old_runs']):
  name='old_run_'+str(i+1).zfill(2);receipt=OUT/(name+'_verified.json')
  if receipt.exists():
   old=read(receipt.name);require(old['complete'] and old['plan_sha256']==base.sha(OUT/'plan_v1.json'),'Prior group verification differs');continue
  desired={x['member']:x for x in p['members'] if x['path'] in run['candidate_paths']}
  remote=OLD_DEST+'/'+run['remote']['Path'];start=time.monotonic()
  with (OUT/(name+'_stderr.txt')).open('x',encoding='utf-8') as err:
   child=subprocess.Popen([str(base.EXE),'--config',str(base.CONFIG),'cat',remote,'--retries','1','--low-level-retries','2','--contimeout','20s','--timeout','1m'],stdout=subprocess.PIPE,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
   raw=HashReader(child.stdout,run['remote']['Size']);reader=io.BufferedReader(raw,buffer_size=1024**2);matched=[];seen=set()
   print(json.dumps({'stream_verify_old_run':run['run'],'candidate_members':len(desired),'remote_bytes':run['remote']['Size'],'pid':child.pid}),flush=True)
   try:
    with tarfile.open(fileobj=reader,mode='r|gz') as tar:
     for member in tar:
      key=member.name.removeprefix('./')
      require(key not in seen,'Duplicate tar member');seen.add(key)
      if key not in desired:continue
      expected=desired[key];require(member.isfile() and member.size==expected['bytes'],'Exact regular tar member required')
      h=hashlib.sha256();f=tar.extractfile(member);total=0
      for block in iter(lambda:f.read(4*1024**2),b''):h.update(block);total+=len(block)
      require(total==expected['bytes'] and h.hexdigest()==expected['sha256'],'Remote member content mismatch '+key)
      matched.append({'path':expected['path'],'member':key,'bytes':total,'sha256':h.hexdigest()})
    for block in iter(lambda:reader.read(4*1024**2),b''):pass
    code=child.wait(timeout=90)
    require(code==0 and raw.count==run['remote']['Size'] and raw.h.hexdigest()==run['source_expected']['sha256'],'Full downloaded archive digest/exit mismatch')
    require(set(x['member'] for x in matched)==set(desired),'Missing exact archived members')
   except BaseException:
    child.kill();child.wait();raise
  result={'complete':True,'plan_sha256':base.sha(OUT/'plan_v1.json'),'run':run['run'],'remote':run['remote'],
   'downloaded_archive_bytes':raw.count,'downloaded_archive_sha256':raw.h.hexdigest(),'members':matched,'seconds':time.monotonic()-start}
  save(receipt.name,result);print(json.dumps({'old_run_verified':run['run'],'members':len(matched),'seconds':result['seconds']}),flush=True)

def remove_exact(row):
 path=(ROOT/base.relative(row['path'])).absolute()
 require(path.is_relative_to(ROOT/'outputs') and path.resolve()==path,'Exact outputs-only removal')
 expected={k:row[k] for k in ('bytes','mtime_ns','inode','dev','nlink','mode','attributes')}
 require(identity(path)==expected and row['nlink']==1,'File identity differs')
 k=ctypes.WinDLL('kernel32',use_last_error=True);create=k.CreateFileW
 create.argtypes=[wintypes.LPCWSTR,wintypes.DWORD,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,wintypes.DWORD,wintypes.HANDLE];create.restype=wintypes.HANDLE
 handle=create(str(path),0x80000000|0x00010000,0,None,3,0x08000000,None)
 require(handle!=ctypes.c_void_p(-1).value,'Exclusive candidate handle unavailable')
 fd=msvcrt.open_osfhandle(handle,os.O_RDONLY|os.O_BINARY)
 with os.fdopen(fd,'rb') as stream:
  h=hashlib.sha256();start=time.monotonic()
  for b in iter(lambda:stream.read(4*1024**2),b''):require(time.monotonic()-start<300,'Finite hash cap');h.update(b)
  require(h.hexdigest()==row['sha256'] and identity(path)==expected,'Exclusive content/identity differs')
  class Disposition(ctypes.Structure):_fields_=[('DeleteFile',ctypes.c_ubyte)]
  op=k.SetFileInformationByHandle;op.argtypes=[wintypes.HANDLE,ctypes.c_int,wintypes.LPVOID,wintypes.DWORD];op.restype=wintypes.BOOL
  flag=Disposition(1);require(op(handle,4,ctypes.byref(flag),ctypes.sizeof(flag)),'Exact native removal failed')
 require(not path.exists(),'Candidate still present')

def apply(p):
 require(read('copy_receipt.json')['complete'] and read('stream_verify_receipt.json')['complete'],'Complete upload/download required')
 require(sorted((OUT/'combined_v1.txt').read_text().splitlines())==sorted('= '+x['path'] for x in p['archives']),'Streamed archive verification set differs')
 proven=[]
 for i,run in enumerate(p['old_runs']):
  v=read('old_run_'+str(i+1).zfill(2)+'_verified.json');require(v['complete'] and v['remote']==run['remote'],'Old member proof differs');proven+=v['members']
 require({x['path']:x['sha256'] for x in proven}=={x['path']:x['sha256'] for x in p['members']},'Member proof union differs')
 remote_check(p,'remote_preapply');verify_protected(p)
 allrows=p['archives']+p['members'];save('guard_preapply_v1.json',guard([x['path'] for x in allrows]))
 for row in allrows:verify_local(row)
 require(read('before_snapshot_v1.json')['complete'],'Full before inventory missing')
 require(not (OUT/'deletion_journal_v1.jsonl').exists(),'Journal exists: audit partial state; do not replay')
 free=shutil.disk_usage(ROOT).free
 with (OUT/'deletion_journal_v1.jsonl').open('x',encoding='utf-8',newline='\n') as journal:
  def event(row):journal.write(json.dumps(row)+'\n');journal.flush();os.fsync(journal.fileno())
  for i,row in enumerate(allrows):
   event({'intent':row});remove_exact(row);event({'removed':row['path'],'bytes':row['bytes']})
   if i%500==0:print(json.dumps({'removed':i+1,'of':len(allrows)}),flush=True)
 save('apply_v1.json',{'complete':True,'files_removed':len(allrows),'logical_bytes_removed':sum(x['bytes'] for x in allrows),
  'archive_files':len(p['archives']),'raw_checkpoint_members':len(p['members']),'free_before':free,'free_after':shutil.disk_usage(ROOT).free,
  'exact_hash_exclusive_handle_deletion':True,'VM_training_inference_calls':0})

def audit(p):
 after=snapshot('after');before=read('before_snapshot_v1.json')
 def ns(label):
  with gzip.open(OUT/(label+'_namespace.jsonl.gz'),'rt',encoding='utf-8') as f:return {x['path']:x for x in map(json.loads,f)}
 a,b=ns('after'),ns('before');expected={x['path'] for x in p['archives']+p['members']}
 missing=set(b)-set(a);changed=[n for n in set(a)&set(b) if a[n]!=b[n]]
 require(missing==expected and not changed,'Unexpected namespace changes: '+json.dumps({'missing_delta':sorted(missing^expected)[:5],'changed':changed[:5]}))
 bm={x['path']:x for x in before['models'] if x['path'] not in expected};am={x['path']:x for x in after['models']}
 require(am==bm,'Retained model content/metadata changed');verify_protected(p)
 save('closure_v1.json',{'complete':True,'missing_exact_planned':len(missing),'unexpected_missing':[],'retained_metadata_changes':[],
  'retained_model_hashes':len(am),'added':sorted(set(a)-set(b)),'retained_namespace_files':len(a),'apply':base.pin(OUT/'apply_v1.json'),
  'all_protected_files_verified':True,'VM_training_inference_calls':0,'overall_goal_complete':False})
 print(json.dumps(read('apply_v1.json')),flush=True)

def main():
 parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['prepare','snapshot','copy','verify','members','apply','audit']);parser.add_argument('--plan-sha',default='');args=parser.parse_args()
 if args.mode=='prepare':return prepare()
 p=plan(args.plan_sha)
 if args.mode=='snapshot':return snapshot('before')
 save('guard_'+args.mode+'_v1.json',guard([x['path'] for x in p['archives']+p['members']]))
 base.connection()
 if args.mode=='copy':
  for row in p['archives']:verify_local(row)
  verify_protected(p)
  command(['copy',str(ROOT),DEST,'--files-from-raw',str(OUT/'files_v1.txt'),'--immutable','--checksum','--transfers','3','--checkers','4','--drive-chunk-size','32M','--tpslimit','6','--retries','2','--low-level-retries','3','--contimeout','20s','--timeout','2m','--max-duration','4h','--cutoff-mode','hard','--stats','30s','--log-level','INFO','--log-file',str(OUT/'copy_v1.log')],'copy')
 elif args.mode=='verify':
  require(read('copy_receipt.json')['complete'],'Copy incomplete')
  command(['check',str(OUT/'sha256_v1.txt'),DEST,'--checkfile','SHA256','--download','--one-way','--combined',str(OUT/'combined_v1.txt'),'--checkers','3','--tpslimit','6','--retries','1','--low-level-retries','3','--contimeout','20s','--timeout','2m','--stats','30s'],'stream_verify')
  require(sorted((OUT/'combined_v1.txt').read_text().splitlines())==sorted('= '+x['path'] for x in p['archives']),'Exact full downloaded SHA set required');remote_check(p,'remote_verified')
 elif args.mode=='members':verify_members(p)
 elif args.mode=='apply':apply(p)
 elif args.mode=='audit':audit(p)

if __name__=='__main__':main()
