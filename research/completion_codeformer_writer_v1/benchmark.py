"""Namespace-scaled pure accounting and bounded write benchmarks; no model calls."""
from pathlib import Path
from datetime import datetime,timezone
from unittest.mock import patch
import hashlib,json,os,sys,tempfile,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.completion_codeformer_coverage_v2 import contracts as old
from research.completion_codeformer_writer_v1 import optimized_budget as new
RET=ROOT/'outputs/completion_codeformer_coverage_v2_return';RUN=RET/'cctv_completion_codeformer_coverage_v2_results';OUT=ROOT/'research/completion_codeformer_writer_v1'
TMP=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs')
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def seed(budget,rows):
 for row in rows:
  path=budget.out/row['path'];category=budget.classify(path) if isinstance(budget,new.OutputBudget) else budget.category(path)
  budget.sizes[path]=row['bytes']
  if isinstance(budget,new.OutputBudget):budget.entries[path]=new.Entry(path,category,row['bytes']);budget.totals[category]+=row['bytes'];budget.extent_total+=row['bytes']
manifest=json.loads((RUN/'export_manifest.json').read_text());all_files=list(RUN.rglob('*'));all_files=[p for p in all_files if p.is_file()]
rows=[{'path':r['path'][len('outputs/'):],'bytes':r['bytes']} for r in manifest['files'] if r['path'].startswith('outputs/')]
assert len(all_files)==624 and len(manifest['files'])==623
# Read-only accounting checks use the actual return output paths and extents.
budgets=[]
for label,cls in [('old',old.OutputBudget),('new',new.OutputBudget)]:
 b=cls(RUN/'outputs',0,clock=lambda:0,disk_free=lambda:100*1024**3);seed(b,rows);b.metadata_external=[RUN/'trainer.log',RUN/'protocol.json']
 category_calls=[0];resolve_calls=[0];original_category=b.category;original_resolve=Path.resolve
 def category(path):category_calls[0]+=1;return original_category(path)
 def resolve(path,*a,**kw):resolve_calls[0]+=1;return original_resolve(path,*a,**kw)
 b.category=category
 target=RUN/'outputs/terminal_training_state.pth.partial';extent=b.sizes[target]+64*1024
 started=time.perf_counter()
 with patch.object(Path,'resolve',resolve):
  for _ in range(3):b.before_write(target,extent,terminal=True)
 elapsed=time.perf_counter()-started
 budgets.append({'label':label,'accounting_checks':3,'actual_output_entries':len(rows),'complete_return_regular_files':len(all_files),'category_calls':category_calls[0],'Path_resolve_calls':resolve_calls[0],'seconds':elapsed,'seconds_per_writecheck':elapsed/3,'writes_performed':0,'actual_return_mutated':False})
# Real bounded8MiB serialization-equivalent chunk writes; seeded namespace isvirtual.
write_results=[]
with tempfile.TemporaryDirectory(prefix='cwbench-',dir=TMP) as text:
 family=Path(text)
 block=bytes((i%251 for i in range(1024*1024)));total=8*1024**2
 for label,cls in [('native',None),('old',old.OutputBudget),('new',new.OutputBudget)]:
  out=family/label;out.mkdir();b=None
  if cls:
   b=cls(out,0,clock=lambda:0,disk_free=lambda:100*1024**3);virtual=[r for r in rows if r['path']!='terminal_training_state.pth.partial'];seed(b,virtual)
  target=out/'terminal_training_state.pth.partial';started=time.perf_counter()
  stream=target.open('xb') if cls is None else b.open(target,terminal=True)
  with stream:
   for _ in range(8):stream.write(block)
   stream.flush()
  elapsed=time.perf_counter()-started;assert target.stat().st_size==total
  if b:assert b.sizes[target.resolve()]==total
  if isinstance(b,new.OutputBudget):b.assert_ledger_consistent()
  write_results.append({'label':label,'bytes_written':total,'callbacks':8,'virtual_namespace_entries_before_open':len(rows)-1,'seconds':elapsed,'MiB_per_second':8/elapsed,'output_sha256':sha(target),'prototype_check_counts':b.check_counts if isinstance(b,new.OutputBudget) else None})
 assert len({r['output_sha256'] for r in write_results})==1
report={'format':'completion-writer-namespace-scale-benchmark-v1','complete':True,'recorded_UTC':datetime.now(timezone.utc).isoformat(),'immutable_old_source':{'path':str(Path(old.__file__)),'sha256':sha(Path(old.__file__))},'prototype_source':{'path':str(Path(new.__file__)),'sha256':sha(Path(new.__file__))},'actual_return_namespace':{'regular_files':len(all_files),'manifest_payloads':len(manifest['files']),'tracked_output_entries':len(rows)},'pure_actual_namespace_checks':budgets,'bounded8MiB_native_old_new_writes':write_results,'operational_inference':'Old accounting repeatedly resolves/categories everytracked output on eachserializedchunk; newcachedtotals eliminate namespace rescans in checks. This localmechanism can contribute to failedVMsave latency; exactsoleVMcause notproven.','limits_unchanged':True,'benchmark_clock_free_are_explicit_fixtures':True,'no_training_packet_change':True,'local_actions':{'model_calls':0,'gradient_calls':0,'optimizer_operations':0,'VM_calls':0,'read_actual_incomplete_state_bytes':False,'old_return_mutated':False}}
p=OUT/'namespace_scale_benchmark_v2.json'
with p.open('x',encoding='utf-8',newline='\n') as stream:json.dump(report,stream,indent=2,allow_nan=False);stream.write('\n')
print(json.dumps({'receipt':str(p),'sha256':sha(p),'accounting':budgets,'writes':write_results},indent=2))