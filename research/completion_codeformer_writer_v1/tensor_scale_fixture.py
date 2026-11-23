"""Bounded32MiB/996tensor serialization fixture at actual620-output namespace.
No network, neural modules, gradients, optimizer objects or VM operations.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,tempfile,time,zipfile
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.completion_codeformer_writer_v1 import optimized_budget as budget_source
RET=ROOT/'outputs/completion_codeformer_coverage_v2_return/cctv_completion_codeformer_coverage_v2_results'
TMP=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs');OUT=ROOT/'research/completion_codeformer_writer_v1'
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()
import torch
torch.set_num_threads(1)
started=time.monotonic()
rows=[{'path':r['path'][8:],'bytes':r['bytes']} for r in json.loads((RET/'export_manifest.json').read_text())['files'] if r['path'].startswith('outputs/')]
assert len(rows)==620
# Fixed independentstorage tensors, not model or optimizer state.
values={}
for i in range(332):
 values[f'weightlike_{i:03d}']=torch.arange(8192,dtype=torch.float32)+i
 values[f'firstmomentlike_{i:03d}']=torch.full((8192,),i/1000,dtype=torch.float32)
 values[f'secondmomentlike_{i:03d}']=torch.full((8192,),i/10000,dtype=torch.float32)
assert len(values)==996 and sum(t.numel()*t.element_size() for t in values.values())==32636928
assert all(not t.requires_grad for t in values.values())
with tempfile.TemporaryDirectory(prefix='cwscale-',dir=TMP) as text:
 family=Path(text);out=family/'out';out.mkdir();b=budget_source.OutputBudget(out,0,clock=lambda:0,disk_free=lambda:100*1024**3)
 for row in rows:
  if row['path']=='terminal_training_state.pth.partial':continue
  path=out/row['path'];category=b.classify(path);entry=budget_source.Entry(path,category,row['bytes']);b.entries[path]=entry;b.sizes[path]=row['bytes'];b.totals[category]+=row['bytes'];b.extent_total+=row['bytes']
 before=b.check_counts.copy();actual=out/'terminal_training_state.pth.partial';save_started=time.perf_counter()
 with b.open(actual,terminal=True) as stream:torch.save({'fixture_state':values},stream);stream.flush()
 save_seconds=time.perf_counter()-save_started
 assert save_seconds<30 and actual.stat().st_size==b.sizes[actual.resolve()]
 assert len(b.entries)==620 and b.assert_ledger_consistent()
 native=family/'native.pth';native_started=time.perf_counter()
 with native.open('xb') as stream:torch.save({'fixture_state':values},stream)
 native_seconds=time.perf_counter()-native_started
 assert sha(actual)==sha(native)
 crc_started=time.perf_counter()
 with zipfile.ZipFile(actual) as archive:
  entries=archive.infolist();assert archive.testzip() is None
  assert len([x for x in entries if '/data/' in x.filename])==996
 crc_seconds=time.perf_counter()-crc_started
 load_started=time.perf_counter();readback=torch.load(actual,map_location='cpu',weights_only=True);load_seconds=time.perf_counter()-load_started
 assert set(readback)=={'fixture_state'} and set(readback['fixture_state'])==set(values)
 assert all(torch.equal(readback['fixture_state'][name],value) for name,value in values.items())
 receipt={'format':'completion-cached-writer-32MiB-996tensor-actual-namespace-v1','complete':True,'recorded_UTC':datetime.now(timezone.utc).isoformat(),'prototype_sha256':sha(Path(budget_source.__file__)),'original_return_namespace_files':624,'tracked_output_entries_afteropen':len(b.entries),'fixture_tensors':996,'fixture_tensor_semantics':'332weightlike+332firstmomentlike+332secondmomentlike independentCPUtensors; not realmodeloroptimizer','raw_tensor_bytes':32636928,'serialized_bytes':actual.stat().st_size,'identical_native_SHA256':sha(actual),'all996tensors_safe_CPU_readback_equal':True,'ZIP_entries':len(entries),'allZIP_CRC_pass':True,'capped_save_seconds':save_seconds,'native_save_seconds':native_seconds,'safe_load_seconds':load_seconds,'CRC_seconds':crc_seconds,'capped_save_stop_seconds':30,'write_callbacks':b.check_counts['write_checks']-before['write_checks'],'canonical_path_calls':b.check_counts['canonical_paths']-before['canonical_paths'],'full_metadata_counts':b.check_counts,'ledger_total_andcategorysums_exact':True,'old_largecallback_benchmark_intentionally_not_run':True,'old_comparison':'See namespace_scale_benchmark.json:8actualcallbacks6.6268s vsnew.00779s; noextrapolatedVMETAclaimed','fixture_clock_free_explicit':True,'no_network_or_VM':True,'no_model_instantiation_or_forward':True,'no_gradient_or_optimizer_operations':True,'no_actual_partial_loaded_or_rebuilt':True,'no_training_packet_modified':True,'total_seconds':time.monotonic()-started}
 path=OUT/'tensor_32MiB_scale_fixture.json'
 with path.open('x',encoding='utf-8',newline='\n') as stream:json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
 print(json.dumps({'receipt':str(path),'sha256':sha(path),**receipt},indent=2))