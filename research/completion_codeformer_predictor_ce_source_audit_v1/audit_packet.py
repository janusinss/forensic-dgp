"""Independent full source/data closure for the prospective manual predictor CE pilot."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import ast,hashlib,importlib.abc,importlib.util,json,sys,time,zipfile
ROOT=Path(__file__).resolve().parents[2]
FOLDER=Path(__file__).resolve().parent
STAGE=ROOT/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/cctv_completion_codeformer_predictor_ce_v1_vm'
EXPECTED='15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b'
checks=[]
def need(ok,label):
 if not ok:raise ValueError(label)
 checks.append(label)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('torch','numpy','PIL'):raise ValueError('Scientific import forbidden during pure source load '+fullname)
def main():
 start=time.monotonic();need(sha(STAGE/'protocol.json')==EXPECTED,'Exact fresh staged protocol');p=read(STAGE/'protocol.json');files={r['path']:r for r in p['files']};need(len(files)==len(p['files'])==670,'Unique670payloads');need({v.relative_to(STAGE).as_posix() for v in STAGE.rglob('*') if v.is_file()}==set(files)|{'protocol.json'},'Closed exact671regular namespace')
 total=(STAGE/'protocol.json').stat().st_size
 for name,row in files.items():
  n=PurePosixPath(name);q=STAGE/name;need(not n.is_absolute() and '..' not in n.parts and ':' not in name and '\\' not in name and str(n)==name,'Safe packet path '+name);need(q.is_file() and not q.is_symlink() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Exact packet asset/source '+name);total+=row['bytes']
 need(total==79488873 and total<256*1024**2,'Compact execution source data under256MiB')
 blocker=Block();sys.meta_path.insert(0,blocker)
 try:
  q=ROOT/'scripts/prepare_completion_codeformer_predictor_ce_v1.py';spec=importlib.util.spec_from_file_location('independent_predictor_packet_builder',q);builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
 finally:sys.meta_path.remove(blocker)
 for name,q in builder.SOURCE_MAP.items():
  need(name in files and (STAGE/name).read_bytes()==q.read_bytes(),'Final canonical/staged source '+name)
  if q.suffix=='.py':
   t=ast.parse(q.read_text(),feature_version=(3,10));need(True,'Python3.10 syntax '+name)
   if name!='face_workflow.py':need(not any(isinstance(n,ast.Attribute) and n.attr=='file_digest' for n in ast.walk(t)),'Python3.10 executed-source chunkedhash '+name)
 need(all('face_workflow' not in (STAGE/name).read_text() for name in builder.SOURCE_MAP if name.endswith('.py') and name not in ('face_workflow.py','scripts/prepare_completion_codeformer_predictor_ce_v1.py')),'Legacy face_workflow.py is pinned unused evidence, never imported by runtime')
 worker=STAGE/'scripts/cctv_completion_codeformer_predictor_ce_v1_vm.py';t=ast.parse(worker.read_text());w=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='worker');sem=read(FOLDER/'independent_source_semantics.json');need(hashlib.sha256(ast.dump(w,include_attributes=False).encode()).hexdigest()==sem['worker_AST_sha256'],'Entire runtime worker matches independent2036 semantic checks')
 rebind=read(FOLDER/'independent_design_binding_recheck.json');need(sha(worker)==rebind['bound_worker_SHA256'] and rebind['whole_worker_AST_unchanged'],'Bound formal design-only source replacement exact')
 for r in sem['source_files']:
  q=Path(r['path'])
  if q.name==worker.name:continue
  need(q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],'Tested helper unchanged '+q.name)
 for key,name in p['evidence'].items():need(name in files and sha(STAGE/name)==p['evidence_sha256'][key],'Frozen evidence '+key)
 need(sha(STAGE/p['evidence']['root_design'])=='077ed97ec5dc4ca75f6b6bdc3caf8def09344113c3b7ce927d13b7b68a4f2b78' and sha(STAGE/p['evidence']['formal_design_admission'])=='691a5ec8151e0102e62f0c54c73770a25277690a7ac848586d33fb47f6ec9fbf' and sha(STAGE/p['evidence']['root_pre_release_clarifications'])=='0495ded17bb5d5d90eb59f50608f72c7cf1b82b6f83a7aaef5a76a9cdaae0664','Formal independent/root clarification chain exact')
 v4root=ROOT/'outputs/completion_codeformer_delivered_objective_v4_packet/draft_001/cctv_completion_codeformer_delivered_objective_v4_vm';v4=read(v4root/'protocol.json');v4files={r['path']:r for r in v4['files']};subset=read(STAGE/p['balanced_subset']);need(p['fit_cases']==subset['cases_in_reference_role_order'] and p['schedule']==subset['schedule'],'Exact independent128 selected cases and balanced32x4 schedule')
 need(p['references']==v4['references'] and p['teacher_targets']==v4['teacher_targets'] and p['monitor_cases']==v4['monitor_cases'] and p['challenge_cases']==v4['challenge_cases'] and p['development_cases']==v4['development_cases'],'Inherited40reference/80target/48paired/32public metadata exact')
 need(len(p['parity_cases'])==4 and all(row in v4['fit_cases'] for row in p['parity_cases']),'Four original fixed fresh parity cases')
 for row in p['fit_cases']+p['monitor_cases']+p['challenge_cases']+p['parity_cases']:
  for field in ('input256','clean256','clean512','mask'):need(files[row[field]]['sha256']==v4files[row[field]]['sha256'],'Exact inherited case bytes '+row['id']+'/'+field)
 for row in p['development_cases']:
  for field in ('input256','mask'):need(files[row[field]]['sha256']==v4files[row[field]]['sha256'],'Exact unpaired public bytes '+row['id']+'/'+field)
 need(p['terms']==v4['terms'] and p['data_limits']==v4['data_limits'] and p['checkpoint']==v4['checkpoint'] and p['parent']==v4['parent'],'Legal/data limits and original/R1 dependency pins unchanged')
 for kind,rows in p['historical_control']['initial_rows'].items():
  need(rows==v4['historical_control']['initial_rows'][kind],'Historical R148baseline exactrows '+kind)
  for row in rows:
   for field in ('result256','PNG'):need(files[row[field]]['sha256']==v4files[row[field]]['sha256'],'Historical raw/PNG immutable '+row['id']+'/'+field)
 # Cached target admission is data-only; neither Torch nor model deserialization.
 import numpy as np
 from PIL import Image
 index=read(STAGE/p['evidence']['cached_label_index']);need(index['complete'] and len(index['labels'])==80,'Exact original cache index80');originals={r['id']:r for r in index['labels']};fit_refs={r['reference_id'] for r in p['fit_cases']};monitor_refs={r['reference_id'] for r in p['monitor_cases']};need(len(fit_refs)==32 and len(monitor_refs)==8 and not fit_refs&monitor_refs,'32FIT and8monitor references disjoint')
 fitkeys={r['teacher_key'] for r in p['fit_cases']};need(len(fitkeys)==64,'All64FIT code sets used without16monitor labels')
 for label in p['cached_labels']:
  key=label['teacher_key'];old=originals[key];q=STAGE/label['path'];need(label['bytes']==old['bytes'] and label['sha256']==old['sha256']==sha(q) and label['clean512']==old['clean512'] and label['clean512_sha256']==old['clean512_sha256'] and label['teacher_original_logical_sha256']==old['teacher_original_logical_sha256'],'Original byteexact cached labels '+key)
  need(label['monitor_labels_never_optimized']==(key not in fitkeys),'FIT/monitor label role '+key)
  with zipfile.ZipFile(q) as z:need(len(z.infolist())==2 and set(z.namelist())=={'prequant_latent.npy','nearest_codes.npy'} and z.testzip() is None,'Two cached arrays and all CRC '+key)
  with np.load(q,allow_pickle=False,max_header_size=512) as z:
   c=z['nearest_codes'];a=z['prequant_latent'];need(c.dtype==np.int64 and c.shape==(1,256) and c.min()>=0 and c.max()<512 and a.dtype==np.float32 and a.shape==(1,256,16,16) and np.isfinite(a).all(),'Numeric cached index/unusedlatent schema '+key)
  target=next(r for r in p['teacher_targets'] if r['id']==key);need(target['clean512']==label['clean512'] and sha(STAGE/target['clean512'])==label['clean512_sha256'],'Cached label exact styled target join '+key)
 for row in p['teacher_targets']:
  if row['style']=='GRAY':
   for field in ('clean256','clean512'):
    with Image.open(STAGE/row[field]) as im:a=np.array(im);need(im.mode=='RGB' and np.array_equal(a[...,0],a[...,1]) and np.array_equal(a[...,1],a[...,2]),'GRAY target exactequal channels '+row['id']+'/'+field)
 contracts_path=ROOT/'research/completion_codeformer_predictor_ce_v1_development/contracts.py';spec=importlib.util.spec_from_file_location('independent_predictor_contract',contracts_path);c=importlib.util.module_from_spec(spec);spec.loader.exec_module(c);need(p['contract']==c.CONTRACT and sum(c.TYPED_CAPS.values())==384 and c.CONTRACT['saved_NPZ_max']==388 and c.CONTRACT['empty_bypasses_planned']==168,'Frozen runtime exact384typed/388generated/168empty contract')
 export=p['export'];need(export['NPZ_generated_max']==388 and export['NPZ_cached_teacher_copies_max']==80 and export['NPZ_total_max']==468 and export['terminal_bytes']==1024**3 and export['maximum_regular_members_including_manifest']==1000,'Export reserves complete scalar/state/cache evidence')
 need(not any(name.endswith(('.pth','.pt')) for name in files) and not any('ORACLE_unavailable' in name for name in files),'No bundled weights or oracle runtime intervention payload')
 need(p['manual_execution_only'] and not p['automatic_resume_or_promotion'] and not p['model_qualified'] and p['source_subset']['FINAL_data_used'] is False and p['release_required_before_archive'],'Manual-only unqualified adaptive scope')
 cli=read(FOLDER/'independent_pure_transfer_CLI_fixture_v1.json');need(cli['complete'] and cli['returncode']==0 and cli['protocol_sha256']==EXPECTED and cli['staged_payload_files']==670 and cli['actual_unmodified_verify_transfer_CLI_branch'] and not cli['actual_VM_preflight'],'Actual guarded transfer-only CLI terminal PASS')
 app=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json');protected=app['source_files']+[app['candidate_manifest'],app['checkpoint']]
 for r in protected:
  q=Path(r['path']);need(q.is_file() and q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],'Current R1app/model preserved '+q.name)
 need(sha(STAGE/'protocol.json')==EXPECTED,'Final protocol stability')
 for name,q in builder.SOURCE_MAP.items():need((STAGE/name).read_bytes()==q.read_bytes(),'Final source stability '+name)
 out=FOLDER/'independent_source_packet_admission.json';need(not out.exists(),'Preserve existing packet admission')
 report={'format':'independent-predictor-CE-source-packet-admission-v1','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'protocol_sha256':EXPECTED,'stage_root':str(STAGE),'payload_files':670,'payload_bytes_including_protocol':total,'checks_passed':len(checks),'checks':checks,'source_files':[pin(STAGE/name) for name in builder.SOURCE_MAP],'formal_design_admission':pin(ROOT/'research/completion_codeformer_predictor_ce_design_independent_review_v1/independent_formal_design_admission.json'),'source_semantics_admission':pin(FOLDER/'independent_source_semantics.json'),'design_binding_recheck':pin(FOLDER/'independent_design_binding_recheck.json'),'actual_guarded_transfer_CLI':pin(FOLDER/'independent_pure_transfer_CLI_fixture_v1.json'),'current_R1_protected_pins_verified':len(protected),'source_data_runtime_admitted':True,'archive_or_manual_guide_admitted':False,'automatic_training_authorized':False,'actual_VM_preflight':False,'unused_legacy_snapshot':{'path':'face_workflow.py','hashlib_file_digest_requires_Python311':True,'not_imported_by_runtime':True,'unchanged_frozen_evidence_not_executable_dependency':True},'local_actions':{'neural_calls':0,'model_objects':0,'optimizer_objects':0,'gradient_calls':0,'training':False,'checkpoint_deserialization':0,'VM_calls':0},'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-start};out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'complete':True,'path':str(out),'sha256':sha(out),'checks':len(checks),'seconds':report['seconds']}))
if __name__=='__main__':main()
