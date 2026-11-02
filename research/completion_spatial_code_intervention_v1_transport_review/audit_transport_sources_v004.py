"""Independent source/transport boundaries with owned tiny-file exports, no model work."""
import ast,contextlib,hashlib,importlib.util,io,json,os,re,shlex,tarfile,types
from pathlib import Path
from unittest.mock import patch
B=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION');H=B/'research/completion_spatial_code_intervention_v1_transport_review';F=Path(r'C:\xampp\htdocs\YEAR 4\Testing\outputs\spatial_transport_policy_004');assert not F.exists();F.mkdir()
BUILDER=B/'scripts/prepare_completion_codeformer_spatial_code_intervention_v1.py';EXPORT=B/'scripts/export_completion_codeformer_spatial_code_intervention_v1.py';RUN=B/'scripts/run_completion_codeformer_spatial_code_intervention_v1.sh'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def pin(path):return {'path':str(path),'bytes':path.stat().st_size,'sha256':sha(path)}
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
builder=load('independent_spatial_builder',BUILDER);exporter=load('independent_spatial_export',EXPORT);initial=[pin(p) for p in (BUILDER,EXPORT,RUN)];checks=[];fixtures=[]
def need(ok,label):
 if not ok:raise ValueError(label)
 checks.append(label)
def write(path,value):path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8',newline='\n')
for path in (BUILDER,EXPORT):ast.parse(path.read_text(),feature_version=(3,10));need('torch' not in {node.names[0].name for node in ast.walk(ast.parse(path.read_text())) if isinstance(node,ast.Import)},'No model import '+path.name)
need(exporter.MAX_BYTES==1280*1024**2 and exporter.OUTPUT_BYTES==1024**3 and exporter.MAX_SECONDS==900 and exporter.STOP_FREE==2*1024**3,'Exact export resources')
shell=RUN.read_text(encoding='utf-8');need('command -v timeout >/dev/null || exit 1' in shell and shell.index('command -v timeout')<shell.index('--run'),'Timeout dependency refuses before neural worker');need('timeout --signal=KILL 900s "" -B -u scripts/export_completion_codeformer_spatial_code_intervention_v1.py --root "" --protocol-sha ""' in shell,'Owned GNUtimeout external900s exporter');need('export_exit=$?' in shell and 'if test "" -ne 0; then exit ""; fi' in shell,'External timeout exit propagated');
rootfix=B/'research/completion_codeformer_spatial_code_intervention_v1_design/root_installer_fixtures.json';need(sha(rootfix)=='610177441433cbb5f282c8264d35a0b63696e13b72cb4c3bcb627eb2c6b58e6e','Exact root installer admission');r=json.loads(rootfix.read_text());need(hashlib.sha256(builder.INSTALL_CODE.encode()).hexdigest()==r['installer_literal_sha256']=='305777c2e412e145bd2059aebc5830e9ff06f374df2ae1f2079b0507136584a3','Current installer equals actually exercised root literal');need(len(r['fixtures'])==9 and all(v['expected_accept']==v['accepted'] and v['source_archive_unchanged'] and v['target_preservation_or_closed_failure_pass'] for v in r['fixtures']),'All nine actual installer fixtures match outcomes');ast.parse(builder.INSTALL_CODE.replace('__PROTOCOL_SHA__','0'*64),feature_version=(3,10))
guide=builder.manual_guide('a'*64,'b'*64,1234,164);need('\r' not in guide and '__INSTALL' not in guide and '__PROTOCOL' not in guide and '__ARCHIVE' not in guide,'Guide LF and all templates resolved')
commands=[line for line in guide.splitlines() if line.startswith('gcloud compute scp ')];need(len(commands)==5,'Exactlyfive SCP commands')
for i,line in enumerate(commands):
 tokens=shlex.split(line);args=[v for v in tokens[3:] if not v.startswith('--')];need(len(args)==2 and sum(v.startswith('janusdominic0@forensic-dgp-thesis:') for v in args)==1,'Each SCP has two arguments and one remote')
 need(line.count('--project=forensic-dgp-thesis')==1 and line.count('--zone=us-central1-a')==1,'Exact project and zone')
 if i<2:need(args[1]=='janusdominic0@forensic-dgp-thesis:/home/janusdominic0/' and args[0] in (builder.STEM+'-execution.tar.gz',builder.STEM+'-execution.tar.gz.sha256'),'Upload direction/file')
 else:need(args[0]=='janusdominic0@forensic-dgp-thesis:/home/janusdominic0/'+builder.STEM+('-results.tar.gz' if i==2 else '-results.tar.gz.sha256' if i==3 else '-export.json') and args[1]=='.','Download direction/file')
need(guide.count('cd /d "C:\\xampp\\htdocs\\YEAR 4\\Testing\\outputs"')==2,'Both Windows current folders exact');need('tmux new-session -A -s completion_spatial_code_intervention_v1' in guide,'Exact tmux session');need("python3 - <<'PY'\n"+builder.INSTALL_CODE.replace('__PROTOCOL_SHA__','a'*64)+'PY\n' in guide,'Literal installer exact in guide');need('sha256sum -c '+builder.STEM+'-execution.tar.gz.sha256 &&' in guide,'Checksum before install')
class Sink:
 def __init__(self,short=False):self.called=0;self.short=short
 def write(self,value):self.called+=1;return len(value)-(1 if self.short else 0)
 def flush(self):pass
sink=Sink();writer=exporter.CountedWriter(sink,lambda:None);writer.count=exporter.MAX_BYTES
try:writer.write(b'x');raise AssertionError('Raw cap bypass')
except ValueError:need(sink.called==0,'1280MiB cap refuses before write')
sink=Sink(short=True)
try:exporter.CountedWriter(sink,lambda:None).write(b'two');raise AssertionError('Short write bypass')
except ValueError:need(sink.called==1,'Short writes rejected')
sink=Sink()
with patch.object(exporter.shutil,'disk_usage',return_value=types.SimpleNamespace(free=exporter.STOP_FREE+1)):
 try:exporter.CountedWriter(sink,lambda:None,disk_directory=F).write(b'xx');raise AssertionError('Disk floor bypass')
 except ValueError:need(sink.called==0,'2GiB floor enforced before write')
def case(name,kind):
 base=F/name;root=base/'pilot';home=base/'home';root.mkdir(parents=True);home.mkdir();(root/'outputs').mkdir();(root/'asset.txt').write_bytes(b'stable exact toy source\n');(root/'trainer.log').write_bytes(b'')
 p={'format':'completion-codeformer-spatial-code-intervention-v1','manual_execution_only':True,'automatic_resume_or_promotion':False,'contract':{'output_bytes':1024**3,'stop_free_bytes':2*1024**3},'files':[{'path':'asset.txt','bytes':(root/'asset.txt').stat().st_size,'sha256':sha(root/'asset.txt')}],'arms':[],'parity_cases':[],'bridge_cases':[],'monitor_cases':[],'challenge_cases':[],'export':{}}
 # Build the independent minimal closed namespace, rather than copying source output.
 names={'outputs/'+v for v in ('state_before.json','oracle_label_receipt.json','reused_control_binding.json','external_control_compatibility.json','arm_H_complete.json','arm_V_complete.json','state_after.json','results.json','failure.json','state_on_failure.json','state_receipt_failure.json','failed_actual_model_state.pth','failed_actual_model_state.pth.partial','failure_state_receipt.json')};names.update(('outputs/fresh_app_parity/summary.json','outputs/external_bridge/summary.json'))
 exitcode=1
 if kind=='members':
  p['parity_cases']=[{'id':'p%03d'%i} for i in range(198)]
  for row in p['parity_cases']:
   for suffix in ('.json','.npz'):
    relative='outputs/fresh_app_parity/'+row['id']+suffix;names.add(relative);path=root/relative;path.parent.mkdir(exist_ok=True);path.write_bytes(b'0')
  write(root/'outputs/failure.json',{'fixture':True})
 if kind=='unknown_file':(root/'outputs/unlisted.txt').write_bytes(b'preserve me')
 if kind=='unknown_directory':(root/'outputs/unlisted').mkdir()
 if kind in ('stable_alias','distinct_alias','success_checkpoint','partial'):
  state=root/'outputs/failed_actual_model_state.pth';partial=Path(str(state)+'.partial');state.write_bytes(b'not a real checkpoint, tiny transport fixture\n')
  if kind=='stable_alias':os.link(state,partial)
  if kind=='distinct_alias':partial.write_bytes(state.read_bytes())
  if kind=='partial':state.rename(partial)
  if kind=='success_checkpoint':exitcode=0;write(root/'outputs/results.json',{'complete':True,'protocol_sha256':'REPLACE','model_qualified':False,'diagnostic_only':True,'new_model_created':False})
 p['export']['allowed_output_paths']=sorted(names);write(root/'protocol.json',p);expected=sha(root/'protocol.json')
 if kind=='success_checkpoint':write(root/'outputs/results.json',{'complete':True,'protocol_sha256':expected,'model_qualified':False,'diagnostic_only':True,'new_model_created':False})
 write(root/'supervisor.json',{'complete':True,'worker_exit':exitcode,'timeout':False})
 if kind=='source_drift':(root/'asset.txt').write_bytes(b'wrong source now')
 before={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()};error=None;receipt=None
 try:
  with contextlib.redirect_stdout(io.StringIO()):receipt=exporter.export(root,expected,home,enforce_vm=False)
 except BaseException as exc:error=repr(exc)
 accept=kind in ('empty_log_valid','stable_alias','partial');need(bool(receipt and receipt['complete'])==accept,'Exportfixture '+name+' expected behavior');need(before=={str(v.relative_to(root)):sha(v) for v in root.rglob('*') if v.is_file()},'Exportfixture '+name+' source evidence untouched')
 if accept:
  with tarfile.open(home/(exporter.STEM+'-results.tar.gz'),'r:gz') as tar:
   members=tar.getmembers();manifest=json.load(tar.extractfile(exporter.MEMBER_ROOT+'/export_manifest.json'));need(len(members)<=400,'Fixture return400cap')
   if kind=='stable_alias':need(len([v for v in members if '.pth' in v.name])==1 and len(manifest['same_inode_failed_state_aliases'])==1 and receipt['source_alias_revalidated_after_tar_immediately_before_publication'],'Actual sameinode payload exported once and finalrevalidated')
   if kind=='partial':need(manifest['partial_files_retained']==['outputs/failed_actual_model_state.pth.partial'],'Corrupt failedpartial retained without checkpoint loading')
 else:need(not(home/(exporter.STEM+'-results.tar.gz')).exists(),'Rejectedfixture neverpublishesfinalarchive')
 fixtures.append({'case':name,'accepted':bool(receipt),'error':error,'pass':True,'root':str(base)})
for name in ('empty_log_valid','stable_alias','distinct_alias','success_checkpoint','partial','unknown_file','unknown_directory','members','source_drift'):case(name,name)
repair=H/'independent_alias_drift_repair_fixture.json';d=json.loads(repair.read_text());need(d['repair_refuses_before_publication'] and d['source']['sha256']==sha(EXPORT),'Independent sourcealiasreplacement repair fixture current')
need(initial==[pin(p) for p in (BUILDER,EXPORT,RUN)],'Source bytes stable through independent fixtures')
result={'format':'spatial-code-intervention-v1-independent-transport-source-review-v1','complete':True,'independent':True,'source_pins':initial,'checks_passed':len(checks),'checks':checks,'actual_export_fixtures':fixtures,'exact_installer_literal_sha256':r['installer_literal_sha256'],'root_installer_admission':pin(rootfix),'independent_alias_repair':pin(repair),'archive_ready_claim':False,'published_archive_not_yet_reviewed':True,'unpublished_guide_template_only':True,'neural_calls':0,'model_construction':False,'VM_calls':0,'scope':'Read-only other-agent builder/exporter/shell source, exact actually-tested installer literal, own tinyfile export fixtures and guide-template commands. No original model/research sources altered; no VM operations or real checkpoint load.'}
path=H/'independent_transport_source_review_v004.json'
with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'pin':pin(path),'checks':len(checks),'fixtures':len(fixtures)},indent=2))
