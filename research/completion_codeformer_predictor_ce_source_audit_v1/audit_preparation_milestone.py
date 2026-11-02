from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,re,time
ROOT=Path(__file__).resolve().parents[2]
def main():
 root=ROOT;folder=Path(__file__).resolve().parent;out=folder/'independent_preparation_milestone_closure.json';assert not out.exists();started=time.monotonic();checks=[]
 def need(ok,label):
  if not ok:raise ValueError(label)
  checks.append(label)
 def sha(p):
  h=hashlib.sha256()
  with Path(p).open('rb') as f:
   for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
  return h.hexdigest()
 def pin(p):p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
 path=root/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/preparation_milestone_v1.json';need(sha(path)=='1b53de172f7ed5cda1cf677a275a0b432d6e07cc9c2d0a38e5a33e0501452f8a','Exact published preparation milestone');m=json.loads(path.read_text());pins=[]
 def visit(v):
  if isinstance(v,dict):
   if set(('path','bytes','sha256'))<=set(v):
    p=Path(v['path']);need(p.is_file() and p.stat().st_size==v['bytes'] and sha(p)==v['sha256'],'Exact milestone artifact '+p.name);pins.append(pin(p))
   for x in v.values():visit(x)
  elif isinstance(v,list):
   for x in v:visit(x)
 visit(m);newtexts=[];oldbytes=[]
 for change in m['document_changes']:
  before=Path(change['before']['path']).read_bytes();afterpath=Path(change['after']['path']);after=afterpath.read_bytes();offset=change['old_bytes_offset'];need(change['old_bytes_preserved_exact'] and after[offset:offset+len(before)]==before,'Exact retained prior document bytes atoffset '+afterpath.name)
  if offset:need(offset==2253 and after[offset:]==before,'Only new handoff2253-byte prefix');new=after[:offset]
  else:need(after.startswith(before),'Specification old prefix exact');new=after[len(before):]
  newtexts.append((afterpath,new.decode('utf-8')));oldbytes.append({'document':str(afterpath),'old_bytes':len(before),'old_bytes_offset':offset,'new_bytes':len(new),'exact_prior_bytes_preserved':True})
 reportpath=Path(m['preparation_report']['path']);need(sha(reportpath)=='7601b78ce2fa1a99da444a274ff1b2f9f9615463f26cf4d211598b0cb2daa375','Exact new preparation report');report=reportpath.read_text();newtexts.append((reportpath,report));links=[]
 for source,text in newtexts:
  for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',text):
   target=target.strip('<>');need(not target.startswith(('http:','https:','javascript:','data:')),'Local-only new documentation link '+label);relative=target.split('#',1)[0];p=(source.parent/relative).resolve();need(p.is_relative_to(root) and p.is_file(),'Confined existing local documentation link '+target);links.append({'source':str(source),'target':str(p)})
 need(len(links)==5,'Five new local documentation links verified');need(m['complete'] and m['status']=='PREPARED_UNRUN_MANUAL_TRANSFER_READY','Preparation only status')
 need(m['actual_VM_actions']==m['actual_training_steps']==0 and not any(m[k] for k in ['RESTORATION_IMPLEMENTATION_edited','new_model_created','model_qualified','goal_complete']) and m['R1_model_app_unchanged'],'No training/adoption or full-goal completion claim')
 for value in ('PREPARED, UNRUN; NO NEW TRAINED MODEL','R1 remains','Learn114','Keep383','32 new updates','128 backwards','256','128 FIT logits','384 network invocations','168 exact empty','Final32','>=1%','same-codebook','No earlier-checkpoint rescue','unavailable clean information','Photograph-ID separation does not prove distinct people','Plain-logit mathematical/autograd fixtures are distinct from model training','No new app candidate exists yet'):
  need(value in report,'Precise preparation scope '+value)
 need('current R1 app/checkpoint remain unchanged' in report and 'blinded independent FINAL' in report and 'minimum Windows CPU/fresh-install checks' in report,'Remaining full-goal scope is explicit')
 need(sha(root/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001/cctv_completion_codeformer_predictor_ce_v1_vm/protocol.json')=='15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b','Sealed source protocol remains exact after live document publication')
 result={'format':'independent-predictor-CE-preparation-milestone-closure-v1','complete':True,'independent':True,'UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(__file__),'milestone':pin(path),'preparation_report':pin(reportpath),'all_milestone_pins_verified':len(pins),'artifact_pins':pins,'prior_document_bytes':oldbytes,'new_local_links':links,'checks_passed':len(checks),'checks':checks,'claims_review':'Prepared unrun manual-only finite predictor CE pilot; no clean oracle inference, achievable oracle gain, new checkpoint, quality qualification or full-goal completion. Earlier immutable history remains exact.','archive_or_neural_replayed':False,'VM_calls':0,'model_objects':0,'optimizer_calls':0,'gradient_calls':0,'actual_training_steps':0,'new_model_created':False,'model_qualified':False,'goal_complete':False,'seconds':time.monotonic()-started};out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8');print(json.dumps({'complete':True,'path':str(out),'sha256':sha(out),'checks':len(checks),'milestone_pins':len(pins),'links':len(links),'seconds':result['seconds']}))
if __name__=='__main__':main()
