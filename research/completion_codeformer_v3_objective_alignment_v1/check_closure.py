"""Independent closure-document/assets/link check after audited V3 quality rejection."""
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import unquote
import hashlib,json,re,time
ROOT=Path(__file__).resolve().parents[2]
RET=ROOT/'outputs/completion_codeformer_fusion_only_v3_return'
OUT=ROOT/'outputs/completion_codeformer_v3_next_objective_independent_review_v1'

def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def main():
 start=time.monotonic();publication=read(RET/'return_document_publication_v1.json');precision=read(RET/'results_precision_review_v2.json');first_precision=read(RET/'results_precision_review_v1.json');preservation=read(RET/'root_current_R1_preservation_after_V3.json')
 result=ROOT/'COMPLETION_CODEFORMER_FUSION_ONLY_V3_RESULTS.md';need(sha(result)==precision['after_sha256']=='106c2a2a2ca3e5ed02e81ef34c5fc27746c19db4728014e70e0139a5df59a473','Revised results changed')
 need(sha(Path(precision['before_exact_backup']))==precision['before_sha256']==first_precision['after_sha256'] and sha(Path(first_precision['before_exact_backup']))==first_precision['before_sha256']==publication['results']['sha256'],'Prior results exact backup differs')
 sections={result:result.read_text()};docs=[]
 for stem in ('handoff','spec'):
  before=Path(publication[stem+'_before']['path']);after=Path(publication[stem+'_after']['path']);old=before.read_bytes();new=after.read_bytes()
  need(sha(before)==publication[stem+'_before']['sha256'] and sha(after)==publication[stem+'_after']['sha256'],'Published doc pin changed '+stem)
  need(new.count(old)==1,'Prior doc bytes not retained exactly once '+stem);addition=new.replace(old,b'',1).decode('utf-8-sig');sections[after]=addition;docs.append({'name':stem,'old_document_preserved_exact':True,'before':pin(before),'after':pin(after)})
 current=[]
 need(preservation['all22pins_exact'] is True and len(preservation['verified_files'])==22,'R1 preservation schema differs')
 for row in preservation['verified_files']:
  p=Path(row['path']);actual=pin(p);need(actual['bytes']==row['bytes'] and actual['sha256']==row['sha256'],'R1 current asset changed '+str(p));current.append(actual)
 retirement=read(RET/'manual_guide_retirement.json');guides=[]
 for row in retirement['guides']:
  p=Path(row['guide']);old=Path(row['prior_bytes_backup']);need(sha(p)==row['new_sha256'] and sha(old)==row['old_sha256'] and p.read_bytes().endswith(old.read_bytes()),'Historical guide backup/prefix differs')
  need(p.read_text().startswith('# HISTORICAL V3 RUN - DO NOT LAUNCH AGAIN'),'Guide still looks launchable');guides.append({'guide':pin(p),'prior_exact_backup':pin(old)})
 stage=ROOT/'outputs/completion_codeformer_fusion_only_v3_packet/draft_002/cctv_completion_codeformer_fusion_only_v3_vm';p=read(stage/'protocol.json');need(sha(stage/'protocol.json')=='3dc71dac3002e82587cbc2cb758f36df604ed4c7131a3e592db4d8dcb5c18033','Sealed protocol changed');names=[]
 for row in p['files']:
  path=stage/row['path'];need(path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],'Sealed payload changed '+row['path']);names.append(row['path'])
 need(len(names)==len(set(names))==831,'Sealed payload namespace count differs');actual={x.relative_to(stage).as_posix() for x in stage.rglob('*') if x.is_file()};need(actual==set(names)|{'protocol.json'},'Sealed namespace not closed')
 links=[]
 for path,text in sections.items():
  for raw in re.findall(r'\]\(([^)]+)\)',text):
   rel=unquote(raw.strip('<>').split('#',1)[0]);
   if not rel or '://' in rel:continue
   target=(path.parent/rel).resolve();need(target.is_file(),'Broken closure Markdown link '+str(path)+' -> '+raw);links.append({'document':str(path),'target':str(target)})
 body=result.read_text();need('aggregate reversal' in body and 'no individual' in body and 'All228NPZ files' in body and 'All224 actual output raw512-to-raw256/PNG replays pass exactly' in body,'Precision distinctions not in revised results')
 need('quality failure' in body and 'No successor training' in body and 'full goal is ACTIVE' in body,'Closure readiness/status wording differs')
 target_report=ROOT/'outputs/completion_codeformer_v3_objective_alignment_v1_r1/FIT256_target_availability.json';need(sha(target_report)=='4dd1c4b8088fc20104937d8b9efebb4aca0c9b6292d3bf2a54973fc2e5921741','Target availability follow-up changed')
 teacher={r['id']:r for r in p['teacher_targets']};fit=p['fit_cases'];need(len(fit)==256 and all(r['clean256']==teacher[r['teacher_key']]['clean256'] for r in fit),'Existing FIT clean256 mapping differs')
 need(len({r['clean256'] for r in fit})==64 and len({r['reference_id'] for r in fit})==32,'Unique FIT target counts differ')
 need(not({r['reference_id'] for r in fit}&{r['reference_id'] for r in p['monitor_cases']+p['challenge_cases']}),'FIT monitor identity source overlap')
 report={'format':'independent-completion-v3-closure-document-asset-check-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'revised_results':pin(result),'precision_transition':pin(RET/'results_precision_review_v2.json'),'exact_prior_documents':docs,'all22_current_R1_pins_independently_rehashed':True,'R1_assets':current,'historical_guides':guides,'sealed_packet':{'protocol_sha256':sha(stage/'protocol.json'),'payloads_rehashed':831,'closed_namespace':True,'source_data_unchanged':True},'current_closure_Markdown_links_checked':links,'FIT256_target_followup':{'source_receipt':pin(target_report),'all256_existing_clean256_fields_map_exact_teacher_key':True,'unique_target_PNGs':64,'FIT_reference_ids':32,'FIT_monitor_challenge_reference_disjoint':True,'new_target_preparation_needed':False},'precision_notes':['224output raw/PNGreplays distinguished from228NPZ schema/CRC reviews.','Clear-lens HQ/PNG reversal explicitly geometry aggregate, noindividual signflip.','Valid recovery state is distinct from failed quality;R1retained;goalactive;no successor transferready.'],'new_model_or_training_packet':False,'model_inference_gradient_optimizer_VM_calls':0,'seconds':time.monotonic()-start}
 out=OUT/'independent_closure_check.json'
 with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(out),'R1_pins':22,'sealed_payloads':831,'links':len(links),'seconds':report['seconds']},indent=2))
if __name__=='__main__':main()
