"""Independent other-agent staged transport bytes/namespace/data binding; not runtime semantics."""
import ast,hashlib,importlib.util,json,os,re
from pathlib import Path,PurePosixPath
B=Path(r'C:\xampp\htdocs\YEAR 4\Testing_FACE_MASK_COMPLETION');H=B/'research/completion_spatial_code_intervention_v1_transport_review';S=B/'outputs/completion_codeformer_spatial_code_intervention_v1_packet/draft_003/cctv_completion_codeformer_spatial_code_intervention_v1_vm'
EXPECTED='cc52d1c5554f7aa0ea3cd61dec9b8348dde37b47cd7ca82640805c4ba658e052';checks=[]
def xp(p):return '\\\\?\\'+str(Path(p).absolute())
def digest(p):
 h=hashlib.sha256()
 with open(xp(p),'rb') as f:
  for block in iter(lambda:f.read(1024**2),b''):h.update(block)
 return h.hexdigest()
def read(p):
 with open(xp(p),'r',encoding='utf-8-sig') as f:return json.load(f)
def pin(p):return {'path':str(p),'bytes':os.stat(xp(p)).st_size,'sha256':digest(p)}
def need(ok,label):
 if not ok:raise ValueError(label)
 checks.append(label)
p=read(S/'protocol.json');need(digest(S/'protocol.json')==EXPECTED,'Final staged protocol pin');need(digest(B/'research/completion_codeformer_spatial_code_intervention_v1/protocol.json')==EXPECTED,'Published research protocol exact');need(p['format']=='completion-codeformer-spatial-code-intervention-v1' and p['manual_execution_only'] is True and p['automatic_resume_or_promotion'] is False and p['new_model_created'] is False and p['model_qualified'] is False,'Unrun diagnostic scope')
records=p['files'];need(len(records)==166 and len({r['path'].casefold() for r in records})==166,'166 unique casefold payloads')
for r in records:
 name=PurePosixPath(r['path']);need(not name.is_absolute() and str(name)==r['path'] and '..' not in name.parts and ':' not in r['path'] and '\\' not in r['path'] and len(r['path'].encode())<=240,'Source canonical relative namespace')
 path=S/Path(*name.parts);need(os.path.isfile(xp(path)) and not os.path.islink(xp(path)) and os.stat(xp(path)).st_size==r['bytes'] and digest(path)==r['sha256'],'Actual staged size/SHA '+r['path'])
 need(path.suffix.lower() not in ('.pt','.pth'),'No checkpoint payload upload')
actual={str(path)[4:] for path in Path(xp(S)).rglob('*') if path.is_file()};expected={str((S/Path(*PurePosixPath(r['path']).parts)).absolute()) for r in records}|{str((S/'protocol.json').absolute())};need(actual==expected,'Exact staged payload namespace');need(sum(r['bytes'] for r in records)+os.stat(xp(S/'protocol.json')).st_size==19491244,'Exact compact staged payload total')
need(len(p['monitor_cases'])==32 and len(p['challenge_cases'])==16 and len(p['parity_cases'])==4 and len(p['bridge_cases'])==4 and len(p['oracle_labels'])==16,'Fixed input/control/label counts')
for key in ('root_design','formal_design_admission','pre_release_clarifications','root_installer_fixtures','author_transport_fixtures','original_probe_protocol','actual_original_probe_transport'):
 relative=p['evidence'][key];need(digest(S/relative)==p['evidence_sha256'][key],'Pinned independent/source evidence '+key)
design=read(S/p['evidence']['root_design']);need([r['id'] for r in p['monitor_cases']+p['challenge_cases']]==design['data']['case_ids_in_fixed_protocol_order'] and [r['id'] for r in p['bridge_cases']]==design['data']['bridge_case_ids'] and p['parity_cases']==design['data']['parity_cases_exact'],'Data exact formal input order/bridges/parities')
external=p['external_controls'];need(external['root']=='../cctv_completion_codeformer_same_codebook_probe_v1_vm/outputs' and external['source_protocol_sha256']=='51782b57fca9ad9cf7161b0f4451a65e545e3eb2f8f58788fa40b64aec48d1ec' and external['copied_to_packet'] is False,'External controls metadata-only exact old source')
old=B/'outputs/completion_codeformer_same_codebook_probe_v1_return/cctv_completion_codeformer_same_codebook_probe_v1_results/outputs';need(len(external['files'])==196 and len({r['path'] for r in external['files']})==196,'196 unique old A/C references')
for r in external['files']:need(os.stat(xp(old/r['path'])).st_size==r['bytes'] and digest(old/r['path'])==r['sha256'],'Actual saved A/C external source pin '+r['path'])
need(set(external['arms'])=={'A','C'},'Only two reused historical control arms')
rootfix=read(S/p['evidence']['root_installer_fixtures']);source_review=read(H/'independent_transport_source_review.json')
for source in source_review['source_pins']:
 relative='scripts/'+Path(source['path']).name;need(pin(Path(source['path']))==source and digest(S/relative)==source['sha256'],'Independent transport source current/staged equality '+relative)
allowed=set(p['export']['allowed_output_paths']);need(len(allowed)<395 and len(allowed)==len(p['export']['allowed_output_paths']) and sum(v.endswith('.npz') for v in allowed)==104 and sum(v.endswith('.png') for v in allowed)==96,'Exact bounded104NPZ96PNG output-name allowlist');need(p['export']['maximum_regular_members_including_manifest']==400,'400 return member contract')
need(p['contract']['terminal_bytes']==384*1024**2 and p['contract']['array_bytes']==560*1024**2 and p['contract']['PNG_bytes_total']==24*1024**2 and p['contract']['metadata_bytes']==8*1024**2 and p['contract']['output_bytes']==1024**3,'Typed output capacities');need(p['contract']['forward_slots_planned']==108 and p['contract']['forward_slots_max']==120 and p['contract']['optimizer_steps_max']==0,'Zero-update finite declared execution')
result={'format':'spatial-code-intervention-v1-independent-staged-transport-admission-v1','complete':True,'independent':True,'protocol_sha256':EXPECTED,'staged_root':str(S),'payload_files':166,'total_members_including_protocol':167,'payload_bytes':19491244,'checks_passed':len(checks),'checks':checks,'source_review_pin':pin(H/'independent_transport_source_review.json'),'external_controls_rehashed':196,'archive_published_or_CRC_checked':False,'runtime_semantics_certified_here':False,'runtime_semantics_separate_independent_agent_required':True,'VM_calls':0,'NN_calls':0,'model_qualified':False,'manual_ready_pending_root_release_and_final_archive_review':True}
path=H/'independent_staged_transport_admission.json'
with path.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps({'pin':pin(path),'checks':len(checks)},indent=2))
