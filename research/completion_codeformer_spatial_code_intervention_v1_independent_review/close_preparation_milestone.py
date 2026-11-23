"""Independent metadata/documents/publication closure; no prior science/CLI rerun.
Checks current milestone pins and preserved changed-document boundaries only.
"""
from pathlib import Path
from datetime import datetime,timezone
from urllib.parse import unquote,urlsplit
import hashlib,json,re
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/completion_codeformer_spatial_code_intervention_v1_packet/draft_004'
PROTOCOL='b27617eb4f2a2225e0969d463acc8048a148203488f0fcd5397e5c5f9a0a0123'
CHECKS=0
SEEN={}

def need(ok,msg):
 global CHECKS
 CHECKS+=1
 if not bool(ok):raise ValueError(msg)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()

def pin(path):return {'path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'sha256':sha(path)}
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def checkpin(row):
 key=(row['path'],row['bytes'],row['sha256'])
 if key not in SEEN:need(pin(row['path'])==row,'Exact current milestone evidence '+row['path']);SEEN[key]=row

def pins_in(value):
 if isinstance(value,dict):
  if all(k in value for k in ('path','bytes','sha256')) and Path(value['path']).is_absolute():checkpin(value)
  for v in value.values():pins_in(v)
 elif isinstance(value,list):
  for v in value:pins_in(v)

def local_links(path,text):
 links=[]
 for label,target in re.findall(r'\[([^\]]+)\]\(([^)]+)\)',text):
  target=target.strip().strip('<>');u=urlsplit(target);need(not u.scheme and not u.netloc,'New milestone links are local '+target);q=(Path(path).parent/unquote(u.path)).resolve();need(q.is_relative_to(ROOT) and q.exists(),'Existing confined new milestone link '+target);links.append({'document':str(Path(path).resolve()),'label':label,'target':str(q)})
 return links

def main():
 mp=BASE/'preparation_milestone_v1.json';need(sha(mp)=='5a6f99b26c7cd22908bc1fc8c5ba84820d6bb6470e7db7d927089980ce65b22b' and mp.stat().st_size==5586,'Exact root preparation milestone');m=read(mp);pins_in(m);need(m['complete'] is True and m['status']=='PREPARED_UNRUN_ZERO_UPDATE_DIAGNOSTIC_MANUAL_TRANSFER_READY' and not m['actual_VM_preflight_or_execution'] and not m['actual_training'] and not m['new_model_created'] and m['neural_calls']==m['VM_calls']==0 and not m['goal_complete'] and m['full_goal_status']=='ACTIVE_INCOMPLETE','Prepared/unrun fullgoal remains active')
 prep=Path(m['preparation']['path']).read_text(encoding='utf-8-sig');guide=Path(m['manual_guide']['path']).read_text(encoding='utf-8-sig');transferguide=Path(m['transfer_guide']['path']);need(Path(m['manual_guide']['path']).read_bytes()==transferguide.read_bytes(),'Both published guides exact');need(PROTOCOL in guide and PROTOCOL in prep and m['protocol']['sha256']==PROTOCOL,'Final004protocol consistently bound')
 rootrelease=read(m['evidence']['root_release']['path']);rootfinal=read(m['evidence']['root_final_archive']['path']);indfinal=read(m['evidence']['independent_final_archive']['path']);source=read(m['evidence']['independent_source_CLI']['path']);runtime=read(m['evidence']['independent_runtime']['path']);pins_in(rootrelease);pins_in(rootfinal);pins_in(indfinal)
 need(rootrelease['released_for_manual_vm_transfer'] is True and not rootrelease['automatic_vm_launch'] and rootrelease['protocol_sha256']==indfinal['protocol_sha256']==source['protocol_sha256']==PROTOCOL,'Manual-only final004 source/transport interface');need(rootfinal['complete'] is True and rootfinal['archive_members']==indfinal['regular_members']==167 and rootfinal['pinned_payload_files']==166 and rootfinal['payload_bytes_including_protocol']==indfinal['payload_bytes']==19491897,'Exact167members166payloads19491897bytes includingprotocol');need(rootfinal['full_stream_all167_SHA_and_gzip_CRC'] and indfinal['full_stream_member_SHA_gzip_CRC_verified'] and indfinal['all167_manifest_members_match'] and indfinal['checks_passed']==1036,'TwoactualfullstreamCRC/member audits alreadyexecuted');need(source['checks_passed']==1669 and source['actual_unmodified_verify_transfer_CLI']['exit_code']==0 and runtime['checks_passed']==2988,'Independent checks statedaccurately');need(source['external_export_hard_cap']['GNUtimeout_owned_export_child_KILL900s'] and source['external_export_hard_cap']['command_v_timeout_guard_precedes_work'],'Independent externalexport900hardstop source admission')
 limits=rootrelease['limits'];need(limits=={'worker_hard_seconds':900,'worker_cooperative_seconds':840,'maximum_new_optimizer_updates':0,'minimum_free_GiB':6,'stop_floor_GiB':2,'unexpected_failure_state_MiB':384,'output_GiB':1,'raw_tar_and_gzip_MiB_each':1280,'export_seconds':900},'Exact finite root publishedlimits');p=read(m['protocol']['path']);c=p['contract'];need(c['forward_slots_planned']==108 and c['forward_slots_max']==120 and c['typed_forward_caps']=={'parity':8,'bridge':4,'H':48,'V':48} and c['new_NPZ_max']==104 and c['new_PNG_max']==96 and c['state_tensors_frozen']==497,'Scientific counts/scopes declared correctly');need(len(p['files'])==166 and len(p['external_controls']['files'])==196 and len(p['oracle_labels'])==16 and len(p['monitor_cases'])==32 and len(p['challenge_cases'])==16,'Exactsourcecase/control/label counts');need(sum(x['bytes'] for x in p['files'])+Path(m['protocol']['path']).stat().st_size==19491897,'Uncompressedtotal includesprotocol exactly')
 need(m['transfer_archive']['bytes']==13610328 and m['transfer_archive']['sha256']=='24f480add44e4fe8176a68a3b67321b11b4baee0ebb43d0a3790fefc434549a5','Published actual archivehash/size');checksum=Path(m['transfer_archive']['path']+'.sha256');expected=m['transfer_archive']['sha256']+'  '+Path(m['transfer_archive']['path']).name+'\n';need(checksum.read_bytes()==expected.encode('ascii'),'Published exactLFchecksum');need(len(rootfinal['SCP_commands'])==5 and all(cmd in guide for cmd in rootfinal['SCP_commands']) and rootfinal['single_remote_per_download'],'AllfiveexactSCPcommands retained in guide');need('tmux new-session -A -s completion_spatial_code_intervention_v1' in guide and '--verify-transfer' in guide and 'bash scripts/run_completion_codeformer_spatial_code_intervention_v1.sh '+PROTOCOL in guide,'Exact manualtmux/sourceprotocol commands');need('command -v timeout >/dev/null' in guide and 'external900-second GNUtimeout KILL' in guide and 'No model is trained.' in guide and 'trainer.log' in guide and 'Do not launch a' in guide,'Exporthardstop/quietworker/unrunmodel truth');need('unavailable-clean-information' in prep and 'cannot be used as app outputs' in prep and 'PREPARED, UNRUN' in prep and 'ZERO TRAINING UPDATES' in prep,'Unavailableoracle andunrunnoqualification truth')
 preserved=[];links=[]
 need(len(m['documents'])==2,'Exacttwohistorydocumentupdates')
 for row in m['documents']:
  old=Path(row['before']['path']).read_bytes();new=Path(row['after']['path']).read_bytes();off=row['old_bytes_offset'];need(new[off:off+len(old)]==old and row['old_bytes_exact'],'Everyoldhistorybyte preserved')
  if row['mode']=='prepend':need(off==2022 and off+len(old)==len(new),'Exacthandoffoldbytes offset2022');changed=new[:off].decode('utf-8-sig')
  else:need(row['mode']=='append' and off==0 and len(new)-len(old)==1135,'Exactspecoldprefix/append1135');changed=new[len(old):].decode('utf-8-sig')
  need('ACTIVE' in changed and 'remain incomplete' in changed and 'R1' in changed and ('NoactualVMrun' in changed or 'No VM' in changed),'Currentdocstatusfullscope activeunrun');links+=local_links(row['after']['path'],changed);preserved.append({'before':row['before'],'after':row['after'],'old_bytes_offset':off,'all_previous_bytes_exact':True})
 links+=local_links(m['preparation']['path'],prep);links+=local_links(m['manual_guide']['path'],guide)
 for text in (prep,guide):need(('not a qualified' in text or 'Neither qualifies a model' in text) and ('not a neural receptive-field map' in text or 'not a quality-selected' in text),'Preparationclaimlimits remainhonest')
 admission=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json');r1=admission['source_files']+[admission['candidate_manifest'],admission['checkpoint']];need(len(r1)==22,'All22currentR1pins')
 for row in r1:checkpin(row)
 need(not (BASE/'cctv_completion_codeformer_spatial_code_intervention_v1_vm/outputs').exists(),'Stagedsource hasno actualrun outputs');need(sha(mp)=='5a6f99b26c7cd22908bc1fc8c5ba84820d6bb6470e7db7d927089980ce65b22b','No rootmilestone modifications')
 result={'format':'independent-spatial-code-intervention-v1-manual-preparation-milestone-closure','complete':True,'independent':True,'created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(__file__),'root_milestone':pin(mp),'protocol_sha256':PROTOCOL,'checks_passed':CHECKS,'all_current_bound_evidence_pins_exact':list(SEEN.values()),'published_archive_checksum_bothguides_exact':True,'prior_fullstream167member_gzipCRC_evidence_hashbound_notrerun':True,'source_and_actualCLI_admissions_final004only':True,'scientific_counts':{'planned_fullforwards':108,'hardcap':120,'newNPZ':104,'newPNG':96,'frozenR1tensors':497,'pairedadaptiveTRAINcases':48,'samebooklabels':16,'oldACcontrolfiles':196,'training_optimizer_backward_teacher':0},'published_resource_limits':limits,'preserved_history_documents':preserved,'new_or_modified_milestone_local_links':links,'link_scope':'Complete newpreparation/guide plus newhandoffprefix/specappend. Priorimmutablehistorybytes preserved; no claim allolderexternal/historicallinks re-audited.','R1_all22_currentpins_exact':True,'manual_transfer_ready':True,'actual_VM_or_model_preflight_verified':False,'actual_new_model_or_training':False,'model_qualified':False,'full_goal_complete':False,'local_actions':{'metadata_and_filehash_checks_only':True,'prior_NN_source_or_CLItests_rerun':False,'model_loads':0,'neural_calls':0,'gradient_or_optimizer_calls':0,'VM_calls':0,'rootdocs_or_oldevidence_modified':False}}
 target=Path(__file__).parent/'independent_preparation_milestone_closure.json'
 with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(target),'checks':CHECKS,'boundpins':len(SEEN),'local_newlinks':len(links),'manual_transfer_ready':True,'goal_complete':False},indent=2),flush=True)
if __name__=='__main__':main()
