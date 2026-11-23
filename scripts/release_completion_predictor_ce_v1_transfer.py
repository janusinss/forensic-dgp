"""Root release for verified manual transfer; never launches training or a VM."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def need(ok,message):
 if not ok:raise ValueError(message)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def main():
 base=ROOT/'outputs/completion_codeformer_predictor_ce_v1_packet/draft_001'
 stage=base/'cctv_completion_codeformer_predictor_ce_v1_vm';protocol=stage/'protocol.json'
 expected='15aa3e1478a67c120b2fcb6af50ecae58d0558eb4e65cef473d0339572bf084b'
 need(sha(protocol)==expected,'Protocol drift')
 p=read(protocol);need(len(p['files'])==670 and p['manual_execution_only'] and not p['automatic_resume_or_promotion'],'Closed manual source scope')
 for r in p['files']:
  q=stage/r['path'];need(q.is_file() and not q.is_symlink() and q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],'Payload drift '+r['path'])
 need({f.relative_to(stage).as_posix() for f in stage.rglob('*') if f.is_file()}=={r['path'] for r in p['files']}|{'protocol.json'},'Unexpected stage namespace')
 evidence={
 'independent_source_admission':(ROOT/'research/completion_codeformer_predictor_ce_source_audit_v1/independent_source_packet_admission.json','5dd1568cda855474d31f4c84175b5cf89c4b6162a1911cbc1c746a77339dc6d3'),
 'actual_guarded_transfer_CLI':(ROOT/'research/completion_codeformer_predictor_ce_source_audit_v1/independent_pure_transfer_CLI_fixture_v1.json','1cabcacfebb8209971e34dee8b15c72730d875b94b23851fb3d06dafe1a6a8e6'),
 'source_semantics':(ROOT/'research/completion_codeformer_predictor_ce_source_audit_v1/independent_source_semantics.json','0eb9cf6ada1554875bf837d01b97a86c7a063887bbfa6608f12f0bffc1d9a0d7'),
 'formal_design':(ROOT/'research/completion_codeformer_predictor_ce_design_independent_review_v1/independent_formal_design_admission.json','691a5ec8151e0102e62f0c54c73770a25277690a7ac848586d33fb47f6ec9fbf'),
 'root_terminal_fixtures':(ROOT/'research/completion_codeformer_predictor_ce_design_v1/root_terminal_admission_fixtures.json','4ce59f5e91e9d68ec52c33d535c1c5acd7daa481fdd277146e7c48099aa41471'),
 'root_installer_fixtures':(ROOT/'research/completion_codeformer_predictor_ce_design_v1/root_installer_fixtures.json','520220a1f7a9e1808e73596f942ad66b7cedaef871ebb5fc37af95a7de0cdbc7'),
 }
 pins={}
 for key,(q,digest) in evidence.items():need(sha(q)==digest,'Admission drift '+key);pins[key]=pin(q)
 admission=read(evidence['independent_source_admission'][0]);need(admission['complete'] and admission['independent'] and admission['protocol_sha256']==expected,'Independent scope differs')
 r1=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json')
 r1pins=r1['source_files']+[r1['candidate_manifest'],r1['checkpoint']]
 for r in r1pins:
  q=Path(r['path']);need(q.stat().st_size==r['bytes'] and sha(q)==r['sha256'],'R1 model/app drift')
 need(len(r1pins)==22,'Exact R1 pins')
 release={'format':'completion-codeformer-predictor-ce-v1-root-release-v1','created_UTC':datetime.now(timezone.utc).isoformat(),'author':pin(Path(__file__)),'released_for_manual_vm_transfer':True,'automatic_vm_launch':False,'protocol_sha256':expected,'independent_source_admission':pins['independent_source_admission'],'evidence':pins,'staged_regular_members':671,'all670_payloads_root_rehashed':True,'payload_bytes_including_protocol':protocol.stat().st_size+sum(r['bytes'] for r in p['files']),'R1_app_model22pins_unchanged':True,'scientific_choice':'Distinct predictor-only CE from exactR1;114learned/383frozen,balanced128FITobservations,32steps/128backs,final32only,alloldhole/boundary plusGRAYstyle gates. Nooracleinference.','limits':{'worker_hard_seconds':1800,'worker_cooperative_seconds':1680,'maximum_new_Adam_updates':32,'minimum_free_GiB':8,'stop_floor_GiB':2,'terminal_GiB':1,'output_GiB':2.5,'raw_tar_and_gzip_GiB_each':2.75},'archive_guide_final_audit_still_required':True,'new_model_created':False,'training_launched':False,'neural_calls':0,'VM_calls':0,'model_qualified':False,'goal_complete':False}
 out=base/'root_manual_transfer_release.json'
 with out.open('x',encoding='utf-8',newline='\n') as f:json.dump(release,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(pin(out)))
if __name__=='__main__':main()
