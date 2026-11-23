"""Focused independent V4 output-receipt/asset closure; saved arrays only, no Torch."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,math,sys,time
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];RET=ROOT/'outputs/completion_codeformer_delivered_objective_v4_return';RUN=RET/'cctv_completion_codeformer_delivered_objective_v4_results';STAGE=ROOT/'outputs/completion_codeformer_delivered_objective_v4_packet/draft_001/cctv_completion_codeformer_delivered_objective_v4_vm'
def need(ok,msg):
 if not ok:raise ValueError(msg)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def pin(p):return {'path':str(p.resolve()),'bytes':p.stat().st_size,'sha256':sha(p)}
def pixels(p):
 with Image.open(p) as im:return np.array(im,copy=True)
def main():
 start=time.monotonic();pixel=RET/'independent_saved_output_audit.json';need(sha(pixel)=='30c6a29c8a7f91044f356139ff691f65833ae1f76f78de0715309b35ecbaac39','Pixel receipt differs');d=read(pixel);need(sha(Path(d['auditor']['path']))=='12c25d2ae015531b001abc0e7d1616d5829b62f6675cc4309aa4cb28b1376ef5','Pixel audit script differs')
 state=RET/'independent_state_audit.json';need(sha(state)=='fcde48ec349e953665259eb56df0da70c8eef980e27e19c058c58a06499051f0','State receipt differs');need(sha(RET/'transport_audit.json')=='187be2e99a5439528288ccbdf3d196752b14dd3e8262429046ec9b1958088570','Transport differs')
 need(d['complete'] is True and d['independent'] is True and d['actual_terminal_cycle']==64 and d['evaluated_raw_PNG_outputs']==224 and d['all_retained_NPZ_count']==228 and d['all_retained_PNG_count']==224,'Complete output counts differ');need(d['saved_float256_to_PNG_all_exact'] is True and d['raw512_to_result_max_FP32_error']==0 and d['raw512_to_PNG_different_channels']==0 and d['outside_source_RGB_bytes_exact']==38414802 and d['outside_source_FP32_and_PNG_all_exact'] is True,'Output preservation/replay status differs')
 semantics=d['objective_contract_audit'];need(semantics['complete'] is True and semantics['FIT_loss_component_receipts_verified']==256 and semantics['loss_divisor_exact_once']==4 and semantics['additive_boundary_loss'] is False and semantics['PNG_rounding_or_STE'] is False and semantics['clipping_detached_semantics'] is True and 'cannot be independently replayed' in semantics['actual_FIT_raw_not_saved'] and semantics['first_last_loss_not_a_matched_trajectory'] is True,'FIT loss evidence limitations missing')
 p=read(STAGE/'protocol.json');need(sha(STAGE/'protocol.json')==sha(RUN/'protocol.json')=='8bbb3dd611e95cfbce76e7e6426ffdadb31ce23dadb1dfdcd0246e501213069d','Protocol binding differs');sourcepins=[]
 for row in p['files']:
  q=STAGE/row['path'];need(q.is_file() and not q.is_symlink() and q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Sealed846 payload differs '+row['path'])
 need(len(p['files'])==846 and {q.relative_to(STAGE).as_posix() for q in STAGE.rglob('*') if q.is_file()}=={r['path'] for r in p['files']}|{'protocol.json'},'Sealed846 namespace differs')
 admission=read(ROOT/'outputs/completion_codeformer_finetune_r1_app/source_admission.json');R1=admission['source_files']+[admission['candidate_manifest'],admission['checkpoint']]
 need(len(R1)==22,'Expected22 R1 assets')
 for row in R1:
  q=Path(row['path']);need(q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],'Selected R1 asset drift '+str(q));sourcepins.append(pin(q))
 kinds=('lower_face','eye_band','image_left_lens','cheek_object');lookup={(r['kind'],r['cycle'],r['id']):r for r in d['all_outputs']};witnesses=[]
 for kind,key in [('monitor','monitor_cases'),('challenge','challenge_cases')]:
  for geometry in kinds:
   case=next(r for r in p[key] if r['mask_kind']==geometry)
   for cycle in (0,64):
    row=lookup[(kind,cycle,case['id'])];arr=Path(row['NPZ']['path']);png=Path(row['PNG']['path']);need(sha(arr)==row['NPZ']['sha256'] and sha(png)==row['PNG']['sha256'],'Witness pin drift')
    with np.load(arr,allow_pickle=False) as z:raw=z['raw512'].copy();source=z['input256'].copy();mask=z['mask256'][0,0].astype(bool);result=z['result256'].copy()
    generated=np.clip((raw+np.float32(1))/np.float32(2),0,1);small=((generated[:,:,0::2,0::2]+generated[:,:,0::2,1::2])+(generated[:,:,1::2,0::2]+generated[:,:,1::2,1::2]))*np.float32(.25);replay=source.copy();replay[:,:,mask]=small[:,:,mask]
    need(np.array_equal(replay,result) and np.array_equal(result[:,:,~mask],source[:,:,~mask]),'Witness replay/outside mismatch');delivered=pixels(png);need(np.array_equal(np.rint(np.clip(result[0].transpose(1,2,0),0,1)*np.float32(255)).astype(np.uint8),delivered),'Witness PNG mismatch')
    target=pixels(STAGE/case['clean256']);float_target=target.astype(np.float32).transpose(2,0,1)[None]/np.float32(255);source8=np.rint(source[0].transpose(1,2,0)*np.float32(255)).astype(np.uint8);float_target=source if np.array_equal(source8,target) else float_target;rawerr=np.abs(result-float_target)[0].transpose(1,2,0).astype(np.float64);pngerr=np.abs(delivered.astype(np.float64)-target.astype(np.float64))/255
    rawL=float(rawerr[mask].mean());pngL=float(pngerr[mask].mean());need(math.isclose(rawL,row['raw_hole_L1'],rel_tol=0,abs_tol=1e-12) and math.isclose(pngL,row['PNG_hole_L1'],rel_tol=0,abs_tol=1e-14),'Witness hole endpoint mismatch');need(row['recomputed_delivered_hole_objective_unscaled']==row['raw_hole_L1'] and row['recomputed_once_divided_microbatch_equivalent']==row['raw_hole_L1']/4,'Witness objective equivalence mismatch');witnesses.append({'kind':kind,'cycle':cycle,'id':case['id'],'geometry':geometry,'raw_hole_L1':rawL,'PNG_hole_L1':pngL,'raw_replay_and_PNG_exact':True})
 gains={}
 for kind,screen in [('monitor',d['clear_incremental_vs_R1']),('challenge',d['challenge_incremental_vs_R1'])]:
  macro={i:sum(r['PNG_hole_L1'] for r in d['all_outputs'] if r['kind']==kind and r['cycle']==i)/sum(r['kind']==kind and r['cycle']==i for r in d['all_outputs']) for i in (0,64)};gain=100*(macro[0]-macro[64])/macro[0];need(math.isclose(gain,screen['gain_percent'],rel_tol=0,abs_tol=1e-12),'Macro gain arithmetic differs');need(screen['prospective_numeric_screen_pass'] is False and all('boundary' not in r['metric'] for r in screen['nonworse_failures']),'Gate/boundary status differs');gains[kind]=gain
 need(d['model_qualified'] is False and d['automatic_app_promotion'] is False and d['new_model_packaged'] is False and d['execution']['Torch_imported'] is False,'No false model/adoption/neural claim')
 report={'format':'independent-completion-V4-focused-artifact-closure-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'auditor':pin(Path(__file__)),'pixel_audit':pin(pixel),'pixel_script':pin(Path(d['auditor']['path'])),'state_audit':pin(state),'source_and_helper_semantic_pins_exact':True,'all256_FIT_loss_receipts_source_schema_not_raw_replay':True,'FIT_loss_numerical_limit_retained':semantics['actual_FIT_raw_not_saved'],'witness_replays':witnesses,'independent_witness_count':16,'macro_gain_percent_recomputed':gains,'all_boundary_groups_nonworse_but_hole_groups_fail':True,'all22_selected_R1_assets_independently_rehashed':True,'selected_R1_assets':sourcepins,'all846_sealed_payloads_rehashed_exact_and_namespace_closed':True,'actual_output_scope':{'outputs':224,'NPZ':228,'outside_RGB_bytes':38414802,'raw_replay_max_error':0,'PNG_replay_different_channels':0},'conclusion':'AuditedV4validstate_failedquality_keepR1;noactualFITrawtrajectoryclaim','local_execution':{'Torch_imported':False,'neural_calls':0,'gradients':0,'optimizer_operations':0,'VM_calls':0,'outputs_modified':False},'model_qualified':False,'seconds':time.monotonic()-start}
 target=RET/'independent_artifact_closure_check.json'
 with target.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
 print(json.dumps({'complete':True,'receipt':pin(target),'witnesses':16,'gains_percent':gains,'R1assets':22,'sealed_payloads':846,'seconds':report['seconds']},indent=2))
if __name__=='__main__':main()
