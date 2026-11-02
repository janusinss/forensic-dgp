"""Independent arithmetic audit of saved HQ TRAIN calibration; never loads a model."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/completion_codeformer_hq_train_probe_v1'
DEST=OUT/'independent_saved_output_audit.json'
PINS={'runner':(ROOT/'scripts/run_completion_codeformer_hq_train_probe_v1.py','e529058e48ab4f668801f5dbc0208ef465be963d0e8b6d993b3a8b8d683168ea'),'controller':(ROOT/'scripts/supervise_completion_codeformer_hq_train_probe_v1.py','33f357a98627994833a9162a8a25c3a3a03f597867e563f7dde3907e980cf08d'),'protocol':(OUT/'protocol.json','38fcd82360e547e32189d6887a3102182ec21a56692e728a632ce955311aaa0a'),'admission':(OUT/'independent_probe_admission.json','c072ca714d3ad9216cf6cfc22323c4affc90fe843ea97f5e11d91971a62ead13'),'root_input_review':(OUT/'root_input_review.json','baa3c4af1371b8f4acac9ebc90516445ff808285eb2657f02762b1f743fd8048')}


def need(ok,label):
 if not ok:raise ValueError(label)


def sha(p):
 with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def pin(p):
 p=Path(p).resolve();return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def verify(p):need(pin(Path(p['path']))==p,'Bound file drift '+p['path'])


def pixels(p,mode='RGB'):
 with Image.open(p) as im:return np.array(im.convert(mode))


def equal(a,b,label):need(a.dtype==b.dtype and a.shape==b.shape and np.array_equal(a,b),label)


def independent_metrics(completed,target,mask):
 if not mask.any():return {'hole_MAE':None,'gradient_edge_error':None,'boundary_edges':0}
 reference=target.astype(np.float32)/np.float32(255);difference=completed-reference
 xedge=mask[:,1:]!=mask[:,:-1];yedge=mask[1:,:]!=mask[:-1,:]
 dx=difference[:,1:,:]-difference[:,:-1,:];dy=difference[1:,:,:]-difference[:-1,:,:]
 vector=np.concatenate([np.abs(dx[xedge]).ravel(),np.abs(dy[yedge]).ravel()])
 return {'hole_MAE':float(np.abs(difference[mask]).mean()),'gradient_edge_error':float(vector.mean()),'boundary_edges':int(xedge.sum()+yedge.sum())}


def main():
 started=time.monotonic();need(not DEST.exists(),'Preserve completed output audit')
 before={k:pin(p) for k,(p,h) in PINS.items()}
 for k,(p,h) in PINS.items():need(before[k]['sha256']==h,'Changed pinned '+k)
 protocol=read(OUT/'protocol.json');admission=read(OUT/'independent_probe_admission.json');results=read(OUT/'results.json');supervisor=read(OUT/'supervisor.json');execution=read(OUT/'execution.json');progress=read(OUT/'progress.json')
 need(results['complete'] is True and supervisor['complete'] is True and supervisor['returncode']==0,'Run not completed successfully')
 need(not(OUT/'failure.json').exists() and not supervisor.get('timed_out',False) and not supervisor.get('interrupted',False),'Retained failure/timeout')
 need(admission['complete'] and admission['independent'] and admission['neural_forwards_authorized']==24,'Admission false')
 need(results['protocol_sha256']==execution['protocol_sha256']==supervisor['protocol_sha256']==PINS['protocol'][1],'Run protocol bindings')
 need(results['review_sha256']==execution['review_sha256']==PINS['admission'][1],'Run admission bindings')
 need(supervisor['worker_sha256']==PINS['runner'][1] and supervisor['root_input_review_sha256']==PINS['root_input_review'][1],'Supervisor source/input review bindings')
 need(0<results['seconds']<=600 and 0<supervisor['seconds']<=630,'Total runtime bounds')
 need(results['requests']==30 and results['neural_forwards']==24 and results['empty_bypasses']==6 and len(results['rows'])==30,'Run count receipts')
 need(not results['optimizer_constructed'] and results['optimizer_updates']==results['gradient_calls']==0 and not results['reserved_FINAL_used'] and not results['quality_qualified'],'Training/quality claims')
 need(results['state_unchanged'] and results['generator_state_before']==results['generator_state_after'],'Frozen state receipt')
 need(progress['phase']=='finished' and progress['complete'] and progress['forward_started']==progress['forward_finished']==24 and progress['finished_requests']==30,'Final progress counts')
 need(supervisor['training_calls']==0 and supervisor['minimum_RAM_certified'] is False and supervisor['runtime_is_local_CPU_calibration_not_training'],'Hardware/training qualification')
 verify(protocol['readiness']);verify(protocol['source_admission']);verify(protocol['checkpoint'])
 for record in protocol['source_files']:verify(record)
 for record in protocol['selected_sources']:
  need(record['original_role']==record['legacy_completion_split']=='train','NonTRAIN source');verify(record['source']);verify(record['target256'])
 need(protocol['budgets']['output_bytes']==167772160,'Output limit')
 need(not any(n=='pretrained_completion' or n.startswith('third_party.codeformer') for n in sys.modules),'Model code already imported')
 # Pure NumPy half-coordinate bilinear resize. No model/runtime import or neural replay.
 outputs=[];arrays=[];outsidebytes=0;emptyids=[];rawcount=0
 expectednames=set();metricrows=[]
 for number,(case,row) in enumerate(zip(protocol['cases'],results['rows']),1):
  name=case['id'];need(name==row['id'] and row['reference_id']==case['reference_id'] and row['mask_kind']==case['mask_kind'] and row['role']==case['role']=='TRAIN_calibration','Case identity/order')
  for field in ['source1024','target256','input256','mask256']:verify(case[field])
  source=pixels(case['input256']['path']);target=pixels(case['target256']['path']);mask8=pixels(case['mask256']['path'],'L');mask=mask8==255
  need(source.shape==target.shape==(256,256,3) and source.dtype==target.dtype==np.uint8 and mask8.shape==(256,256) and set(np.unique(mask8))<={0,255},'Input/mask shape')
  expectedinput=target.copy();expectedinput[mask]=255;equal(source,expectedinput,'Whitehole input differs')
  empty=not mask.any();need(empty==row['empty_bypass'] and int(mask.sum())==row['mask_pixels']==case['mask_pixels'] and row['neural_calls']==int(not empty),'Support/bypass counts')
  need(0<=row['seconds']<=120,'Case120s bound');need(read(OUT/('progress_%02d.json'%number))==row,'Per-case receipt differs')
  op=OUT/row['output'];st=OUT/row['stages'];need(op.resolve().is_relative_to(OUT.resolve()) and st.resolve().is_relative_to(OUT.resolve()),'Output escapes scope')
  need(sha(op)==row['output_sha256'] and sha(st)==row['stages_sha256'],'Output/stage hashes');final=pixels(op)
  with np.load(st,allow_pickle=False) as z:
   need(set(z.files)==({'input_rgb','target_rgb','removal_mask','completed256','output_rgb'} if empty else {'input_rgb','target_rgb','removal_mask','completed256','output_rgb','raw512'}),'Stage members')
   equal(z['input_rgb'],source,'Stage input');equal(z['target_rgb'],target,'Stage target');equal(z['removal_mask'],mask.astype(np.uint8),'Stage mask');equal(z['output_rgb'],final,'Stage PNGbytes');completed=z['completed256'].copy()
   need(completed.dtype==np.float32 and completed.shape==(256,256,3) and np.isfinite(completed).all() and completed.min()>=0 and completed.max()<=1,'Invalid completion array')
   if empty:equal(completed,source.astype(np.float32)/np.float32(255),'Empty completed float');equal(final,source,'Empty delivered PNG');emptyids.append(name)
   else:
    raw=z['raw512'].copy();need(raw.dtype==np.float32 and raw.shape==(512,512,3) and np.isfinite(raw).all(),'Invalid internal512')
    # 512->256 align_corners=False samples each2x2 block at itscenter.
    # Pair horizontalneighbors then verticalrows, matching declared4thread FP32 arithmetic.
    decoded=np.clip((raw+np.float32(1))/np.float32(2),np.float32(0),np.float32(1))
    top=decoded[0::2,0::2]+decoded[0::2,1::2]
    bottom=decoded[1::2,0::2]+decoded[1::2,1::2]
    resized=(top+bottom)*np.float32(0.25)
    reconstructed=source.astype(np.float32)/np.float32(255);reconstructed[mask]=resized[mask];equal(completed,reconstructed,'Exact savedraw512-to-composed256');rawcount+=1
  canonical=source.astype(np.float32)/np.float32(255);equal(completed[~mask],canonical[~mask],'Visible raw floats');expectedfinal=np.rint(completed*np.float32(255)).clip(0,255).astype(np.uint8);expectedfinal[~mask]=source[~mask];equal(final,expectedfinal,'RintPNG replay');equal(final[~mask],source[~mask],'Source byte preservation');equal(final[~mask],target[~mask],'Paired target known bytes')
  count=int((~mask).sum())*3;need(count==row['outside_source_bytes_exact'] and row['outside_changed_pixels']==0,'Outside byte count');outsidebytes+=count
  independent=independent_metrics(completed,target,mask);need(independent==row['paired_metrics'],'Paired metrics exact arithmetic');metricrows.append((name,independent,empty))
  outputs.append({'id':name,'reference_id':case['reference_id'],'mask_kind':case['mask_kind'],'role':'TRAIN_calibration','file':pin(op),'stages':pin(st),'empty_mask_bypass':empty,'reported_neural_calls':row['neural_calls'],'exact_outside_reviewed_mask':True,'raw512_to_completed256_to_PNG_exact':True,'outside_source_bytes':count,'paired_metrics':independent,'seconds_reported':row['seconds']});arrays.extend([pin(op),pin(st)]);expectednames.add(name)
 need(rawcount==24 and len(emptyids)==6,'Actual saved generated/bypass count')
 need({f.stem for f in (OUT/'images').iterdir()}==expectednames and {f.stem for f in (OUT/'stages').iterdir()}==expectednames,'Output/stage inventory')
 summary={}
 for metric in ['hole_MAE','gradient_edge_error']:
  values=[(n,m[metric]) for n,m,e in metricrows if not e];worst=max(values,key=lambda v:v[1]);summary[metric]={'mean':float(np.mean([v for n,v in values])),'worst':worst[0],'maximum':worst[1]}
 need(summary==results['paired_TRAIN_summary'],'Summary metric arithmetic')
 pages=[]
 for reference in protocol['selected_sources']:
  name=reference['reference_id'];path=OUT/'preview'/(name+'.png')
  with Image.open(path) as im:sheet=np.array(im.convert('RGB'));need(im.size==(1024,1148),'Output sheet geometry')
  cases=[c for c in protocol['cases'] if c['reference_id']==name and c['mask_pixels']];need(len(cases)==4,'Per-source4geometries')
  for slot,case in enumerate(cases):
   mask=pixels(case['mask256']['path'],'L');cells=[pixels(case['target256']['path']),pixels(case['input256']['path']),np.repeat(mask[:,:,None],3,axis=2),pixels(OUT/'images'/(case['id']+'.png'))]
   for col,expected in enumerate(cells):equal(sheet[28+slot*280:284+slot*280,col*256:(col+1)*256],expected,'Output sheet actualcell')
  pages.append({'file':pin(path),'actual_cells_exact':16});arrays.append(pin(path))
 need({f.stem for f in (OUT/'preview').iterdir()}=={r['reference_id'] for r in protocol['selected_sources']},'Six output sheet inventory')
 allbytes=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());need(allbytes<=167772160,'Output artifact160MiBcap')
 after={k:pin(p) for k,(p,h) in PINS.items()};need(before==after,'Dependency drift during arithmetic audit')
 for r in arrays:verify(r)
 need(not any(n=='pretrained_completion' or n.startswith('third_party.codeformer') for n in sys.modules),'Model module imported by audit')
 report={'format':'codeformer-HQ-TRAIN-paired-output-independent-audit-v1','complete':True,'independent':True,'audited_UTC':datetime.now(timezone.utc).isoformat(),'protocol':pin(OUT/'protocol.json'),'results':pin(OUT/'results.json'),'supervisor':pin(OUT/'supervisor.json'),'admission':pin(OUT/'independent_probe_admission.json'),'auditor':pin(Path(__file__)),'cases':30,'outputs':outputs,'raw512_to_composed256_rebuilds_exact':24,'PNGcompositions_exact':30,'empty_bypasses_exact':6,'empty_ids':emptyids,'outside_source_bytes_exact':outsidebytes,'paired_TRAIN_metrics_independently_recomputed':summary,'output_pages':pages,'sheet_cells_verified_exact':96,'source_roles':'Six existing originalTRAIN/legacycompletionTRAIN references only. NoVAL/FINAL pixels used.','actual_neural_calls_reported':24,'actual_neural_calls_evidence':'Pinned runner captures actual networkforward and requires24; immutable per-case receipts and terminalstate/report cross-checked. Auditor makes no neural calls.','state_integrity_evidence':{'before_reported':results['generator_state_before'],'after_reported':results['generator_state_after'],'reported_equal':True,'model_state_loaded_by_auditor':False},'runtime':{'worker_seconds_reported':results['seconds'],'supervisor_seconds_reported':supervisor['seconds'],'maximum_case_seconds_reported':max(r['seconds'] for r in results['rows']),'observed_elapsed_times_within_declared_numeric_bounds':True,'actual_worker_hard_timeout_boundary_verified':False,'launcher_pid':supervisor['child_pid'],'actual_worker_pid':execution['pid'],'launcher_worker_PID_mismatch':supervisor['child_pid']!=execution['pid'],'hardware_and_memory_original_receipt':supervisor,'model_RAM_measurement_valid':False,'memory_limit':'Original supervisor sampled the Windows venv launcher rather than the actual model worker; RSS/private/working-set values cannot measure model RAM. Successful elapsed observations remain valid, but owned-launcher termination does not establish termination of the actual worker. No rerun was performed.','minimum_RAM_certified':False},'all_output_bytes_before_audit':allbytes,'maximum_output_bytes':167772160,'pins_before':before,'pins_after':after,'arithmetical_resize_runtime':{'implementation':'Explicit NumPy FP32 2x2 half-coordinate bilinear reduction; horizontalrowpairs then verticalrowpair times0.25','torch_imported':False,'usage':'Savedtensor normalization/clamp/bilinearresize only; no network construction, checkpointload, inferencecall, gradients or modelsource import'},'initial_arithmetic_audit_stop':{'failed_source':pin(OUT/'audit_arithmetic_attempt1.py'),'receipt_published':False,'cause':'Initial auditor forced one Torch CPU thread; original probe usedfour. Torch bilinear accumulation differed by atmost5.960464477539063e-8. Separate saved-array-only diagnostic found fourthread replay and explicit NumPyrowpair reduction bitexact; finalaudit uses declared-arithmetic rowpair reconstruction without tolerance relaxation.','model_calls':0},'neural_calls_by_auditor':0,'model_loads':0,'gradient_calls':0,'training_calls':0,'VM_calls':0,'reserved_FINAL_used':False,'validation_pixels_used':False,'visual_quality_review_performed_by_auditor':False,'model_qualified':False,'generalization_or_DEV_qualification':False,'automatic_qualified':False,'assisted_fullfamily_qualified':False,'scope':'Integrity and paired proceduralTRAIN calibration only. Metric values do not establish hiddenidentity, wholefamilyquality or benefit of anytrainedchange. All6actualoutputpages require separate visualreview.','audit_seconds':time.monotonic()-started}
 with DEST.open('x',encoding='utf-8',newline='\n') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({'audit':pin(DEST),'cases':30,'raw_rebuilds':24,'empty_exact':6,'cells':96,'outside_source_bytes':outsidebytes,'metrics':summary,'audit_seconds':report['audit_seconds'],'model_calls':0},indent=2))


if __name__=='__main__':main()
